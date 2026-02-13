import logging
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig

from camera import Camera
from face_landmark_estimator import LandmarkEstimator
from face_model import Face, FaceModel
from normalization_utils import LandmarkNormalizer

# Import the Siamese model and training config
from Train_siameseMLP import SiameseGazeNet, Config as SiameseTrainConfig

logger = logging.getLogger(__name__)


class GazeEstimatorNormalizedSiamese:
    """
    Estimates gaze using the Siamese architecture (Left/Right eye independence).
    
    This class mirrors the structure of GazeEstimatorNormalized but prepares
    input features for the SiameseGazeNet (two separate branches for eyes)
    instead of the single GazeNetVector.
    """

    def __init__(self, config: DictConfig):
        self._config = config
        self.face_model3d = FaceModel()
        self.camera = Camera(config.gaze_estimator.camera_params)
        self._landmark_estimator = LandmarkEstimator(config)

        # Standard Normalization Params (Must match the Generator/Training Normalization)
        self._normalized_camera = Camera(self._config.gaze_estimator.normalized_camera_params)
        self.scale_factor = self._normalized_camera.width
        # Initialize Shared Normalizer
        self.normalizer = LandmarkNormalizer(self._normalized_camera)

        # Load Landmark Indices from the Siamese Training Config
        self.left_indices = SiameseTrainConfig.LEFT_EYE_LANDMARKS
        self.right_indices = SiameseTrainConfig.RIGHT_EYE_LANDMARKS
        self.head_indices = SiameseTrainConfig.HEAD_ANCHORS
        self.landmark_indices = self.left_indices + self.right_indices + self.head_indices

        # Indices of eye corners WITHIN the left/right landmarks
        self.left_inner_idx = self.left_indices.index(SiameseTrainConfig.LEFT_INNER_CORNER) 
        self.left_outer_idx = self.left_indices.index(SiameseTrainConfig.LEFT_OUTER_CORNER)
        self.right_inner_idx = self.right_indices.index(SiameseTrainConfig.RIGHT_INNER_CORNER)
        self.right_outer_idx = self.right_indices.index(SiameseTrainConfig.RIGHT_OUTER_CORNER)

        self._gaze_estimation_model = self._load_model()

    def _load_model(self):
        try:
            model_path = self._config.gaze_estimator.model_path
            
            # Calculate input size per eye based on landmarks count * 2 (x, y)
            input_size_per_eye = len(self.left_indices) * 2

            # Initialize the Siamese Model
            model = SiameseGazeNet(
                input_size_per_eye=input_size_per_eye,
                config=SiameseTrainConfig
            )

            device = torch.device(self._config.device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {model_path}")
            raise e

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        """Detects faces and their landmarks from an image."""
        return self._landmark_estimator.detect_faces(image)

    def _process_eye_features(self, landmarks_norm, inner_pt, outer_pt):
        """
        Normalizes a specific set of landmarks relative to eye corners.
        
        This duplicates the logic from 'SiameseGazeDataset._process_eye' 
        in ModelTrainingNormalized3_siamese.py.
        """
        xs = landmarks_norm[:, 0]
        ys = landmarks_norm[:, 1]

        # Calculate local centroid (center of the eye)
        centroid_x = (inner_pt[0] + outer_pt[0]) / 2.0
        centroid_y = (inner_pt[1] + outer_pt[1]) / 2.0

        # Normalize: Center at (0,0) and scale
        xs_norm = (xs - centroid_x) / self.scale_factor
        ys_norm = (ys - centroid_y) / self.scale_factor

        # Interleave [x1, y1, x2, y2...]
        features = np.empty((len(xs) * 2,), dtype=np.float32)
        features[0::2] = xs_norm
        features[1::2] = ys_norm
        
        return features, (centroid_x, centroid_y), self.scale_factor 

    def estimate_gaze(self, face: Face) -> None:
        # 1. Estimate Head Pose (Standard OpenCV PnP)
        self.face_model3d.estimate_head_pose(face, self.camera)

        if face.head_pose_rot is None or face.head_position is None:
            logger.error("Head pose not estimated. Cannot proceed with gaze estimation.")
            return

        # PnP Vectors
        tvec = face.head_position
        head_rot_mat = face.head_pose_rot.as_matrix()

        # 2. Compute Normalization Matrix
        # This computes the Matrix M that warps the image/landmarks to the normalized camera view
        M, R_norm = self.normalizer.compute_normalization_matrix(
            head_rot_mat, tvec, self.camera.camera_matrix
        )

        # 3. Normalize Landmarks
        # Construct the full landmark set to index into (Mesh + Iris)
        all_landmarks = np.concatenate([face.landmarks, face.landmarks_eyes])
        target_landmarks = all_landmarks[self.landmark_indices].astype(np.float32)
        normalized_landmarks = self.normalizer.normalize_landmarks(target_landmarks, M)

        # 4 Estimate normalized gaze vector from selected normalized landmarks
        g_n = self.estimate_norm_gaze_from_norm_lmks(self, normalized_landmarks)

        # 5. Denormalize
        # Transform Normalized Frame -> Camera Frame (OpenCV)
        # v_norm = R_norm @ v_cam  =>  v_cam = R_norm.T @ v_norm
        g_cam = R_norm.T @ g_n
        g_cam /= np.linalg.norm(g_cam)

        # 6. Store results
        face.normalized_gaze_vector = g_n
        face.gaze_vector = g_cam

        # Head Relative (OpenCV Head -> OpenCV Gaze)
        face.gaze_vector_head = head_rot_mat.T @ g_cam

        self.face_model3d.compute_3d_pose(face)
        self.face_model3d.compute_face_eye_centers(face)

    @torch.no_grad()
    def estimate_norm_gaze_from_norm_lmks(self, normalized_landmarks):
        Nl = len(self.left_indices)
        Nr = len(self.right_indices)

        left_norm_coords  = normalized_landmarks[:Nl,:]
        right_norm_coords = normalized_landmarks[Nl:Nl+Nr,:]
        head_norm_coords = normalized_landmarks[Nl+Nr:,:]
        
        left_inner_corner = left_norm_coords[self.left_inner_idx]
        left_outer_corner = left_norm_coords[self.left_outer_idx]
        right_inner_corner = right_norm_coords[self.right_inner_idx]
        right_outer_corner = right_norm_coords[self.right_outer_idx]
        
        # 1. Feature Pre-processing (Siamese Style)
        # Center and scale each eye independently
        feat_left, l_center, l_scale  = self._process_eye_features(left_norm_coords,left_inner_corner,left_outer_corner)
        feat_right, r_center, r_scale = self._process_eye_features(right_norm_coords,right_inner_corner,right_outer_corner)

        # --- Calculate Relative Position Vector ---
        # Vector from Left Eye Center to Right Eye Center
        # Normalized by the average scale of the eyes to keep units consistent
        avg_scale = (l_scale + r_scale) / 2.0
        delta_x = (r_center[0] - l_center[0]) / avg_scale
        delta_y = (r_center[1] - l_center[1]) / avg_scale
        
        relative_pos = np.array([delta_x, delta_y], dtype=np.float32)

        # Head anchors
        Hx_norm = (head_norm_coords[:, 0] - (l_center[0]+r_center[0])/2.) / avg_scale
        Hy_norm = (head_norm_coords[:, 1] - (l_center[1]+r_center[1])/2.) / avg_scale

        # Interleave features [x1, y1, x2, y2...]
        feat_head = np.empty((len(self.head_indices) * 2,), dtype=np.float32)
        feat_head[0::2] = Hx_norm
        feat_head[1::2] = Hy_norm
        
        # 2. Inference
        device = torch.device(self._config.device)
        
        # Create batches of size 1
        input_l = torch.tensor(feat_left).unsqueeze(0).to(device)
        input_r = torch.tensor(feat_right).unsqueeze(0).to(device)
        input_rel = torch.tensor(relative_pos).unsqueeze(0).to(device)
        input_h = torch.tensor(feat_head).unsqueeze(0).to(device)

        # Predict Gaze (g_n) - Vector in the Normalized Frame
        # Pass both eye inputs to the Siamese model
        g_n = self._gaze_estimation_model(input_l, input_r, input_rel, input_h).cpu().numpy().flatten()
        g_n /= np.linalg.norm(g_n)
        
        return g_n


    def update_camera_parameters(self, width, height, camera_matrix, dist_coeffs=None):
        """
        Updates the camera intrinsics.
        """
        self.camera.width = width
        self.camera.height = height
        self.camera.camera_matrix = camera_matrix
        if dist_coeffs is not None:
            self.camera.dist_coefficients = dist_coeffs
