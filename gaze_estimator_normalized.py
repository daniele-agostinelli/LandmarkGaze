import logging
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig

from camera import Camera
from face_landmark_estimator import LandmarkEstimator
from face_model import Face, FaceModel
from Train_MLP import GazeNetVector, Config as TrainConfig
from normalization_utils import LandmarkNormalizer

logger = logging.getLogger(__name__)


class GazeEstimatorNormalized:
    """
    Estimates gaze using the Data Normalization method.
    Uses 'normalization_utils' to share exact math with dataset generation.
    """

    def __init__(self, config: DictConfig):
        self._config = config
        self.face_model3d = FaceModel()
        self.camera = Camera(config.gaze_estimator.camera_params)
        self._landmark_estimator = LandmarkEstimator(config)

        # Standard Normalization Params (Must match Generator)
        self._normalized_camera = Camera(self._config.gaze_estimator.normalized_camera_params)
        self.scale_factor = self._normalized_camera.width
        # Initialize Shared Normalizer
        self.normalizer = LandmarkNormalizer(self._normalized_camera)

        self.landmark_indices = TrainConfig.LANDMARK_INDICES
        # Indices of eye corners WITHIN the selected landmarks
        self.left_inner_idx = self.landmark_indices.index(TrainConfig.LEFT_INNER_CORNER) 
        self.left_outer_idx = self.landmark_indices.index(TrainConfig.LEFT_OUTER_CORNER)
        self.right_inner_idx = self.landmark_indices.index(TrainConfig.RIGHT_INNER_CORNER)
        self.right_outer_idx = self.landmark_indices.index(TrainConfig.RIGHT_OUTER_CORNER)
        
        self.num_landmarks = len(self.landmark_indices)
        self._gaze_estimation_model = self._load_model()

    def _load_model(self):
        try:
            model_path = self._config.gaze_estimator.model_path
            input_size = self.num_landmarks * 2

            model = GazeNetVector(
                input_size=input_size,
                output_size=3,
                hidden_width=TrainConfig.HIDDEN_WIDTH,
                num_blocks=TrainConfig.NUM_BLOCKS,
                dropout_rate=0.0
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

    def detect_faces(self, image: np.ndarray)-> List[Face]:
        """Detects faces and their landmarks from an image."""
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, face: Face) -> None:
        # 1. Estimate Head Pose (Standard OpenCV PnP)
        self.face_model3d.estimate_head_pose(face, self.camera)

        if face.head_pose_rot is None or face.head_position is None:
            logger.error("Head pose not estimated. Cannot proceed with gaze estimation.")
            return

        # PnP Vectors (OpenCV Frame)
        #rvec = face.head_pose_rot.as_rotvec()
        tvec = face.head_position
        head_rot_mat = face.head_pose_rot.as_matrix() #cv2.Rodrigues(rvec)

        # 2. Compute Normalization Matrix
        # Using the clean OpenCV coordinates directly
        M, R_norm = self.normalizer.compute_normalization_matrix(
            head_rot_mat, tvec, self.camera.camera_matrix
        )

        # 3. Normalize Landmarks
        all_landmarks = np.concatenate([face.landmarks, face.landmarks_eyes])
        target_landmarks = all_landmarks[self.landmark_indices].astype(np.float32)
        normalized_landmarks = self.normalizer.normalize_landmarks(target_landmarks, M)

        # 4. Estimate normalized gaze vector from selected normalized landmarks
        g_n = self.estimate_norm_gaze_from_norm_lmks(self, normalized_landmarks)

        # 5. Denormalize
        # Transform Normalized Frame -> Camera Frame (OpenCV)
        # v_norm = R_norm @ v_cam  =>  v_cam = R_norm.T @ v_norm
        g_cam = R_norm.T @ g_n
        g_cam /= np.linalg.norm(g_cam)

        # 6. Store Results
        face.normalized_gaze_vector = g_n
        face.gaze_vector = g_cam

        # Head Relative (OpenCV Head -> OpenCV Gaze)
        face.gaze_vector_head = head_rot_mat.T @ g_cam

        self.face_model3d.compute_3d_pose(face)
        self.face_model3d.compute_face_eye_centers(face)

    @torch.no_grad()
    def estimate_norm_gaze_from_norm_lmks(self, normalized_landmarks):
        # 1. Feature Pre-processing
        xs = normalized_landmarks[:, 0]
        ys = normalized_landmarks[:, 1]
        
        xs_eye = normalized_landmarks[[self.left_inner_idx,self.left_outer_idx,self.right_inner_idx,self.right_outer_idx], 0]
        ys_eye = normalized_landmarks[[self.left_inner_idx,self.left_outer_idx,self.right_inner_idx,self.right_outer_idx], 1]
        centroid_x = np.mean(xs_eye)
        centroid_y = np.mean(ys_eye)
        xs_norm = (xs - centroid_x) / self.scale_factor
        ys_norm = (ys - centroid_y) / self.scale_factor

        features = np.empty((len(xs) * 2,), dtype=np.float32)
        features[0::2] = xs_norm
        features[1::2] = ys_norm

        # 2. Inference
        device = torch.device(self._config.device)
        input_tensor = torch.tensor(features).to(device).unsqueeze(0)

        # Predict Gaze (g_n) - Vector in the Normalized Frame
        g_n = self._gaze_estimation_model(input_tensor).cpu().numpy().flatten()
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
