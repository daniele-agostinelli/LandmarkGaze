import logging
from typing import List

import cv2
import numpy as np
import joblib
import os
from omegaconf import DictConfig

# Import existing dependencies to maintain project structure
from camera import Camera
from face_landmark_estimator import LandmarkEstimator
from face_model import Face, FaceModel
from Train_MLP import Config as TrainConfig
from normalization_utils import LandmarkNormalizer

logger = logging.getLogger(__name__)


class GazeEstimatorXGBoost:
    """
    Estimates gaze using the XGBoost model instead of the Neural Network.
    Maintains exact scientific compatibility with the normalization method.
    """

    def __init__(self, config: DictConfig):
        self._config = config
        self.face_model3d = FaceModel()
        self.camera = Camera(config.gaze_estimator.camera_params)
        self._landmark_estimator = LandmarkEstimator(config)

        # Standard Normalization Params (Must match Generator and Training)
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

        # Load the XGBoost model
        self._gaze_estimation_model = self._load_model()

    def _load_model(self):
        try:
            # Look for the model in the models directory
            model_path = self._config.gaze_estimator.model_path

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"XGBoost model file not found at: {model_path}")

            logger.info(f"Loading XGBoost model from {model_path}")
            model = joblib.load(model_path)
            return model

        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            raise e

    def detect_faces(self, image: np.ndarray)-> List[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, face: Face) -> None:
        # 1. Estimate Head Pose (Standard OpenCV PnP)
        self.face_model3d.estimate_head_pose(face, self.camera)

        if face.head_pose_rot is None or face.head_position is None:
            return

        # PnP Vectors (OpenCV Frame)
        rvec = face.head_pose_rot.as_rotvec()
        tvec = face.head_position
        head_rot_mat, _ = cv2.Rodrigues(rvec)

        # 2. Compute Normalization Matrix
        M, R_norm = self.normalizer.compute_normalization_matrix(
            head_rot_mat, tvec, self.camera.camera_matrix
        )

        # 3. Normalize Landmarks
        all_landmarks = np.concatenate([face.landmarks, face.landmarks_eyes])
        target_landmarks = all_landmarks[self.landmark_indices].astype(np.float32)
        normalized_landmarks = self.normalizer.normalize_landmarks(target_landmarks, M)

        g_n = self.estimate_norm_gaze_from_norm_lmks(normalized_landmarks)

        # 6. Denormalize
        # Transform Normalized Frame -> Camera Frame (OpenCV)
        # v_norm = R_norm @ v_cam  =>  v_cam = R_norm.T @ v_norm
        g_cam = R_norm.T @ g_n
        g_cam /= np.linalg.norm(g_cam)

        # 7. Store Result
        face.normalized_gaze_vector = g_n
        face.gaze_vector = g_cam

        # Head Relative (OpenCV Head -> OpenCV Gaze)
        face.gaze_vector_head = head_rot_mat.T @ g_cam
        
        self.face_model3d.compute_3d_pose(face)
        self.face_model3d.compute_face_eye_centers(face)

    def estimate_norm_gaze_from_norm_lmks(self, normalized_landmarks):
        # 1. Feature Pre-processing
        xs = normalized_landmarks[:, 0]
        ys = normalized_landmarks[:, 1]

        # --- IMPORTANT: Exact same scaling as in Training ---
        xs_eye = normalized_landmarks[[self.left_inner_idx,self.left_outer_idx,self.right_inner_idx,self.right_outer_idx], 0]
        ys_eye = normalized_landmarks[[self.left_inner_idx,self.left_outer_idx,self.right_inner_idx,self.right_outer_idx], 1]
        centroid_x = np.mean(xs_eye)
        centroid_y = np.mean(ys_eye)
        xs_norm = (xs - centroid_x) / self.scale_factor
        ys_norm = (ys - centroid_y) / self.scale_factor

        features = np.empty((len(xs) * 2,), dtype=np.float32)
        features[0::2] = xs_norm # Even columns are X
        features[1::2] = ys_norm # Odd columns are Y

        # 2. Inference (XGBoost)
        # XGBoost via sklearn expects (N_samples, N_features)
        input_features = features.reshape(1, -1)

        # Predict Gaze (g_n) - Vector in the Normalized Frame
        # Returns shape (1, 3) -> flatten to (3,)
        g_n = self._gaze_estimation_model.predict(input_features).flatten()
        # Normalize vector (Model predicts direction, but magnitude might not be perfectly 1.0)
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
