import logging
import os
from typing import List

import cv2
import joblib
import numpy as np
from omegaconf import DictConfig

from camera import Camera
from face_landmark_estimator import LandmarkEstimator
from face_model import Face, FaceModel
from normalization_utils import LandmarkNormalizer
from Train_MLP import Config as TrainConfig

logger = logging.getLogger(__name__)


class GazeEstimatorXGBoost:
    """
    Estimates gaze using the XGBoost regressor.
    """

    def __init__(self, config: DictConfig):
        self._config = config
        self.face_model3d = FaceModel()
        self.camera = Camera(config.gaze_estimator.camera_params)
        self._landmark_estimator = None
        try:
            self._landmark_estimator = LandmarkEstimator(config)
        except Exception as e:
            logger.warning(
                "LandmarkEstimator initialization failed (%s). "
                "CSV-based inference remains available; face detection routines are disabled.",
                e,
            )

        self._normalized_camera = Camera(self._config.gaze_estimator.normalized_camera_params)
        self.scale_factor = self._normalized_camera.width
        self.normalizer = LandmarkNormalizer(self._normalized_camera)

        self.landmark_indices = TrainConfig.LANDMARK_INDICES
        self.left_inner_idx = self.landmark_indices.index(TrainConfig.LEFT_INNER_CORNER)
        self.left_outer_idx = self.landmark_indices.index(TrainConfig.LEFT_OUTER_CORNER)
        self.right_inner_idx = self.landmark_indices.index(TrainConfig.RIGHT_INNER_CORNER)
        self.right_outer_idx = self.landmark_indices.index(TrainConfig.RIGHT_OUTER_CORNER)
        self.num_landmarks = len(self.landmark_indices)
        self._gaze_estimation_model = self._load_model()

    def _load_model(self):
        model_path = self._config.gaze_estimator.model_path
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"XGBoost model file not found at: {model_path}")

            logger.info("Loading XGBoost model from %s", model_path)
            return joblib.load(model_path)
        except Exception as e:
            logger.error("Failed to load XGBoost model: %s", e)
            raise e

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        if self._landmark_estimator is None:
            raise RuntimeError("LandmarkEstimator is not available in this environment.")
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, face: Face) -> None:
        self.face_model3d.estimate_head_pose(face, self.camera)
        if face.head_pose_rot is None or face.head_position is None:
            return

        rvec = face.head_pose_rot.as_rotvec()
        tvec = face.head_position
        head_rot_mat, _ = cv2.Rodrigues(rvec)

        warp_matrix, rotation_norm = self.normalizer.compute_normalization_matrix(
            head_rot_mat,
            tvec,
            self.camera.camera_matrix,
        )

        all_landmarks = np.concatenate([face.landmarks, face.landmarks_eyes])
        target_landmarks = all_landmarks[self.landmark_indices].astype(np.float32)
        normalized_landmarks = self.normalizer.normalize_landmarks(target_landmarks, warp_matrix)

        g_n = self.estimate_norm_gaze_from_norm_lmks(normalized_landmarks)
        g_cam = rotation_norm.T @ g_n
        g_cam /= np.linalg.norm(g_cam)

        face.normalized_gaze_vector = g_n
        face.gaze_vector = g_cam
        face.gaze_vector_head = head_rot_mat.T @ g_cam

        self.face_model3d.compute_3d_pose(face)
        self.face_model3d.compute_face_eye_centers(face)

    def estimate_norm_gaze_from_norm_lmks(self, normalized_landmarks):
        xs = normalized_landmarks[:, 0]
        ys = normalized_landmarks[:, 1]

        xs_eye = normalized_landmarks[
            [self.left_inner_idx, self.left_outer_idx, self.right_inner_idx, self.right_outer_idx],
            0,
        ]
        ys_eye = normalized_landmarks[
            [self.left_inner_idx, self.left_outer_idx, self.right_inner_idx, self.right_outer_idx],
            1,
        ]
        centroid_x = np.mean(xs_eye)
        centroid_y = np.mean(ys_eye)
        xs_norm = (xs - centroid_x) / self.scale_factor
        ys_norm = (ys - centroid_y) / self.scale_factor

        features = np.empty((len(xs) * 2,), dtype=np.float32)
        features[0::2] = xs_norm
        features[1::2] = ys_norm

        g_n = self._gaze_estimation_model.predict(features.reshape(1, -1)).flatten()
        g_n /= np.linalg.norm(g_n)
        return g_n

    def update_camera_parameters(self, width, height, camera_matrix, dist_coeffs=None):
        self.camera.width = width
        self.camera.height = height
        self.camera.camera_matrix = camera_matrix
        if dist_coeffs is not None:
            self.camera.dist_coefficients = dist_coeffs
