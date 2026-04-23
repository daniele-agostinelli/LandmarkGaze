import logging
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig

from camera import Camera
from face_landmark_estimator import LandmarkEstimator
from face_model import Face, FaceModel
from normalization_utils import LandmarkNormalizer
from Train_MLP import Config as TrainConfig
from Train_MLP import GazeNetVector

logger = logging.getLogger(__name__)


class GazeEstimatorNormalized:
    """
    Estimates gaze using normalized landmark coordinates.
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
        device = torch.device(self._config.device)

        try:
            try:
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
            except TypeError:
                state_dict = torch.load(model_path, map_location=device)

            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            if isinstance(state_dict, dict) and all(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            if not isinstance(state_dict, dict):
                raise RuntimeError("Loaded checkpoint is not a state_dict dictionary.")

            input_size, hidden_width, num_blocks = self._infer_model_shape_from_state_dict(state_dict)
            model = GazeNetVector(
                input_size=input_size,
                output_size=3,
                hidden_width=hidden_width,
                num_blocks=num_blocks,
                dropout_rate=0.0,
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
            logger.info(
                "Loaded MLP checkpoint with inferred shape: input_size=%d, hidden_width=%d, blocks=%d",
                input_size,
                hidden_width,
                num_blocks,
            )
            return model
        except FileNotFoundError as e:
            logger.error("Model file not found: %s", model_path)
            raise e

    def _infer_model_shape_from_state_dict(self, state_dict):
        key_in = "input_layer.0.weight"
        if key_in not in state_dict:
            raise RuntimeError(f"Checkpoint missing required key for MLP shape inference: {key_in}")

        hidden_width = int(state_dict[key_in].shape[0])
        input_size = int(state_dict[key_in].shape[1])

        block_ids = set()
        for key in state_dict.keys():
            if not key.startswith("res_blocks."):
                continue
            parts = key.split(".")
            if len(parts) < 2:
                continue
            try:
                block_ids.add(int(parts[1]))
            except ValueError:
                continue

        num_blocks = (max(block_ids) + 1) if block_ids else int(TrainConfig.NUM_BLOCKS)
        return input_size, hidden_width, num_blocks

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        if self._landmark_estimator is None:
            raise RuntimeError("LandmarkEstimator is not available in this environment.")
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, face: Face) -> None:
        self.face_model3d.estimate_head_pose(face, self.camera)
        if face.head_pose_rot is None or face.head_position is None:
            logger.error("Head pose not estimated. Cannot proceed with gaze estimation.")
            return

        tvec = face.head_position
        head_rot_mat = face.head_pose_rot.as_matrix()
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

    @torch.no_grad()
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

        device = torch.device(self._config.device)
        input_tensor = torch.tensor(features).to(device).unsqueeze(0)
        g_n = self._gaze_estimation_model(input_tensor).cpu().numpy().flatten()
        g_n /= np.linalg.norm(g_n)
        return g_n

    def update_camera_parameters(self, width, height, camera_matrix, dist_coeffs=None):
        self.camera.width = width
        self.camera.height = height
        self.camera.camera_matrix = camera_matrix
        if dist_coeffs is not None:
            self.camera.dist_coefficients = dist_coeffs
