import logging
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig

from camera import Camera
from face_landmark_estimator import LandmarkEstimator
from face_model import Face, FaceModel
from normalization_utils import LandmarkNormalizer
from Train_siameseMLP import Config as SiameseTrainConfig
from Train_siameseMLP import SiameseGazeNet

logger = logging.getLogger(__name__)


def _resolve_camera_params_path(config: DictConfig) -> str:
    camera_params = config.gaze_estimator.get("camera_params", None)
    if camera_params is not None:
        return camera_params

    normalized_camera_params = config.gaze_estimator.get("normalized_camera_params", None)
    if normalized_camera_params is None:
        raise ValueError(
            "Both gaze_estimator.camera_params and normalized_camera_params are missing."
        )

    logger.info(
        "gaze_estimator.camera_params is missing; falling back to normalized_camera_params."
    )
    return normalized_camera_params


class GazeEstimatorNormalizedSiamese:
    """
    Estimates gaze using the Siamese landmark architecture.
    """

    def __init__(self, config: DictConfig):
        self._config = config
        self.face_model3d = FaceModel()
        self.camera = Camera(_resolve_camera_params_path(config))
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

        self.left_indices = SiameseTrainConfig.LEFT_EYE_LANDMARKS
        self.right_indices = SiameseTrainConfig.RIGHT_EYE_LANDMARKS
        self.head_indices = SiameseTrainConfig.HEAD_ANCHORS
        self.landmark_indices = self.left_indices + self.right_indices + self.head_indices

        self.left_inner_idx = self.left_indices.index(SiameseTrainConfig.LEFT_INNER_CORNER)
        self.left_outer_idx = self.left_indices.index(SiameseTrainConfig.LEFT_OUTER_CORNER)
        self.right_inner_idx = self.right_indices.index(SiameseTrainConfig.RIGHT_INNER_CORNER)
        self.right_outer_idx = self.right_indices.index(SiameseTrainConfig.RIGHT_OUTER_CORNER)

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
            if not isinstance(state_dict, dict):
                raise RuntimeError("Loaded checkpoint is not a state_dict dictionary.")
            if all(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

            input_size_per_eye, model_cfg = self._infer_model_shape_from_state_dict(state_dict)
            model = SiameseGazeNet(
                input_size_per_eye=input_size_per_eye,
                config=model_cfg,
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
            logger.info(
                "Loaded Siamese checkpoint with inferred shape: input_per_eye=%d, branch_hidden=%d, fusion_hidden=%d, blocks=%d",
                input_size_per_eye,
                model_cfg.BRANCH_HIDDEN_WIDTH,
                model_cfg.FUSION_HIDDEN_WIDTH,
                model_cfg.NUM_BLOCKS,
            )
            return model
        except FileNotFoundError as e:
            logger.error("Model file not found: %s", model_path)
            raise e

    def _infer_model_shape_from_state_dict(self, state_dict):
        key_in = "left_branch.input.0.weight"
        key_fusion = "fusion.0.weight"
        if key_in not in state_dict or key_fusion not in state_dict:
            raise RuntimeError(
                "Checkpoint missing required keys for Siamese shape inference: "
                f"{key_in}, {key_fusion}"
            )

        input_size_per_eye = int(state_dict[key_in].shape[1])
        branch_hidden = int(state_dict[key_in].shape[0])
        fusion_hidden = int(state_dict[key_fusion].shape[0])

        block_ids = set()
        for key in state_dict.keys():
            if not key.startswith("left_branch.blocks."):
                continue
            parts = key.split(".")
            if len(parts) < 3:
                continue
            try:
                block_ids.add(int(parts[2]))
            except ValueError:
                continue
        num_blocks = (max(block_ids) + 1) if block_ids else int(SiameseTrainConfig.NUM_BLOCKS)

        model_cfg = SiameseTrainConfig()
        model_cfg.BRANCH_HIDDEN_WIDTH = branch_hidden
        model_cfg.FUSION_HIDDEN_WIDTH = fusion_hidden
        model_cfg.NUM_BLOCKS = num_blocks
        model_cfg.DROPOUT_RATE = float(SiameseTrainConfig.DROPOUT_RATE)
        return input_size_per_eye, model_cfg

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        if self._landmark_estimator is None:
            raise RuntimeError("LandmarkEstimator is not available in this environment.")
        return self._landmark_estimator.detect_faces(image)

    def _process_eye_features(self, landmarks_norm, inner_pt, outer_pt):
        xs = landmarks_norm[:, 0]
        ys = landmarks_norm[:, 1]

        centroid_x = (inner_pt[0] + outer_pt[0]) / 2.0
        centroid_y = (inner_pt[1] + outer_pt[1]) / 2.0
        xs_norm = (xs - centroid_x) / self.scale_factor
        ys_norm = (ys - centroid_y) / self.scale_factor

        features = np.empty((len(xs) * 2,), dtype=np.float32)
        features[0::2] = xs_norm
        features[1::2] = ys_norm
        return features, (centroid_x, centroid_y), self.scale_factor

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
        left_count = len(self.left_indices)
        right_count = len(self.right_indices)

        left_norm_coords = normalized_landmarks[:left_count, :]
        right_norm_coords = normalized_landmarks[left_count : left_count + right_count, :]
        head_norm_coords = normalized_landmarks[left_count + right_count :, :]

        left_inner_corner = left_norm_coords[self.left_inner_idx]
        left_outer_corner = left_norm_coords[self.left_outer_idx]
        right_inner_corner = right_norm_coords[self.right_inner_idx]
        right_outer_corner = right_norm_coords[self.right_outer_idx]

        feat_left, l_center, l_scale = self._process_eye_features(
            left_norm_coords,
            left_inner_corner,
            left_outer_corner,
        )
        feat_right, r_center, r_scale = self._process_eye_features(
            right_norm_coords,
            right_inner_corner,
            right_outer_corner,
        )

        avg_scale = (l_scale + r_scale) / 2.0
        relative_pos = np.array(
            [
                (r_center[0] - l_center[0]) / avg_scale,
                (r_center[1] - l_center[1]) / avg_scale,
            ],
            dtype=np.float32,
        )

        head_xn = (head_norm_coords[:, 0] - (l_center[0] + r_center[0]) / 2.0) / avg_scale
        head_yn = (head_norm_coords[:, 1] - (l_center[1] + r_center[1]) / 2.0) / avg_scale

        feat_head = np.empty((len(self.head_indices) * 2,), dtype=np.float32)
        feat_head[0::2] = head_xn
        feat_head[1::2] = head_yn

        device = torch.device(self._config.device)
        input_l = torch.tensor(feat_left).unsqueeze(0).to(device)
        input_r = torch.tensor(feat_right).unsqueeze(0).to(device)
        input_rel = torch.tensor(relative_pos).unsqueeze(0).to(device)
        input_h = torch.tensor(feat_head).unsqueeze(0).to(device)

        g_n = self._gaze_estimation_model(input_l, input_r, input_rel, input_h).cpu().numpy().flatten()
        g_n /= np.linalg.norm(g_n)
        return g_n

    def update_camera_parameters(self, width, height, camera_matrix, dist_coeffs=None):
        self.camera.width = width
        self.camera.height = height
        self.camera.camera_matrix = camera_matrix
        if dist_coeffs is not None:
            self.camera.dist_coefficients = dist_coeffs
