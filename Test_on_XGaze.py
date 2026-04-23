import csv
import datetime
import logging
import os

import numpy as np
from omegaconf import OmegaConf

try:
    from gaze_estimator_normalized import GazeEstimatorNormalized
except ImportError:
    print("Warning: Could not import GazeEstimatorNormalized.")
    GazeEstimatorNormalized = None

try:
    from gaze_estimator_normalized_XGBoost import GazeEstimatorXGBoost as GazeEstimatorNormalizedXGBoost
except ImportError:
    print("Warning: Could not import GazeEstimatorNormalizedXGBoost.")
    GazeEstimatorNormalizedXGBoost = None

try:
    from gaze_estimator_normalized_siamese import GazeEstimatorNormalizedSiamese
except ImportError:
    print("Warning: Could not import GazeEstimatorNormalizedSiamese.")
    GazeEstimatorNormalizedSiamese = None

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class CSVBenchmarkRunner:
    def __init__(self, config, model_def, output_csv_path=None, summary_csv_path=None):
        self.config = config
        self.model_type = model_def["type"]
        self.model_name = model_def["name"]
        self.output_csv_path = output_csv_path
        self.summary_csv_path = summary_csv_path

        logger.info("Loading model: %s (%s)", self.model_name, self.model_type)
        self.config.gaze_estimator.model_path = model_def["path"]

        if self.model_type == "normalized":
            if GazeEstimatorNormalized is None:
                raise ImportError("GazeEstimatorNormalized not found.")
            self.estimator = GazeEstimatorNormalized(self.config)
        elif self.model_type == "normalized_xgboost":
            if GazeEstimatorNormalizedXGBoost is None:
                raise ImportError("GazeEstimatorNormalizedXGBoost not found.")
            self.estimator = GazeEstimatorNormalizedXGBoost(self.config)
        elif self.model_type == "normalized_siamese":
            if GazeEstimatorNormalizedSiamese is None:
                raise ImportError("GazeEstimatorNormalizedSiamese not found.")
            self.estimator = GazeEstimatorNormalizedSiamese(self.config)
        else:
            raise ValueError(f"Unsupported model type for CSV benchmarking: {self.model_type}")

        self.landmark_indices = self.estimator.landmark_indices
        logger.info("Model expects %d landmarks.", len(self.landmark_indices))

        self.fieldnames = [
            "subject",
            "camera",
            "frame",
            "gt_gaze_x",
            "gt_gaze_y",
            "gt_gaze_z",
            "pred_gaze_x",
            "pred_gaze_y",
            "pred_gaze_z",
            "angular_error_deg",
        ]

    def compute_angular_error(self, v1, v2):
        if v1 is None or v2 is None:
            return np.nan

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return np.nan

        v1 = v1 / n1
        v2 = v2 / n2
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    def closest_angular_distance(self, a, b):
        if a is None or b is None:
            return np.nan
        diff = abs(a - b) % 360
        return min(diff, 360 - diff)

    def vector_to_pitch_yaw(self, vector):
        if vector is None:
            return None, None
        norm = np.linalg.norm(vector)
        if norm == 0:
            return 0.0, 0.0
        v = vector / norm
        pitch = np.arcsin(np.clip(-v[1], -1.0, 1.0))
        yaw = np.arctan2(v[0], v[2])
        return np.degrees(pitch), np.degrees(yaw)

    def save_summary_results(self, valid_count, mean_angle, std_error, mean_yaw, mean_pitch):
        if not self.summary_csv_path:
            return

        file_exists = os.path.isfile(self.summary_csv_path)
        model_name = os.path.basename(self.config.gaze_estimator.model_path)
        if not model_name:
            model_name = os.path.basename(os.path.dirname(self.config.gaze_estimator.model_path))

        fieldnames = [
            "timestamp",
            "model_name",
            "model_type",
            "images_processed",
            "mean_gaze_error_deg",
            "std_gaze_error",
            "mean_yaw_error_deg",
            "mean_pitch_error_deg",
        ]

        with open(self.summary_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": model_name,
                    "model_type": self.model_type,
                    "images_processed": valid_count,
                    "mean_gaze_error_deg": f"{mean_angle:.4f}",
                    "std_gaze_error": f"{std_error:.4f}",
                    "mean_yaw_error_deg": f"{mean_yaw:.4f}",
                    "mean_pitch_error_deg": f"{mean_pitch:.4f}",
                }
            )
        logger.info("Summary saved to %s", self.summary_csv_path)

    def run_benchmark(self, csv_path):
        logger.info("Processing CSV: %s", csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        file_exists = os.path.exists(self.output_csv_path) if self.output_csv_path else False
        out_file = None
        writer = None
        if self.output_csv_path:
            os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
            out_file = open(self.output_csv_path, "a", newline="")
            writer = csv.DictWriter(out_file, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()

        gaze_errors = []
        yaw_errors = []
        pitch_errors = []

        try:
            with open(csv_path, "r", newline="") as input_file:
                reader = csv.DictReader(input_file, delimiter=";")
                headers = reader.fieldnames or []
                required_headers = ["subject", "frame", "camera", "gaze_x", "gaze_y", "gaze_z"]
                for header in required_headers:
                    if header not in headers:
                        raise ValueError(f"Missing required column '{header}' in {csv_path}")
                for idx in self.landmark_indices:
                    if f"{idx}_x" not in headers or f"{idx}_y" not in headers:
                        raise ValueError(f"Missing landmark columns for {idx} in {csv_path}")

                for row_idx, row in enumerate(reader):
                    try:
                        gt_vec = np.array(
                            [float(row["gaze_x"]), float(row["gaze_y"]), float(row["gaze_z"])],
                            dtype=np.float32,
                        )
                        gt_pitch, gt_yaw = self.vector_to_pitch_yaw(gt_vec)

                        landmarks = []
                        for idx in self.landmark_indices:
                            landmarks.append([float(row[f"{idx}_x"]), float(row[f"{idx}_y"])])
                        norm_lmks = np.array(landmarks, dtype=np.float32)

                        est_vec = self.estimator.estimate_norm_gaze_from_norm_lmks(norm_lmks)
                        est_pitch, est_yaw = self.vector_to_pitch_yaw(est_vec)

                        angle_err = self.compute_angular_error(gt_vec, est_vec)
                        if np.isnan(angle_err):
                            continue

                        gaze_errors.append(angle_err)
                        yaw_errors.append(self.closest_angular_distance(gt_yaw, est_yaw))
                        pitch_errors.append(self.closest_angular_distance(gt_pitch, est_pitch))

                        if writer:
                            writer.writerow(
                                {
                                    "subject": row.get("subject", ""),
                                    "camera": row.get("camera", ""),
                                    "frame": row.get("frame", ""),
                                    "gt_gaze_x": gt_vec[0],
                                    "gt_gaze_y": gt_vec[1],
                                    "gt_gaze_z": gt_vec[2],
                                    "pred_gaze_x": est_vec[0],
                                    "pred_gaze_y": est_vec[1],
                                    "pred_gaze_z": est_vec[2],
                                    "angular_error_deg": angle_err,
                                }
                            )
                    except Exception as e:
                        logger.warning("Error processing row %d: %s", row_idx, e)

                    if row_idx > 0 and row_idx % 1000 == 0:
                        print(f"Processed {row_idx} rows...", end="\r")
                        if out_file:
                            out_file.flush()
        finally:
            if out_file:
                out_file.close()

        valid_count = len(gaze_errors)
        if valid_count > 0:
            mean_error = float(np.mean(gaze_errors))
            std_error = float(np.std(gaze_errors))
            mean_yaw = float(np.mean(yaw_errors)) if yaw_errors else 0.0
            mean_pitch = float(np.mean(pitch_errors)) if pitch_errors else 0.0

            print(f"\n--- Results for {self.model_name} ---")
            print(f"Samples: {valid_count}")
            print(f"Mean Angular Error: {mean_error:.4f} deg")
            print(f"Std Dev: {std_error:.4f} deg")
            print(f"Mean Yaw Error: {mean_yaw:.4f} deg")
            print(f"Mean Pitch Error: {mean_pitch:.4f} deg")
            print("---------------------------------------")
            self.save_summary_results(valid_count, mean_error, std_error, mean_yaw, mean_pitch)
            return {
                "model": self.model_name,
                "error": mean_error,
                "yaw_error": mean_yaw,
                "pitch_error": mean_pitch,
            }

        print(f"No valid predictions for {self.model_name}")
        return None


if __name__ == "__main__":
    config = OmegaConf.load("configs/default_config.yaml")
    config.device = getattr(config, "device", "cpu")
    config.gaze_estimator.camera_params = config.gaze_estimator.normalized_camera_params

    models_to_test = [
        {"name": "xgaze_mlp", "path": "models/xgaze_MLP.pth", "type": "normalized"},
        {"name": "xgaze_siamese", "path": "models/xgaze_siameseMLP.pth", "type": "normalized_siamese"},
        {"name": "xgaze_xgboost", "path": "models/xgaze_xgboost.pkl", "type": "normalized_xgboost"},
    ]
    csv_path = "datasets/XGaze_448/training_xgaze_dataset_normalized_det_conf_0_8_with_lm168_TEST.csv"

    for model_def in models_to_test:
        if not os.path.exists(model_def["path"]):
            logger.warning("Skipping missing model: %s", model_def["path"])
            continue

        runner = CSVBenchmarkRunner(
            config=config.copy(),
            model_def=model_def,
            output_csv_path=None,
            summary_csv_path="models/stats/summary_benchmark_xgaze_448.csv",
        )
        runner.run_benchmark(csv_path)
