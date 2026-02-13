import os
import csv
import numpy as np
import logging
import torch
import math
import datetime
import pathlib
from omegaconf import OmegaConf

# --- IMPORTS ---
try:
    from gaze_estimator_normalized import GazeEstimatorNormalized
except ImportError:
    print("Warning: Could not import GazeEstimatorNormalized.")
    GazeEstimatorNormalized = None

try:
    from gaze_estimator_normalized_siamese import GazeEstimatorNormalizedSiamese
except ImportError:
    print("Warning: Could not import GazeEstimatorNormalizedSiamese.")
    GazeEstimatorNormalizedSiamese = None

try:
    from gaze_estimator_normalized_XGBoost import GazeEstimatorXGBoost as GazeEstimatorNormalizedXGBoost
except ImportError:
    print("Warning: Could not import GazeEstimatorNormalizedXGBoost.")
    GazeEstimatorNormalizedXGBoost = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GazeGeneBenchmarkFromCSV:
    def __init__(self, config, model_type='normalized', output_csv_path=None, summary_csv_path=None):
        self.config = config
        self.model_type = model_type
        self.output_csv_path = output_csv_path
        self.summary_csv_path = summary_csv_path
        
        # Initialize the estimator based on type
        logger.info(f"Initializing {self.model_type} model from {config.gaze_estimator.model_path}...")
        
        if self.model_type == 'normalized_siamese':
            if GazeEstimatorNormalizedSiamese is None: raise ImportError("Siamese class not loaded.")
            self.estimator = GazeEstimatorNormalizedSiamese(config)
        elif self.model_type == 'normalized_xgboost':
            if GazeEstimatorNormalizedXGBoost is None: raise ImportError("XGBoost class not loaded.")
            self.estimator = GazeEstimatorNormalizedXGBoost(config)
        else:
            if GazeEstimatorNormalized is None: raise ImportError("Normalized class not loaded.")
            self.estimator = GazeEstimatorNormalized(config)
        
        self.landmark_indices = self.estimator.landmark_indices
        logger.info(f"Model expects {len(self.landmark_indices)} landmarks.")

        # Prepare Output CSV Headers
        self.fieldnames = [
            "subject", "camera", "image_path",
            "gt_gaze_x", "gt_gaze_y", "gt_gaze_z",
            "pred_gaze_x", "pred_gaze_y", "pred_gaze_z",
            "angular_error_deg"
        ]

    def compute_angular_error(self, v1, v2):
        """Computes the angle in degrees between two 3D vectors."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        if n1 < 1e-6 or n2 < 1e-6:
            return np.nan

        v1_u = v1 / n1
        v2_u = v2 / n2
        
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        return np.degrees(angle_rad)

    def closest_angular_distance(self, a, b):
        """Computes the shortest distance between two angles in degrees."""
        if a is None or b is None: return np.nan
        diff = abs(a - b) % 360
        return min(diff, 360 - diff)

    def vector_to_pitch_yaw(self, vector):
        """
        Converts a 3D vector to pitch and yaw in degrees.
        Assumption: OpenCV Coordinate System
        Z: Forward, Y: Down, X: Right
        """
        if vector is None:
            return None, None
        norm = np.linalg.norm(vector)
        if norm == 0:
            return 0.0, 0.0
        v = vector / norm
        # Pitch: Angle with X-Z plane (Rotation around X). Positive Y is down, so negative sin is pitch up.
        pitch = np.arcsin(np.clip(-v[1], -1.0, 1.0))
        # Yaw: Angle projected on X-Z plane (Rotation around Y).
        yaw = np.arctan2(v[0], v[2])
        return np.degrees(pitch), np.degrees(yaw)

    def save_summary_results(self, valid_count, mean_angle, std_error, mean_yaw, mean_pitch):
        file_exists = os.path.isfile(self.summary_csv_path)
        model_name = os.path.basename(self.config.gaze_estimator.model_path)
        if not model_name:
            model_name = os.path.basename(os.path.dirname(self.config.gaze_estimator.model_path))

        fieldnames = ["timestamp", "model_name", "model_type", "images_processed",
                      "mean_gaze_error_deg", "std_gaze_error", "mean_yaw_error_deg", "mean_pitch_error_deg"]

        try:
            with open(self.summary_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists: writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": model_name,
                    "model_type": self.model_type,
                    "images_processed": valid_count,
                    "mean_gaze_error_deg": f"{mean_angle:.4f}",
                    "std_gaze_error": f"{std_error:.4f}",
                    "mean_yaw_error_deg": f"{mean_yaw:.4f}",
                    "mean_pitch_error_deg": f"{mean_pitch:.4f}"
                })
            logger.info(f"Summary saved to {self.summary_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")

    def run_benchmark(self, input_csv_path):
        if not os.path.exists(input_csv_path):
            logger.error(f"Input CSV file not found: {input_csv_path}")
            return

        logger.info(f"Processing CSV: {input_csv_path}")
        
        # Setup Output CSV
        file_exists = os.path.exists(self.output_csv_path) if self.output_csv_path else False
        out_f = None
        writer = None
        
        if self.output_csv_path:
            os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
            out_f = open(self.output_csv_path, 'a', newline='')
            writer = csv.DictWriter(out_f, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()
                logger.info(f"Created new output file: {self.output_csv_path}")
            else:
                logger.info(f"Appending to existing output file: {self.output_csv_path}")

        gaze_errors = []
        yaw_errors = []
        pitch_errors = []
        valid_count = 0
        skipped_count = 0

        try:
            with open(input_csv_path, 'r') as f:
                reader = csv.DictReader(f, delimiter=';')
                    
                headers = reader.fieldnames
                for idx in self.landmark_indices:
                    if f"{idx}_x" not in headers or f"{idx}_y" not in headers:
                        logger.error(f"Missing columns for landmark {idx} in Input CSV.")
                        return

                for row_idx, row in enumerate(reader):
                    # 1. Parse Ground Truth Gaze
                    gt_gaze = np.array([
                        float(row['gaze_x']),
                        float(row['gaze_y']),
                        float(row['gaze_z'])], dtype=np.float32)
                    gt_pitch, gt_yaw = self.vector_to_pitch_yaw(gt_gaze)

                    # 2. Parse Landmarks
                    landmarks = []
                    for idx in self.landmark_indices:
                        lx = float(row[f"{idx}_x"])
                        ly = float(row[f"{idx}_y"])
                        landmarks.append([lx, ly])
                            
                    landmarks_np = np.array(landmarks, dtype=np.float32)

                    # 3. Inference
                    predicted_gaze = self.estimator.estimate_norm_gaze_from_norm_lmks(landmarks_np)
                    est_pitch, est_yaw = self.vector_to_pitch_yaw(predicted_gaze)
                    
                    # 4. Compute Error
                    error = self.compute_angular_error(gt_gaze, predicted_gaze)
                    yaw_error = self.closest_angular_distance(gt_yaw, est_yaw)
                    pitch_error = self.closest_angular_distance(gt_pitch, est_pitch)
                            
                    if not np.isnan(error):
                        gaze_errors.append(error)
                        yaw_errors.append(yaw_error)
                        pitch_errors.append(pitch_error)
                        valid_count += 1
                                
                        # 5. Write Result
                        if writer:
                            out_row = {
                                "subject": row.get('subject', ''),
                                "camera": row.get('camera', ''),
                                "image_path": row.get('image_path', ''),
                                "gt_gaze_x": gt_gaze[0],
                                "gt_gaze_y": gt_gaze[1],
                                "gt_gaze_z": gt_gaze[2],
                                "pred_gaze_x": predicted_gaze[0],
                                "pred_gaze_y": predicted_gaze[1],
                                "pred_gaze_z": predicted_gaze[2],
                                "angular_error_deg": error
                                }
                            writer.writerow(out_row)
                        else:
                            skipped_count += 1

                        
                    if row_idx % 1000 == 0:
                        print(f"Processed {row_idx} rows...", end='\r')
                        if out_f: out_f.flush()

        finally:
            if out_f:
                out_f.close()

        # --- Report Results ---
        print("\n" + "="*40)
        print(f"BENCHMARK RESULTS: {os.path.basename(input_csv_path)}")
        print(f"Model: {os.path.basename(self.config.gaze_estimator.model_path)}")
        print("="*40)
        
        if valid_count > 0:
            mean_error = np.mean(gaze_errors)
            std_error = np.std(gaze_errors)
            yaw_mean_err   = np.mean(yaw_errors)
            pitch_mean_err = np.mean(pitch_errors)
            print(f"Total Samples:   {valid_count}")
            print(f"Mean Angular Error:   {mean_error:.4f} degrees")
            print(f"Std Dev:              {std_error:.4f} degrees")
            self.save_summary_results(valid_count, mean_error, std_error, yaw_mean_err, pitch_mean_err)
        else:
            print("No valid samples found.")
        print("="*40)

        return np.mean(gaze_errors) if valid_count > 0 else None


def _expanduser(path: str) -> str:
    if not path: return path
    p = pathlib.Path(path).expanduser()
    if not p.is_absolute():
        cwd = pathlib.Path(os.getcwd()).resolve()
        p = cwd / p
    return p.as_posix()

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # Path to the GazeGene Normalized CSV
    CSV_PATH_CROSS_TEST = './datasets/GazeGene/gazegene_normalized_det_conf_0_8_ALL.csv'
    CSV_PATH_WITHIN_TEST = './datasets/GazeGene/gazagene_normalized_det_conf_0_8_TEST.csv'
    CONFIG_PATH = 'configs/default_config.yaml'

    # --- DEFINE MODELS TO BENCHMARK ---
    models_to_benchmark = [
        {
            'name': 'normalized_gazegene_0_8',
            'path': 'models/gazegene_MLP.pth',
            'type': 'normalized',
            'test_type': 'within'
        },{
            'name': 'normalized_siamese_gazegene_0_8',
            'path': 'models/gazegene_siameseMLP.pth',
            'type': 'normalized_siamese',
            'test_type': 'within'
        },{
            'name': 'normalized_xgboost_gazegene_0_8',
            'path': 'models/gazegene_xgboost.pkl',
            'type': 'normalized_xgboost',
            'test_type': 'within'
        },{
            'name': 'normalized_xgaze_0_8',
            'path': 'models/xgaze_MLP.pth',
            'type': 'normalized',
            'test_type': 'cross'
        },{
            'name': 'normalized_siamese_xgaze_0_8',
            'path': 'models/xgaze_siameseMLP.pth',
            'type': 'normalized_siamese',
            'test_type': 'cross'
        },{
            'name': 'normalized_xgboost_xgaze_0_8',
            'path': 'models/xgaze_xgboost.pkl',
            'type': 'normalized_xgboost',
            'test_type': 'cross'
        },{
            'name': 'normalized_gaze360_0_8',
            'path': 'models/gaze360_MLP.pth',
            'type': 'normalized',
            'test_type': 'cross'
        },{
            'name': 'normalized_siamese_gaze360_0_8',
            'path': 'models/gaze360_siameseMLP.pth',
            'type': 'normalized_siamese',
            'test_type': 'cross'
        },{
            'name': 'normalized_xgboost_gaze360_0_8',
            'path': 'models/gaze360_xgboost.pkl',
            'type': 'normalized_xgboost',
            'test_type': 'cross'
        }
    ]

    if not os.path.exists(CONFIG_PATH):
        print(f"Config file not found at {CONFIG_PATH}")
        exit(1)
    elif not os.path.exists(CSV_PATH_CROSS_TEST):
        print(f"Input CSV file not found at {CSV_PATH_CROSS_TEST}")
        exit(1)
    elif not os.path.exists(CSV_PATH_WITHIN_TEST):
        print(f"Input CSV file not found at {CSV_PATH_WITHIN_TEST}")
        exit(1)
    
    base_config = OmegaConf.load(CONFIG_PATH)

    for model_def in models_to_benchmark:
        print(f"\nRunning Model: {model_def['name']} ({model_def['type']})")

        if model_def['test_type'] == 'within':
            CSV_PATH_TO_TEST = CSV_PATH_WITHIN_TEST
        elif model_def['test_type'] == 'cross':
            CSV_PATH_TO_TEST = CSV_PATH_CROSS_TEST
        else:
            logger.error(f"Invalid test type {model_def['test_type']}") 
        
        # Prepare Config
        config = base_config.copy()
        config.gaze_estimator.model_path = _expanduser(model_def['path'])
        
        if not hasattr(config, 'device'): config.device = 'cpu'

        # Prepare Output Path
        output_csv = f'./models/stats/gazegene/CSV files/CSV files/benchmark_results_csv_{model_def["name"]}.csv'
        summary_file = './models/stats/summary_benchmark_gazegene.csv'

        try:
            benchmark = GazeGeneBenchmarkFromCSV(
                config, 
                model_type=model_def['type'],
                output_csv_path=output_csv,
                summary_csv_path=summary_file
            )
            benchmark.run_benchmark(CSV_PATH_TO_TEST)
            
        except Exception as e:
            logger.error(f"Failed to run model {model_def['name']}: {e}")
            import traceback
            traceback.print_exc()
