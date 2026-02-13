import os
import csv
import numpy as np
import logging
import pandas as pd
from omegaconf import OmegaConf
import datetime

# --- 1. IMPORTS & CONFIGURATION ---

# Import gaze estimator for normalized models
try:
    from gaze_estimator_normalized import GazeEstimatorNormalized
except ImportError:
    print("Warning: Could not import GazeEstimatorNormalized.")
    GazeEstimatorNormalized = None

# Import gaze estimator for XGBoost normalized models
try:
    from gaze_estimator_normalized_XGBoost import GazeEstimatorXGBoost as GazeEstimatorNormalizedXGBoost
except ImportError:
    print("Warning: Could not import GazeEstimatorNormalizedXGBoost.")
    GazeEstimatorNormalizedXGBoost = None

# Import gaze estimator for Siamese models
try:
    from gaze_estimator_normalized_siamese import GazeEstimatorNormalizedSiamese
except ImportError:
    print("Warning: Could not import GazeEstimatorNormalizedSiamese.")
    GazeEstimatorNormalizedSiamese = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class CSVBenchmarkRunner:
    def __init__(self, config, model_def, output_csv_path=None, summary_csv_path=None):
        self.config = config
        self.model_type = model_def['type']
        self.model_name = model_def['name']
        self.output_csv_path = output_csv_path
        self.summary_csv_path = summary_csv_path
        
        # Load the specific estimator
        logger.info(f"Loading model: {self.model_name} ({self.model_type})")
        
        # Point config to the specific model path
        self.config.gaze_estimator.model_path = model_def['path']
        
        if self.model_type == 'normalized':
            if GazeEstimatorNormalized is None: raise ImportError("GazeEstimatorNormalized not found.")
            self.estimator = GazeEstimatorNormalized(self.config)
        elif self.model_type == 'normalized_xgboost':
            if GazeEstimatorNormalizedXGBoost is None: raise ImportError("GazeEstimatorNormalizedXGBoost not found.")
            self.estimator = GazeEstimatorNormalizedXGBoost(self.config)
        elif self.model_type == 'normalized_siamese':
            if GazeEstimatorNormalizedSiamese is None: raise ImportError("GazeEstimatorNormalizedSiamese not found.")
            self.estimator = GazeEstimatorNormalizedSiamese(self.config)
        else:
            raise ValueError(f"Unsupported model type for CSV benchmarking: {self.model_type}")

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
        """Computes angular error in degrees between two vectors."""
        if v1 is None or v2 is None: return np.nan
        
        # Normalize vectors
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6: return np.nan
        
        v1 = v1 / n1
        v2 = v2 / n2
        
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

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

    def run_benchmark(self, csv_path):
        logger.info(f"Reading dataset: {csv_path}")
        
        # 1. Load Data
        try:
            df = pd.read_csv(csv_path, delimiter=';')
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return

        # Check if required columns exist
        required_cols = ['gaze_x', 'gaze_y', 'gaze_z']
        if not all(col in df.columns for col in required_cols):
            logger.error("CSV missing ground truth gaze columns.")
            return

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

        # 2. Prepare Landmarks
        # We need to extract columns like "468_x", "468_y" in the exact order of LANDMARK_INDICES
        # to reconstruct the (N, 20, 2) array expected by the model.
        landmark_data = []
        try:
            for idx in self.landmark_indices:
                x_col = df[f"{idx}_x"].values
                y_col = df[f"{idx}_y"].values
                # Stack x and y for this landmark
                lm_xy = np.stack([x_col, y_col], axis=1) # (N, 2)
                landmark_data.append(lm_xy)
            
            # Stack all landmarks: Result shape (N, Num_Landmarks, 2)
            all_landmarks = np.stack(landmark_data, axis=1)
            
        except KeyError as e:
            logger.error(f"Missing landmark column in CSV: {e}")
            return

        # 3. Ground Truth
        gt_vectors = df[['gaze_x', 'gaze_y', 'gaze_z']].values
        subjects = df['subject'].values
        frames   = df['frame'].values
        cameras  = df['camera'].values

        gaze_errors = []
        yaw_errors = []
        pitch_errors = []

        logger.info(f"Processing {len(df)} samples...")

        # 4. Inference Loop
        for i in range(len(df)):
            gt_vec = gt_vectors[i]
            gt_p, gt_y = self.vector_to_pitch_yaw(gt_vec)
            norm_lmks = all_landmarks[i]
            subject = subjects[i]
            frame = frames[i]
            cam = cameras[i]
            
            try:
                # Inference of normalized gaze vector
                est_vec = self.estimator.estimate_norm_gaze_from_norm_lmks(norm_lmks)
                est_p, est_y = self.vector_to_pitch_yaw(est_vec)

                # Compute Error
                angle_err = self.compute_angular_error(gt_vec, est_vec)
                yaw_error = self.closest_angular_distance(gt_y, est_y)
                pitch_error = self.closest_angular_distance(gt_p, est_p)
                gaze_errors.append(angle_err)
                    
                if not np.isnan(gt_p) and not np.isnan(est_p):
                    pitch_errors.append(pitch_error)
                    yaw_errors.append(yaw_error)

                    # Write Result
                    if writer:
                        out_row = {
                            "subject": subject,
                            "camera": cam,
                            "frame": frame,
                            "gt_gaze_x": gt_vec[0],
                            "gt_gaze_y": gt_vec[1],
                            "gt_gaze_z": gt_vec[2],
                            "pred_gaze_x": est_vec[0],
                            "pred_gaze_y": est_vec[1],
                            "pred_gaze_z": est_vec[2],
                            "angular_error_deg": angle_err
                        }
                        writer.writerow(out_row)                

            except Exception as e:
                # Fail silently for individual bad rows, but log if needed
                # print(f"Error on row {i}: {e}")
                pass

            if i > 0 and i % 1000 == 0:
                print(f"Processed {i}/{len(df)}...", end='\r')

        if out_f:
            out_f.close()

        # 5. Results
        gaze_errors = np.array(gaze_errors)
        gaze_errors = gaze_errors[~np.isnan(gaze_errors)] # Remove NaNs

        valid_count = len(gaze_errors)
        if valid_count > 0:
            mean_error = np.mean(gaze_errors)
            std_error = np.std(gaze_errors)
            mean_yaw = np.mean(yaw_errors) if yaw_errors else 0
            mean_pitch = np.mean(pitch_errors) if pitch_errors else 0
            
            print(f"\n--- Results for {self.model_name} ---")
            print(f"Samples: {valid_count}")
            print(f"Mean Angular Error: {mean_error:.4f}°")
            print(f"Std Dev: {std_error:.4f}°")
            print(f"Mean Yaw Error:     {mean_yaw:.4f}°")
            print(f"Mean Pitch Error:   {mean_pitch:.4f}°")
            print("---------------------------------------")
            self.save_summary_results(valid_count, mean_error, std_error, mean_yaw, mean_pitch)
            return {
                "model": self.model_name,
                "error": mean_error,
                "yaw_error": mean_yaw,
                "pitch_error": mean_pitch
            }
        else:
            print(f"No valid predictions for {self.model_name}")
            return None

if __name__ == "__main__":
    # --- CONFIGURATION ---
    
    # 1. Global Config (Used for initialization)
    CONFIG_PATH = 'configs/default_config.yaml'
   
    # 2. Define Models to Test
    models_to_test = [
        {
            'name': 'normalized_gazegene_0_8',
            'path': 'models/gazegene_MLP.pth',
            'type': 'normalized',
            'test_type': 'cross'
        },{
            'name': 'normalized_siamese_gazegene_0_8',
            'path': 'models/gazegene_siameseMLP.pth',
            'type': 'normalized_siamese',
            'test_type': 'cross'
        },{
            'name': 'normalized_xgboost_gazegene_0_8',
            'path': 'models/gazegene_xgboost.pkl',
            'type': 'normalized_xgboost',
            'test_type': 'cross'
        },{
            'name': 'normalized_xgaze_0_8',
            'path': 'models/xgaze_MLP.pth',
            'type': 'normalized',
            'test_type': 'within'
        },{
            'name': 'normalized_siamese_xgaze_0_8',
            'path': 'models/xgaze_siameseMLP.pth',
            'type': 'normalized_siamese',
            'test_type': 'within'
        },{
            'name': 'normalized_xgboost_xgaze_0_8',
            'path': 'models/xgaze_xgboost.pkl',
            'type': 'normalized_xgboost',
            'test_type': 'within'
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

    # 3. Define Dataset Paths
    CSV_PATH_CROSS_TEST = './datasets/XGaze_448/xgaze_normalized_det_conf_0_8_ALL.csv'
    CSV_PATH_WITHIN_TEST = './datasets/XGaze_448/xgaze_normalized_det_conf_0_8_TEST.csv'


    # --- EXECUTION ---
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

    for model_def in models_to_test:
        # Prepare Output Path
        output_csv = f'./models/stats/xgaze_448/CSV files/benchmark_results_csv_{model_def["name"]}.csv'
        summary_file = './models/stats/summary_benchmark_xgaze_448.csv'
        
        print(f"\n==================================================")
        print(f"BENCHMARKING ON: {model_def['name']} ({model_def['test_type']})")
        print(f"==================================================")

        if model_def['test_type'] == 'within':
            CSV_PATH_TO_TEST = CSV_PATH_WITHIN_TEST
        elif model_def['test_type'] == 'cross':
            CSV_PATH_TO_TEST = CSV_PATH_CROSS_TEST
        else:
            logger.error(f"Invalid test type {model_def['test_type']}") 
        
        # Create a clean config copy for each model run
        config = base_config.copy()
            
        try:
            runner = CSVBenchmarkRunner(config, model_def, output_csv_path=output_csv,summary_csv_path=summary_file)
            runner.run_benchmark(CSV_PATH_TO_TEST)
        except Exception as e:
            print(f"Failed to run model {model_def['name']}: {e}")
            import traceback
            traceback.print_exc()
