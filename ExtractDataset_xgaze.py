import os
import cv2
import numpy as np
import logging
import csv
import h5py
from omegaconf import OmegaConf

# --- IMPORTS FROM LOCAL FILES ---
try:
    from face_landmark_estimator import LandmarkEstimator
except ImportError:
    print("Error: Could not import LandmarkEstimator.")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONSTANTS ---
# Landmark Indices
# 1. Iris (Center + 4 circumference)
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
# 2. Contours
LEFT_INNER_CORNER = 133
LEFT_OUTER_CORNER = 33
LEFT_EYE_CONTOUR  = [LEFT_OUTER_CORNER, LEFT_INNER_CORNER, 159, 145] 
RIGHT_INNER_CORNER = 362
RIGHT_OUTER_CORNER = 263
RIGHT_EYE_CONTOUR  = [RIGHT_OUTER_CORNER, RIGHT_INNER_CORNER, 386, 374] 
# 3. Head Anchors
HEAD_ANCHORS = [1,9]

LEFT_EYE_LANDMARKS = LEFT_IRIS + LEFT_EYE_CONTOUR
#[LEFT_OUTER_CORNER, 246, 161, 160, 159, 158, 157, 173, LEFT_INNER_CORNER, 155, 154, 153, 145, 144,
#163, 7, 468, 469, 470, 471, 472, 27, 190, 243, 233, 232, 230, 31, 25, 110, 113, 247, 225]
RIGHT_EYE_LANDMARKS = RIGHT_IRIS + RIGHT_EYE_CONTOUR
#[475, 473, 474, 476, 477, RIGHT_INNER_CORNER, 398, 382, 381, 380, 374, 373, 390, 249,
#RIGHT_OUTER_CORNER, 466, 388, 387, 386, 385, 384, 257, 445, 342, 467, 255, 339, 450, 452, 453, 463, 414, 261]
LANDMARK_INDICES = LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS + HEAD_ANCHORS

# Output columns
GAZE_COLS = ['gaze_x', 'gaze_y', 'gaze_z', 'gaze_yaw', 'gaze_pitch']
LANDMARK_COLS_XY = [f"{idx}_x" for idx in LANDMARK_INDICES] + [f"{idx}_y" for idx in LANDMARK_INDICES]

class H5LandmarkDatasetGenerator:
    def __init__(self, config, dataset_root, output_csv_path, batch_size=100):
        self.config = config
        self.dataset_root = dataset_root
        self.output_csv_path = output_csv_path
        self.batch_size = batch_size

        # Initialize Normalizer and Estimator
        logger.info("Initializing LandmarkEstimator...")
        self.landmark_estimator = LandmarkEstimator(self.config)

        # CSV Headers
        self.fieldnames = ['subject', 'frame', 'camera'] + GAZE_COLS + LANDMARK_COLS_XY

        # Initialize CSV
        self._init_csv()

    def _init_csv(self):
        """Initializes the CSV file if it doesn't exist."""
        if not os.path.exists(self.output_csv_path):
            with open(self.output_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, delimiter=';')
                writer.writeheader()
            logger.info(f"Initialized new CSV at {self.output_csv_path}")
        else:
            logger.info(f"Appending to existing CSV at {self.output_csv_path}")

    def pitchyaw_to_vector(self, pitchyaws):
        """Convert pitch/yaw (radians) to 3D vector (x, y, z)."""
        pitchyaws = np.array(pitchyaws)
        if pitchyaws.ndim == 1:
            pitchyaws = pitchyaws[np.newaxis, :]

        sin = np.sin(pitchyaws)
        cos = np.cos(pitchyaws)
        out = np.empty((pitchyaws.shape[0], 3))
        # ETH-XGaze Convention:
        # x = cos(pitch) * sin(yaw)
        # y = sin(pitch)
        # z = cos(pitch) * cos(yaw)
        out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
        out[:, 1] = sin[:, 0]
        out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
        
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / norm

    def vector_to_pitch_yaw(self, v):
        x, y, z = v[0], v[1], v[2]
        norm = np.linalg.norm(v)
        if norm > 0: x, y, z = x / norm, y / norm, z / norm
        pitch = np.arcsin(np.clip(y, -1.0, 1.0))
        yaw = np.arctan2(x, -z)  # Note: This assumes Z-Forward convention for yaw
        return yaw, pitch

    def process_dataset(self):
        h5_files = [f for f in os.listdir(self.dataset_root) if f.endswith('.h5')]
        h5_files.sort()

        if not h5_files:
            logger.error(f"No .h5 files found in {self.dataset_root}")
            return

        total_processed = 0
        total_skipped = 0
        buffer = []

        for h5_file in h5_files:
            full_path = os.path.join(self.dataset_root, h5_file)
            subject_id = h5_file.split('.')[0]
            logger.info(f"Processing {subject_id}...")

            try:
                with h5py.File(full_path, 'r') as fid:
                    if "face_patch" not in fid:
                        logger.warning(f"face_patch not found in {h5_file}")
                        continue

                    num_data = fid["face_patch"].shape[0]
                        
                    # Pre-load data to minimize disk I/O lag in loop
                    # Note: typically one H5 per subject fits in RAM (few GBs).
                    face_patches = fid['face_patch'][:]
                    face_gazes = fid['face_gaze'][:] # (N, 2) Pitch, Yaw
                    frame_indices = fid['frame_index'][:, 0]
                    cam_indices = fid['cam_index'][:, 0]

                    for i in range(num_data):
                        # 1. Get Image
                        img = np.array(face_patches[i]).copy() # BGR
                            
                        # Histogram equalization (Logic from BenchmarkXgaze.py)
                        if frame_indices[i] > 524:
                            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                        # 2. Get Ground Truth Gaze
                        # face_gaze is (2,) pitch/yaw
                        gt_pitch_rad = face_gazes[i, 0]
                        gt_yaw_rad = face_gazes[i, 1]

                        #Convert to 3D vector
                        gt_vector = self.pitchyaw_to_vector(np.array([gt_pitch_rad, gt_yaw_rad]))[0]

                        # 3. Detect Faces and landmarks
                        faces = self.landmark_estimator.detect_faces(img)
                            
                        if faces:
                            face = faces[0]

                            all_landmarks = np.concatenate([face.landmarks, face.landmarks_eyes])
                            normalized_landmarks = all_landmarks[LANDMARK_INDICES]
                                
                            gaze_vec_normalized = -gt_vector
                            gaze_vec_normalized /= np.linalg.norm(gaze_vec_normalized)
                            yaw, pitch = self.vector_to_pitch_yaw(gaze_vec_normalized)

                            # 4. Prepare Row
                            row_data = {
                                'subject': subject_id,
                                'frame': frame_indices[i],
                                'camera': cam_indices[i],
                                'gaze_x': gaze_vec_normalized[0],
                                'gaze_y': gaze_vec_normalized[1],
                                'gaze_z': gaze_vec_normalized[2],
                                'gaze_yaw': yaw,
                                'gaze_pitch': pitch
                                }

                            # Add flattened landmarks
                            for idx, lm_idx in enumerate(LANDMARK_INDICES):
                                row_data[f"{lm_idx}_x"] = normalized_landmarks[idx, 0]
                                row_data[f"{lm_idx}_y"] = normalized_landmarks[idx, 1]

                            buffer.append(row_data)
                            total_processed += 1
                        else:
                            total_skipped += 1

                        # Write Buffer
                        if len(buffer) >= self.batch_size:
                            self.write_buffer(buffer)
                            buffer = []
                            print(f"Processed: {total_processed} | Skipped: {total_skipped}", end='\r')

            except Exception as e:
                logger.error(f"Failed processing {h5_file}: {e}")
                continue

        # Final write
        if buffer:
            self.write_buffer(buffer)
        
        print(f"\nComplete. Total Processed: {total_processed}, Total Skipped: {total_skipped}")

    def write_buffer(self, buffer):
        if not buffer: return
        with open(self.output_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, delimiter=';')
            writer.writerows(buffer)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    CONFIG_PATH = 'configs/default_config.yaml'
    RES = 448
    
    # Path to the folder containing .h5 files (e.g., standard ETH-XGaze train folder)
    DATASET_ROOT = f'./xgaze_{RES}_link/train' 
    
    # Output file
    OUTPUT_CSV = f'./datasets/XGaze_{RES}/xgaze_det_conf_0_8_ALL.csv'
    
    # Parameters
    DET_CONF = 0.8
    PADDING = 0.25

    if not os.path.exists(CONFIG_PATH):
        print(f"Config file not found at {CONFIG_PATH}")
    elif not os.path.exists(DATASET_ROOT):
        print(f"Dataset root not found at {DATASET_ROOT}")
    else:
        # Load Config
        config = OmegaConf.load(CONFIG_PATH)
        
        # Override specific settings for detection quality
        config.face_detector.mediapipe_min_det_conf = DET_CONF
        config.face_detector.padding = PADDING
        
        # ETH-XGaze patch dimensions
        config.image.height = RES
        config.image.width = RES

        # Run Generator
        generator = H5LandmarkDatasetGenerator(config, DATASET_ROOT, OUTPUT_CSV)
        generator.process_dataset()
