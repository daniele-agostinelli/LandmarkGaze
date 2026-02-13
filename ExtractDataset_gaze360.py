# THIS SCRIPT SERVES TO GENERATE A NORMALIZED DATASET FROM GAZE360
# It generates 4 CSV files: All, Train, Validation, Test.
# It handles dynamic image sizes and converts Gaze360 coordinates to OpenCV standards.

import os
import cv2
import numpy as np
import logging
import csv
import gc
import scipy.io
from omegaconf import OmegaConf

# --- IMPORTS FROM LOCAL FILES ---
try:
    from face_landmark_estimator import LandmarkEstimator
    from normalization_utils import LandmarkNormalizer
    from camera import Camera
    from face_model import FaceModel
except ImportError:
    print("Error: Could not import LandmarkEstimator, LandmarkNormalizer, Camera or FaceModel.")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- USER SETTINGS ---
DEBUG_MODE = True  # Set to True to save visualization images
VIS_INTERVAL = 100 # Save debug image every N successful frames

# --- CONSTANTS ---
# Landmark Indices (Same as GazeGene)
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_INNER_CORNER = 133
LEFT_OUTER_CORNER = 33
LEFT_EYE_CONTOUR  = [LEFT_OUTER_CORNER, LEFT_INNER_CORNER, 159, 145] 
RIGHT_INNER_CORNER = 362
RIGHT_OUTER_CORNER = 263
RIGHT_EYE_CONTOUR  = [RIGHT_OUTER_CORNER, RIGHT_INNER_CORNER, 386, 374] 
HEAD_ANCHORS = [1,9]

LEFT_EYE_LANDMARKS = LEFT_IRIS + LEFT_EYE_CONTOUR
RIGHT_EYE_LANDMARKS = RIGHT_IRIS + RIGHT_EYE_CONTOUR
LANDMARK_INDICES = LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS + HEAD_ANCHORS

# Output columns
GAZE_COLS = ['gaze_x', 'gaze_y', 'gaze_z', 'gaze_yaw', 'gaze_pitch']
LANDMARK_COLS_XY = [f"{idx}_x" for idx in LANDMARK_INDICES] + [f"{idx}_y" for idx in LANDMARK_INDICES]
FIELDNAMES = ['subject', 'recording', 'frame', 'split', 'image_path'] + GAZE_COLS + LANDMARK_COLS_XY

class Gaze360DatasetGenerator:
    def __init__(self, config, dataset_root, output_dir, batch_size=50):
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.config = config
        self.face_model3d = FaceModel()

        # Initialize Normalizer (Target Normalized Camera)
        self._normalized_camera = Camera(self.config.gaze_estimator.normalized_camera_params)
        self.normalizer = LandmarkNormalizer(self._normalized_camera)
        
        self.landmark_estimator = None # Lazy init

        # Buffers for writing to CSVs
        self.buffers = {
            'all': [],
            'train': [],
            'val': [],
            'test': [],
            'all_used': []
        }
        
        # File paths
        self.csv_paths = {
            'all': os.path.join(output_dir, "gaze360_normalized_ALL.csv"),
            'train': os.path.join(output_dir, "gaze360_normalized_TRAIN.csv"),
            'val': os.path.join(output_dir, "gaze360_normalized_VAL.csv"),
            'test': os.path.join(output_dir, "gaze360_normalized_TEST.csv"),
            'all_used': os.path.join(output_dir, "gaze360_normalized_ALL_USED.csv")
        }

        # Initialize files (Write headers)
        os.makedirs(output_dir, exist_ok=True)
        for key, path in self.csv_paths.items():
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=';')
                writer.writeheader()

    def _init_estimator(self):
        if self.landmark_estimator is None:
            logger.info("Initializing LandmarkEstimator...")
            self.landmark_estimator = LandmarkEstimator(self.config)

    def vector_to_pitch_yaw(self, v):
        x, y, z = v[0], v[1], v[2]
        norm = np.linalg.norm(v)
        if norm > 0: x, y, z = x / norm, y / norm, z / norm
        pitch = np.arcsin(np.clip(y, -1.0, 1.0))
        yaw = np.arctan2(x, -z)  # Z-Forward convention
        return yaw, pitch

    def get_dynamic_camera_params(self, w, h):
        """
        Generates camera parameters for Gaze360 images.
        Approximation: Focal length ~ Image Width (FOV ~53 deg).
        """
        c_x, c_y = w / 2.0, h / 2.0
        f_x, f_y = float(w), float(w)
        
        camera_matrix = np.array([
            [f_x, 0., c_x],
            [0., f_y, c_y],
            [0., 0., 1.]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((1, 5), dtype=np.float32)
        return camera_matrix, dist_coeffs

    def estimate_pose_pnp_scratch(self, face, camera_matrix, dist_coeffs=None):
        """
        Estimates head pose (R, t) from scratch using the 3D FaceModel.
        We do NOT use a guess here because Gaze360 doesn't provide head pose GT.
        """
        success, rvec, tvec = cv2.solvePnP(
            self.face_model3d.LANDMARKS,  # 3D points
            face.landmarks,               # 2D points
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP       # Robust starting method
        )
        
        # Refine with Iterative if EPNP succeeds
        if success:
             success, rvec, tvec = cv2.solvePnP(
                self.face_model3d.LANDMARKS,
                face.landmarks,
                camera_matrix,
                dist_coeffs,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        if not success:
            return None, None
        return rvec, tvec

    def process_dataset(self, limit_images=None):
        self._init_estimator()
        
        # 1. Load Metadata
        metadata_path = os.path.join(self.dataset_root, 'metadata.mat')
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata not found at {metadata_path}")
            return

        logger.info("Loading metadata.mat...")
        try:
            mat_data = scipy.io.loadmat(metadata_path, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return

        # Extract fields
        recordings = mat_data['recordings']
        recording_indices = mat_data['recording']
        person_ids = mat_data['person_identity']
        frame_ids = mat_data['frame']
        splits = mat_data['split'] # 0: train, 1: val, 2: test, 3:unused
        gaze_dirs_g360 = mat_data['gaze_dir'] # (N, 3)

        total_indices = len(frame_ids)
        if limit_images:
            total_indices = min(total_indices, limit_images)
            logger.info(f"Limiting processing to first {limit_images} images.")

        total_processed = 0
        total_skipped = 0
        
        for idx in range(total_indices):
            try:
                # 2. Get Info
                rec_idx = recording_indices[idx]
                rec_name = recordings[rec_idx]
                pid = person_ids[idx]
                fid = frame_ids[idx]
                split_val = splits[idx] # 0=Train, 1=Val, 2=Test, 3=Other

                # Construct Path: gaze360Dataset/imgs/rec_name/head/pid_06d/fid_06d.jpg
                rel_path = os.path.join('imgs', str(rec_name), 'head', f"{pid:06d}", f"{fid:06d}.jpg")
                full_path = os.path.join(self.dataset_root, rel_path)

                if not os.path.exists(full_path):
                    total_skipped += 1
                    continue

                # 3. Load Image
                image = cv2.imread(full_path)
                if image is None:
                    total_skipped += 1
                    continue

                h, w = image.shape[:2]
                
                # 4. Camera & Gaze Setup
                camera_matrix, dist_coeffs = self.get_dynamic_camera_params(w, h)
                
                # Convert Gaze360 (X-Left, Y-Up, Z-Back) -> OpenCV (X-Right, Y-Down, Z-Forward)
                g360_vec = gaze_dirs_g360[idx]
                gt_gaze_cam = np.array([-g360_vec[0], -g360_vec[1], g360_vec[2]])
                norm_val = np.linalg.norm(gt_gaze_cam)
                
                if norm_val < 1e-6 or np.isnan(g360_vec).any():
                    total_skipped += 1
                    continue
                
                gt_gaze_cam /= norm_val # Normalize vector

                # 5. Face Detection
                # Note: We don't undistort first because we assume Gaze360 images are effectively pinhole/rectilinear 
                # crops and we generated a camera matrix matching that assumption.
                faces = self.landmark_estimator.detect_faces(image)

                if faces:
                    face = faces[0]
                    
                    # 6. Pose Estimation (PnP) - No previous guess available
                    rvec, tvec = self.estimate_pose_pnp_scratch(face, camera_matrix, dist_coeffs)
                    
                    if rvec is None:
                        total_skipped += 1
                        continue

                    head_R_cv = cv2.Rodrigues(rvec)[0]
                    head_T_cv = tvec

                    all_landmarks = np.concatenate([face.landmarks, face.landmarks_eyes])
                    landmarks_subset = all_landmarks[LANDMARK_INDICES]

                    # 7. Normalization
                    M, R_norm = self.normalizer.compute_normalization_matrix(head_R_cv, head_T_cv, camera_matrix)
                    normalized_landmarks = self.normalizer.normalize_landmarks(landmarks_subset, M)
                    
                    # Rotate Gaze Vector into Normalized Space
                    gaze_vec_normalized = R_norm @ gt_gaze_cam
                    gaze_vec_normalized /= np.linalg.norm(gaze_vec_normalized)
                    
                    # Convert to spherical
                    yaw, pitch = self.vector_to_pitch_yaw(gaze_vec_normalized)

                    # 8. Prepare Data Row
                    row_data = {
                        'subject': f"{pid:06d}",
                        'recording': rec_name,
                        'frame': fid,
                        'split': split_val,
                        'image_path': rel_path,
                        'gaze_x': gaze_vec_normalized[0],
                        'gaze_y': gaze_vec_normalized[1],
                        'gaze_z': gaze_vec_normalized[2],
                        'gaze_yaw': yaw,
                        'gaze_pitch': pitch
                    }

                    for i, lm_idx in enumerate(LANDMARK_INDICES):
                        row_data[f"{lm_idx}_x"] = normalized_landmarks[i, 0]
                        row_data[f"{lm_idx}_y"] = normalized_landmarks[i, 1]

                    # 9. Add to Buffers
                    self.buffers['all'].append(row_data)
                    
                    if split_val == 0:
                        self.buffers['train'].append(row_data)
                    elif split_val == 1:
                        self.buffers['val'].append(row_data)
                    elif split_val == 2:
                        self.buffers['test'].append(row_data)

                    if split_val in range(3):
                        self.buffers['all_used'].append(row_data)

                    total_processed += 1

                    # 10. Debug Visualization
                    if DEBUG_MODE and total_processed % VIS_INTERVAL == 0:
                        self._save_debug_images(pid, rec_name, fid, image, M, normalized_landmarks, gaze_vec_normalized, yaw, pitch)

                else:
                    total_skipped += 1

                del image
                
                # Batch Write
                if len(self.buffers['all']) >= self.batch_size:
                    self.flush_buffers()
                    print(f"Processed: {total_processed} | Skipped: {total_skipped}", end='\r')
                    gc.collect()

            except Exception as e:
                logger.warning(f"Error processing {idx}: {e}")
                total_skipped += 1
                continue

        # Final Flush
        self.flush_buffers()
        print(f"\nDone. Total Processed: {total_processed}. Total Skipped: {total_skipped}")

    def flush_buffers(self):
        """Writes accumulated data to all relevant CSV files."""
        for key, buffer in self.buffers.items():
            if not buffer: continue
            
            try:
                with open(self.csv_paths[key], 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=';')
                    writer.writerows(buffer)
                self.buffers[key] = [] # Clear after writing
            except Exception as e:
                logger.error(f"Error flushing buffer {key}: {e}")

    def _save_debug_images(self, pid, rec, fid, raw_img, M, norm_landmarks, gaze_vec, yaw, pitch):
        # Create folder: processed_data/debug_normalized_img/recording/pid
        debug_dir = os.path.join(self.output_dir, "debug_normalized_img", str(rec), f"{pid:06d}")
        os.makedirs(debug_dir, exist_ok=True)
        
        h_norm = self._normalized_camera.height
        w_norm = self._normalized_camera.width

        # Warp
        warped_img = cv2.warpPerspective(raw_img, M, (w_norm, h_norm))

        # Draw Landmarks (Green)
        for (lx, ly) in norm_landmarks:
            cv2.circle(warped_img, (int(lx), int(ly)), 2, (0, 255, 0), -1)

        # Draw Gaze Arrow (Red)
        eye_center = np.mean(norm_landmarks, axis=0)
        arrow_len = 50
        start_point = (int(eye_center[0]), int(eye_center[1]))
        end_point = (
            int(start_point[0] + gaze_vec[0] * arrow_len),
            int(start_point[1] + gaze_vec[1] * arrow_len)
        )
        cv2.arrowedLine(warped_img, start_point, end_point, (0, 0, 255), 2, tipLength=0.3)
        
        # Text Info
        cv2.putText(warped_img, f"Y:{np.degrees(yaw):.1f} P:{np.degrees(pitch):.1f}", 
                   (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Reference Lines (Cyan)
        cx, cy = w_norm // 2, h_norm // 2
        color_grid = (255, 255, 0)
        cv2.line(warped_img, (0, cy), (w_norm, cy), color_grid, 1) # Horizontal center
        cv2.line(warped_img, (cx, 0), (cx, h_norm), color_grid, 1) # Vertical center
        
        cv2.imwrite(os.path.join(debug_dir, f"{fid:06d}.jpg"), warped_img)
        del warped_img

if __name__ == "__main__":
    CONFIG_PATH = 'configs/default_config.yaml'
    
    # --- CONFIGURATION ---
    # Gaze360 faces are not cropped tightly, so we need padding=0 usually, 
    # but the landmarks are small, so detection might need tuning.
    PADDING = 0.0 
    DET_CONF = 0.8 
    
    # Path settings
    DATASET_ROOT = './gaze360Dataset' # Must contain metadata.mat and imgs/
    OUTPUT_DIR = './datasets/Gaze360/'
    BATCH_SIZE = 50
    LIMIT_IMAGES = None # Set to None for full dataset, or int for testing

    if not os.path.exists(CONFIG_PATH):
        print(f"Config file not found at {CONFIG_PATH}. Please check path.")
    elif not os.path.exists(DATASET_ROOT):
        print(f"Dataset root not found at {DATASET_ROOT}. Please check path.")
    else:
        config = OmegaConf.load(CONFIG_PATH)
        config.face_detector.padding = PADDING
        config.face_detector.mediapipe_min_det_conf = DET_CONF
        
        # Gaze360 images are variable size, config height/width is irrelevant for input
        config.image.height, config.image.width = (None, None) 
        
        print("Starting Gaze360 Normalization Process...")
        generator = Gaze360DatasetGenerator(config, DATASET_ROOT, OUTPUT_DIR, BATCH_SIZE)
        generator.process_dataset(limit_images=LIMIT_IMAGES)
