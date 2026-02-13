# THIS SCRIPT SERVES TO GENERATE A DATASET FROM A DATASET OF LABELLED IMAGES
# This version uses the normalization approach via NormalizationUtils.
# It converts Blender Coordinates to Standard OpenCV Coordinates BEFORE normalization.

import os
import cv2
import numpy as np
import pickle
import logging
import csv
import gc
from omegaconf import OmegaConf

# --- IMPORTS FROM LOCAL FILES ---
try:
    from face_landmark_estimator import LandmarkEstimator
    from normalization_utils import LandmarkNormalizer
    from camera import Camera
    from face_model import FaceModel
except ImportError:
    print("Error: Could not import LandmarkEstimator, LandmarkNormalizer or Camera.")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- USER SETTINGS ---
DEBUG_MODE = True  # Set to True to save visualization images

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

class DatasetGenerator:
    def __init__(self, config, dataset_root, output_csv_path, output_img_root, batch_size=50):
        self.dataset_root = dataset_root
        self.output_csv_path = output_csv_path
        self.output_img_root = output_img_root
        self.batch_size = batch_size
        self.config = config
        self.face_model3d = FaceModel()

        # Initialize Normalizer
        self._normalized_camera = Camera(self.config.gaze_estimator.normalized_camera_params)
        self.normalizer = LandmarkNormalizer(self._normalized_camera)
        self.images_processed_total = 0
        self.fieldnames = ['subject', 'camera', 'image_path'] + GAZE_COLS + LANDMARK_COLS_XY

        # Coordinate Transformation Matrix: Blender -> OpenCV
        self.CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

        # Lazy initialization of estimator
        self.landmark_estimator = None

    def _init_estimator(self):
        if self.landmark_estimator is None:
            logger.info("Initializing LandmarkEstimator...")
            self.landmark_estimator = LandmarkEstimator(self.config)

    def reset_estimator(self):
        """Destroys and recreates the landmark estimator to free leaked memory."""
        logger.info(f"Routine Maintenance: Resetting LandmarkEstimator after {self.images_processed_total} images...")
        if self.landmark_estimator is not None:
            del self.landmark_estimator
            self.landmark_estimator = None

        gc.collect()  # Force Garbage Collection
        self._init_estimator()   # Re-init

    def vector_to_pitch_yaw(self, v):
        x, y, z = v[0], v[1], v[2]
        norm = np.linalg.norm(v)
        if norm > 0: x, y, z = x / norm, y / norm, z / norm
        pitch = np.arcsin(np.clip(y, -1.0, 1.0))
        yaw = np.arctan2(x, -z)  # Note: This assumes Z-Forward convention for yaw
        return yaw, pitch

    def prune_and_get_last_state(self):
        """
        Reads the last line of the CSV to find where we left off.
        Truncates the file to remove that last line (so we can re-process it safely).
        Returns: (last_subject, last_camera, last_image_path) or None
        """
        if not os.path.exists(self.output_csv_path) or os.path.getsize(self.output_csv_path) == 0:
            # Initialize new file
            with open(self.output_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, delimiter=';')
                writer.writeheader()
            return None

        # Read backwards to find the last complete line
        with open(self.output_csv_path, 'rb+') as f:
            f.seek(0, os.SEEK_END)
            end_pos = f.tell()

            if end_pos == 0:
                return None

            # Go back one byte to skip the very last newline if it exists
            pos = end_pos - 1
            while pos > 0 and f.read(1) != b'\n':
                pos -= 1
                f.seek(pos, os.SEEK_SET)

            # Check if we are at the very beginning (only header or empty)
            if pos <= 0:
                return None

            # Now pos is at the newline character BEFORE the last row
            # We want to read the last row to know what it was
            f.seek(pos + 1, os.SEEK_SET)
            last_line = f.readline().decode('utf-8').strip()

            if not last_line:
                return None

            # Truncate the file at 'pos' (removing the last line entirely)
            # We add 1 to keep the newline of the previous row
            logger.info("Found existing progress. Pruning last entry and resuming...")
            f.seek(pos + 1, os.SEEK_SET)
            f.truncate()

        # Parse the last line to get identifiers
        try:
            parts = last_line.split(';')
            # subject is idx 0, camera is idx 1, image_path is idx 2 based on fieldnames
            return parts[0], parts[1], parts[2]
        except Exception as e:
            logger.warning(f"Could not parse last line for resume: {e}. Starting over.")
            return None

    def estimate_pose_pnp(self, face, rvec_guess_cv, tvec_guess_cv, camera_matrix, dist_coeffs=None):
        """
        Estimates the head pose (R, t) of the FaceModel relative to the camera
        using solvePnP. This effectively finds the position of the FaceModel's
        origin (which is now the face center defined as the average between eyes and nose)
        in the camera frame.
        """
        success, rvec, tvec = cv2.solvePnP(
            self.face_model3d.LANDMARKS,  # 3D points subset
            face.landmarks,  # 2D points subset
            camera_matrix,
            dist_coeffs,
            rvec=rvec_guess_cv,  # Use CORRECTED CV space pose
            tvec=tvec_guess_cv,  # Use CORRECTED CV space pose
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None

        return rvec, tvec

    def process_dataset(self, subjects=None, cameras=None):
        # 1. Determine Resume State
        resume_state = self.prune_and_get_last_state()
        skipping_mode = (resume_state is not None)

        if skipping_mode:
            target_subj, target_cam, target_path = resume_state
            logger.info(f"RESUMING from: {target_subj} - {target_cam} - {target_path}")

        # 2. Init estimator
        self._init_estimator()

        if subjects is None:
            subjects = [d for d in os.listdir(self.dataset_root) if
                        os.path.isdir(os.path.join(self.dataset_root, d)) and d.startswith('subject')]
            subjects.sort()

        if cameras is None:
            cameras = [f'camera{i}' for i in range(9)]

        total_processed = 0
        total_skipped = 0
        buffer = []
        images_since_reset = 0

        for subject in subjects:
            # Optimization: Skip whole subjects if we are resuming and haven't reached the target subject
            if skipping_mode and subject != target_subj and subject < target_subj:
                continue

            for camera in cameras:
                # Optimization: Skip cameras if we are in the target subject but haven't reached target cam
                if skipping_mode and subject == target_subj and camera != target_cam and camera < target_cam:
                    continue

                logger.info(f"Processing {subject} - {camera}...")

                label_dir = os.path.join(self.dataset_root, subject, 'labels')
                complex_label_path = os.path.join(label_dir, f'complex_label_{camera}.pkl')
                gaze_label_path = os.path.join(label_dir, f'gaze_label_{camera}.pkl')

                if not os.path.exists(complex_label_path) or not os.path.exists(gaze_label_path):
                    # Only log warning if we aren't skipping (to reduce console spam)
                    if not skipping_mode:
                        logger.warning(f"Labels not found for {subject} {camera}, skipping.")
                    continue

                try:
                    with open(complex_label_path, 'rb') as f:
                        complex_data = pickle.load(f)
                    with open(gaze_label_path, 'rb') as f:
                        gaze_data = pickle.load(f)
                except Exception as e:
                    logger.error(f"Error loading pickle for {subject} {camera}: {e}")
                    continue

                num_entries = len(complex_data['img_path'])

                for idx in range(num_entries):
                    rel_img_path = complex_data['img_path'][idx]

                    # --- RESUME LOGIC ---
                    if skipping_mode:
                        # Check if this is the target
                        if subject == target_subj and camera == target_cam and rel_img_path == target_path:
                            logger.info("Found resume point! Resuming processing...")
                            skipping_mode = False
                            # We found the last one processed. We want to REDO it.
                            # So we proceed to process this current 'idx'.
                        else:
                            # Still searching
                            continue
                    # --------------------

                    try:
                        full_img_path = os.path.join(self.dataset_root, subject, rel_img_path)

                        if not os.path.exists(full_img_path):
                            total_skipped += 1
                            continue

                        # --- 1. Load & Convert Data ---
                        head_R_blender = gaze_data['head_R_mat'][idx] # in CCS
                        head_R_cv = head_R_blender @ self.CV_TO_BLENDER
                        rvec_guess_cv, _ = cv2.Rodrigues(head_R_cv)

                        head_T_raw = gaze_data['head_T_vec'][idx] # (cm) in CCS
                        # --- UNIT CORRECTION (cm -> m) ---
                        head_T_guess_cv = head_T_raw.copy()/100.0 # (m)

                        if head_T_raw is None or not np.isfinite(head_T_raw).all():
                            logger.warning(f"Skipping {rel_img_path}: Invalid Head Translation Vector (NaN/Inf)")
                            total_skipped += 1
                            continue
                            
                        if np.linalg.norm(head_T_raw) < 1e-6:
                            logger.warning(f"Skipping {rel_img_path}: Head Translation Vector is Zero")
                            total_skipped += 1
                            continue

                        try:
                            gt_visual_L = gaze_data['visual_axis_L'][idx]
                            gt_visual_R = gaze_data['visual_axis_R'][idx]
                            gaze_cam = (gt_visual_L + gt_visual_R) / 2.0
                            gaze_cam /= np.linalg.norm(gaze_cam)
                        except KeyError:
                            total_skipped += 1
                            continue

                        # --- 2. Load and Undistort ---
                        image = cv2.imread(full_img_path)
                        if image is None:
                            total_skipped += 1
                            continue

                        crop_intrinsics = complex_data['intrinsic_matrix_cropped'][idx]
                        undistorted = cv2.undistort(image, crop_intrinsics, np.zeros(5))
                        del image  # Free raw image immediately

                        # --- 3. Detect Faces ---
                        faces = self.landmark_estimator.detect_faces(undistorted)

                        if faces:
                            face = faces[0]
                            rvec_est, head_T_cv = self.estimate_pose_pnp(face, rvec_guess_cv, head_T_guess_cv, crop_intrinsics)
                            head_R_cv = cv2.Rodrigues(rvec_est)[0]

                            all_landmarks = np.concatenate([face.landmarks, face.landmarks_eyes])
                            landmarks_subset = all_landmarks[LANDMARK_INDICES]

                            # --- 4. Apply Normalization ---
                            M, R_norm = self.normalizer.compute_normalization_matrix(head_R_cv, head_T_cv,
                                                                                     crop_intrinsics)
                            normalized_landmarks = self.normalizer.normalize_landmarks(landmarks_subset, M)
                            gaze_vec_normalized = R_norm @ gaze_cam
                            gaze_vec_normalized /= np.linalg.norm(gaze_vec_normalized)
                            yaw, pitch = self.vector_to_pitch_yaw(gaze_vec_normalized)

                            row_data = {
                                'subject': subject,
                                'camera': camera,
                                'image_path': rel_img_path,
                                'gaze_x': gaze_vec_normalized[0],
                                'gaze_y': gaze_vec_normalized[1],
                                'gaze_z': gaze_vec_normalized[2],
                                'gaze_yaw': yaw,
                                'gaze_pitch': pitch
                            }

                            for i, lm_idx in enumerate(LANDMARK_INDICES):
                                row_data[f"{lm_idx}_x"] = normalized_landmarks[i, 0]
                                row_data[f"{lm_idx}_y"] = normalized_landmarks[i, 1]

                            buffer.append(row_data)
                            total_processed += 1
                            self.images_processed_total += 1
                            images_since_reset += 1

                            # --- 5. DEBUG VISUALIZATION ---
                            if DEBUG_MODE and total_processed % 100 == 0:
                                self._save_debug_images(subject, camera, idx, undistorted, M, normalized_landmarks, gaze_vec_normalized, yaw, pitch)
                        else:
                            total_skipped += 1

                        # Explicit Cleanup for heavy objects
                        del undistorted
                        del faces

                    except Exception as e:
                        logger.warning(f"Error processing image {idx} in {subject}/{camera}: {e}")
                        total_skipped += 1
                        continue

                    if len(buffer) >= self.batch_size:
                        self.write_buffer(buffer)
                        buffer = []
                        print(f"Processed: {total_processed} | Skipped: {total_skipped}", end='\r')
                        gc.collect()

                # End of Camera Loop cleanup
                del complex_data
                del gaze_data
                gc.collect()

        if buffer:
            self.write_buffer(buffer)

        print(f"\nDone. Total Processed: {total_processed}. Total Skipped: {total_skipped}")

    def write_buffer(self, buffer):
        if not buffer: return
        with open(self.output_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, delimiter=';')
            writer.writerows(buffer)

    def _save_debug_images(self, subject, camera, idx, undistorted, M, norm_landmarks, gaze_vec, yaw, pitch):
        debug_dir = os.path.join(self.output_img_root, f"debug_normalized_img/{subject}/{camera}")
        os.makedirs(debug_dir, exist_ok=True)
        h_norm = self._normalized_camera.height
        w_norm = self._normalized_camera.width

        # 1. Warped Image (Verify Scale & Orientation)
        warped_img = cv2.warpPerspective(undistorted, M, (w_norm, h_norm))

        # Draw Landmarks
        for (lx, ly) in norm_landmarks:
            cv2.circle(warped_img, (int(lx), int(ly)), 2, (0, 255, 0), -1)

        # Draw Gaze
        eye_center = np.mean(norm_landmarks, axis=0)
        arrow_len = 50
        start_point = (int(eye_center[0]), int(eye_center[1]))
        end_point = (
            int(start_point[0] + gaze_vec[0] * arrow_len),
            int(start_point[1] + gaze_vec[1] * arrow_len)
        )
        cv2.arrowedLine(warped_img, start_point, end_point, (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(warped_img, f"Yaw: {np.degrees(yaw):.1f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),1)

        # Draw reference lines in cyan
        cx, cy = w_norm // 2, h_norm // 2
        color_grid = (255, 255, 0)  # Cyan
        # Horizontal Reference Line - Eye Line should be parallel to this line
        cv2.line(warped_img, (0, cy), (w_norm, cy), color_grid, 1)
        # Vertical Reference Line - Face center should be at the image center
        cv2.line(warped_img, (cx, 0), (cx, h_norm), color_grid, 1)
        
        cv2.imwrite(os.path.join(debug_dir, f"dbg_{subject}_{camera}_{idx}.jpg"), warped_img)

        # Clean up
        del warped_img

if __name__ == "__main__":
    CONFIG_PATH = 'configs/default_config.yaml'
    PADDING = 0.25 # pad cropped image to enhance face detection
    DET_CONF = 0.8 # mediapipe detection confidence
    IMAGE_SIZE = (448,448)
    DATASET_ROOT = './GazeGeneDataset/GazeGene_FaceCrops'
    OUTPUT_CSV = f'./datasets/GazeGene/gazegene_normalized_ALL.csv'
    OUTPUT_IMG = './GazeGeneDataset/Processed data'
    BATCH_SIZE = 50

    if not os.path.exists(CONFIG_PATH):
        print(f"Config file not found at {CONFIG_PATH}. Please check path.")
    elif not os.path.exists(DATASET_ROOT):
        print(f"Dataset root not found at {DATASET_ROOT}. Please check path.")
    else:
        config = OmegaConf.load(CONFIG_PATH)
        config.face_detector.padding = PADDING
        config.face_detector.mediapipe_min_det_conf = DET_CONF
        config.image.height, config.image.width = IMAGE_SIZE
        generator = DatasetGenerator(config, DATASET_ROOT, OUTPUT_CSV, OUTPUT_IMG, BATCH_SIZE)
        generator.process_dataset()
