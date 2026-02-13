from typing import List
import mediapipe
import numpy as np
import cv2

from omegaconf import DictConfig
from face_model import Face

class LandmarkEstimator:
    def __init__(self, config: DictConfig):
        self.padding = config.face_detector.padding # padding for improving face detection of cropped images
        self.detector = mediapipe.solutions.face_mesh.FaceMesh(
                max_num_faces=config.face_detector.mediapipe_max_num_faces,
                static_image_mode=config.face_detector.mediapipe_static_image_mode,
                refine_landmarks=True,
                min_detection_confidence=config.face_detector.mediapipe_min_det_conf,  # Minimum confidence for face detection
                min_tracking_confidence=config.face_detector.mediapipe_min_track_conf  # Minimum confidence for landmark tracking
                )

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        if self.padding > 0: #  Modify image with padding
            pad_ratio = self.padding
            h, w = image.shape[:2]
            pad_h, pad_w = int(h * pad_ratio), int(w * pad_ratio)
            padded_img = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))

            faces = self._detect_faces_mediapipe(padded_img)
        
            valid_faces = []
            for face in faces: # Shift landmarks back accounting for padding
                face.landmarks -= np.array([pad_w, pad_h])
                face.landmarks_eyes -= np.array([pad_w, pad_h])
                face.bbox -= np.array([[pad_w, pad_h], [pad_w, pad_h]])
                valid_faces.append(face)
                
            return valid_faces
        else:
            return self._detect_faces_mediapipe(image)

    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Face]:
        detected = []
        h, w = image.shape[:2]
        predictions = self.detector.process(image[:, :, ::-1])
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h)
                                for pt in prediction.landmark],
                               dtype=np.float64)
                pts3D = np.array([(pt.x, pt.y, pt.z)
                                for pt in prediction.landmark],
                               dtype=np.float64)
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                detected.append(Face(bbox, pts[:-10],pts[-10:],pts3D))
        return detected
