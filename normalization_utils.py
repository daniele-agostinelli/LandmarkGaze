import numpy as np
from camera import Camera

class LandmarkNormalizer:
    """
    A unified class for Data Normalization (perspective warping) used in
    both Dataset Generation and Gaze Estimation.

    This ensures that the same procedure is applied during training and inference.
    Based on the method by Zhang et al. (MPIIGaze/ETH-XGaze).
    """

    def __init__(self, normalized_camera: Camera):
        """
        Args:
            normalized_camera_params: camera
            initializes dict containing 'focal_length', 'distance', 'size', and 'matrix'
        """
        self.norm_camera_params = {
                                    'focal_length': normalized_camera.focal_length,  # Virtual focal length - (pixels)
                                    'distance': normalized_camera.distance,  # Virtual distance - (m)
                                    'size': (normalized_camera.width, normalized_camera.height),  # Normalized image size (width, height) - (pixels)
                                    'matrix': normalized_camera.camera_matrix
                                    }

    def compute_normalization_matrix(self, head_rot_mat, head_pos_vec, camera_matrix):
        """
        Computes the normalization matrix M and the rotation matrix R_norm.

        Input:
            head_rot_mat: Rotation Matrix (3x3) in OpenCV Frame (Camera -> Head)
            head_pos_vec: Translation Vector (3,) in OpenCV Frame (Camera Center -> Face)
            camera_matrix: Intrinsic Matrix (3x3) of the source camera

        Returns:
            M: Homography matrix (3x3)
            R_norm: Rotation matrix (3x3) representing the rotation of the Normalized Camera
        """
        # 1. Z-Axis: Vector from Camera to Face
        # In OpenCV, Z is forward (positive).
        z_axis = head_pos_vec.flatten()
        z_axis = z_axis / np.linalg.norm(z_axis)

        # 2. X-Axis: Derived from Head Rotation to cancel Roll
        # We take the Head's X-axis (Right) projected onto the image plane
        head_x_axis = head_rot_mat[:, 0]

        # 3. Y-Axis: Perpendicular to Z and Head-Right
        # y = z (cross) head_right  <-- Note: order matters for coordinate system orientation
        # This defines Y pointing 'down' relative to the new Z
        y_axis = np.cross(z_axis, head_x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # 4. Re-orthogonalize X-Axis
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # 5. Construct Rotation Matrix R_norm (Rows are the new basis vectors)
        # This rotates World(Camera) -> Normalized Frame
        R_norm = np.vstack([x_axis, y_axis, z_axis])

        # Scale S
        dist = np.linalg.norm(head_pos_vec)
        norm_dist = self.norm_camera_params['distance']
        scale = 1.0
        if norm_dist > 0:
            scale = dist / norm_dist  # We scale the image down if face is far

        S = np.eye(3, dtype=np.float32)
        S[0, 0] = scale
        S[1, 1] = scale

        # Normalized Intrinsics
        K_norm = self.norm_camera_params['matrix']

        # 6. Compute M (Homography)
        M = K_norm @ S @ R_norm @ np.linalg.inv(camera_matrix)

        return M, R_norm

    def normalize_landmarks(self, landmarks, M):
        """
        Applies perspective warp (M) to 2D landmarks.

        Args:
            landmarks: (N, 2) numpy array
            M: (3, 3) Homography matrix

        Returns:
            normalized_landmarks: (N, 2) numpy array
        """
        num_pts = landmarks.shape[0]
        ones = np.ones((num_pts, 1))
        lm_hom = np.hstack([landmarks, ones])  # (N, 3)

        # Apply M: p_new = M @ p_old
        # Transpose logic: (M @ N_pts.T).T
        lm_new_hom = (M @ lm_hom.T).T

        # Project back to 2D (Perspective Division)
        # Avoid division by zero
        div = lm_new_hom[:, 2:3]
        #div[div < 1e-6] = 1e-6

        lm_new = lm_new_hom[:, :2] / div
        return lm_new