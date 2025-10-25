"""
Fixed-offset IMU-based video stabilization model.
Uses a predefined camera-IMU offset without calibration.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
import time
from scipy.spatial.transform import Rotation
from .imu_fusion import quaternion_to_rotation_matrix

class IMUFixedModel:
    """
    Video stabilization model using IMU data with fixed camera offset.
    Camera is assumed to be 20cm in front of the IMU on the X axis.
    """
    
    def __init__(self, camera_offset: float = 0.35):
        """
        Initialize IMU fixed model.
        
        Args:
            camera_offset: Camera offset from IMU in meters (default: 0.35m = 35cm)
        """
        # Fixed translation from IMU to camera (20cm on X axis)
        self.t_camera_imu = np.array([camera_offset, 0.0, 0.0])
        
        # Identity rotation between camera and IMU frames
        self.R_camera_imu = np.eye(3)
        
    def _get_stabilization_transform(self, frame: np.ndarray, imu_data: Dict) -> np.ndarray:
        """
        Calculate stabilization transform for a frame using IMU data.
        
        Args:
            frame: Input frame
            imu_data: IMU measurement data
            
        Returns:
            3x3 homography transformation matrix
        """
        h, w = frame.shape[:2]
        
        # Get IMU rotation
        R_imu = quaternion_to_rotation_matrix(
            imu_data['qw'], imu_data['qx'], imu_data['qy'], imu_data['qz'])
            
        # Convert IMU rotation to camera frame
        R_camera = self.R_camera_imu @ R_imu @ self.R_camera_imu.T
        
        # Calculate translation due to rotation around offset point
        t = -R_camera @ self.t_camera_imu + self.t_camera_imu
        
        # Convert to pixel coordinates (assuming camera focal length = frame width)
        focal = w
        K = np.array([
            [focal, 0, w/2],
            [0, focal, h/2],
            [0, 0, 1]
        ])
        
        # Build homography
        H = K @ (R_camera + (t[:, np.newaxis] @ np.array([[0, 0, 1]]))) @ np.linalg.inv(K)
        return H
        
    def stabilize_frames(
        self,
        frames: List[np.ndarray],
        sensor_data: Optional[List[Dict]] = None,
        ref_idx: Optional[int] = None
    ) -> Dict:
        """
        Stabilize frames using IMU data with fixed offset.
        
        Args:
            frames: List of BGR images
            sensor_data: List of IMU measurements for each frame
            ref_idx: Optional reference frame index
            
        Returns:
            dict with warped frames and transformation data
        """
        # Validate inputs: frames must be provided and sensor_data must be a list
        if not frames:
            return {
                "warped": [], "orig": [], "scales": [],
                "translations": [], "rotations": [], "transforms": [],
                "inliers": [], "ref_idx": ref_idx
            }

        # sensor_data must be a list with one entry per frame; ensure no None elements
        if sensor_data is None:
            print("IMUFixedModel: sensor_data is None -> cannot stabilize.")
            return {
                "warped": [], "orig": [], "scales": [],
                "translations": [], "rotations": [], "transforms": [],
                "inliers": [], "ref_idx": ref_idx
            }

        if not isinstance(sensor_data, list) or len(sensor_data) < len(frames):
            print(f"IMUFixedModel: sensor_data length ({len(sensor_data) if isinstance(sensor_data, list) else 'N/A'}) does not match frames ({len(frames)}). Skipping.")
            return {"warped": [], "orig": [], "scales": [], "translations": [], "rotations": [], "transforms": [], "inliers": [], "ref_idx": ref_idx}

        missing_indices = [i for i, s in enumerate(sensor_data) if s is None]
        if missing_indices:
            print(f"IMUFixedModel: missing sensor entries at indices: {missing_indices} -> skipping this window")
            return {"warped": [], "orig": [], "scales": [], "translations": [], "rotations": [], "transforms": [], "inliers": [], "ref_idx": ref_idx}

        # Validate presence of required quaternion keys in each sensor dict
        required_keys = {'qw', 'qx', 'qy', 'qz'}
        for i, s in enumerate(sensor_data):
            if not isinstance(s, dict) or not required_keys.issubset(s.keys()):
                print(f"IMUFixedModel: invalid sensor data at index {i}: missing quaternion keys -> skipping window")
                return {"warped": [], "orig": [], "scales": [], "translations": [], "rotations": [], "transforms": [], "inliers": [], "ref_idx": ref_idx}
            
        n = len(frames)
        if ref_idx is None:
            ref_idx = n // 2
        if ref_idx < 0 or ref_idx >= n:
            raise ValueError("ref_idx out of range")
            
        # Reference shape
        h, w = frames[ref_idx].shape[:2]
        
        # Get reference frame transform
        H_ref = self._get_stabilization_transform(frames[ref_idx], sensor_data[ref_idx])
        H_ref_inv = np.linalg.inv(H_ref)
        
        # Initialize result arrays
        transforms = []
        warped = []
        scales = [1.0] * n
        translations = [(0.0, 0.0)] * n
        rotations = [0.0] * n
        inliers_list = [np.zeros((0,), dtype=np.uint8) for _ in range(n)]
        
        t1 = time.time()
        
        # Process each frame
        for i in range(n):
            # Get current frame transform
            H = self._get_stabilization_transform(frames[i], sensor_data[i])
            
            # Calculate relative transform
            H_relative = H_ref_inv @ H
            transforms.append(H_relative)
            
            # Extract scale and rotation from the upper 2x2 block
            A = H_relative[:2, :2]
            s = np.sqrt(np.mean(np.sum(A**2, axis=0)))
            scales[i] = s
            
            # Extract rotation angle from the 2x2 block
            cos_theta = (A[0,0] + A[1,1]) / (2 * s)
            sin_theta = (A[1,0] - A[0,1]) / (2 * s)
            angle = np.arctan2(sin_theta, cos_theta)
            rotations[i] = angle
            
            # Extract translation
            translations[i] = (float(H_relative[0,2]), float(H_relative[1,2]))
            
            # We already extracted rotations[i] and translations[i] above
            
            # Apply stabilization with automatic zoom to maintain valid pixels
            margin = 0.2  # 20% margin for zoom
            pts = np.array([[0,0,1], [w,0,1], [w,h,1], [0,h,1]]).T
            pts_warped = H_relative @ pts
            pts_warped = pts_warped[:2] / pts_warped[2]
            
            min_x, max_x = np.min(pts_warped[0]), np.max(pts_warped[0])
            min_y, max_y = np.min(pts_warped[1]), np.max(pts_warped[1])
            
            # Calculate zoom to keep all pixels valid with margin
            zoom_x = (1 - 2*margin) * w / (max_x - min_x)
            zoom_y = (1 - 2*margin) * h / (max_y - min_y)
            zoom = min(zoom_x, zoom_y)
            
            # Center the frame
            tx = (w - zoom * (max_x + min_x)) / 2
            ty = (h - zoom * (max_y + min_y)) / 2
            
            # Apply zoom and centering
            M_zoom = np.array([
                [zoom, 0, tx],
                [0, zoom, ty],
                [0, 0, 1]
            ])
            
            H_final = M_zoom @ H_relative
            
            # Apply transformation with border padding
            warped_frame = cv2.warpPerspective(
                frames[i], H_final, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0,0,0)
            )
            warped.append(warped_frame)
            
        t2 = time.time()
        print(f"IMU fixed stabilization time: {(t2 - t1)*1000:.3f}ms")
            
        return {
            "warped": warped,
            "orig": list(frames),
            "scales": scales,
            "translations": translations,
            "rotations": rotations,
            "transforms": transforms,
            "inliers": inliers_list,
            "ref_idx": ref_idx
        }