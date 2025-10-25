"""
IMU-based video stabilization model.

This implementation combines IMU data with visual features for robust video stabilization,
following the approach described in the paper "Efficient real-time video stabilization for UAVs
using only IMU data".
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from scipy.spatial.transform import Rotation
from .base import StabilizationModel


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix."""
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def estimate_camera_imu_transform(frames: List[np.ndarray], 
                                imu_data: List[Dict],
                                max_features: int = 2000) -> Tuple[np.ndarray, float]:
    """
    Estimate the transformation between camera and IMU coordinate systems.
    Uses visual feature tracking and IMU rotation data to find the best alignment.
    
    Args:
        frames: List of video frames
        imu_data: List of IMU measurements
        max_features: Maximum number of features to track
    
    Returns:
        R_c_i: 3x3 rotation matrix from IMU to camera coordinates
        time_offset: Estimated time offset between camera and IMU timestamps
    """
    # Extract feature tracks
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    
    tracks = []
    p0 = cv2.goodFeaturesToTrack(gray_frames[0], maxCorners=max_features, 
                                qualityLevel=0.01, minDistance=7, blockSize=7)
    
    if p0 is None:
        return np.eye(3), 0.0
        
    for i in range(1, len(frames)):
        p1, st, _ = cv2.calcOpticalFlowPyrLK(gray_frames[i-1], gray_frames[i], p0, None,
                                            winSize=(21, 21), maxLevel=3)
        if p1 is None:
            continue
            
        # Keep only good tracks
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        tracks.append((good_old, good_new))
        p0 = good_new.reshape(-1, 1, 2)
    
    # Extract IMU rotations
    R_imu = []
    for data in imu_data:
        R = quaternion_to_rotation_matrix(data['qw'], data['qx'], data['qy'], data['qz'])
        R_imu.append(R)
    
    # Estimate camera motion from feature tracks
    R_cam = []
    for (pts1, pts2) in tracks:
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.), 
                                     method=cv2.RANSAC, prob=0.999, threshold=3.0)
        _, R, _, _ = cv2.recoverPose(E, pts1, pts2, mask=mask)
        R_cam.append(R)
    
    # Find best alignment between camera and IMU rotations
    # Using Kabsch algorithm
    R_cam = np.array(R_cam)
    R_imu = np.array(R_imu[1:])  # Skip first frame to match R_cam length
    
    H = np.zeros((3,3))
    for Rc, Ri in zip(R_cam, R_imu):
        H += Rc @ Ri.T
        
    U, _, Vt = np.linalg.svd(H)
    R_c_i = U @ Vt
    
    # Estimate time offset using cross-correlation of rotation angles
    time_offset = 0.0  # TODO: Implement time offset estimation if needed
    
    return R_c_i, time_offset


class IMUFusionModel(StabilizationModel):
    """
    Video stabilization model that fuses IMU data with visual features.
    
    Key features:
    - Uses IMU orientation for initial motion estimation
    - Refines with visual features when available
    - Handles IMU-camera calibration
    - Supports online processing (no future frames needed)
    """
    
    def __init__(self, max_features: int = 2000):
        """
        Initialize IMU fusion model.
        
        Args:
            max_features: Maximum number of features to track
        """
        self.max_features = max_features
        self.R_camera_imu = np.eye(3)  # Camera to IMU transformation
        self.time_offset = 0.0  # Camera-IMU time offset
        self.is_calibrated = False
        
    def calibrate(self, frames: List[np.ndarray], imu_data: List[Dict]) -> None:
        """
        Calibrate camera-IMU transformation using a sequence of frames and IMU data.
        """
        self.R_camera_imu, self.time_offset = estimate_camera_imu_transform(
            frames, imu_data, self.max_features)
        self.is_calibrated = True
        
    def _get_imu_transform(self, imu_data: Dict) -> np.ndarray:
        """Convert IMU measurement to camera frame transformation."""
        R_imu = quaternion_to_rotation_matrix(
            imu_data['qw'], imu_data['qx'], imu_data['qy'], imu_data['qz'])
        
        # Convert IMU rotation to camera frame
        R_camera = self.R_camera_imu @ R_imu @ self.R_camera_imu.T
        return R_camera
        
    def stabilize_frames(
        self,
        frames: List[np.ndarray],
        sensor_data: Optional[List[Dict]] = None,
        ref_idx: Optional[int] = None
    ) -> Dict:
        """
        Stabilize frames using IMU data and visual features.
        
        Args:
            frames: List of BGR images
            sensor_data: List of IMU measurements for each frame
            ref_idx: Optional reference frame index
            
        Returns:
            dict with warped frames and transformation data
        """
        if not frames or sensor_data is None:
            return {
                "warped": [], "orig": [], "scales": [], 
                "translations": [], "rotations": [], "transforms": [],
                "inliers": [], "ref_idx": ref_idx
            }
            
        n = len(frames)
        if ref_idx is None:
            ref_idx = n // 2
        if ref_idx < 0 or ref_idx >= n:
            raise ValueError("ref_idx out of range")
            
        # Calibrate camera-IMU transform if needed
        if not self.is_calibrated:
            self.calibrate(frames, sensor_data)
            
        # Reference shape
        h, w = frames[ref_idx].shape[:2]
        
        # Initialize result arrays
        scales = [1.0] * n
        translations = [(0.0, 0.0)] * n
        rotations = [0.0] * n
        transforms = []
        warped = []
        inliers_list = [np.zeros((0,), dtype=np.uint8) for _ in range(n)]
        
        # Get reference frame IMU orientation
        R_ref = self._get_imu_transform(sensor_data[ref_idx])
        
        t1 = time.time()
        
        # Process each frame
        for i in range(n):
            if i == ref_idx:
                warped.append(frames[i].copy())
                transforms.append(np.eye(3))
                continue
                
            # Get IMU-based rotation
            R_current = self._get_imu_transform(sensor_data[i])
            R_relative = R_ref @ R_current.T
            
            # Convert to angle-axis representation
            angle = np.arccos((np.trace(R_relative) - 1) / 2)
            if abs(angle) > 1e-10:
                axis = np.array([
                    R_relative[2,1] - R_relative[1,2],
                    R_relative[0,2] - R_relative[2,0],
                    R_relative[1,0] - R_relative[0,1]
                ]) / (2 * np.sin(angle))
            else:
                axis = np.array([0, 0, 1])
                
            rotations[i] = angle if axis[2] >= 0 else -angle
            
            # Estimate translation and scale using visual features
            gray_current = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray_ref = cv2.cvtColor(frames[ref_idx], cv2.COLOR_BGR2GRAY)
            
            # Find matching features
            p0 = cv2.goodFeaturesToTrack(gray_current, maxCorners=self.max_features,
                                       qualityLevel=0.01, minDistance=7, blockSize=7)
            if p0 is not None:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(gray_current, gray_ref, p0, None,
                                                    winSize=(21, 21), maxLevel=3)
                if p1 is not None:
                    st = st.flatten()
                    pts_src = p0[st==1].reshape(-1, 2)
                    pts_dst = p1[st==1].reshape(-1, 2)
                    
                    if len(pts_src) >= 3:
                        # Estimate scale and translation
                        M, inliers = cv2.estimateAffinePartial2D(
                            pts_src, pts_dst, method=cv2.RANSAC,
                            ransacReprojThreshold=3.0, maxIters=2000)
                            
                        if M is not None:
                            inliers = inliers.flatten().astype(bool)
                            inliers_list[i] = inliers
                            
                            if np.count_nonzero(inliers) >= 3:
                                scales[i] = np.sqrt(M[0,0]**2 + M[0,1]**2)
                                translations[i] = (float(M[0,2]), float(M[1,2]))
            
            # Build full transformation matrix
            cos_theta = np.cos(rotations[i])
            sin_theta = np.sin(rotations[i])
            s = scales[i]
            tx, ty = translations[i]
            
            transform = np.array([
                [s*cos_theta, -s*sin_theta, tx],
                [s*sin_theta, s*cos_theta, ty],
                [0, 0, 1]
            ])
            transforms.append(transform)
            
            # Calculate required zoom to maintain valid pixels
            margin = 0.2  # 20% margin for zoom
            pts = np.array([[0,0,1], [w,0,1], [w,h,1], [0,h,1]]).T
            pts_warped = transform @ pts
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
            
            transform_final = M_zoom @ transform
            
            # Apply transformation with black borders
            warped_frame = cv2.warpPerspective(
                frames[i], transform_final, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0,0,0)
            )
            warped.append(warped_frame)
            
        t2 = time.time()
        print(f"IMU fusion stabilization time: {(t2 - t1)*1000:.3f}ms")
            
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