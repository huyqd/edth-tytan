"""
Sensor Fusion Stabilization Model.

This model combines visual (optical flow) and IMU sensor data for improved
video stabilization. It uses:
- Quaternions and angular rates from IMU for rotation estimation
- Optical flow for translation and scale estimation
- Complementary filtering to fuse both sources
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional
from .base import StabilizationModel
from .baseline import get_matches_kp


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        q: Quaternion as [qw, qx, qy, qz] or dict with keys 'qw', 'qx', 'qy', 'qz'

    Returns:
        3x3 rotation matrix
    """
    if isinstance(q, dict):
        qw, qx, qy, qz = q['qw'], q['qx'], q['qy'], q['qz']
    else:
        qw, qx, qy, qz = q

    # Normalize quaternion
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if norm < 1e-8:
        return np.eye(3)

    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

    return R


def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw) in radians.
    Using ZYX convention (yaw-pitch-roll).

    Args:
        R: 3x3 rotation matrix

    Returns:
        (roll, pitch, yaw) in radians
    """
    # Check for gimbal lock
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return roll, pitch, yaw


def estimate_imu_rotation(sensor_data_src, sensor_data_dst):
    """
    Estimate rotation between two frames using IMU quaternion data.

    Args:
        sensor_data_src: Sensor data dict for source frame
        sensor_data_dst: Sensor data dict for destination frame

    Returns:
        Rotation angle in radians (around Z-axis for 2D image plane)
        If sensor data is missing, returns None
    """
    if sensor_data_src is None or sensor_data_dst is None:
        return None

    # Check if quaternion data is available
    quat_keys = ['qw', 'qx', 'qy', 'qz']
    if not all(k in sensor_data_src and k in sensor_data_dst for k in quat_keys):
        return None

    try:
        # Get rotation matrices from quaternions
        R_src = quaternion_to_rotation_matrix(sensor_data_src)
        R_dst = quaternion_to_rotation_matrix(sensor_data_dst)

        # Compute relative rotation: R_rel = R_dst * R_src^T
        R_rel = R_dst @ R_src.T

        # Extract yaw angle (rotation around Z-axis, which affects the image plane)
        # For UAV stabilization, we primarily care about roll for the image plane
        roll, pitch, yaw = rotation_matrix_to_euler(R_rel)

        # Use roll as the primary rotation for image stabilization
        # (camera looking down, roll rotates the image)
        return roll

    except Exception as e:
        print(f"Warning: Failed to compute IMU rotation: {e}")
        return None


def estimate_rotation_from_optical_flow(pts_src, pts_dst, center):
    """
    Estimate rotation angle from optical flow point correspondences.

    Args:
        pts_src: Source points (N, 2)
        pts_dst: Destination points (N, 2)
        center: Image center (cx, cy)

    Returns:
        Rotation angle in radians
    """
    if len(pts_src) < 3:
        return 0.0

    # Compute vectors from center to each point
    v_src = pts_src - center
    v_dst = pts_dst - center

    # Compute angles for each point pair
    angles = []
    for vs, vd in zip(v_src, v_dst):
        # Skip points too close to center (unstable)
        if np.linalg.norm(vs) < 10 or np.linalg.norm(vd) < 10:
            continue

        # Compute angle between vectors
        angle = np.arctan2(vd[1], vd[0]) - np.arctan2(vs[1], vs[0])

        # Normalize to [-pi, pi]
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        angles.append(angle)

    if not angles:
        return 0.0

    # Use median to be robust to outliers
    return float(np.median(angles))


def fuse_rotations(imu_rotation, optical_rotation, imu_confidence=0.7):
    """
    Fuse IMU and optical flow rotation estimates using complementary filtering.

    Args:
        imu_rotation: Rotation from IMU (radians) or None
        optical_rotation: Rotation from optical flow (radians)
        imu_confidence: Weight for IMU data (0 to 1)

    Returns:
        Fused rotation angle in radians
    """
    if imu_rotation is None:
        # No IMU data, use optical flow only
        return optical_rotation

    # Complementary filter: trust IMU for high-frequency, optical flow for low-frequency
    # Simple weighted average for this implementation
    fused = imu_confidence * imu_rotation + (1 - imu_confidence) * optical_rotation

    return fused


def estimate_transform_with_fusion(src_gray, dst_gray, sensor_data_src, sensor_data_dst,
                                   img_shape, imu_weight=0.7):
    """
    Estimate transformation between frames using sensor fusion.

    Combines optical flow (for translation and scale) with IMU data (for rotation).

    Args:
        src_gray: Source grayscale image
        dst_gray: Destination grayscale image
        sensor_data_src: IMU sensor data for source frame
        sensor_data_dst: IMU sensor data for destination frame
        img_shape: Image shape (H, W)
        imu_weight: Weight for IMU rotation (0 to 1)

    Returns:
        (rotation_angle, scale, translation, inliers)
        - rotation_angle: rotation in radians
        - scale: scale factor
        - translation: (tx, ty) tuple
        - inliers: boolean mask of inlier points
    """
    # Get optical flow correspondences
    pts_src, pts_dst, _ = get_matches_kp(src_gray, dst_gray)

    if len(pts_src) < 3:
        # Not enough points, return identity transform
        return 0.0, 1.0, (0.0, 0.0), np.zeros((0,), dtype=np.uint8)

    # Estimate transformation using RANSAC
    # Try to get a similarity transform (scale + rotation + translation)
    M, inliers = cv2.estimateAffinePartial2D(
        pts_src, pts_dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000
    )

    if M is None or inliers is None:
        return 0.0, 1.0, (0.0, 0.0), np.zeros((0,), dtype=np.uint8)

    inliers = inliers.flatten().astype(bool)

    if np.count_nonzero(inliers) < 3:
        return 0.0, 1.0, (0.0, 0.0), inliers

    # Extract transformation parameters from affine matrix
    # M = [[a, -b, tx],
    #      [b,  a, ty]]
    # where a = s*cos(θ), b = s*sin(θ)
    a, b = M[0, 0], M[1, 0]
    tx, ty = M[0, 2], M[1, 2]

    # Extract scale and rotation from optical flow
    optical_scale = np.sqrt(a**2 + b**2)
    optical_rotation = np.arctan2(b, a)

    # Get rotation from IMU
    imu_rotation = estimate_imu_rotation(sensor_data_src, sensor_data_dst)

    # Fuse rotations
    fused_rotation = fuse_rotations(imu_rotation, optical_rotation, imu_confidence=imu_weight)

    # Use fused rotation with optical flow scale
    scale = optical_scale
    rotation_angle = fused_rotation
    translation = (float(tx), float(ty))

    return rotation_angle, scale, translation, inliers


def warp_with_transform(img, rotation, scale, tx, ty, output_shape):
    """
    Warp image with rotation, scale, and translation.

    Args:
        img: Input image
        rotation: Rotation angle in radians
        scale: Scale factor
        tx, ty: Translation
        output_shape: Output image shape (H, W)

    Returns:
        Warped image
    """
    h, w = output_shape[:2]
    center = (w / 2, h / 2)

    # Create rotation + scale matrix around image center
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)

    # Combined transformation matrix: T_center * R * S * T_center_inv * T_translate
    # First translate to origin, then scale+rotate, then translate back and apply translation
    M = np.array([
        [scale * cos_r, -scale * sin_r, tx + center[0] * (1 - scale * cos_r) + center[1] * scale * sin_r],
        [scale * sin_r,  scale * cos_r, ty + center[1] * (1 - scale * cos_r) - center[0] * scale * sin_r]
    ], dtype=np.float32)

    warped = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped


class FusionModel(StabilizationModel):
    """
    Sensor fusion stabilization model.

    Combines visual (optical flow) and IMU sensor data for improved stabilization:
    - Uses IMU quaternions and angular rates for rotation estimation
    - Uses optical flow for translation and scale estimation
    - Fuses both sources using complementary filtering
    """

    def __init__(self, max_features: int = 2000, imu_weight: float = 0.7, enable_profiling: bool = False):
        """
        Initialize fusion model.

        Args:
            max_features: Maximum number of features to track (default: 2000)
            imu_weight: Weight for IMU rotation data (0 to 1, default: 0.7)
                       Higher values trust IMU more, lower values trust optical flow more
            enable_profiling: Enable detailed performance profiling (default: False)
        """
        self.max_features = max_features
        self.imu_weight = imu_weight
        self.enable_profiling = enable_profiling

        # Performance tracking
        self.perf_stats = {
            'optical_flow_ms': [],
            'imu_processing_ms': [],
            'transform_fusion_ms': [],
            'warping_ms': [],
            'total_ms': []
        }

    def stabilize_frames(
        self,
        frames: List[np.ndarray],
        sensor_data: Optional[List[Dict]] = None,
        ref_idx: Optional[int] = None
    ) -> Dict:
        """
        Stabilize frames using sensor fusion.

        Args:
            frames: List of BGR images
            sensor_data: List of sensor data dicts (one per frame)
            ref_idx: Optional reference frame index

        Returns:
            dict with warped frames, transformations, and metadata
        """
        if not frames:
            return {
                "warped": [], "orig": [], "scales": [], "translations": [],
                "rotations": [], "inliers": [], "ref_idx": ref_idx, "transforms": []
            }

        n = len(frames)
        if ref_idx is None:
            ref_idx = n // 2
        if ref_idx < 0 or ref_idx >= n:
            raise ValueError("ref_idx out of range")

        # Reference frame properties
        h, w = frames[ref_idx].shape[:2]

        t_start = time.perf_counter()
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        t_gray = time.perf_counter()

        ref_gray = grays[ref_idx]
        ref_sensor = sensor_data[ref_idx] if sensor_data else None

        # Initialize output lists
        scales = [1.0] * n
        translations = [(0.0, 0.0)] * n
        rotations = [0.0] * n
        inliers_list = [np.zeros((0,), dtype=np.uint8) for _ in range(n)]

        # Detailed timing breakdown
        time_optical_flow = 0.0
        time_imu = 0.0
        time_fusion = 0.0

        # Estimate transform from each frame to reference
        for i in range(n):
            if i == ref_idx:
                continue

            sensor_i = sensor_data[i] if sensor_data else None

            t_frame_start = time.perf_counter()

            # Estimate transform with sensor fusion (with internal timing)
            rotation, scale, translation, inliers = estimate_transform_with_fusion(
                grays[i], ref_gray,
                sensor_i, ref_sensor,
                (h, w),
                imu_weight=self.imu_weight
            )

            t_frame_end = time.perf_counter()
            time_optical_flow += (t_frame_end - t_frame_start)

            rotations[i] = rotation
            scales[i] = scale
            translations[i] = translation
            inliers_list[i] = inliers

        t_estimate_end = time.perf_counter()

        # Warp all frames to reference
        warped = [None] * n
        t_warp_start = time.perf_counter()

        for i in range(n):
            if i == ref_idx:
                warped[i] = frames[i].copy()
            else:
                t_single_warp_start = time.perf_counter()
                warped[i] = warp_with_transform(
                    frames[i],
                    rotations[i],
                    scales[i],
                    translations[i][0],
                    translations[i][1],
                    (h, w)
                )
                t_single_warp_end = time.perf_counter()
                time_fusion += (t_single_warp_end - t_single_warp_start)

        t_warp_end = time.perf_counter()
        t_total = time.perf_counter()

        # Calculate timings in milliseconds
        time_total_ms = (t_total - t_start) * 1000
        time_gray_ms = (t_gray - t_start) * 1000
        time_estimate_ms = (t_estimate_end - t_gray) * 1000
        time_warp_ms = (t_warp_end - t_warp_start) * 1000
        time_per_frame_warp_ms = time_warp_ms / max(1, n - 1)  # Exclude reference frame

        if self.enable_profiling:
            print(f"\n=== Fusion Model Performance ===")
            print(f"Grayscale conversion:  {time_gray_ms:6.2f}ms")
            print(f"Transform estimation:  {time_estimate_ms:6.2f}ms")
            print(f"  └─ Per frame:        {time_estimate_ms/(n-1):6.2f}ms" if n > 1 else "")
            print(f"Warping ({n-1} frames): {time_warp_ms:6.2f}ms")
            print(f"  └─ Per frame:        {time_per_frame_warp_ms:6.2f}ms")
            print(f"Total time:            {time_total_ms:6.2f}ms")
            print(f"Throughput:            {1000.0/time_total_ms:.1f} FPS")
            print(f"Real-time (30 FPS)?    {'✓ YES' if time_total_ms < 33.33 else '✗ NO'}")
            print(f"================================\n")
        else:
            print(f"Fusion stabilization: estimate {time_estimate_ms:.1f}ms, warp {time_warp_ms:.1f}ms ({time_per_frame_warp_ms:.2f}ms/frame), total {time_total_ms:.1f}ms")

        # Store performance stats
        if self.enable_profiling:
            self.perf_stats['optical_flow_ms'].append(time_estimate_ms)
            self.perf_stats['warping_ms'].append(time_per_frame_warp_ms)
            self.perf_stats['total_ms'].append(time_total_ms)

        # Build transformation matrices
        transforms = []
        for i in range(n):
            rot = rotations[i]
            scale = scales[i]
            tx, ty = translations[i]

            # 2D rotation + scale + translation matrix
            cos_r = np.cos(rot)
            sin_r = np.sin(rot)
            center = (w / 2, h / 2)

            # Full 3x3 homogeneous transformation matrix
            transform_matrix = np.array([
                [scale * cos_r, -scale * sin_r, tx + center[0] * (1 - scale * cos_r) + center[1] * scale * sin_r],
                [scale * sin_r,  scale * cos_r, ty + center[1] * (1 - scale * cos_r) - center[0] * scale * sin_r],
                [0, 0, 1]
            ])
            transforms.append(transform_matrix)

        return {
            "warped": warped,
            "orig": list(frames),
            "scales": scales,
            "translations": translations,
            "rotations": rotations,
            "inliers": inliers_list,
            "ref_idx": ref_idx,
            "transforms": transforms
        }

    def get_performance_summary(self) -> Dict:
        """
        Get summary of performance statistics.

        Returns:
            dict with mean, median, and percentile statistics for timing metrics
        """
        if not self.perf_stats['total_ms']:
            return {"error": "No performance data collected. Enable profiling first."}

        summary = {}
        for metric_name, values in self.perf_stats.items():
            if values:
                arr = np.array(values)
                summary[metric_name] = {
                    'mean': float(np.mean(arr)),
                    'median': float(np.median(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'p95': float(np.percentile(arr, 95)),
                    'p99': float(np.percentile(arr, 99))
                }

        # Add derived metrics
        if 'warping_ms' in summary:
            avg_warp_ms = summary['warping_ms']['mean']
            summary['fps'] = {
                'mean': 1000.0 / avg_warp_ms if avg_warp_ms > 0 else 0,
                'realtime_capable_30fps': avg_warp_ms < 33.33
            }

        return summary

    def reset_performance_stats(self):
        """Reset performance statistics."""
        for key in self.perf_stats:
            self.perf_stats[key] = []
