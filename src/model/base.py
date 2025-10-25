"""
Base model interface for video stabilization.

All stabilization models should inherit from StabilizationModel and implement
the stabilize_frames method.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np


class StabilizationModel(ABC):
    """
    Abstract base class for video stabilization models.

    This interface allows different stabilization algorithms to be used
    interchangeably in the inference pipeline.
    """

    @abstractmethod
    def stabilize_frames(
        self,
        frames: List[np.ndarray],
        sensor_data: Optional[List[Dict]] = None,
        ref_idx: Optional[int] = None
    ) -> Dict:
        """
        Stabilize a list of video frames.

        Args:
            frames: List of BGR images (numpy arrays, shape: [H, W, 3])
            sensor_data: Optional list of sensor data dicts for each frame.
                        Each dict contains keys like:
                        - 'qw', 'qx', 'qy', 'qz': quaternion orientation
                        - 'wx_radDs', 'wy_radDs', 'wz_radDs': angular rates (rad/s)
                        - 'ax_mDs2', 'ay_mDs2', 'az_mDs2': accelerations (m/s²)
                        - 'timestamp': frame timestamp
            ref_idx: Optional index of reference frame (defaults to central frame)

        Returns:
            dict with keys:
                - "warped": list of stabilized frames (numpy arrays)
                - "orig": original input frames (same order)
                - "scales": list of estimated scales (1.0 for reference)
                - "translations": list of (tx, ty) tuples (0,0 for reference)
                - "rotations": list of rotation angles in radians
                - "inliers": list of inlier masks (if applicable)
                - "ref_idx": chosen reference index
                - "transforms": list of 3x3 transformation matrices (optional)
        """
        pass

    def __call__(self, frames: List[np.ndarray], sensor_data: Optional[List[Dict]] = None, **kwargs) -> Dict:
        """
        Call the model to stabilize frames.

        This allows using the model as a callable: model(frames, sensor_data).
        """
        return self.stabilize_frames(frames, sensor_data, **kwargs)
