# Sensor Fusion Stabilization Model

## Overview

The `FusionModel` is an advanced video stabilization algorithm that combines visual (optical flow) and IMU sensor data for improved stabilization performance. It addresses limitations of purely vision-based approaches by incorporating inertial measurements.

## Key Features

### 1. Multi-Modal Sensor Fusion
- **Visual Component**: Lucas-Kanade optical flow with RANSAC for robust feature tracking
- **IMU Component**: Quaternion-based orientation and angular rate measurements
- **Fusion Strategy**: Complementary filtering to combine both modalities

### 2. Transformation Estimation

The model estimates the following transformations:

- **Rotation**: Fused from IMU quaternions and optical flow
  - IMU provides fast, high-frequency rotation estimates
  - Optical flow provides low-frequency drift correction
  - Configurable fusion weight (default: 0.7 for IMU)

- **Scale**: Estimated from optical flow feature correspondences

- **Translation**: Estimated from optical flow feature correspondences

### 3. Robust to Missing Data

The model gracefully handles missing or unreliable sensor data:
- Falls back to vision-only mode if IMU data is unavailable
- Uses RANSAC to filter outliers in optical flow
- Median filtering for robust rotation estimation

## Algorithm Details

### IMU-Based Rotation Estimation

1. **Quaternion Processing**:
   - Converts quaternions to 3x3 rotation matrices
   - Computes relative rotation between frames: `R_rel = R_dst * R_src^T`
   - Extracts Euler angles (roll, pitch, yaw)

2. **Camera-Centric Rotation**:
   - For downward-facing UAV cameras, uses roll angle as primary image rotation
   - Roll rotates the image plane as the UAV banks

### Visual Motion Estimation

1. **Feature Detection**: Shi-Tomasi corner detection (`cv2.goodFeaturesToTrack`)
2. **Feature Tracking**: Pyramidal Lucas-Kanade optical flow
3. **Transformation Fitting**: `cv2.estimateAffinePartial2D` with RANSAC
4. **Parameter Extraction**: Decomposes affine matrix into rotation, scale, translation

### Complementary Filtering

```
fused_rotation = α * imu_rotation + (1 - α) * optical_rotation
```

where α = `imu_weight` (default: 0.7)

**Rationale**:
- IMU excels at high-frequency motion (fast rotations, vibrations)
- Optical flow excels at low-frequency drift correction
- Weighted combination leverages strengths of both

## Usage

### Basic Usage

```python
from model import FusionModel

# Initialize model
model = FusionModel(max_features=2000, imu_weight=0.7)

# Stabilize frames with sensor data
result = model.stabilize_frames(
    frames=frame_list,           # List of BGR images
    sensor_data=sensor_data_list # List of sensor dicts
)

# Access results
stabilized_frames = result["warped"]
rotations = result["rotations"]
scales = result["scales"]
translations = result["translations"]
```

### Command-Line Usage

```bash
# Run fusion model on test set
uv run python src/inference.py --model fusion --split-set test

# Run on validation set with custom output name
uv run python src/inference.py --model fusion --split-set val --output-name fusion_v1

# Run on all data without split
uv run python src/inference.py --model fusion --split ""

# Adjust IMU weight (requires code modification)
# In your script:
# model = FusionModel(imu_weight=0.8)  # Trust IMU more
# model = FusionModel(imu_weight=0.5)  # Equal weight
```

### Sensor Data Format

The model expects sensor data as a list of dictionaries, one per frame:

```python
sensor_data = [
    {
        'qw': 0.999, 'qx': 0.001, 'qy': 0.002, 'qz': 0.003,  # Quaternion
        'wx_radDs': 0.01, 'wy_radDs': 0.02, 'wz_radDs': 0.03,  # Angular rates (rad/s)
        'ax_mDs2': 0.5, 'ay_mDs2': 0.3, 'az_mDs2': 9.8,  # Accelerations (m/s²)
        'timestamp': 1234567.890  # Optional timestamp
    },
    # ... one dict per frame
]
```

**Note**: The fusion model automatically enables `--use-sensor-data` flag in inference.py.

## Parameters

### Constructor Parameters

- **max_features** (int, default=2000): Maximum number of features to track for optical flow
- **imu_weight** (float, default=0.7): Weight for IMU rotation (0 to 1)
  - Higher values trust IMU more (better for high-frequency motion)
  - Lower values trust optical flow more (better for low-drift scenarios)

### Tuning Guidelines

**Increase `imu_weight` (e.g., 0.8-0.9)** when:
- Fast rotations or vibrations are dominant
- Visual features are sparse or unreliable
- IMU calibration is known to be accurate

**Decrease `imu_weight` (e.g., 0.5-0.6)** when:
- IMU exhibits significant drift
- Scene has rich visual features
- Motion is mostly translational

## Output Format

The `stabilize_frames` method returns a dictionary:

```python
{
    "warped": List[np.ndarray],      # Stabilized frames
    "orig": List[np.ndarray],         # Original frames
    "scales": List[float],            # Scale factors per frame
    "translations": List[Tuple],      # (tx, ty) per frame
    "rotations": List[float],         # Rotation angles (radians) per frame
    "inliers": List[np.ndarray],      # RANSAC inlier masks
    "ref_idx": int,                   # Reference frame index
    "transforms": List[np.ndarray]    # 3x3 transformation matrices
}
```

## Performance Considerations

### Computational Cost
- **Similar to baseline**: Optical flow is the dominant cost
- **IMU processing**: Negligible overhead (~1ms per frame)
- **Overall**: Expect ~20-50ms per frame depending on image resolution

### Memory Usage
- Same as baseline model
- Sensor data adds minimal memory overhead (~100 bytes per frame)

## Comparison with Baseline

| Aspect | Baseline | Fusion |
|--------|----------|--------|
| Rotation estimation | Optical flow only | IMU + Optical flow |
| High-frequency motion | Struggles | Handles well |
| Sensor requirements | Images only | Images + IMU |
| Robustness to blur | Limited | Improved |
| Computational cost | Baseline | ~Same |

## Evaluation

To evaluate the fusion model:

```bash
# Run stabilization
uv run python src/inference.py --model fusion --split-set test

# Evaluate results
uv run python src/evaluate.py --model fusion

# Compare with baseline
uv run python src/evaluate.py --model baseline
uv run python src/evaluate.py --model fusion
```

## Troubleshooting

### Issue: Model falls back to vision-only

**Symptom**: Warning messages about missing IMU data

**Solution**:
- Verify sensor data CSVs contain required columns: `qw, qx, qy, qz`
- Check frame-to-sensor synchronization
- Ensure `--use-sensor-data` flag is enabled (automatic for fusion model)

### Issue: Excessive rotation correction

**Symptom**: Over-rotated or jittery stabilized frames

**Solution**:
- Reduce `imu_weight` to 0.5-0.6
- Check IMU calibration
- Verify quaternion convention (Hamilton vs JPL)

### Issue: Insufficient rotation correction

**Symptom**: Residual rotation in stabilized video

**Solution**:
- Increase `imu_weight` to 0.8-0.9
- Verify optical flow has sufficient features
- Check if motion exceeds stabilization limits

## Future Improvements

Potential enhancements for the fusion model:

1. **Adaptive Fusion Weights**: Adjust IMU weight based on motion characteristics
2. **Accelerometer Integration**: Use accelerometer for translation estimation
3. **Kalman Filtering**: Replace complementary filter with Extended Kalman Filter
4. **Temporal Smoothing**: Add motion smoothing across longer time windows
5. **Learning-Based Fusion**: Learn optimal fusion weights from data

## References

The fusion approach is inspired by:
- Complementary filtering in IMU-camera fusion
- Visual-inertial odometry (VIO) systems
- Hybrid stabilization in modern smartphones

## License

Same as parent project.
