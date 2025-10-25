# Baseline Video Stabilization Algorithm

## Overview

This baseline implementation provides **online video stabilization** using optical flow and geometric transformation estimation. The algorithm aligns multiple frames to a common reference frame without requiring future frames, making it suitable for real-time applications.

## Algorithm Pipeline

```
Input: List of frames → Feature Detection → Optical Flow Tracking →
Transform Estimation (RANSAC) → Frame Warping → Output: Stabilized frames
```

## Core Functions

### 1. `get_matches_kp(img1_gray, img2_gray, max_features=2000)`

**Purpose**: Find corresponding points between two frames using optical flow.

**Method**: Shi-Tomasi corner detection + Lucas-Kanade optical flow

**Steps**:
1. Detect up to `max_features` strong corners in `img1` using `cv2.goodFeaturesToTrack()`
   - `qualityLevel=0.01`: Only corners with quality > 1% of best corner
   - `minDistance=7`: Minimum 7-pixel spacing between corners
   - `blockSize=7`: 7×7 neighborhood for corner detection

2. Track detected corners from `img1` to `img2` using pyramidal Lucas-Kanade optical flow
   - `winSize=(21, 21)`: 21×21 pixel search window
   - `maxLevel=3`: Use 3-level image pyramid for coarse-to-fine tracking
   - Tracks points efficiently without descriptor matching

3. Filter successfully tracked points (where tracking status `st == 1`)

**Returns**:
- `pts_src`: Feature locations in frame 1
- `pts_dst`: Corresponding locations in frame 2
- Empty list (placeholder for compatibility)

**Why this approach?**
- **Fast**: Optical flow is much faster than descriptor-based matching (SIFT/ORB)
- **Accurate**: Pyramidal LK handles moderate motion well
- **Dense**: Can track thousands of points in real-time

---

### 2. `estimate_scale_translation(src_gray, dst_gray)`

**Purpose**: Estimate the scale and translation that best aligns source frame to destination frame.

**Method**: RANSAC-based robust estimation

**Steps**:

1. Get feature correspondences using `get_matches_kp()`

2. Use `cv2.estimateAffinePartial2D()` with RANSAC to estimate similarity transform
   - **Model**: Scale + rotation + translation (4 DOF)
   - **RANSAC**: Filters out outliers from bad feature tracks
   - `ransacReprojThreshold=3.0`: Points within 3 pixels are inliers
   - `maxIters=2000`: Maximum RANSAC iterations

3. Extract scale and translation from inlier points:
   - Compute centroids of source and destination inlier points
   - **Scale** = ratio of average distances from centroid:
     ```
     scale = sum(||dst_points - dst_centroid||) / sum(||src_points - src_centroid||)
     ```
   - **Translation** = shift after scaling:
     ```
     t = dst_centroid - scale * src_centroid
     ```

**Returns**:
- `scale`: Uniform scaling factor
- `(tx, ty)`: Translation in x and y
- `inliers`: Boolean mask of inlier points

**Why simplify to scale + translation?**
- UAV footage often has rotation, but estimating rotation separately can introduce artifacts
- The affine model from RANSAC captures rotation implicitly, but we extract only scale/translation for stability
- This is a **design choice** - you could extract full rotation for potentially better results

---

### 3. `warp_with_scale_translation(img, scale, tx, ty, output_shape)`

**Purpose**: Apply estimated transformation to warp a frame.

**Method**: Affine transformation matrix

**Transform Matrix**:
```
M = [scale   0      tx]
    [0       scale  ty]
```

**Warping**:
- Uses `cv2.warpAffine()` with bilinear interpolation
- `borderMode=cv2.BORDER_REPLICATE`: Replicates edge pixels for areas outside the frame

**Returns**: Warped frame matching `output_shape`

---

### 4. `stabilize_frames(frames, ref_idx=None)`

**Purpose**: Main stabilization function that aligns all frames to a reference frame.

**Algorithm**:

1. **Select reference frame**:
   - Default: Middle frame (`ref_idx = n // 2`)
   - This minimizes average motion to the reference

2. **Convert to grayscale**: All frames converted for feature detection

3. **Estimate transforms**: For each frame (except reference):
   - Compute scale and translation to align it with the reference
   - Store transformation parameters

4. **Warp frames**: Apply transformations to align all frames to reference coordinate system

5. **Return results**: Dictionary containing:
   - `warped`: Stabilized frames
   - `orig`: Original input frames
   - `scales`: Scaling factors for each frame
   - `translations`: Translation vectors for each frame
   - `inliers`: RANSAC inlier masks
   - `ref_idx`: Index of reference frame

**Timing**:
- Reports separate timing for transform estimation and warping phases

---

## Usage Example

```python
from model.baseline import stabilize_frames
import cv2

# Load a sequence of frames
frames = [cv2.imread(f"frame_{i:04d}.png") for i in range(10)]

# Stabilize to middle frame
result = stabilize_frames(frames, ref_idx=None)

# Access stabilized frames
stabilized = result["warped"]
for i, frame in enumerate(stabilized):
    cv2.imwrite(f"stabilized_{i:04d}.png", frame)

# Check transformation parameters
print(f"Scales: {result['scales']}")
print(f"Translations: {result['translations']}")
```

## Inference Mode (Sliding Window)

See `inference.py` for production usage with sliding windows:

```python
window = 10  # Process 10 frames at a time
stride = 5   # Advance by 5 frames each step
ref_idx = stride // 2  # Reference is at position 2 in the window
```

This creates **overlapping windows** for smooth stabilization across long videos.

## Algorithm Characteristics

### ✅ Strengths
- **Fast**: Optical flow is computationally efficient (~100-200ms for 10 frames)
- **Online**: Only uses past/current frames, no future frame dependency
- **Robust**: RANSAC handles outliers from moving objects or tracking failures
- **No training required**: Purely geometric algorithm

### ⚠️ Limitations
- **Image-only**: Does not use IMU sensor data (accelerations, gyroscope, quaternions)
- **No rotation compensation**: Only scale + translation, ignores roll/pitch/yaw
- **Drift**: Long sequences may accumulate error without global optimization
- **Local minima**: Optical flow can fail on large motions or low-texture regions
- **No motion model**: Treats each frame independently, no temporal smoothing

## Potential Improvements

1. **Incorporate IMU data**:
   - Use gyroscope to predict rotation between frames
   - Use accelerometer for motion priors
   - Fuse visual and inertial measurements (VIO)

2. **Add rotation estimation**:
   - Extract rotation from affine matrix instead of ignoring it
   - Use quaternions from sensor data for rotation initialization

3. **Temporal smoothing**:
   - Apply Kalman filter or low-pass filter to transformation parameters
   - Smooth trajectories instead of treating frames independently

4. **Rolling shutter correction**:
   - UAV footage often has rolling shutter distortion
   - Model and compensate for per-scanline timing

5. **Feature quality**:
   - Use more robust features (e.g., ORB with descriptors)
   - Implement feature prediction based on IMU

6. **Global optimization**:
   - Bundle adjustment over longer windows
   - Loop closure detection for drift correction

## Performance Notes

- **Feature detection**: ~10-50ms depending on image resolution
- **Optical flow tracking**: ~20-100ms per frame pair
- **RANSAC estimation**: ~5-10ms per frame pair
- **Warping**: ~10-20ms per frame

**Total**: ~50-200ms for a 10-frame window on typical hardware

## References

- Lucas-Kanade Optical Flow: [Wikipedia](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method)
- Shi-Tomasi Corner Detection: "Good Features to Track" (Shi & Tomasi, 1994)
- RANSAC: "Random Sample Consensus" (Fischler & Bolles, 1981)
