# Video Stabilization Evaluation Metrics

This document explains the quantitative metrics used to evaluate video stabilization quality in this project.

## Overview

Video stabilization aims to remove unwanted camera shake while preserving intentional camera motion. A good stabilization algorithm should:
1. Reduce high-frequency jitter and shake
2. Preserve low-frequency intentional motion (pans, tilts)
3. Minimize content loss (cropping)
4. Avoid introducing distortion or artifacts

Our evaluation pipeline computes multiple metrics to assess these qualities quantitatively.

---

## Core Metrics

### 1. Stability Score

**What it measures:** The smoothness of the camera path by analyzing motion in the frequency domain.

**Intuition:** Camera shake manifests as high-frequency motion (rapid jitter), while intentional camera movements (pans, tilts) are low-frequency. A good stabilizer removes the high-frequency components while preserving the low-frequency intentional motion.

**How it's computed:**

1. **Extract Motion Parameters:** For each frame, decompose the transformation matrix to extract:
   - Translation in x-direction: `dx` (pixels)
   - Translation in y-direction: `dy` (pixels)
   - Rotation angle: `da` (radians)

   This creates three time-series representing the camera's motion path.

2. **Perform Fourier Transform:** Apply Fast Fourier Transform (FFT) to each time-series:
   ```python
   fft_dx = np.fft.fft(dx)
   fft_dy = np.fft.fft(dy)
   fft_da = np.fft.fft(da)
   ```

3. **Calculate Energy Ratio:**
   - Define low-frequency band (e.g., 0.5 Hz - 2 Hz) representing smooth intentional motion
   - Define high-frequency band (e.g., > 2 Hz) representing shake/jitter
   - Stability Score = Energy(low-freq) / Energy(total)

   A **higher score** indicates more stable video with less high-frequency jitter.

**Interpretation:**
- Score ≈ 0.8-1.0: Excellent stability, mostly smooth motion
- Score ≈ 0.5-0.8: Good stability, some residual shake
- Score < 0.5: Poor stability, significant jitter remains

**Note:** This metric requires transformation parameters from the stabilization algorithm.

---

### 2. Cropping Ratio

**What it measures:** How much of the original field of view is preserved after stabilization.

**Intuition:** Stabilization typically requires cropping the image because warping introduces black borders at the edges. Less cropping means more content is preserved.

**How it's computed:**

1. **Get Original Frame Area:**
   ```python
   original_area = original_width × original_height
   ```

2. **Find Largest Valid Rectangle:** After warping, find the largest rectangle that contains no black pixels (the valid region).

3. **Calculate Ratio:**
   ```python
   cropping_ratio = final_area / original_area
   ```

4. **Average Across Video:** Compute the mean cropping ratio across all frames.

**Interpretation:**
- Ratio ≈ 0.9-1.0: Excellent, minimal cropping
- Ratio ≈ 0.7-0.9: Good, moderate cropping
- Ratio < 0.7: Poor, significant content loss

**Trade-off:** Aggressive stabilization often requires more cropping. The best algorithms find the right balance between stability and content preservation.

---

### 3. Distortion Score

**What it measures:** How much non-rigid warping (stretching, shearing) is introduced by stabilization.

**Intuition:** A rigid transformation (rotation + translation + uniform scaling) preserves the shape of objects. Non-rigid transformations can make the video look unnatural by stretching or shearing content.

**How it's computed:**

1. **Get Transformation Matrix:** For each frame, obtain the 3×3 homography matrix `H` used for warping.

2. **Extract Affine Component:** Take the top-left 2×2 sub-matrix:
   ```python
   A = H[0:2, 0:2]  # [[h00, h01], [h10, h11]]
   ```

3. **Calculate Eigenvalues:** Compute the two eigenvalues λ₁, λ₂ of matrix A.

4. **Compute Distortion Ratio:**
   ```python
   distortion = max(λ₁, λ₂) / min(λ₁, λ₂)
   ```

5. **Average Across Video:** Compute the mean distortion score.

**Interpretation:**
- Distortion ≈ 1.0: Perfect, only rigid transformations
- Distortion ≈ 1.0-1.2: Excellent, minimal stretching
- Distortion ≈ 1.2-1.5: Good, noticeable but acceptable
- Distortion > 1.5: Poor, significant visual artifacts

**Note:** This metric requires the transformation matrices from the stabilization algorithm.

---

## Supplementary Metrics

These additional metrics provide complementary information about stabilization quality:

### 4. Inter-frame Difference

**What it measures:** Average pixel-level difference between consecutive frames.

**Computation:**
```python
diff = mean(|frame[t] - frame[t-1]|)
```

**Interpretation:**
- Lower values indicate less frame-to-frame variation (more stability)
- Compute for both original and stabilized videos
- Report improvement percentage

---

### 5. Optical Flow Magnitude

**What it measures:** Average magnitude of optical flow vectors between consecutive frames.

**Computation:**
```python
flow = calcOpticalFlowFarneback(frame[t-1], frame[t])
magnitude = mean(sqrt(flow_x² + flow_y²))
```

**Interpretation:**
- Lower magnitude indicates smoother motion between frames
- Captures motion smoothness better than raw pixel differences
- Should be significantly reduced in stabilized video

---

### 6. PSNR (Peak Signal-to-Noise Ratio)

**What it measures:** Similarity between consecutive frames.

**Computation:**
```python
PSNR = 20 × log₁₀(MAX_PIXEL / sqrt(MSE))
```

**Interpretation:**
- Higher PSNR indicates more similar consecutive frames (less motion)
- Typical values: 25-35 dB for stabilized video
- Should be higher for stabilized vs original

---

### 7. Sharpness Score

**What it measures:** Image detail preservation (detects blur introduced by warping).

**Computation:**
```python
sharpness = variance(Laplacian(image))
```

**Interpretation:**
- Higher values indicate sharper images
- Stabilization should not significantly reduce sharpness
- Bilinear interpolation during warping can introduce blur

---

## Metric Summary Table

| Metric | Type | Better Value | Primary Purpose |
|--------|------|--------------|-----------------|
| **Stability Score** | Frequency analysis | Higher (→1.0) | Measures jitter removal |
| **Cropping Ratio** | Content preservation | Higher (→1.0) | Measures field-of-view retention |
| **Distortion Score** | Geometric quality | Lower (→1.0) | Measures shape preservation |
| **Inter-frame Diff** | Pixel-level stability | Lower | Quick stability indicator |
| **Optical Flow Mag** | Motion smoothness | Lower | Advanced stability indicator |
| **PSNR** | Frame similarity | Higher | Consecutive frame similarity |
| **Sharpness** | Image quality | Higher | Blur detection |

---

## Usage

### Running Evaluation

```bash
# Basic evaluation
python src/evaluate.py --model baseline

# Save detailed results
python src/evaluate.py --model baseline --save-json results.json

# Evaluate different model
python src/evaluate.py --model mymodel
```

### Comparing Models

```bash
# Evaluate multiple models
python src/evaluate.py --model baseline --save-json baseline.json
python src/evaluate.py --model improved --save-json improved.json

# Compare results (manual or write comparison script)
```

### Understanding Output

The evaluation script outputs:
1. **Per-flight metrics** - Individual performance for each flight
2. **Aggregate metrics** - Average performance across all flights
3. **Improvement percentages** - Relative improvement vs. original video

---

## Implementation Notes

### Transformation Matrix Requirements

To compute **Stability Score** and **Distortion Score**, the stabilization algorithm must save transformation parameters. The inference script should output:

1. A JSON/NPZ file per flight containing:
   - Frame indices
   - Transformation matrices (3×3 homography or 2×3 affine)
   - Or decomposed parameters: `(dx, dy, da, scale)`

Example structure:
```python
{
  "Flight1": {
    "frames": [0, 1, 2, ...],
    "transforms": [H0, H1, H2, ...],  # 3x3 matrices
    "translations": [(dx0, dy0), (dx1, dy1), ...],
    "rotations": [da0, da1, da2, ...],
    "scales": [s0, s1, s2, ...]
  }
}
```

### Frequency Band Selection

For Stability Score, the frequency bands should be tuned based on:
- **Frame rate** of the video (fps)
- **Typical shake frequency** (usually 5-15 Hz for handheld cameras)
- **Intended camera motion speed** (pans are typically < 2 Hz)

Default settings:
- Low-freq band: 0.5 - 2.0 Hz (intentional motion)
- High-freq band: > 2.0 Hz (shake/jitter)

---

## Best Practices

### When Evaluating Algorithms

1. **Always compare to original:** Report both absolute metrics and improvement percentages
2. **Check multiple flights:** Performance may vary across different flight characteristics
3. **Look for trade-offs:** High stability often means more cropping
4. **Visual inspection:** Metrics are helpful but not perfect - always watch the videos
5. **Consider use case:** Real-time applications may prioritize speed over perfect metrics

### Metric Limitations

- **Stability Score:** May not capture all types of stabilization artifacts
- **Cropping Ratio:** Doesn't account for where cropping occurs (center vs edges)
- **Distortion Score:** Doesn't capture all types of visual artifacts
- **PSNR:** Can be misleading for videos with intentional motion

**Bottom line:** Use a combination of metrics and visual assessment for comprehensive evaluation.

---

## References

1. Grundmann et al., "Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths", CVPR 2011
2. Liu et al., "Content-Preserving Warps for 3D Video Stabilization", SIGGRAPH 2009
3. Matsushita et al., "Full-Frame Video Stabilization with Motion Inpainting", TPAMI 2006

---

## Questions?

For implementation details, see:
- `src/evaluate.py` - Evaluation pipeline implementation
- `src/model/baseline.py` - Baseline stabilization algorithm
- `CLAUDE.md` - Project documentation
