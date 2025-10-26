# Real-Time UAV Video Stabilization

**Hackathon Challenge: Online Video Stabilization for UAV Footage**

> **🚀 Quick Start:** See [SUBMISSION.md](SUBMISSION.md) for step-by-step instructions to run the pipeline.

## Overview

This repository contains our solution for real-time UAV video stabilization using **IMU sensor fusion**. The approach combines roll correction, high-frequency pitch filtering, and intelligent freeze frame detection to produce smooth, stable video while preserving intentional camera movements.

## Problem Statement

UAV video footage suffers from:
- **Horizon tilt** due to aircraft roll during turns
- **High-frequency vibrations** from motors and turbulence
- **Frozen/duplicate frames** in the video stream
- **Real-time processing constraints** (no access to future frames)

## Our Solution

### Three-Part Approach

1. **Roll Correction**: Fully corrects horizon tilt using IMU quaternion data
2. **Bandpass Pitch Filtering**: Removes high-frequency vibrations (0.3-10 Hz) while preserving smooth flight path
3. **Freeze Frame Detection**: Intelligently handles duplicate frames to avoid processing artifacts

### Key Innovation: Frequency-Domain Stabilization

We apply a **Butterworth bandpass filter (0.3-10 Hz)** to the pitch signal:
- **Below 0.3 Hz**: Remove slow drift and sensor bias
- **0.3-10 Hz**: **Preserve intentional flight movements** (terrain following, maneuvers)
- **Above 10 Hz**: **Remove motor vibrations and turbulence**

This allows the drone to naturally follow the terrain while eliminating high-frequency jitter.

### Why This Works

- **Sensor Fusion**: Combines visual (frames) + inertial (IMU) data for robust stabilization
- **Frequency Separation**: Bandpass filtering separates intentional movements from vibrations
- **Real-Time Capable**: Processes frames causally without future frame access (~30 FPS)
- **Handles Edge Cases**: Detects and handles frozen/duplicate frames gracefully

## Algorithm Details

### 1. Roll Correction

Full correction applied to level the horizon:

```python
# Extract roll from quaternion
roll, pitch, yaw = quaternion_to_euler(qw, qx, qy, qz, degrees=True)

# Rotate image to counteract roll
rotated = rotate_image(image, -roll, output_size=(1620, 1296))
```

- Input: Roll angle from IMU quaternion
- Output: Image rotated to level the horizon
- Canvas: 1.2x larger (1620×1296) to prevent black borders

### 2. Pitch Stabilization

Vertical translation based on **bandpass-filtered** pitch:

```python
# Apply bandpass filter to pitch signal
pitch_filtered = butterworth_bandpass(pitch, lowcut=0.3, highcut=10, fs=327.7, order=2)

# Translate image vertically
dy = -28 * pitch_filtered  # pixels/degree × filtered pitch
dy_clipped = np.clip(dy, -108, 108)  # Stay within canvas borders
translated = translate_image_centered(image, dx=0, dy=dy_clipped)
```

- **28 pixels/degree**: Calibrated for 1350×1080 resolution, ~40° FOV
- **Clipped to ±108 pixels**: Prevents image from leaving canvas (canvas border = 108px)
- **Result**: Removes high-freq jitter, preserves smooth terrain following

### 3. Freeze Frame Detection

Two-stage detection for efficiency:

```python
# Stage 1: Fast file size check
size_diff = abs(size1 - size2) / max(size1, size2)
if size_diff < 0.02:  # 2% tolerance
    # Stage 2: Pixel-wise comparison
    diff = np.mean(np.abs(frame1 - frame2))
    is_frozen = (diff < threshold)  # default: 0.5

if is_frozen:
    output = last_valid_frame.copy()  # Reuse previous stabilization
```

This avoids artifacts from re-processing identical frames.

## Performance

- **Processing speed**: ~30 FPS on consumer hardware (Intel i7)
- **Real-time capable**: Yes (causal processing, no future frames)
- **Output quality**: Level horizon, smooth motion, preserved intentional movements

## Data Structure

```
data/
├── images/Flight1/          # Input video frames (JPG)
│   ├── 00000000.jpg
│   ├── 00000001.jpg
│   └── ...
├── labels/Flight1.csv       # Frame-synchronized IMU data (qw, qx, qy, qz)
└── raw/logs/Flight1.csv     # Raw flight logs (high-rate IMU)

output/
├── rawWithEulerAndFiltAndYawFix/Flight1.csv        # Filtered Euler angles
├── relevantWithEulerAndFiltAndYawFix/Flight1/      # Frame-matched filtered data
│   └── Flight1.csv
└── rotated_freeze_filt/Flight1/                    # Stabilized output frames
    ├── 00000000.jpg
    └── ...
```

## Pipeline Overview

```
Raw IMU Logs → ProcessRawData.py → Filtered Euler Angles
                                           ↓
Video Frames ← BringFilteredDataAndVideoFrameDataTogeth.py → Frame-Matched Data
     ↓                                                              ↓
ProcessPictureFreeze.py ←───────────────────────────────────────────┘
     ↓
Stabilized Frames
```

## Main Scripts

### ProcessPictureFreeze.py
**Main stabilization algorithm**

```bash
python ProcessPictureFreeze.py --flight Flight1
```

Reads video frames and IMU data, applies roll correction + filtered pitch translation, handles frozen frames.

### ProcessRawData.py
**IMU preprocessing**

Converts quaternions to Euler angles, applies bandpass filter (0.3-10 Hz), unwraps yaw discontinuities.

### BringFilteredDataAndVideoFrameDataTogeth.py
**Timestamp synchronization**

Matches filtered sensor data to video frame timestamps using nearest-neighbor (5ms tolerance).

## Evaluation & Visualization

### Quantitative Metrics

```bash
python src/evaluate.py \
    --original-dir data/images/Flight1 \
    --stabilized-dir output/rotated_freeze_filt/Flight1
```

Computes:
- Inter-frame stability (motion smoothness)
- Optical flow magnitude (residual jitter)
- PSNR (frame quality)
- Sharpness (detail retention)

### Video Creation

```bash
python src/create_video_from_images.py \
    --input-dir output/rotated_freeze_filt/Flight1 \
    --output-path output/videos/Flight1_stabilized.mp4 \
    --fps 30
```

### Interactive Visualization

```bash
python app.py
# Access at: http://localhost:7860
```

Gradio app with side-by-side comparison, playback controls, and metric display.

## Dependencies

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

Core requirements:
- Python 3.10+
- numpy
- opencv-python (cv2)
- pandas
- scipy (signal processing)
- tqdm (progress bars)

See [pyproject.toml](pyproject.toml) for full dependency list.

## Results

Our stabilization pipeline:
- ✅ Removes horizon tilt (roll correction)
- ✅ Eliminates high-frequency pitch vibrations (0.3-10 Hz bandpass)
- ✅ Preserves intentional camera movements (terrain following)
- ✅ Handles frozen/duplicate frames gracefully
- ⚡ Real-time capable (~30 FPS on consumer hardware)

## Usage Instructions

**For detailed step-by-step instructions**, including:
- Required data format
- Complete workflow examples
- Video creation commands
- Evaluation and visualization

See **[SUBMISSION.md](SUBMISSION.md)**

## Repository Structure

```
.
├── ProcessPictureFreeze.py          # Main stabilization script
├── ProcessRawData.py                 # IMU preprocessing (Euler + filtering)
├── BringFilteredDataAndVideoFrameDataTogeth.py  # Timestamp matching
├── README.md                         # This file (algorithm overview)
├── SUBMISSION.md                     # Step-by-step usage guide
├── app.py                            # Gradio visualization app
├── pyproject.toml                    # Dependencies
├── src/
│   ├── evaluate.py                   # Quantitative evaluation
│   ├── create_video_from_images.py   # Video creation
│   └── utils/                        # Helper functions
└── data/                             # Input data (not included in repo)
```

## Future Improvements

Potential enhancements:
- [ ] Adaptive filter parameters based on flight dynamics
- [ ] GPU acceleration for real-time 4K video
- [ ] Online calibration of pixel-per-degree scaling
- [ ] Learning-based filter design from training data
- [ ] Yaw stabilization with horizon detection

## Acknowledgments

- Challenge organizers for providing synchronized video + IMU data
- OpenCV community for robust computer vision tools
- SciPy for signal processing utilities
