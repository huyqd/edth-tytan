# Real-Time UAV Video Stabilization

**Hackathon Challenge Solution: Online Video Stabilization for UAV Footage**

## Overview

This project implements a real-time video stabilization system for UAV (drone) footage using IMU sensor fusion. The solution combines roll correction, high-frequency pitch filtering, and intelligent freeze frame detection to produce smooth, stable video while preserving intentional camera movements.

## Problem Statement

UAV video footage suffers from:
- **Horizon tilt** due to aircraft roll during turns
- **High-frequency vibrations** from motors and turbulence
- **Frozen/duplicate frames** in the video stream
- **Real-time processing constraints** (must work without future frames)

## Our Solution

### Key Components

1. **Roll Correction**: Fully corrects horizon tilt using IMU quaternion data
2. **Bandpass Pitch Filtering**: Removes high-frequency vibrations (0.3-10 Hz) while preserving smooth flight path
3. **Freeze Frame Detection**: Intelligently handles duplicate frames to avoid processing artifacts

### Why This Works

- **Sensor Fusion**: Combines visual (frames) + inertial (IMU) data for robust stabilization
- **Frequency-Domain Approach**: Bandpass filtering removes vibrations while keeping intentional movements
- **Real-Time Capable**: Processes frames sequentially without looking ahead (~30 FPS)

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Running the Pipeline

**Step 1: Preprocess IMU data**

```bash
# Convert quaternions to Euler angles and apply bandpass filter
python ProcessRawData.py
```

**Step 2: Match filtered data to video frames**

```bash
python BringFilteredDataAndVideoFrameDataTogeth.py --flight Flight1
```

**Step 3: Run stabilization**

```bash
python ProcessPictureFreeze.py --flight Flight1
```

Stabilized frames will be saved to `output/rotated_freeze_filt/Flight1/`

## Detailed Usage

### Main Script: ProcessPictureFreeze.py

```bash
python ProcessPictureFreeze.py \
    --flight Flight1 \
    --start-frame 0 \
    --freeze-threshold 0.5 \
    --pixel-per-deg 28 \
    --output-dir output/rotated_freeze_filt
```

**Command-line arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--flight` | Flight name (Flight1, Flight2, Flight3) | Flight1 |
| `--start-frame` | First frame number to process | 0 |
| `--freeze-threshold` | Freeze detection sensitivity (lower = stricter) | 0.5 |
| `--pixel-per-deg` | Pixels per degree for pitch translation | 28 |
| `--output-dir` | Output directory for stabilized frames | output/rotated_freeze_filt |

### Preprocessing Scripts

**ProcessRawData.py**
- Converts quaternions to Euler angles (roll, pitch, yaw)
- Applies Butterworth bandpass filter (0.3-10 Hz) to remove vibrations
- Unwraps yaw to avoid 360° discontinuities
- Outputs to `output/rawWithEulerAndFiltAndYawFix/`

**BringFilteredDataAndVideoFrameDataTogeth.py**
- Matches filtered sensor data to video frame timestamps
- Uses nearest-neighbor matching with configurable tolerance (default: 5ms)
- Ensures frame-sensor synchronization

```bash
python BringFilteredDataAndVideoFrameDataTogeth.py \
    --flight Flight1 \
    --tolerance 5.0
```

## Algorithm Details

### Bandpass Filter Design

```python
# Butterworth bandpass filter
lowcut = 0.3 Hz   # Remove slow drift (< 0.3 Hz)
highcut = 10 Hz   # Remove fast vibrations (> 10 Hz)
order = 2         # Second-order for smooth response
fs = 327.7 Hz     # IMU sampling rate
```

**Why these frequencies?**
- Below 0.3 Hz: Slow drift and sensor bias
- 0.3-10 Hz: **Intentional flight movements** (preserved)
- Above 10 Hz: **Motor vibrations and turbulence** (removed)

### Roll Correction

Full correction applied to level the horizon:

```python
rotated = rotate_image(image, -roll, output_size=(1620, 1296))
```

- Input roll angle from IMU quaternion
- Negated to counteract aircraft rotation
- 1.2x larger canvas (1620x1296) prevents black borders

### Pitch Stabilization

Vertical translation based on **filtered** pitch:

```python
dy = -28 * pitch_filtered  # pixels/degree × filtered pitch
dy_clipped = np.clip(dy, -108, 108)  # Stay within borders
translated = translate_image_centered(image, dx=0, dy=dy_clipped)
```

- 28 pixels/degree calibrated for 1350x1080 resolution, ~40° FOV
- Clipped to ±108 pixels (canvas border width)
- Removes high-freq jitter, preserves terrain following

### Freeze Frame Detection

Two-stage detection for efficiency:

1. **Fast check**: Compare file sizes (2% tolerance)
2. **Pixel check**: Mean absolute difference < threshold

If frozen:
```python
stabilized_frame = last_valid_frame.copy()  # Reuse previous stabilization
```

This avoids artifacts from re-processing identical frames.

## Data Structure

```
data/
├── images/
│   ├── Flight1/
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   └── ...
│   ├── Flight2/
│   └── Flight3/
├── labels/
│   ├── Flight1.csv  # Frame-matched IMU data (qw, qx, qy, qz, etc.)
│   ├── Flight2.csv
│   └── Flight3.csv
└── raw/
    └── logs/
        ├── Flight1.csv  # Raw IMU logs (high sampling rate)
        ├── Flight2.csv
        └── Flight3.csv

output/
├── rawWithEulerAndFiltAndYawFix/        # Filtered Euler angles
│   └── Flight1.csv
├── relevantWithEulerAndFiltAndYawFix/   # Frame-matched filtered data
│   └── Flight1/
│       └── Flight1.csv
└── rotated_freeze_filt/                  # Stabilized output frames
    └── Flight1/
        ├── 00000000.jpg
        ├── 00000001.jpg
        └── ...
```

## CSV Data Format

**data/labels/FlightX.csv** (frame-synchronized IMU data):
- `qw, qx, qy, qz`: Orientation quaternion (Hamilton convention)
- `ax_mDs2, ay_mDs2, az_mDs2`: Accelerations (m/s²)
- `wx_radDs, wy_radDs, wz_radDs`: Angular rates (rad/s)

**output/.../FlightX.csv** (preprocessed data):
- `roll, pitch, yaw`: Euler angles (degrees)
- `roll_filtered, pitch_filtered, yaw_filtered`: Bandpass filtered angles

## Performance

- **Processing speed**: ~30 FPS on consumer hardware (Intel i7)
- **Real-time capable**: Yes (causal filter, no future frames needed)
- **Output quality**: Stable horizon, smooth motion, preserved intentional movements

## Dependencies

Core requirements:
- Python 3.10+
- numpy
- opencv-python (cv2)
- pandas
- scipy
- tqdm

See [pyproject.toml](pyproject.toml) for full dependency list.

## Repository Structure

```
.
├── ProcessPictureFreeze.py          # Main stabilization script
├── ProcessRawData.py                 # IMU preprocessing (Euler + filtering)
├── BringFilteredDataAndVideoFrameDataTogeth.py  # Frame-sensor matching
├── pyproject.toml                    # Dependencies
├── README.md                         # This file
├── SUBMISSION.md                     # Detailed submission documentation
├── README_evaluate.md                # Evaluation metrics documentation
├── app.py                            # Interactive visualization tool (Gradio)
├── src/                              # Additional utilities
│   ├── model/                        # Baseline models (optical flow, fusion)
│   ├── utils/                        # Dataset loaders, plotting
│   ├── inference.py                  # Batch inference script
│   ├── evaluate.py                   # Quantitative evaluation
│   └── split_data.py                 # Train/val/test splitting
├── config/                           # Configuration files
├── docs/                             # Additional documentation
└── notebooks/                        # Exploratory data analysis
```

## Visualization

Launch the interactive Gradio app to compare original vs stabilized videos:

```bash
python app.py
```

Features:
- Side-by-side video comparison
- Model selector (baseline, fusion, raw)
- Evaluation metrics display
- Frame-by-frame navigation
- Video playback mode

## Evaluation

Run quantitative evaluation:

```bash
# Evaluate stabilized output
python src/evaluate.py --model rotated_freeze_filt
```

Metrics computed:
- Inter-frame stability (lower = better)
- Optical flow smoothness
- PSNR (peak signal-to-noise ratio)
- Sharpness preservation
- Cropping ratio
- Distortion score

See [README_evaluate.md](README_evaluate.md) for detailed metric descriptions.

## Additional Models

The repository also includes baseline optical flow models in `src/model/`:

- **baseline.py**: Lucas-Kanade optical flow + RANSAC stabilization
- **fusion.py**: Optical flow + IMU sensor fusion

These can be run using:

```bash
python src/inference.py --model baseline
python src/inference.py --model fusion
```

However, our **primary submission** is the `ProcessPictureFreeze.py` script which focuses on roll correction and high-frequency pitch stabilization.

## Future Improvements

Potential enhancements:
- [ ] Adaptive filter parameters based on flight dynamics
- [ ] GPU acceleration for real-time 4K video
- [ ] Online calibration of pixel-per-degree scaling
- [ ] Learning-based filter design from training data
- [ ] Yaw stabilization with horizon detection

## License

This project was developed for the UAV Video Stabilization Hackathon Challenge.

## Acknowledgments

- Challenge organizers for providing synchronized video + IMU data
- OpenCV community for robust computer vision tools
- SciPy for signal processing utilities
