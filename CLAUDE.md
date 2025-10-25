# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hackathon challenge for online (real-time) UAV video stabilization. The goal is to develop methods that stabilize video frames without relying on future frames at inference time. The project includes synchronized video frames and IMU sensor data (accelerations, angular rates, quaternions).

## Key Commands

### Testing and Validation
```bash
# Test dataloader and visualize data
python baseline/test.py

# Run stabilization inference
python baseline/inference.py
```

### Environment Setup
The project uses Python 3.13 with dependencies managed via `pyproject.toml`:
```bash
# Install using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

Legacy baseline requirements are in `baseline/requirements.txt`.

## Data Structure

Expected data directory layout:
```
data/
├── images/
│   ├── <flight_name>/
│   │   ├── 00000000.png
│   │   ├── 00000001.png
│   │   └── ...
├── labels/
│   ├── <flight_name>.csv
│   └── ...
└── raw/  # Additional logs and raw video files
```

### Sensor Data Format
CSV files in `data/labels/` contain per-frame sensor readings:
- `ax_mDs2, ay_mDs2, az_mDs2`: accelerations (m/s²) in body frame
- `wx_radDs, wy_radDs, wz_radDs`: angular rates (rad/s) in body frame
- `qw, qx, qy, qz`: orientation quaternion (Hamilton convention)

**Important**: Frame-to-sensor synchronization has small time shifts and may need improvement.

## Code Architecture

### Baseline Implementation (`baseline/`)
- **`model/baseline.py`**: Core stabilization algorithm using optical flow (Lucas-Kanade + RANSAC)
  - `stabilize_frames()`: Main function that warps frames to a reference frame
  - `estimate_scale_translation()`: Estimates scale + translation transform between frames
  - Uses `cv2.goodFeaturesToTrack()` + `cv2.calcOpticalFlowPyrLK()` for feature tracking

- **`utils/datasets.py`**: PyTorch dataset for temporal video clips + sensor data
  - `LoadClipsAndLabels`: Handles multi-frame sampling with temporal skip rates
  - Returns: `(imgs, labels, paths, shapes, main_frame_ids, label_paths)`
    - `imgs`: (B×T, C, H, W) tensor
    - `labels`: (B, T, F) sensor feature vectors per frame
  - Frame sampling configured via `hyp` dict (`num_frames`, `skip_rate`, etc.)

- **`utils/plots.py`**: Visualization utilities for debugging sensor data on frames

- **`inference.py`**: Sliding window inference over full video sequences
  - Processes frames in overlapping windows (default: window=10, stride=5)
  - Outputs stabilized frames to `data_res/`

### Development Notebooks
- `eda.ipynb`: Exploratory data analysis
- `notebooks/stabilize_flight1.ipynb`: Flight-specific stabilization experiments

## Development Notes

- **Online constraint**: Prefer methods that use only past frames (causal filtering). Using future frames reduces real-time viability.
- **Sensor-frame sync**: Matching between video and sensor logs is approximate. Improving synchronization can be part of the solution.
- **No train/val/test split provided**: Decide your own data usage strategy.
- The baseline uses only image data (optical flow). Sensor data is loaded but not used in stabilization—this is an opportunity for improvement.
