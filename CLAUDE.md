# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hackathon challenge for online (real-time) UAV video stabilization. The goal is to develop methods that stabilize video frames without relying on future frames at inference time. The project includes synchronized video frames and IMU sensor data (accelerations, angular rates, quaternions).

## Key Commands

### Data Splitting

```bash
# Split data into train/val/test (80/10/10) with temporal strategy
# Saves to data/data_split.json by default
python src/split_data.py --strategy temporal --ratios 0.8 0.1 0.1

# Other splitting strategies
python src/split_data.py --strategy random --ratios 0.8 0.1 0.1  # Random (breaks temporal order)
python src/split_data.py --strategy flight --ratios 0.8 0.1 0.1  # Entire flights (needs many flights)

# Custom output location
python src/split_data.py --strategy temporal --output data/my_split.json
```

### Testing and Validation

```bash
# Test dataloader and visualize data
python src/test.py

# Run stabilization inference on test set (default: uses data/data_split.json)
python src/inference.py --split-set test

# Run inference on all data (disable split)
python src/inference.py --split ""

# Run inference on validation set
python src/inference.py --split-set val --output-name baseline_val

# Evaluate stabilization quality (default: loads data/data_split.json if exists)

# RECOMMENDED WORKFLOW: First compute original metrics once
python src/evaluate.py --model original --split-set test

# Then evaluate models (will reuse cached original metrics)
python src/evaluate.py --model baseline
python src/evaluate.py --model baseline_v2
python src/evaluate.py --model mymodel

# Save to custom JSON location
python src/evaluate.py --model baseline --save-json results.json

# Evaluate without split information
python src/evaluate.py --model baseline --split ""
```

### Visualization

```bash
# Launch interactive Gradio app for side-by-side comparison
python app.py

# Launch on specific port
python app.py --port 7860

# Create public shareable link
python app.py --share

# Make accessible from other machines on network
python app.py --server-name 0.0.0.0

# Note: The app automatically loads evaluation metrics from output/{model}/evaluation_results.json if available
```

### Environment Setup

The project uses Python 3.13 with dependencies managed via `pyproject.toml`:


```bash
# Install using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Data Structure

Expected data directory layout:
```
data/
├── images/
│   ├── Flight1/
│   ├── Flight2/
│   ├── Flight3/
│   │   ├── 00000000.png
│   │   ├── 00000001.png
│   │   └── ...
├── labels/
│   ├── Flight1.csv
│   ├── Flight2.csv
│   ├── Flight3.csv
│   └── labels.cache  # Cached dataset metadata
└── raw/
    ├── logs/       # Raw flight logs
    └── videos/     # Original video files

output/          # Stabilized output frames
vis/             # Visualization results
data_split.json  # Train/val/test split configuration (generated)
```

### Data Split Format

The `data_split.json` file contains:


```json
{
  "metadata": {
    "strategy": "temporal",
    "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
    "seed": 42
  },
  "splits": {
    "train": {"Flight1": [0, 1, 2, ..., 799], ...},
    "val": {"Flight1": [800, ..., 899], ...},
    "test": {"Flight1": [900, ..., 999], ...}
  }
}
```

### Sensor Data Format

CSV files in `data/labels/` contain per-frame sensor readings:


- `ax_mDs2, ay_mDs2, az_mDs2`: accelerations (m/s²) in body frame
- `wx_radDs, wy_radDs, wz_radDs`: angular rates (rad/s) in body frame
- `qw, qx, qy, qz`: orientation quaternion (Hamilton convention)

**Important**: Frame-to-sensor synchronization has small time shifts and may need improvement.

## Code Architecture

### Core Implementation (`src/`)

#### Model (`src/model/`)

- **`baseline.py`**: Core stabilization algorithm using optical flow (Lucas-Kanade + RANSAC)
  - `stabilize_frames()`: Main function that warps frames to a reference frame
  - `estimate_scale_translation()`: Estimates scale + translation transform between frames
  - Uses `cv2.goodFeaturesToTrack()` + `cv2.calcOpticalFlowPyrLK()` for feature tracking
  - See `src/model/README.md` for detailed algorithm documentation

#### Utilities (`src/utils/`)

- **`datasets.py`**: PyTorch dataset for temporal video clips + sensor data
  - `LoadClipsAndLabels`: Handles multi-frame sampling with temporal skip rates
  - Returns: `(imgs, labels, paths, shapes, main_frame_ids, label_paths)`
    - `imgs`: (B×T, C, H, W) tensor
    - `labels`: (B, T, F) sensor feature vectors per frame
  - Frame sampling configured via `hyp` dict (`num_frames`, `skip_rate`, etc.)

- **`plots.py`**: Visualization utilities for debugging sensor data on frames

- **`match_logs_to_frames.py`**: Tools for synchronizing sensor logs with video frames

- **`create_clip_from_frames.py`**: Video clip creation utilities

- **`create_frames_from_clip.py`**: Video frame extraction utilities

- **`augmentations.py`**: Data augmentation functions

- **`torch_utils.py`**: PyTorch utility functions for distributed training and model operations

#### Scripts

- **`split_data.py`**: Split dataset into train/validation/test sets
  - Three strategies: temporal (preserves order), random (shuffles), flight (entire flights)
  - Configurable ratios (default: 80/10/10)
  - Saves split configuration to `data/data_split.json`
  - Supports reproducible splits with random seed

- **`inference.py`**: Sliding window inference over full video sequences
  - Processes frames in overlapping windows (default: window=10, stride=5)
  - Outputs stabilized frames to `output/<model_name>/Flight*/*.jpg`
  - Saves transformation parameters to `output/<model_name>/Flight*/<FlightName>.json`
  - Transformation data includes: translations, rotations, scales, and full transform matrices
  - **Supports data splits**: Default split file is `data/data_split.json`
  - Use `--split-set test/val/train` to choose which set to process
  - Use `--split ""` to disable split and process all data

- **`test.py`**: Test dataloader and visualize data

- **`evaluate.py`**: Evaluation pipeline for stabilization quality assessment
  - Compares original and stabilized videos using quantitative metrics
  - **Optimized performance**: Only loads frames that have corresponding stabilized outputs
  - **Cached original metrics**: Use `--model original` to pre-compute original video metrics once
    - Saves to `data/original_metrics.json` by default
    - When evaluating models, automatically loads and reuses these metrics to avoid redundant computation
    - **Workflow**: Run `--model original` once, then evaluate multiple models without recomputing original metrics
  - **Core metrics**: inter-frame stability, optical flow smoothness, PSNR, sharpness
  - **Advanced metrics** (require transformation data):
    - **Stability Score (FFT)**: Frequency-domain analysis of camera motion
    - **Cropping Ratio**: Largest inscribed rectangle with valid pixels
    - **Distortion Score**: Eigenvalue-based measure of non-rigid warping
  - CLI-based with `--model` argument for easy comparison between different algorithms
  - **Automatic JSON export**: Results saved to `output/{model}/evaluation_results.json` by default
  - Custom JSON path: Use `--save-json <path>` for custom location
  - Default: loads split info from `data/data_split.json` if exists
  - Use `--split ""` to disable split info loading
  - Use `--split-set train/val/test` to specify which set to process for original metrics
  - Outputs per-flight and aggregate statistics with both raw values and improvements
  - **Progress bars**: Shows real-time progress for frame loading and metric computation
  - See `README_evaluate.md` for detailed metric explanations

### Development Notebooks (`notebooks/`)

- **`eda_labels.ipynb`**: Exploratory data analysis of sensor labels
- **`eda_logs.ipynb`**: Exploratory data analysis of flight logs

## Evaluation Metrics

The `evaluate.py` script computes the following metrics to assess stabilization quality:

### Stability Metrics (Lower is Better)

- **Inter-frame Difference**: Mean absolute pixel difference between consecutive frames. Lower values indicate less motion/jitter.
- **Optical Flow Magnitude**: Average magnitude of optical flow vectors between frames. Lower values indicate smoother motion.

### Quality Metrics (Higher is Better)

- **PSNR**: Peak Signal-to-Noise Ratio between consecutive frames. Higher values indicate less frame-to-frame variation.
- **Sharpness**: Laplacian variance measuring image detail preservation. Higher values indicate sharper, less blurred images.

### Distortion Metrics

- **Cropping Ratio**: Area of largest inscribed rectangle without black borders, relative to original frame. Higher is better (less cropping).
- **Distortion Score**: Ratio of transformation matrix eigenvalues. Values near 1.0 indicate rigid transformations; higher values indicate non-rigid warping (stretching/shearing).

### Improvement Metrics

- **Improvement Percentage**: Relative improvement compared to original video for inter-frame difference and flow magnitude.

## Visualization Tools

### Interactive Gradio App (`app.py`)

A web-based viewer for comparing stabilization results:

**Features:**
- Side-by-side comparison of raw vs stabilized frames
- Dropdown selectors for left/right models (including "Raw")
- **Automatic metrics display**: Shows evaluation metrics below each model if available
- **Video playback mode**: Play through frames as a video with adjustable FPS
- Frame navigation with slider and prev/next buttons
- Loop mode for continuous playback
- Only shows test set frames (respects data split)
- Compare multiple stabilization models
- Responsive web interface

**Usage:**
```bash
python app.py                    # Launch on http://localhost:7860
python app.py --share            # Create public link
python app.py --port 8080        # Custom port
```

**Requirements:** The app expects:
- `data/images/Flight*/` - Original frames
- `output/baseline/Flight*/` - Stabilized frames from inference
- `data/data_split.json` - Data split configuration (optional)
- `output/baseline/evaluation_results.json` - Evaluation metrics (auto-generated by `evaluate.py`)

## Development Notes

- **Online constraint**: Prefer methods that use only past frames (causal filtering). Using future frames reduces real-time viability.
- **Sensor-frame sync**: Matching between video and sensor logs is approximate. Improving synchronization can be part of the solution.
- **Data splitting**: Use `split_data.py` to create train/val/test splits with configurable ratios and strategies.
- The baseline uses only image data (optical flow). Sensor data is loaded but not used in stabilization—this is an opportunity for improvement.
- **Visualization**: Use `app.py` to launch an interactive viewer for qualitative assessment of stabilization results.

### Evaluation Workflow

#### With Data Splits (Recommended)

```bash
# 1. Create data split (one-time setup)
python src/split_data.py --strategy temporal --ratios 0.8 0.1 0.1

# 2. Compute original video metrics ONCE (saves time when evaluating multiple models)
python src/evaluate.py --model original --split-set test

# 3. Run stabilization on test set only (default split file: data/data_split.json)
python src/inference.py --split-set test

# 4. Evaluate test set results (will reuse cached original metrics from step 2)
python src/evaluate.py --model baseline

# 5. Evaluate additional models (no need to recompute original metrics!)
python src/inference.py --split-set test --output-name baseline_v2
python src/evaluate.py --model baseline_v2

# Optional: Process validation set for hyperparameter tuning
python src/evaluate.py --model original --split-set val  # Compute val set original metrics
python src/inference.py --split-set val --output-name baseline_val
python src/evaluate.py --model baseline_val
```

#### Without Data Splits (Process All Data)

```bash
# 1. Compute original metrics for all data
python src/evaluate.py --model original --split ""

# 2. Run stabilization on all data (disable split with empty string)
python src/inference.py --split ""

# 3. Evaluate results (will use cached original metrics)
python src/evaluate.py --model baseline --split ""
```

### Notes

- **Performance optimization**: Computing original metrics once and caching them (`--model original`) can save **significant time** when evaluating multiple models, as original metrics computation is the most expensive operation (optical flow on all frames).
- **Advanced metrics**: The FFT-based Stability Score and Distortion Score require transformation parameters saved by `inference.py`. Without these files, evaluation will compute only the basic metrics.
- **Baseline is unsupervised**: The optical flow baseline doesn't require training, but the split infrastructure supports future learning-based methods.
- **Temporal continuity**: For video stabilization, the `temporal` split strategy is recommended as it preserves temporal order within each flight.
