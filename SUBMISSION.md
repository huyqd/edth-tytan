# UAV Video Stabilization Solution

## Team Submission

This repository contains our solution for real-time UAV video stabilization using IMU sensor fusion.

## Approach

Our solution combines:
1. **Roll correction** using IMU quaternion data to level the horizon
2. **High-frequency pitch stabilization** using bandpass-filtered pitch angles
3. **Freeze frame detection** to handle dropped/duplicate frames in the video stream

### Key Innovation

We apply a **bandpass filter (0.3-10 Hz)** to the pitch signal to remove only high-frequency vibrations while preserving intentional camera movements. This allows the drone to naturally follow terrain changes while eliminating jitter caused by motor vibrations and turbulence.

## Pipeline

### Required Input Data

**Before running the pipeline, you must provide the following data:**

**1. Video Frames** (JPG images)
```
data/images/Flight1/00000000.jpg
data/images/Flight1/00000001.jpg
data/images/Flight1/00000002.jpg
...
```
- Format: Sequential JPG images with 8-digit zero-padded filenames
- Resolution: 1350×1080 pixels (or similar)
- Flight folders: `Flight1/`, `Flight2/`, `Flight3/`, etc.

**2. Frame-Synchronized IMU Data** (CSV with quaternions)
```
data/labels/Flight1.csv
```
- **Required columns**: `qw, qx, qy, qz` (orientation quaternions in Hamilton convention)
- Optional columns: `ax_mDs2, ay_mDs2, az_mDs2` (accelerations), `wx_radDs, wy_radDs, wz_radDs` (angular rates)
- **One row per video frame** (synchronized by frame index)

**3. Raw Flight Logs** (CSV with high-rate IMU data - for preprocessing only)
```
data/raw/logs/Flight1.csv
```
- **Required columns**: `timestamp, qw, qx, qy, qz`
- Higher sampling rate than video frames (~300 Hz)
- Used only for preprocessing steps (converts to filtered Euler angles)

---

### 1. Data Preprocessing

**Step 1a: Extract Euler angles and apply bandpass filtering**

```bash
python ProcessRawData.py --flight Flight1
```

This script:
- Converts quaternions from raw flight logs to Euler angles (roll, pitch, yaw)
- Unwraps yaw to avoid 360° discontinuities
- Applies a Butterworth bandpass filter (0.3-10 Hz) to remove high-frequency vibrations
- **Automatically creates output directories** if they don't exist
- Outputs to `output/rawWithEulerAndFiltAndYawFix/<Flight>.csv`

Optional parameters:
- `--lowcut 0.3` - Low cutoff frequency (Hz)
- `--highcut 10` - High cutoff frequency (Hz)
- `--plot` - Show comparison plots of original vs filtered data

**Step 1b: Match filtered data to video frames by timestamp**

```bash
python BringFilteredDataAndVideoFrameDataTogeth.py \
    --raw-dir output/rawWithEulerAndFiltAndYawFix \
    --reference-dir data/labels \
    --output-dir output/relevantWithEulerAndFiltAndYawFix \
    --flight Flight1 \
    --tolerance 5.0
```

This script:
- Matches the filtered sensor data to video frame timestamps
- Uses nearest-neighbor matching with 5ms tolerance
- Outputs synchronized CSV files used by the stabilization script

### 2. Video Stabilization

**Run the main stabilization script:**

```bash
python ProcessPictureFreeze.py \
    --flight Flight1 \
    --start-frame 0 \
    --freeze-threshold 0.5 \
    --pixel-per-deg 28 \
    --output-dir output/rotated_freeze_filt
```

**Parameters:**
- `--flight`: Flight name (Flight1, Flight2, Flight3)
- `--start-frame`: First frame to process (default: 0)
- `--freeze-threshold`: Sensitivity for detecting duplicate frames (default: 0.5, lower = stricter)
- `--pixel-per-deg`: Pixels per degree for pitch correction (default: 28, calibrated for 1350x1080 @ ~40° FOV)
- `--output-dir`: Output directory for stabilized frames

**What it does:**
1. Reads video frames from `data/images/<Flight>/`
2. Reads IMU data from `data/labels/<Flight>.csv`
3. Reads filtered pitch from `output/relevantWithEulerAndFiltAndYawFix/<Flight>/<Flight>.csv`
4. For each frame:
   - Detects if frame is frozen (duplicate) compared to previous frame
   - If frozen: copies the last stabilized frame
   - If not frozen:
     - Translates image vertically based on filtered pitch (removes high-freq jitter)
     - Rotates image to correct roll (levels horizon)
   - Outputs to larger canvas (1620x1296) to accommodate rotation without cropping
5. Saves stabilized frames to `output/rotated_freeze_filt/<Flight>/`

## Algorithm Details

### Roll Correction
- Full correction applied: `-roll` angle from quaternion data
- Removes all horizon tilt to create level footage

### Pitch Stabilization
- **Bandpass filtered** pitch signal (0.3-10 Hz)
- Translation: `-28 pixels/degree × filtered_pitch`
- Clipped to ±108 pixels to stay within canvas borders
- This removes **only** high-frequency vibrations while preserving smooth flight path

### Freeze Frame Handling
- File size comparison first (fast check)
- If similar size, pixel-wise comparison (MAD < threshold)
- Frozen frames reuse the last valid stabilized frame to avoid processing artifacts

### Output Format
- Input: 1350x1080 frames
- Output: 1620x1296 (1.2x upscale) to prevent black borders from rotation
- Format: JPG images matching input filenames

## Dependencies

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

Required packages:
- numpy
- opencv-python (cv2)
- pandas
- scipy (for bandpass filtering)
- tqdm (progress bars)

## Results

The stabilization pipeline:
- ✅ Removes horizon tilt (roll correction)
- ✅ Eliminates high-frequency pitch vibrations
- ✅ Preserves intentional camera movements (terrain following)
- ✅ Handles frozen/duplicate frames gracefully
- ⚡ Real-time capable (~30 FPS on consumer hardware)

## Repository Structure

```
.
├── ProcessPictureFreeze.py          # Main stabilization script
├── ProcessRawData.py                 # Preprocessing: Euler conversion + filtering
├── BringFilteredDataAndVideoFrameDataTogeth.py  # Timestamp matching
├── data/
│   ├── images/<Flight>/             # Input video frames
│   ├── labels/<Flight>.csv          # IMU sensor data (quaternions)
│   └── raw/logs/<Flight>.csv        # Raw flight logs
├── output/
│   ├── rawWithEulerAndFiltAndYawFix/      # Filtered Euler angles
│   ├── relevantWithEulerAndFiltAndYawFix/ # Frame-matched filtered data
│   └── rotated_freeze_filt/<Flight>/      # Stabilized output frames
├── src/                              # Additional utilities (optional)
└── README.md                         # General project info
```

## Quick Start

### Running the Pipeline

**Note:** For Flight1, preprocessed data is already included in `output/relevantWithEulerAndFiltAndYawFix/Flight1/`, so you can skip steps 2-3 and go directly to step 4!

```bash
# STEP 1: Install dependencies
uv sync  # or: pip install -e .

# STEP 2 (OPTIONAL - already done for Flight1): Preprocess IMU data
python ProcessRawData.py --flight Flight1

# STEP 3 (OPTIONAL - already done for Flight1): Match filtered data to frames
python BringFilteredDataAndVideoFrameDataTogeth.py --flight Flight1

# STEP 4: Run stabilization (main algorithm!)
python ProcessPictureFreeze.py --flight Flight1

# Output will be in: output/rotated_freeze_filt/Flight1/
```

### Creating Videos from Output

After stabilization, you can create viewable MP4 videos:

```bash
# Create video from stabilized frames
python src/create_video_from_images.py \
    --input-dir output/rotated_freeze_filt/Flight1 \
    --output-path output/videos/Flight1_stabilized.mp4 \
    --fps 30

# Create video from original frames (for comparison)
python src/create_video_from_images.py \
    --input-dir data/images/Flight1 \
    --output-path output/videos/Flight1_original.mp4 \
    --fps 30
```

### Quantitative Evaluation

Compute stabilization quality metrics:

```bash
# Evaluate stabilized output vs original
python src/evaluate.py \
    --original-dir data/images/Flight1 \
    --stabilized-dir output/rotated_freeze_filt/Flight1 \
    --output-json output/evaluation_results.json

# View metrics in the JSON file
cat output/evaluation_results.json
```

Metrics include:
- Inter-frame stability (motion smoothness)
- Optical flow magnitude (residual jitter)
- PSNR (frame quality preservation)
- Sharpness (detail retention)

### Interactive Visualization

Launch the Gradio web app to compare original vs stabilized videos side-by-side:

```bash
python app.py

# Access at: http://localhost:7860
```

The app provides:
- Side-by-side video comparison
- Frame-by-frame navigation
- Playback controls
- Automatic metric display (if evaluation_results.json exists)

## Complete Example Workflow

```bash
# 1. Install dependencies
uv sync

# 2. Run stabilization (uses pre-included preprocessed data for Flight1)
python ProcessPictureFreeze.py --flight Flight1

# 3. Create comparison videos
python src/create_video_from_images.py \
    --input-dir data/images/Flight1 \
    --output-path output/videos/Flight1_original.mp4 \
    --fps 30

python src/create_video_from_images.py \
    --input-dir output/rotated_freeze_filt/Flight1 \
    --output-path output/videos/Flight1_stabilized.mp4 \
    --fps 30

# 4. Evaluate results
python src/evaluate.py \
    --original-dir data/images/Flight1 \
    --stabilized-dir output/rotated_freeze_filt/Flight1 \
    --output-json output/evaluation_Flight1.json

# 5. Launch interactive viewer
python app.py
```

## Processing Additional Flights

For Flight2, Flight3, or custom data:

```bash
# 1. Run preprocessing
python ProcessRawData.py --flight Flight2

# 2. Match to frames
python BringFilteredDataAndVideoFrameDataTogeth.py --flight Flight2

# 3. Run stabilization
python ProcessPictureFreeze.py --flight Flight2
```

## Contact

For questions about this solution, please contact the team.
