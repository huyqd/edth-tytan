"""
Two-step stabilization pipeline:
1. Apply freeze-aware roll correction (ProcessPictureFreeze.py logic)
2. Apply baseline optical flow stabilization to the roll-corrected frames

This creates a temporary directory with roll-corrected frames, then runs baseline inference on them.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Configuration
FLIGHT = "Flight1"
FREEZE_THRESHOLD = 0.5
OUTPUT_NAME = "roll_then_baseline"
TEMP_DATA_DIR = f"data/{OUTPUT_NAME}"  # Temporary location (mirrors output structure)

print("="*80)
print("Two-Step Stabilization Pipeline")
print("="*80)
print(f"Flight: {FLIGHT}")
print(f"Freeze threshold: {FREEZE_THRESHOLD}")
print()

# Step 1: Run ProcessPictureFreeze to create roll-corrected frames
print("Step 1: Creating freeze-aware roll-corrected frames...")
print("-"*80)

# Check if ProcessPictureFreeze.py has already been run
rotated_dir = f"output/rotated_freeze/{FLIGHT}"
if not os.path.exists(rotated_dir) or len(list(Path(rotated_dir).glob("*.jpg"))) == 0:
    print(f"Running ProcessPictureFreeze.py...")
    result = subprocess.run([sys.executable, "ProcessPictureFreeze.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR in ProcessPictureFreeze.py:")
        print(result.stderr)
        sys.exit(1)
else:
    print(f"Roll-corrected frames already exist in {rotated_dir}")
    num_frames = len(list(Path(rotated_dir).glob("*.jpg")))
    print(f"Found {num_frames} frames")

print()

# Step 2: Create temporary symlink so inference.py can read from rotated frames
print("Step 2: Setting up temporary data directory...")
print("-"*80)

temp_flight_dir = Path(TEMP_DATA_DIR) / FLIGHT
temp_flight_dir.parent.mkdir(parents=True, exist_ok=True)

# Remove existing symlink/directory if it exists
if temp_flight_dir.exists():
    if temp_flight_dir.is_symlink():
        temp_flight_dir.unlink()
    else:
        shutil.rmtree(temp_flight_dir)

# Create symlink to rotated frames
rotated_abs_path = Path(rotated_dir).absolute()
try:
    temp_flight_dir.symlink_to(rotated_abs_path, target_is_directory=True)
    print(f"Created symlink: {temp_flight_dir} -> {rotated_abs_path}")
except OSError:
    # Symlinks might not work on Windows without admin rights, fallback to copy
    print("Symlink failed, copying files instead (this may take a moment)...")
    shutil.copytree(rotated_abs_path, temp_flight_dir)
    print(f"Copied {rotated_dir} -> {temp_flight_dir}")

print()

# Step 3: Temporarily modify inference.py's images_root
print("Step 3: Running baseline inference on roll-corrected frames...")
print("-"*80)

# We need to modify inference.py to read from temp directory
# The cleanest way is to create a modified copy

inference_script = """
import cv2
import numpy as np
import os
import json
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, 'src')

from utils.datasets import create_dataloader
from model import BaselineModel

# Override images_root
images_root = '{temp_data_dir}'
labels_root = 'data/labels'
output_dir = 'output/{output_name}/{flight}'

os.makedirs(output_dir, exist_ok=True)

# Initialize baseline model
model = BaselineModel(max_features=2000)

# Create dataloader
hyp = {{
    "num_frames": 2,
    "skip_rate": [0, 1],
    "val_skip_rate": [0, 1],
    "debug_data": False,
    "frame_wise": 0,
}}

dataloader, dataset = create_dataloader(
    path=images_root,
    annotation_path=labels_root,
    image_root_path=images_root,
    imgsz=320,
    batch_size=1,
    stride=32,
    hyp=hyp,
    augment=False,
    is_training=False,
    img_ext="jpg",  # Changed from png to jpg
    debug_dir=None,
)

n = len(dataset.img_files)
window, stride = 10, 5

print(f"Found {{n}} frames")
print(f"Processing with baseline stabilization...")

pbar = tqdm(range(window, n, stride), desc="Processing windows")

for i in pbar:
    window_frames = [dataset.img_files[j] for j in range(i - window, i)]

    frames = []
    for p in window_frames:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            frames = []
            break
        frames.append(img)

    if len(frames) != window:
        continue

    # Stabilize
    res_dict = model.stabilize_frames(frames, sensor_data=None, ref_idx=stride // 2)
    warped = res_dict["warped"]

    if not warped:
        continue

    # Save stabilized frames
    for j_idx, j in enumerate(range(i - stride, i)):
        img_path = dataset.img_files[j]
        filename = os.path.basename(img_path)
        frame_name = os.path.splitext(filename)[0]

        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, warped[window - stride + j_idx])

print(f"Saved stabilized frames to {{output_dir}}")
""".format(temp_data_dir=TEMP_DATA_DIR, output_name=OUTPUT_NAME, flight=FLIGHT)

# Write temporary inference script
temp_inference_path = "temp_inference_roll_baseline.py"
with open(temp_inference_path, 'w') as f:
    f.write(inference_script)

# Run it
result = subprocess.run([sys.executable, temp_inference_path], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERROR in baseline inference:")
    print(result.stderr)
    os.remove(temp_inference_path)
    sys.exit(1)

# Clean up
os.remove(temp_inference_path)

print()
print("="*80)
print("Pipeline Complete!")
print("="*80)
print(f"Output saved to: output/{OUTPUT_NAME}/{FLIGHT}")
print()
print("To create a video, run:")
print(f"  python src/create_video_from_images.py --model {OUTPUT_NAME} {FLIGHT}")
