"""
Inference script using NVIDIA Hardware Optical Flow.

This script performs stabilization using:
1. Pre-processed images (roll-corrected with freeze detection)
2. IMU-EMA smoothing for rotational jitter
3. NVIDIA HW optical flow for residual translation/scale

Usage:
    # Check GPU support first
    python scripts/check_nvidia_of_support.py

    # Preprocess test split
    python scripts/preprocess_test_split.py

    # Run inference with NVIDIA HW flow
    python scripts/inference_nvidia_flow.py --split-set test

    # Run with classical flow (fallback)
    python scripts/inference_nvidia_flow.py --split-set test --flow-method classical
"""

import cv2
import numpy as np
import os
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional

# Add src to path
import sys
sys.path.insert(0, 'src')

from model.nvidia_optical_flow import create_optical_flow_estimator, NVIDIA_OF_AVAILABLE


def load_split_config(split_path):
    """Load data split configuration."""
    with open(split_path, 'r') as f:
        return json.load(f)


def get_test_frame_indices(split_config):
    """Get test split frame indices."""
    return {
        flight: set(indices)
        for flight, indices in split_config['splits']['test'].items()
    }


def load_sensor_data(labels_dir, flight_name):
    """Load sensor data CSV for a flight."""
    csv_path = os.path.join(labels_dir, f'{flight_name}.csv')
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    return df


def get_sensor_for_frame(df, frame_idx):
    """Get sensor data for a specific frame index."""
    if df is None or frame_idx >= len(df):
        return None

    row = df.iloc[frame_idx]
    return {
        'qw': row[2], 'qx': row[3], 'qy': row[4], 'qz': row[5],
        'ax': row[6], 'ay': row[7], 'az': row[8],
        'wx': row[9], 'wy': row[10], 'wz': row[11]
    }


def smooth_quaternions_ema(quaternions, alpha=0.3):
    """
    Apply causal EMA smoothing to quaternions.

    Args:
        quaternions: List of [qw, qx, qy, qz]
        alpha: Smoothing factor (0-1)

    Returns:
        List of smoothed quaternions
    """
    from scipy.spatial.transform import Rotation, Slerp

    if not quaternions:
        return []

    smoothed = []
    q_smooth = None

    for q_current in quaternions:
        if q_smooth is None:
            q_smooth = np.array(q_current)
            smoothed.append(q_smooth.tolist())
        else:
            # Convert to scipy format [qx, qy, qz, qw]
            q_smooth_scipy = [q_smooth[1], q_smooth[2], q_smooth[3], q_smooth[0]]
            q_current_scipy = [q_current[1], q_current[2], q_current[3], q_current[0]]

            r_smooth = Rotation.from_quat(q_smooth_scipy)
            r_current = Rotation.from_quat(q_current_scipy)

            # SLERP interpolation
            slerp = Slerp([0, 1], Rotation.concatenate([r_smooth, r_current]))
            r_blended = slerp(alpha)

            # Convert back to [qw, qx, qy, qz]
            q_scipy = r_blended.as_quat()
            q_smooth = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
            smoothed.append(q_smooth.tolist())

    return smoothed


def process_flight_with_nvidia_flow(
    flight_name,
    test_indices,
    preprocessed_dir,
    labels_dir,
    output_dir,
    flow_estimator,
    alpha,
    window_size,
    stride,
    profile
):
    """
    Process a single flight using NVIDIA HW optical flow.

    Args:
        flight_name: Name of flight
        test_indices: Set of test frame indices
        preprocessed_dir: Directory with preprocessed images
        labels_dir: Directory with sensor CSV files
        output_dir: Output directory for stabilized frames
        flow_estimator: Optical flow estimator object
        alpha: EMA smoothing alpha
        window_size: Sliding window size
        stride: Sliding window stride
        profile: Enable profiling

    Returns:
        dict: Processing statistics
    """
    # Create output directory
    output_flight_dir = os.path.join(output_dir, flight_name)
    os.makedirs(output_flight_dir, exist_ok=True)

    # Load preprocessed frames
    preprocessed_flight_dir = os.path.join(preprocessed_dir, flight_name)
    if not os.path.exists(preprocessed_flight_dir):
        print(f"Warning: No preprocessed images for {flight_name}")
        return {'processed': 0, 'errors': 0}

    # Get test frames only
    all_frames = sorted([
        os.path.join(preprocessed_flight_dir, f)
        for f in os.listdir(preprocessed_flight_dir)
        if f.endswith(('.jpg', '.png'))
    ])

    test_frames = []
    for frame_path in all_frames:
        filename = os.path.basename(frame_path)
        frame_idx = int(os.path.splitext(filename)[0])
        if frame_idx in test_indices:
            test_frames.append((frame_idx, frame_path))

    test_frames.sort(key=lambda x: x[0])

    if not test_frames:
        print(f"Warning: No test frames for {flight_name}")
        return {'processed': 0, 'errors': 0}

    # Load sensor data
    sensor_df = load_sensor_data(labels_dir, flight_name)

    print(f"\n{flight_name}: Processing {len(test_frames)} test frames...")
    print(f"  Method: {'NVIDIA HW' if hasattr(flow_estimator, 'nv_of') else 'Classical'} optical flow")
    print(f"  EMA alpha: {alpha}")

    # Sliding window processing
    processed_count = 0
    error_count = 0
    transform_data = {}

    frame_indices = [idx for idx, _ in test_frames]
    frame_paths = [path for _, path in test_frames]

    # Process in sliding windows
    num_windows = (len(frame_indices) - window_size) // stride + 1

    for window_idx in tqdm(range(num_windows), desc=f"  {flight_name}"):
        start_idx = window_idx * stride
        end_idx = min(start_idx + window_size, len(frame_indices))

        # Get window frames
        window_frame_indices = frame_indices[start_idx:end_idx]
        window_frame_paths = frame_paths[start_idx:end_idx]

        # Load frames
        frames = []
        for frame_path in window_frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                error_count += 1
                continue
            frames.append(frame)

        if len(frames) != len(window_frame_paths):
            continue

        # Load sensor data
        quaternions = []
        for frame_idx in window_frame_indices:
            sensor = get_sensor_for_frame(sensor_df, frame_idx)
            if sensor:
                quaternions.append([sensor['qw'], sensor['qx'], sensor['qy'], sensor['qz']])
            else:
                quaternions.append([1.0, 0.0, 0.0, 0.0])  # Identity

        # Apply EMA smoothing to quaternions (if available)
        if len(quaternions) == len(frames):
            quaternions_smooth = smooth_quaternions_ema(quaternions, alpha=alpha)
        else:
            quaternions_smooth = quaternions

        # Reference frame (middle of window)
        ref_idx = len(frames) // 2
        ref_frame = frames[ref_idx]

        # Convert to grayscale for optical flow
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

        # Process each frame in window
        for i, (frame, frame_idx) in enumerate(zip(frames, window_frame_indices)):
            if i == ref_idx:
                # Reference frame - save as is
                output_path = os.path.join(output_flight_dir, f'{frame_idx:08d}.jpg')
                cv2.imwrite(output_path, frame)
                transform_data[frame_idx] = {
                    'scale': 1.0,
                    'translation': [0.0, 0.0],
                    'inliers': 0
                }
                processed_count += 1
                continue

            # Estimate optical flow to reference
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            try:
                scale, (tx, ty), inliers = flow_estimator.estimate_scale_translation(
                    gray, ref_gray
                )

                # Apply transform
                h, w = frame.shape[:2]
                M = np.array([
                    [scale, 0.0, tx],
                    [0.0, scale, ty]
                ], dtype=np.float32)

                warped = cv2.warpAffine(
                    frame, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )

                # Save result
                output_path = os.path.join(output_flight_dir, f'{frame_idx:08d}.jpg')
                cv2.imwrite(output_path, warped)

                # Store transform data
                transform_data[frame_idx] = {
                    'scale': float(scale),
                    'translation': [float(tx), float(ty)],
                    'inliers': int(np.sum(inliers)) if inliers is not None else 0
                }

                processed_count += 1

            except Exception as e:
                if profile:
                    print(f"Error processing frame {frame_idx}: {e}")
                error_count += 1

    # Save transform data
    transform_json_path = os.path.join(output_flight_dir, f'{flight_name}.json')
    with open(transform_json_path, 'w') as f:
        json.dump(transform_data, f, indent=2)

    return {'processed': processed_count, 'errors': error_count}


def main():
    parser = argparse.ArgumentParser(
        description="Stabilization inference using NVIDIA Hardware Optical Flow"
    )

    parser.add_argument(
        '--preprocessed-dir',
        type=str,
        default='data/images_preprocessed',
        help='Directory with preprocessed images (default: data/images_preprocessed)'
    )

    parser.add_argument(
        '--labels-dir',
        type=str,
        default='data/labels',
        help='Labels directory with CSV files (default: data/labels)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/nvidia_hw_flow',
        help='Output directory (default: output/nvidia_hw_flow)'
    )

    parser.add_argument(
        '--split-file',
        type=str,
        default='data/data_split.json',
        help='Data split JSON file (default: data/data_split.json)'
    )

    parser.add_argument(
        '--split-set',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Which split to process (default: test)'
    )

    parser.add_argument(
        '--flow-method',
        type=str,
        default='nvidia_hw',
        choices=['nvidia_hw', 'classical'],
        help='Optical flow method (default: nvidia_hw)'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.3,
        help='EMA smoothing alpha for quaternions (default: 0.3)'
    )

    parser.add_argument(
        '--window',
        type=int,
        default=10,
        help='Sliding window size (default: 10)'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=5,
        help='Sliding window stride (default: 5)'
    )

    parser.add_argument(
        '--grid-size',
        type=int,
        default=1,
        choices=[1, 2, 4],
        help='NVIDIA OF grid size: 1=dense, 2=half, 4=quarter (default: 1)'
    )

    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable profiling (timing information)'
    )

    args = parser.parse_args()

    # Load split config
    if not os.path.exists(args.split_file):
        print(f"Error: Split file not found: {args.split_file}")
        return

    print("="*80)
    print("NVIDIA HARDWARE OPTICAL FLOW INFERENCE")
    print("="*80)
    print(f"Preprocessed images: {args.preprocessed_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split: {args.split_set}")
    print(f"Flow method: {args.flow_method}")
    print(f"EMA alpha: {args.alpha}")
    print(f"Window/stride: {args.window}/{args.stride}")
    if args.flow_method == 'nvidia_hw':
        print(f"Grid size: {args.grid_size}")
    print("="*80)

    # Check preprocessed directory
    if not os.path.exists(args.preprocessed_dir):
        print(f"\nError: Preprocessed directory not found: {args.preprocessed_dir}")
        print("Please run preprocessing first:")
        print("  python scripts/preprocess_test_split.py")
        return

    # Create optical flow estimator
    flow_estimator = create_optical_flow_estimator(
        method=args.flow_method,
        grid_size=args.grid_size,
        enable_profiling=args.profile
    )

    # Load split config
    split_config = load_split_config(args.split_file)
    test_indices = {
        flight: set(indices)
        for flight, indices in split_config['splits'][args.split_set].items()
    }

    print(f"\n{args.split_set.upper()} split contains:")
    for flight, indices in test_indices.items():
        print(f"  {flight}: {len(indices)} frames")

    # Process each flight
    total_processed = 0
    total_errors = 0

    for flight_name, indices in test_indices.items():
        if not indices:
            continue

        stats = process_flight_with_nvidia_flow(
            flight_name,
            indices,
            args.preprocessed_dir,
            args.labels_dir,
            args.output_dir,
            flow_estimator,
            args.alpha,
            args.window,
            args.stride,
            args.profile
        )

        total_processed += stats['processed']
        total_errors += stats['errors']

        print(f"  Processed: {stats['processed']}, Errors: {stats['errors']}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total frames processed: {total_processed}")
    print(f"Errors: {total_errors}")
    print(f"\nStabilized frames saved to: {args.output_dir}")
    print("="*80)

    # Save metadata
    metadata = {
        'flow_method': args.flow_method,
        'preprocessed_dir': args.preprocessed_dir,
        'split_set': args.split_set,
        'alpha': args.alpha,
        'window': args.window,
        'stride': args.stride,
        'grid_size': args.grid_size if args.flow_method == 'nvidia_hw' else None,
        'frames_processed': total_processed,
        'errors': total_errors
    }

    metadata_path = os.path.join(args.output_dir, 'inference_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    main()
