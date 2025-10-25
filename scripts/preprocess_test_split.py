"""
Pre-process test split images with roll correction and freeze detection.

This script processes only the test split frames from all flights,
applying roll correction (based on IMU quaternions) and freeze detection.

Usage:
    python scripts/preprocess_test_split.py
    python scripts/preprocess_test_split.py --output-dir data/images_preprocessed
    python scripts/preprocess_test_split.py --freeze-threshold 0.7
"""

import numpy as np
import cv2
import pandas as pd
import math
import os
import json
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def quaternion_to_euler(qw, qx, qy, qz, degrees=False):
    """
    Convert Hamilton quaternion to Euler angles (roll, pitch, yaw).

    Args:
        qw: Scalar component of quaternion
        qx: x component of quaternion
        qy: y component of quaternion
        qz: z component of quaternion
        degrees: If True, return angles in degrees. Otherwise radians (default: False)

    Returns:
        tuple: (roll, pitch, yaw) in radians or degrees
    """
    # Normalize quaternion
    norm = np.sqrt(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx ** 2 + qy ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy ** 2 + qz ** 2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    if degrees:
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)

    return roll, pitch, yaw


def rotate_image(image, angle_degrees, output_size=None, fill_color=(0, 0, 0)):
    """
    Rotate an image by a certain angle around its center.

    Args:
        image: Input image (numpy array)
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
        output_size: Tuple (width, height) for output image size.
                     If None, uses original size. If larger than original,
                     fills extra space with fill_color. (default: None)
        fill_color: Color to fill empty areas (default: black (0, 0, 0))

    Returns:
        numpy array: Rotated image
    """
    # Get original image dimensions
    orig_height, orig_width = image.shape[:2]

    # Determine output dimensions
    if output_size is None:
        out_width, out_height = orig_width, orig_height
    else:
        out_width, out_height = output_size

    # Calculate center point of the OUTPUT image
    center = (out_width / 2, out_height / 2)

    # Get rotation matrix for the output center
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)

    # If output is larger than input, adjust translation to center the original image
    if out_width > orig_width or out_height > orig_height:
        # Calculate offset to center the original image in the larger canvas
        dx = (out_width - orig_width) / 2
        dy = (out_height - orig_height) / 2

        # Create a larger canvas and place original image in center
        canvas = np.full((out_height, out_width, image.shape[2] if image.ndim == 3 else 1),
                         fill_color, dtype=image.dtype)

        # Calculate paste position
        y1 = int(dy)
        y2 = int(dy + orig_height)
        x1 = int(dx)
        x2 = int(dx + orig_width)

        if image.ndim == 3:
            canvas[y1:y2, x1:x2, :] = image
        else:
            canvas[y1:y2, x1:x2] = image

        # Now rotate the canvas
        rotated = cv2.warpAffine(
            canvas,
            rotation_matrix,
            (out_width, out_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill_color
        )
    else:
        # Output size is same or smaller, rotate directly
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (out_width, out_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill_color
        )

    return rotated


def detect_frozen_frame(frame1, frame2, threshold=0.5):
    """
    Detect if consecutive frames are frozen (camera stopped updating).

    Args:
        frame1, frame2: consecutive frames (numpy arrays)
        threshold: mean absolute difference threshold (lower = more similar)

    Returns:
        bool: True if frames are frozen (nearly identical)
    """
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        frame1_gray = frame1

    if len(frame2.shape) == 3:
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        frame2_gray = frame2

    diff = np.mean(np.abs(frame1_gray.astype(float) - frame2_gray.astype(float)))
    return diff < threshold


def load_split_config(split_path):
    """Load data split configuration."""
    with open(split_path, 'r') as f:
        return json.load(f)


def get_test_frame_indices(split_config):
    """Get test split frame indices for all flights."""
    return {
        flight: set(indices)
        for flight, indices in split_config['splits']['test'].items()
    }


def process_flight(flight_name, test_indices, data_dir, labels_dir, output_dir,
                   freeze_threshold, file_size_tolerance, output_size):
    """
    Process a single flight's test frames.

    Args:
        flight_name: Name of flight (e.g., 'Flight1')
        test_indices: Set of frame indices in test split
        data_dir: Input images directory
        labels_dir: Labels directory with CSV files
        output_dir: Output directory for processed images
        freeze_threshold: Threshold for freeze detection
        file_size_tolerance: File size difference tolerance
        output_size: Tuple (width, height) for output images

    Returns:
        dict: Statistics about processing
    """
    # Paths
    flight_dir = os.path.join(data_dir, flight_name)
    output_flight_dir = os.path.join(output_dir, flight_name)
    os.makedirs(output_flight_dir, exist_ok=True)

    # Load sensor data
    csv_path = os.path.join(labels_dir, f'{flight_name}.csv')
    if not os.path.exists(csv_path):
        print(f"Warning: No sensor data found for {flight_name} at {csv_path}")
        return {'processed': 0, 'frozen': 0, 'errors': 0}

    df = pd.read_csv(csv_path)

    # Get all frame paths and filter to test split
    all_frames = sorted(glob.glob(os.path.join(flight_dir, '*.jpg')) +
                       glob.glob(os.path.join(flight_dir, '*.png')))

    # Filter to only test split frames
    test_frames = []
    for frame_path in all_frames:
        filename = os.path.basename(frame_path)
        frame_idx = int(os.path.splitext(filename)[0])
        if frame_idx in test_indices:
            test_frames.append(frame_path)

    if not test_frames:
        print(f"Warning: No test frames found for {flight_name}")
        return {'processed': 0, 'frozen': 0, 'errors': 0}

    # Sort by frame index
    test_frames.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    # Process frames
    last_unfrozen_path = None
    last_unfrozen_frame = None
    last_unfrozen_rotated = None
    frozen_count = 0
    processed_count = 0
    error_count = 0

    print(f"\n{flight_name}: Processing {len(test_frames)} test frames...")

    for frame_path in tqdm(test_frames, desc=f"  {flight_name}"):
        try:
            filename = os.path.basename(frame_path)
            frame_idx = int(os.path.splitext(filename)[0])

            # Check if frame index exists in CSV
            if frame_idx >= len(df):
                print(f"Warning: Frame {frame_idx} not in CSV for {flight_name}")
                error_count += 1
                continue

            # Get IMU data
            roll, pitch, yaw = quaternion_to_euler(
                df.iloc[frame_idx, 2], df.iloc[frame_idx, 3],
                df.iloc[frame_idx, 4], df.iloc[frame_idx, 5],
                degrees=True
            )

            # Load image
            image = cv2.imread(frame_path)
            if image is None:
                print(f"Warning: Failed to load {frame_path}")
                error_count += 1
                continue

            # Freeze detection
            is_frozen = False
            if last_unfrozen_path is not None:
                # Quick file size check
                size1 = os.path.getsize(last_unfrozen_path)
                size2 = os.path.getsize(frame_path)
                size_diff = abs(size1 - size2) / max(size1, size2) if max(size1, size2) > 0 else 1.0

                if size_diff < file_size_tolerance:
                    # Pixel comparison
                    is_frozen = detect_frozen_frame(last_unfrozen_frame, image, freeze_threshold)

            if is_frozen:
                # Use last unfrozen frame
                rotated_image = last_unfrozen_rotated.copy()
                frozen_count += 1
            else:
                # Rotate by negative roll angle
                rotated_image = rotate_image(
                    image,
                    -roll,
                    output_size=output_size
                )

                # Update tracking
                last_unfrozen_path = frame_path
                last_unfrozen_frame = image.copy()
                last_unfrozen_rotated = rotated_image.copy()

            # Save result
            output_path = os.path.join(output_flight_dir, filename)
            cv2.imwrite(output_path, rotated_image)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            error_count += 1

    return {
        'processed': processed_count,
        'frozen': frozen_count,
        'errors': error_count
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process test split images with roll correction and freeze detection"
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/images',
        help='Input images directory (default: data/images)'
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
        default='data/images_preprocessed',
        help='Output directory for preprocessed images (default: data/images_preprocessed)'
    )

    parser.add_argument(
        '--split-file',
        type=str,
        default='data/data_split.json',
        help='Path to data split JSON file (default: data/data_split.json)'
    )

    parser.add_argument(
        '--freeze-threshold',
        type=float,
        default=0.5,
        help='Freeze detection threshold (default: 0.5)'
    )

    parser.add_argument(
        '--file-size-tolerance',
        type=float,
        default=0.02,
        help='File size difference tolerance for freeze detection (default: 0.02 = 2%%)'
    )

    parser.add_argument(
        '--output-width',
        type=int,
        default=1620,  # 1350 * 1.2
        help='Output image width (default: 1620)'
    )

    parser.add_argument(
        '--output-height',
        type=int,
        default=1296,  # 1080 * 1.2
        help='Output image height (default: 1296)'
    )

    args = parser.parse_args()

    # Load split configuration
    if not os.path.exists(args.split_file):
        print(f"Error: Split file not found: {args.split_file}")
        print("Please run: python src/split_data.py --strategy temporal")
        return

    print("="*80)
    print("TEST SPLIT PRE-PROCESSING")
    print("="*80)
    print(f"Input directory: {args.data_dir}")
    print(f"Labels directory: {args.labels_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split file: {args.split_file}")
    print(f"Freeze threshold: {args.freeze_threshold}")
    print(f"Output size: {args.output_width}x{args.output_height}")
    print("="*80)

    split_config = load_split_config(args.split_file)
    test_indices = get_test_frame_indices(split_config)

    print(f"\nTest split contains:")
    for flight, indices in test_indices.items():
        print(f"  {flight}: {len(indices)} frames")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each flight
    output_size = (args.output_width, args.output_height)
    total_stats = defaultdict(int)

    for flight_name, indices in test_indices.items():
        if not indices:
            print(f"\nSkipping {flight_name}: No test frames")
            continue

        stats = process_flight(
            flight_name,
            indices,
            args.data_dir,
            args.labels_dir,
            args.output_dir,
            args.freeze_threshold,
            args.file_size_tolerance,
            output_size
        )

        for key, value in stats.items():
            total_stats[key] += value

        # Print flight summary
        if stats['processed'] > 0:
            freeze_pct = 100.0 * stats['frozen'] / stats['processed']
            print(f"  Processed: {stats['processed']}, Frozen: {stats['frozen']} ({freeze_pct:.1f}%), Errors: {stats['errors']}")

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total frames processed: {total_stats['processed']}")
    print(f"Frozen frames detected: {total_stats['frozen']} ({100.0*total_stats['frozen']/max(1,total_stats['processed']):.1f}%)")
    print(f"Errors: {total_stats['errors']}")
    print(f"\nPreprocessed images saved to: {args.output_dir}")
    print("="*80)

    # Save processing metadata
    metadata = {
        'input_dir': args.data_dir,
        'output_dir': args.output_dir,
        'split_file': args.split_file,
        'freeze_threshold': args.freeze_threshold,
        'output_size': [args.output_width, args.output_height],
        'stats': dict(total_stats),
        'flights_processed': list(test_indices.keys())
    }

    metadata_path = os.path.join(args.output_dir, 'preprocessing_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    main()
