import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from utils.datasets import create_dataloader
from model.baseline import stabilize_frames


def load_data_split(split_path):
    """Load data split configuration from JSON file."""
    with open(split_path, 'r') as f:
        split_config = json.load(f)
    return split_config


def get_split_frame_indices(split_config, split_set='test'):
    """
    Get frame indices for a specific split set.

    Args:
        split_config: Loaded split configuration
        split_set: 'train', 'val', or 'test'

    Returns:
        dict: {flight_name: set of frame indices}
    """
    split_data = split_config['splits'][split_set]

    # Convert lists to sets for fast lookup
    return {
        flight: set(indices) if indices else set()
        for flight, indices in split_data.items()
    }


def should_process_frame(frame_path, split_indices):
    """
    Check if a frame should be processed based on split.

    Args:
        frame_path: Path to frame image
        split_indices: dict of {flight_name: set of frame indices}

    Returns:
        bool: True if frame should be processed
    """
    if split_indices is None:
        return True  # No split specified, process all

    # Extract flight name and frame index from path
    path_parts = frame_path.split(os.sep)
    flight_name = path_parts[-2]
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]

    try:
        frame_idx = int(frame_name)
    except ValueError:
        return False

    if flight_name not in split_indices:
        return False

    return frame_idx in split_indices[flight_name]


def main():
    parser = argparse.ArgumentParser(
        description="Run video stabilization inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all data
  python src/inference.py

  # Process only test set
  python src/inference.py --split data/data_split.json --split-set test

  # Process validation set
  python src/inference.py --split data/data_split.json --split-set val
        """
    )

    parser.add_argument(
        '--split',
        type=str,
        default='data/data_split.json',
        help='Path to data split JSON file (default: data/data_split.json). Set to empty string to disable.'
    )

    parser.add_argument(
        '--split-set',
        type=str,
        choices=['train', 'val', 'test'],
        default='test',
        help='Which split to process (default: test)'
    )

    parser.add_argument(
        '--output-name',
        type=str,
        default='baseline',
        help='Output folder name under output/ (default: baseline)'
    )

    args = parser.parse_args()

    # repo root (one level up from model/)
    repo_root = os.path.dirname(os.path.dirname(__file__))
    mock_root = os.path.join(repo_root, "data")
    images_root = os.path.join(mock_root, "images")
    labels_root = os.path.join(mock_root, "labels")

    # Load data split if provided
    split_indices = None
    if args.split and args.split.strip():  # Check if split is not empty string
        split_path = os.path.join(repo_root, args.split)
        if os.path.exists(split_path):
            print(f"Loading data split from: {split_path}")
            split_config = load_data_split(split_path)
            split_indices = get_split_frame_indices(split_config, args.split_set)

            print(f"Processing split: {args.split_set}")
            print(f"Flights in split:")
            for flight, indices in split_indices.items():
                if indices:
                    print(f"  {flight}: {len(indices)} frames")
            print("-" * 80)
        else:
            print(f"Warning: Split file not found: {split_path}")
            print("Processing all data without split...")
            print("-" * 80)
    else:
        print("No split specified, processing all data...")
        print("-" * 80)

    # dataloader settings: window of 2 frames
    hyp = {
        "num_frames": 2,
        "skip_rate": [0, 1],
        "val_skip_rate": [0, 1],
        "debug_data": False,
        "frame_wise": 0,
    }

    # create dataloader + dataset (batch_size 1, we will iterate pairs manually)
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
        img_ext="png",
        debug_dir=None,
    )

    out_root = os.path.join(repo_root, "output", args.output_name)
    os.makedirs(out_root, exist_ok=True)

    n = len(dataset.img_files)
    saved = 0
    window, stride = 10, 5
    use_stabilized_frames = False
    cached_stabilized_frames = []

    # Track transformation data per flight
    flight_transforms = defaultdict(lambda: {
        "frames": [],
        "translations": [],
        "rotations": [],
        "scales": [],
        "transforms": []
    })

    for i in range(window, n, stride):
        # Check if any frames in this window are in the split
        window_frames = [dataset.img_files[j] for j in range(i - window, i)]

        # Skip window if no frames should be processed
        if split_indices is not None:
            frames_to_process = [f for f in window_frames if should_process_frame(f, split_indices)]
            if not frames_to_process:
                continue  # Skip this window entirely

        mid_idx = i - (window // 2 + 1) if window > 2 else i - window // 2
        ref_frame_path = dataset.img_files[mid_idx]
        if use_stabilized_frames and cached_stabilized_frames:
            frames_stabilized = cached_stabilized_frames[(window - stride) :]
            frames_unstabilized = [
                dataset.img_files[j] for j in range(i - window + (window - stride), i)
            ]
            frames = frames_stabilized + frames_unstabilized
        else:
            frames = window_frames
        loaded = []
        for p in frames:
            try:
                img = cv2.imread(p, cv2.IMREAD_COLOR)
            except Exception as e:
                img = p
            if img is None:
                print(f"Failed to read frame {p}")
                loaded = []
                break
            loaded.append(img)

        # ensure we have the expected number of frames
        if len(loaded) != window:
            print(f"Skipping set due to read failure or missing frames: {frames}")
            continue

        frames = loaded

        # call existing stabilize function from this module
        res_dict = stabilize_frames(frames, ref_idx=stride // 2)
        warped = res_dict["warped"]
        orig = res_dict["orig"]
        if not warped:
            print(f"Stabilization failed for set {frames}")
            continue

        # Save stabilized frames and transformation data
        for j_idx, j in enumerate(range(i - stride, i)):
            img_path = dataset.img_files[j]

            # Skip if frame not in split
            if split_indices is not None and not should_process_frame(img_path, split_indices):
                continue

            # Extract flight name from path (e.g., "data/images/Flight1/00000000.png" -> "Flight1")
            flight_name = img_path.split(os.sep)[-2]
            frame_name = os.path.splitext(os.path.basename(img_path))[0]
            frame_idx = int(frame_name)  # Assuming frame names are numbers like "00000000"

            # Create flight-specific output directory
            flight_out_dir = os.path.join(out_root, flight_name)
            os.makedirs(flight_out_dir, exist_ok=True)

            # Save as .jpg
            out_path = os.path.join(flight_out_dir, f"{frame_name}.jpg")
            cv2.imwrite(out_path, warped[window - stride + j_idx])

            # Store transformation data
            # Map from window index to actual frame index
            window_idx = window - stride + j_idx
            scale = res_dict["scales"][window_idx]
            translation = res_dict["translations"][window_idx]

            # Build transformation matrix (scale + translation)
            transform_matrix = np.array([
                [scale, 0, translation[0]],
                [0, scale, translation[1]],
                [0, 0, 1]
            ])

            # Baseline doesn't use rotation, so rotation angle is 0
            rotation_angle = 0.0

            # Only add if not already present (avoid duplicates from overlapping windows)
            if frame_idx not in flight_transforms[flight_name]["frames"]:
                flight_transforms[flight_name]["frames"].append(frame_idx)
                flight_transforms[flight_name]["translations"].append(list(translation))
                flight_transforms[flight_name]["rotations"].append(rotation_angle)
                flight_transforms[flight_name]["scales"].append(float(scale))
                flight_transforms[flight_name]["transforms"].append(transform_matrix.tolist())

        cached_stabilized_frames = warped
        saved += 1

    # Save transformation data for each flight
    print("\nSaving transformation data...")
    for flight_name, transform_data in flight_transforms.items():
        # Sort by frame index
        sorted_indices = np.argsort(transform_data["frames"])
        sorted_data = {
            "frames": [transform_data["frames"][i] for i in sorted_indices],
            "translations": [transform_data["translations"][i] for i in sorted_indices],
            "rotations": [transform_data["rotations"][i] for i in sorted_indices],
            "scales": [transform_data["scales"][i] for i in sorted_indices],
            "transforms": [transform_data["transforms"][i] for i in sorted_indices],
        }

        # Save to JSON
        transform_json_path = os.path.join(out_root, flight_name, f"{flight_name}.json")
        with open(transform_json_path, 'w') as f:
            json.dump(sorted_data, f, indent=2)
        print(f"  Saved {flight_name} transforms: {len(sorted_data['frames'])} frames")

    print(f"\nSaved {saved} stabilized pair images to {out_root}")

    if args.split:
        print(f"Processed split: {args.split_set}")
        print(f"From split file: {args.split}")


if __name__ == "__main__":
    main()
