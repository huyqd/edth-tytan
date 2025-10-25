"""
Data splitting script for video stabilization dataset.

Splits data into train/validation/test sets with configurable ratios.
Supports two splitting strategies:
1. Temporal splitting: Split frames within each flight into temporal segments
2. Flight-level splitting: Split entire flights (requires many flights)

Usage:
    python src/split_data.py --strategy temporal --ratios 0.8 0.1 0.1
    python src/split_data.py --strategy temporal --ratios 0.8 0.1 0.1 --seed 42
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


def verify_labels_exist(labels_dir, flight_name):
    """
    Verify that labels CSV exists for a flight.

    Args:
        labels_dir: Path to labels directory
        flight_name: Name of the flight

    Returns:
        Path to labels CSV if exists, None otherwise
    """
    labels_path = Path(labels_dir) / f"{flight_name}.csv"
    return labels_path if labels_path.exists() else None


def get_label_frame_ids(labels_csv_path):
    """
    Get frame IDs from labels CSV file.

    Args:
        labels_csv_path: Path to labels CSV

    Returns:
        set of frame IDs (as integers)
    """
    try:
        df = pd.read_csv(labels_csv_path)
        # Convert frame_id strings like "00000001" to integers
        frame_ids = set(int(fid) if isinstance(fid, (int, str)) else int(fid)
                       for fid in df['frame_id'])
        return frame_ids
    except Exception as e:
        print(f"Warning: Error reading {labels_csv_path}: {e}")
        return set()


def get_flight_frame_counts(data_dir, labels_dir=None, verify_labels=True):
    """
    Get the number of frames for each flight and verify labels if requested.

    Args:
        data_dir: Path to images directory
        labels_dir: Optional path to labels directory for verification
        verify_labels: Whether to verify labels exist and match frames

    Returns:
        dict: {flight_name: {'num_frames': int, 'has_labels': bool, 'label_coverage': float}}
    """
    data_path = Path(data_dir)
    flight_info = {}

    for flight_dir in sorted(data_path.iterdir()):
        if flight_dir.is_dir():
            flight_name = flight_dir.name

            # Count image files
            frame_files = (
                list(flight_dir.glob("*.png")) +
                list(flight_dir.glob("*.jpg")) +
                list(flight_dir.glob("*.jpeg"))
            )

            # Get frame indices from filenames
            frame_indices = set()
            for frame_file in frame_files:
                try:
                    frame_idx = int(frame_file.stem)
                    frame_indices.add(frame_idx)
                except ValueError:
                    continue

            num_frames = len(frame_indices)
            has_labels = False
            label_coverage = 0.0

            # Verify labels if requested
            if verify_labels and labels_dir:
                labels_csv = verify_labels_exist(labels_dir, flight_name)
                if labels_csv:
                    label_frame_ids = get_label_frame_ids(labels_csv)
                    has_labels = True

                    # Calculate coverage (what percentage of frames have labels)
                    if num_frames > 0:
                        matching_frames = frame_indices & label_frame_ids
                        label_coverage = len(matching_frames) / num_frames

                    if label_coverage < 1.0:
                        missing_frames = frame_indices - label_frame_ids
                        extra_labels = label_frame_ids - frame_indices
                        print(f"  Warning: {flight_name} label coverage: {label_coverage:.1%}")
                        if missing_frames:
                            print(f"    Missing labels for {len(missing_frames)} frames")
                        if extra_labels:
                            print(f"    Extra labels for {len(extra_labels)} frames (no corresponding images)")
                else:
                    print(f"  Warning: No labels CSV found for {flight_name} (expected: {flight_name}.csv)")

            flight_info[flight_name] = {
                'num_frames': num_frames,
                'has_labels': has_labels,
                'label_coverage': label_coverage
            }

    return flight_info


def split_temporal(flight_info, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Split frames within each flight into temporal segments.

    This preserves temporal continuity within each split.

    Args:
        flight_info: dict of {flight_name: {'num_frames': int, ...}}
        train_ratio: Fraction for training (e.g., 0.8)
        val_ratio: Fraction for validation (e.g., 0.1)
        test_ratio: Fraction for test (e.g., 0.1)
        seed: Random seed for reproducibility

    Returns:
        dict with 'train', 'val', 'test' keys, each containing
        dict of {flight_name: list of frame indices}
    """
    np.random.seed(seed)

    split_data = {
        'train': {},
        'val': {},
        'test': {}
    }

    for flight_name, info in flight_info.items():
        num_frames = info['num_frames']

        # Calculate split points
        train_end = int(num_frames * train_ratio)
        val_end = train_end + int(num_frames * val_ratio)

        # Create temporal segments
        all_indices = np.arange(num_frames)

        split_data['train'][flight_name] = all_indices[:train_end].tolist()
        split_data['val'][flight_name] = all_indices[train_end:val_end].tolist()
        split_data['test'][flight_name] = all_indices[val_end:].tolist()

        labels_status = f" (labels: {info['label_coverage']:.0%})" if info['has_labels'] else " (no labels)"
        print(f"{flight_name}{labels_status}:")
        print(f"  Total frames: {num_frames}")
        print(f"  Train: {len(split_data['train'][flight_name])} frames (indices 0-{train_end-1})")
        print(f"  Val:   {len(split_data['val'][flight_name])} frames (indices {train_end}-{val_end-1})")
        print(f"  Test:  {len(split_data['test'][flight_name])} frames (indices {val_end}-{num_frames-1})")

    return split_data


def split_temporal_random(flight_counts, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Split frames within each flight randomly (non-temporal).

    Warning: This breaks temporal continuity and may not be suitable
    for video stabilization tasks.

    Args:
        flight_counts: dict of {flight_name: num_frames}
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed

    Returns:
        dict with split data
    """
    np.random.seed(seed)

    split_data = {
        'train': {},
        'val': {},
        'test': {}
    }

    for flight_name, num_frames in flight_counts.items():
        # Calculate split sizes
        train_size = int(num_frames * train_ratio)
        val_size = int(num_frames * val_ratio)

        # Shuffle indices
        all_indices = np.arange(num_frames)
        np.random.shuffle(all_indices)

        split_data['train'][flight_name] = sorted(all_indices[:train_size].tolist())
        split_data['val'][flight_name] = sorted(all_indices[train_size:train_size + val_size].tolist())
        split_data['test'][flight_name] = sorted(all_indices[train_size + val_size:].tolist())

        print(f"{flight_name}:")
        print(f"  Total frames: {num_frames}")
        print(f"  Train: {len(split_data['train'][flight_name])} frames (random)")
        print(f"  Val:   {len(split_data['val'][flight_name])} frames (random)")
        print(f"  Test:  {len(split_data['test'][flight_name])} frames (random)")

    return split_data


def split_flights(flight_counts, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Split entire flights into train/val/test sets.

    This is suitable when you have many flights.

    Args:
        flight_counts: dict of {flight_name: num_frames}
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed

    Returns:
        dict with split data
    """
    np.random.seed(seed)

    flight_names = list(flight_counts.keys())
    num_flights = len(flight_names)

    # Calculate split sizes
    train_size = max(1, int(num_flights * train_ratio))
    val_size = max(1, int(num_flights * val_ratio))

    # Shuffle flights
    shuffled_flights = flight_names.copy()
    np.random.shuffle(shuffled_flights)

    # Split flights
    train_flights = shuffled_flights[:train_size]
    val_flights = shuffled_flights[train_size:train_size + val_size]
    test_flights = shuffled_flights[train_size + val_size:]

    # Create split data with all frames for each flight
    split_data = {
        'train': {},
        'val': {},
        'test': {}
    }

    for flight in train_flights:
        split_data['train'][flight] = list(range(flight_counts[flight]))

    for flight in val_flights:
        split_data['val'][flight] = list(range(flight_counts[flight]))

    for flight in test_flights:
        split_data['test'][flight] = list(range(flight_counts[flight]))

    print(f"Flight-level split:")
    print(f"  Total flights: {num_flights}")
    print(f"  Train flights: {train_flights} ({len(train_flights)} flights)")
    print(f"  Val flights:   {val_flights} ({len(val_flights)} flights)")
    print(f"  Test flights:  {test_flights} ({len(test_flights)} flights)")

    return split_data


def print_split_summary(split_data):
    """Print summary statistics for the split."""
    print("\n" + "=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)

    for split_name in ['train', 'val', 'test']:
        total_frames = sum(len(frames) for frames in split_data[split_name].values())
        num_flights = len(split_data[split_name])
        print(f"\n{split_name.upper()}:")
        print(f"  Total frames: {total_frames}")
        print(f"  Flights: {num_flights}")
        for flight, frames in split_data[split_name].items():
            if len(frames) > 0:
                print(f"    {flight}: {len(frames)} frames")

    # Calculate overall ratios
    all_frames = sum(
        sum(len(frames) for frames in split_data[split_name].values())
        for split_name in ['train', 'val', 'test']
    )

    train_frames = sum(len(frames) for frames in split_data['train'].values())
    val_frames = sum(len(frames) for frames in split_data['val'].values())
    test_frames = sum(len(frames) for frames in split_data['test'].values())

    print(f"\nActual ratios:")
    print(f"  Train: {train_frames / all_frames:.2%} ({train_frames}/{all_frames})")
    print(f"  Val:   {val_frames / all_frames:.2%} ({val_frames}/{all_frames})")
    print(f"  Test:  {test_frames / all_frames:.2%} ({test_frames}/{all_frames})")


def save_split_csv_files(split_data, labels_dir, output_base_dir):
    """
    Save split-specific CSV files for each flight.

    Args:
        split_data: dict with 'train', 'val', 'test' keys containing frame indices per flight
        labels_dir: Path to original labels directory
        output_base_dir: Base directory for split CSV files (e.g., data/labels_split)
    """
    print("\nCreating split CSV files...")

    for split_name in ['train', 'val', 'test']:
        split_dir = Path(output_base_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for flight_name, frame_indices in split_data[split_name].items():
            if not frame_indices:
                continue

            # Read original CSV
            csv_path = Path(labels_dir) / f"{flight_name}.csv"
            if not csv_path.exists():
                print(f"  Warning: CSV not found for {flight_name}, skipping")
                continue

            try:
                df = pd.read_csv(csv_path)

                # Convert frame_id to int for filtering
                df['frame_id_int'] = df['frame_id'].apply(lambda x: int(x) if isinstance(x, (int, str)) else int(x))

                # Filter rows for this split
                frame_indices_set = set(frame_indices)
                df_split = df[df['frame_id_int'].isin(frame_indices_set)].copy()
                df_split = df_split.drop(columns=['frame_id_int'])

                # Save split CSV
                output_csv = split_dir / f"{flight_name}.csv"
                df_split.to_csv(output_csv, index=False)

                print(f"  {split_name}/{flight_name}.csv: {len(df_split)} rows")

            except Exception as e:
                print(f"  Error processing {flight_name}: {e}")

    print(f"\nSplit CSV files saved to: {output_base_dir}/{{train,val,test}}/")


def save_split(split_data, output_path, metadata, labels_dir=None):
    """
    Save split data to JSON file and optionally create split CSV files.

    Args:
        split_data: dict with 'train', 'val', 'test' splits
        output_path: Path to output JSON file
        metadata: Metadata dict to save
        labels_dir: Optional path to labels directory for creating split CSV files
    """
    output = {
        'metadata': metadata,
        'splits': split_data
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSplit data saved to: {output_path}")

    # Also create split CSV files if labels_dir is provided
    if labels_dir:
        output_base_dir = Path(output_path).parent / "labels_split"
        save_split_csv_files(split_data, labels_dir, output_base_dir)

        # Add labels_split info to metadata
        output['metadata']['labels_split_dir'] = str(output_base_dir)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Split video stabilization data into train/val/test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Temporal split (preserves temporal order within each flight)
  python src/split_data.py --strategy temporal --ratios 0.8 0.1 0.1

  # Random split (shuffles frames, breaks temporal continuity)
  python src/split_data.py --strategy random --ratios 0.8 0.1 0.1

  # Flight-level split (entire flights in each set)
  python src/split_data.py --strategy flight --ratios 0.8 0.1 0.1

  # Custom seed for reproducibility
  python src/split_data.py --strategy temporal --ratios 0.8 0.1 0.1 --seed 42
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/images',
        help='Path to data directory (default: data/images)'
    )

    parser.add_argument(
        '--labels-dir',
        type=str,
        default='data/labels',
        help='Path to labels directory (default: data/labels)'
    )

    parser.add_argument(
        '--split-csv',
        action='store_true',
        help='Also create split CSV files for sensor data'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        choices=['temporal', 'random', 'flight'],
        default='temporal',
        help='Splitting strategy (default: temporal)'
    )

    parser.add_argument(
        '--ratios',
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='Split ratios for train/val/test (default: 0.8 0.1 0.1)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/data_split.json',
        help='Output path for split file (default: data/data_split.json)'
    )

    args = parser.parse_args()

    # Validate ratios
    train_ratio, val_ratio, test_ratio = args.ratios
    total_ratio = sum(args.ratios)
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio})")
        return

    # Normalize ratios
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    # Get repository root
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / args.data_dir
    labels_dir = repo_root / args.labels_dir
    output_path = repo_root / args.output

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    if args.split_csv and not labels_dir.exists():
        print(f"Warning: Labels directory not found: {labels_dir}")
        print("Continuing without CSV splitting...")
        args.split_csv = False

    print(f"Analyzing data in: {data_dir}")
    if args.split_csv:
        print(f"Labels directory: {labels_dir}")
    print(f"Strategy: {args.strategy}")
    print(f"Ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    print(f"Random seed: {args.seed}")
    print(f"Split CSV files: {'Yes' if args.split_csv else 'No'}")
    print("=" * 80)

    # Get flight frame counts (verify labels if splitting CSV)
    flight_counts = get_flight_frame_counts(data_dir, labels_dir if args.split_csv else None, verify_labels=args.split_csv)

    if not flight_counts:
        print(f"Error: No flights found in {data_dir}")
        return

    print(f"\nFound {len(flight_counts)} flight(s)")
    print("-" * 80)

    # Perform split based on strategy
    if args.strategy == 'temporal':
        split_data = split_temporal(flight_counts, train_ratio, val_ratio, test_ratio, args.seed)
    elif args.strategy == 'random':
        print("Warning: Random split breaks temporal continuity!")
        split_data = split_temporal_random(flight_counts, train_ratio, val_ratio, test_ratio, args.seed)
    elif args.strategy == 'flight':
        if len(flight_counts) < 3:
            print(f"Warning: Only {len(flight_counts)} flights available. Flight-level split may not work well.")
        split_data = split_flights(flight_counts, train_ratio, val_ratio, test_ratio, args.seed)

    # Print summary
    print_split_summary(split_data)

    # Save split
    metadata = {
        'strategy': args.strategy,
        'ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'seed': args.seed,
        'data_dir': str(args.data_dir),
        'labels_dir': str(args.labels_dir) if args.split_csv else None,
        'num_flights': len(flight_counts),
        'flight_counts': flight_counts
    }

    save_split(split_data, output_path, metadata, labels_dir=labels_dir if args.split_csv else None)

    print("\n" + "=" * 80)
    print("USAGE:")
    print(f"  python src/inference.py --split {output_path} --split-set test")
    print("=" * 80)


if __name__ == "__main__":
    main()
