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


def get_flight_frame_counts(data_dir):
    """
    Get the number of frames for each flight.

    Returns:
        dict: {flight_name: num_frames}
    """
    data_path = Path(data_dir)
    flight_counts = {}

    for flight_dir in sorted(data_path.iterdir()):
        if flight_dir.is_dir():
            # Count image files
            frame_files = (
                list(flight_dir.glob("*.png")) +
                list(flight_dir.glob("*.jpg")) +
                list(flight_dir.glob("*.jpeg"))
            )
            flight_counts[flight_dir.name] = len(frame_files)

    return flight_counts


def split_temporal(flight_counts, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Split frames within each flight into temporal segments.

    This preserves temporal continuity within each split.

    Args:
        flight_counts: dict of {flight_name: num_frames}
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

    for flight_name, num_frames in flight_counts.items():
        # Calculate split points
        train_end = int(num_frames * train_ratio)
        val_end = train_end + int(num_frames * val_ratio)

        # Create temporal segments
        all_indices = np.arange(num_frames)

        split_data['train'][flight_name] = all_indices[:train_end].tolist()
        split_data['val'][flight_name] = all_indices[train_end:val_end].tolist()
        split_data['test'][flight_name] = all_indices[val_end:].tolist()

        print(f"{flight_name}:")
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


def save_split(split_data, output_path, metadata):
    """Save split data to JSON file."""
    output = {
        'metadata': metadata,
        'splits': split_data
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSplit data saved to: {output_path}")


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
    output_path = repo_root / args.output

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    print(f"Analyzing data in: {data_dir}")
    print(f"Strategy: {args.strategy}")
    print(f"Ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)

    # Get flight frame counts
    flight_counts = get_flight_frame_counts(data_dir)

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
        'num_flights': len(flight_counts),
        'flight_counts': flight_counts
    }

    save_split(split_data, output_path, metadata)

    print("\n" + "=" * 80)
    print("USAGE:")
    print(f"  python src/inference.py --split {output_path} --split-set test")
    print("=" * 80)


if __name__ == "__main__":
    main()
