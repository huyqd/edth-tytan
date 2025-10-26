import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def match_by_timestamp(raw_csv_path, reference_csv_path, output_csv_path,
                       tolerance_ms=0.1):
    """
    Match raw data to reference data based on timestamp field.

    Args:
        raw_csv_path: Path to raw CSV with full data
        reference_csv_path: Path to reference CSV (data/labels/Flight1.csv)
        output_csv_path: Path to save matched CSV
        tolerance_ms: Maximum time difference in milliseconds for matching
    """
    print(f"Loading raw data from: {raw_csv_path}")
    df_raw = pd.read_csv(raw_csv_path)
    print(f"  Total rows: {len(df_raw)}")

    print(f"\nLoading reference data from: {reference_csv_path}")
    df_ref = pd.read_csv(reference_csv_path)
    print(f"  Total rows: {len(df_ref)}")

    # Check if timestamp column exists
    if 'timestamp' not in df_raw.columns:
        print("Error: 'timestamp' column not found in raw data")
        return

    if 'timestamp' not in df_ref.columns:
        print("Error: 'timestamp' column not found in reference data")
        return

    # Get reference timestamps
    ref_timestamps = df_ref['timestamp'].values
    print(f"\nReference timestamps range: {ref_timestamps.min():.6f} to {ref_timestamps.max():.6f}")

    # Match timestamps using nearest neighbor within tolerance
    matched_indices = []
    matched_timestamps = []
    time_diffs = []

    tolerance_sec = tolerance_ms / 1000.0

    for ref_ts in ref_timestamps:
        # Find closest timestamp in raw data
        time_diff = np.abs(df_raw['timestamp'].values - ref_ts)
        closest_idx = np.argmin(time_diff)
        min_diff = time_diff[closest_idx]

        if min_diff <= tolerance_sec:
            matched_indices.append(closest_idx)
            matched_timestamps.append(ref_ts)
            time_diffs.append(min_diff * 1000)  # Convert to ms
        else:
            print(f"Warning: No match found for timestamp {ref_ts:.6f} (min diff: {min_diff * 1000:.2f} ms)")

    print(f"\nMatching results:")
    print(f"  Reference timestamps: {len(ref_timestamps)}")
    print(f"  Matched timestamps:   {len(matched_indices)}")
    print(f"  Unmatched:            {len(ref_timestamps) - len(matched_indices)}")

    if time_diffs:
        print(f"\nTime difference statistics:")
        print(f"  Mean:   {np.mean(time_diffs):.3f} ms")
        print(f"  Median: {np.median(time_diffs):.3f} ms")
        print(f"  Max:    {np.max(time_diffs):.3f} ms")
        print(f"  Std:    {np.std(time_diffs):.3f} ms")

    # Create matched dataframe
    df_matched = df_raw.iloc[matched_indices].copy()

    # Reset index to match reference
    df_matched.reset_index(drop=True, inplace=True)

    # Create matched dataframe
    df_matched = df_raw.iloc[matched_indices].copy()

    # Reset index to match reference
    df_matched.reset_index(drop=True, inplace=True)

    # ADD frame_id from reference data as FIRST column
    if 'frame_id' in df_ref.columns:
        # Get frame_id from reference
        frame_ids = df_ref['frame_id'].iloc[:len(df_matched)].values

        # Remove frame_id if it already exists in matched data
        if 'frame_id' in df_matched.columns:
            df_matched = df_matched.drop('frame_id', axis=1)

        # Insert frame_id as first column
        df_matched.insert(0, 'frame_id', frame_ids)
        print(f"\n✓ Added 'frame_id' as first column from reference data")
    # Save
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_matched.to_csv(output_csv_path, index=False)

    print(f"\n✓ Saved matched data to: {output_csv_path}")
    print(f"  Rows: {len(df_matched)}")
    print(f"  Columns: {list(df_matched.columns)}")

    return df_matched


def process_all_flights(raw_dir, reference_dir, output_dir, tolerance_ms=5.0):
    """Process all flights."""
    raw_path = Path(raw_dir)
    csv_files = sorted(raw_path.glob('Flight*.csv'))

    if not csv_files:
        print(f"No Flight*.csv files found in {raw_dir}")
        return

    print(f"Found {len(csv_files)} flight CSV files\n")

    for csv_file in csv_files:
        flight_name = csv_file.stem
        ref_csv = Path(reference_dir) / f'{flight_name}.csv'
        output_csv = Path(output_dir) / f'{flight_name}.csv'

        print(f"{'=' * 70}")
        print(f"Processing {flight_name}")
        print(f"{'=' * 70}")

        if ref_csv.exists():
            match_by_timestamp(
                str(csv_file),
                str(ref_csv),
                str(output_csv),
                tolerance_ms=tolerance_ms
            )
            print()
        else:
            print(f"Warning: Reference file not found: {ref_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match raw data to reference data based on timestamp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all flights (default tolerance: 5ms)
  python BringFilteredDataAndVideoFrameDataTogether.py

  # Process specific flight
  python BringFilteredDataAndVideoFrameDataTogether.py --flight Flight1

  # Custom tolerance (10ms)
  python BringFilteredDataAndVideoFrameDataTogether.py --tolerance 10.0

  # Custom directories
  python BringFilteredDataAndVideoFrameDataTogether.py \\
      --raw-dir output/rawWithEulerAndFiltAndYawFix \\
      --reference-dir data/labels \\
      --output-dir data/labels_matched
        """
    )

    parser.add_argument(
        '--raw-dir',
        type=str,
        default='output/rawWithEulerAndFiltAndYawFix',
        help='Directory with raw (full) CSV files'
    )

    parser.add_argument(
        '--reference-dir',
        type=str,
        default='data/labels',
        help='Directory with reference CSV files (frame-matched)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/labels',
        help='Output directory for matched CSVs (default: overwrites reference)'
    )

    parser.add_argument(
        '--flight',
        type=str,
        help='Process specific flight only (e.g., Flight1)'
    )

    parser.add_argument(
        '--tolerance',
        type=float,
        default=5.0,
        help='Maximum time difference for matching in milliseconds (default: 5.0)'
    )

    args = parser.parse_args()

    if args.flight:
        # Process single flight
        raw_csv = Path(args.raw_dir) / f'{args.flight}.csv'
        ref_csv = Path(args.reference_dir) / f'{args.flight}.csv'
        output_csv = Path(args.output_dir) / f'{args.flight}.csv'

        if raw_csv.exists() and ref_csv.exists():
            match_by_timestamp(
                str(raw_csv),
                str(ref_csv),
                str(output_csv),
                tolerance_ms=args.tolerance
            )
        else:
            print(f"Error: Required files not found")
            print(f"  Raw CSV: {raw_csv} (exists: {raw_csv.exists()})")
            print(f"  Ref CSV: {ref_csv} (exists: {ref_csv.exists()})")
    else:
        # Process all flights
        process_all_flights(
            args.raw_dir,
            args.reference_dir,
            args.output_dir,
            tolerance_ms=args.tolerance
        )



# Evaluate with python BringFilteredDataAndVideoFrameDataTogeth.py --output-dir output/relevantWithEulerAndFiltAndYawFix/Flight1 --flight Flight1