import numpy as np
import pandas as pd
from scipy import signal
import argparse
from pathlib import Path
import os


def quaternion_to_euler_vectorized(qw, qx, qy, qz, degrees=False):
    """
    Vectorized version for arrays of quaternions.

    Args:
        qw, qx, qy, qz: Arrays of quaternion components
        degrees: If True, return angles in degrees

    Returns:
        tuple: (roll_array, pitch_array, yaw_array)
    """
    # Normalize quaternions
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


def add_euler_angles_to_dataframe(df, degrees=True):
    """
    Add Euler angles to a DataFrame containing quaternion columns.

    Args:
        df: DataFrame with columns 'qw', 'qx', 'qy', 'qz'
        degrees: If True, angles in degrees. Otherwise radians.

    Returns:
        DataFrame: Original DataFrame with added 'roll', 'pitch', 'yaw' columns
    """
    roll, pitch, yaw = quaternion_to_euler_vectorized(
        df['qw'].values,
        df['qx'].values,
        df['qy'].values,
        df['qz'].values,
        degrees=degrees
    )

    df['roll'] = roll
    df['pitch'] = pitch
    df['yaw'] = yaw

    return df


def unwrap_yaw(yaw_angles, degrees=True):
    """
    Unwrap yaw angles to create a continuous curve without 360° jumps.

    Args:
        yaw_angles: Array of yaw angles
        degrees: If True, input is in degrees. If False, in radians.

    Returns:
        numpy array: Unwrapped continuous yaw angles
    """
    yaw = np.array(yaw_angles, dtype=float)

    # Convert to radians if needed
    if degrees:
        yaw_rad = np.deg2rad(yaw)
    else:
        yaw_rad = yaw.copy()

    # Use numpy's unwrap to remove 2π (360°) discontinuities
    unwrapped_rad = np.unwrap(yaw_rad)

    # Convert back to degrees if needed
    if degrees:
        return np.rad2deg(unwrapped_rad)
    else:
        return unwrapped_rad


def unwrap_yaw_relative(yaw_angles, degrees=True):
    """
    Unwrap yaw angles and normalize to start at 0 (relative yaw).

    Args:
        yaw_angles: Array of yaw angles
        degrees: If True, input is in degrees. If False, in radians.

    Returns:
        numpy array: Unwrapped continuous yaw angles starting from 0
    """
    # Unwrap
    unwrapped = unwrap_yaw(yaw_angles, degrees=degrees)

    # Normalize to start at 0
    unwrapped_relative = unwrapped - unwrapped[0]

    return unwrapped_relative


def apply_bandpass_filter(data, lowcut, highcut, fs, order):
    """
    Apply a Butterworth bandpass filter to the data.

    Args:
        data: 1D array of data to filter
        lowcut: Low cutoff frequency (Hz) - high-pass component
        highcut: High cutoff frequency (Hz) - low-pass component
        fs: Sampling frequency (Hz) - frame rate of the video
        order: Filter order

    Returns:
        Filtered data array
    """
    # Nyquist frequency
    nyq = 0.5 * fs

    # Normalize frequencies
    low = lowcut / nyq
    high = highcut / nyq

    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')

    # Apply filter
    filtered_data = signal.lfilter(b, a, data)

    return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description="Process raw IMU data: convert quaternions to Euler angles and apply bandpass filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process Flight1
  python ProcessRawData.py --flight Flight1

  # Process Flight2 with custom filter parameters
  python ProcessRawData.py --flight Flight2 --lowcut 0.5 --highcut 15

  # Process and show plots
  python ProcessRawData.py --flight Flight1 --plot
        """
    )

    parser.add_argument(
        '--flight',
        type=str,
        required=True,
        help='Flight name (e.g., Flight1, Flight2, Flight3)'
    )

    parser.add_argument(
        '--lowcut',
        type=float,
        default=0.3,
        help='Low cutoff frequency in Hz (default: 0.3)'
    )

    parser.add_argument(
        '--highcut',
        type=float,
        default=10.0,
        help='High cutoff frequency in Hz (default: 10.0)'
    )

    parser.add_argument(
        '--fs',
        type=float,
        default=327.7,
        help='IMU sampling frequency in Hz (default: 327.7)'
    )

    parser.add_argument(
        '--order',
        type=int,
        default=2,
        help='Butterworth filter order (default: 2)'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Show comparison plots of original vs filtered data'
    )

    args = parser.parse_args()

    # Input path
    input_csv = f'data/raw/logs/{args.flight}.csv'

    if not os.path.exists(input_csv):
        print(f"Error: Input file not found: {input_csv}")
        print(f"Please ensure {input_csv} exists before running preprocessing.")
        return 1

    print(f"Processing {args.flight}...")
    print(f"Input: {input_csv}")

    # Read CSV
    df = pd.read_csv(input_csv)

    # Convert quaternions to Euler angles
    print("Converting quaternions to Euler angles...")
    df = add_euler_angles_to_dataframe(df, degrees=True)

    # Create output directories
    output_dir_1 = Path('output/rawWithEulerAndYawFix')
    output_dir_2 = Path('output/rawWithEulerAndFiltAndYawFix')
    output_dir_1.mkdir(parents=True, exist_ok=True)
    output_dir_2.mkdir(parents=True, exist_ok=True)

    # Save intermediate output (Euler angles only)
    output_path_1 = output_dir_1 / f'{args.flight}.csv'
    df.to_csv(output_path_1, index=False)
    print(f"Saved Euler angles to: {output_path_1}")

    # Unwrap yaw to avoid steps
    print("Unwrapping yaw angles...")
    df['yaw'] = unwrap_yaw_relative(df['yaw'], degrees=True)

    # Apply bandpass filter
    print(f"Applying bandpass filter ({args.lowcut}-{args.highcut} Hz)...")
    df['roll_filtered'] = apply_bandpass_filter(
        df['roll'].values, args.lowcut, args.highcut, args.fs, args.order
    )
    df['pitch_filtered'] = apply_bandpass_filter(
        df['pitch'].values, args.lowcut, args.highcut, args.fs, args.order
    )
    df['yaw_filtered'] = apply_bandpass_filter(
        df['yaw'].values, args.lowcut, args.highcut, args.fs, args.order
    )

    # Save final output (Euler + filtered)
    output_path_2 = output_dir_2 / f'{args.flight}.csv'
    df.to_csv(output_path_2, index=False)
    print(f"Saved filtered data to: {output_path_2}")

    # Optional plotting
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            print("Generating plots...")
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))

            time = df['system_time_s'].values

            # Roll plot
            axes[0].plot(time, df['roll'], label='Original', alpha=0.7)
            axes[0].plot(time, df['roll_filtered'], label='Filtered', linewidth=2)
            axes[0].set_ylabel('Roll (deg)')
            axes[0].set_title('Roll: Original vs Filtered')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Pitch plot
            axes[1].plot(time, df['pitch'], label='Original', alpha=0.7)
            axes[1].plot(time, df['pitch_filtered'], label='Filtered', linewidth=2)
            axes[1].set_ylabel('Pitch (deg)')
            axes[1].set_title('Pitch: Original vs Filtered')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Yaw plot
            axes[2].plot(time, df['yaw'], label='Original', alpha=0.7)
            axes[2].plot(time, df['yaw_filtered'], label='Filtered', linewidth=2)
            axes[2].set_ylabel('Yaw (deg)')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_title('Yaw: Original vs Filtered')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Warning: matplotlib not available. Skipping plots.")

    print("\nProcessing complete!")
    print(f"Bandpass filter: {args.lowcut}-{args.highcut} Hz")
    print(f"Next step: Run BringFilteredDataAndVideoFrameDataTogeth.py --flight {args.flight}")

    return 0


if __name__ == "__main__":
    exit(main())
