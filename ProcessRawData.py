import numpy as np
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


# Example usage with pandas DataFrame (for your Flight CSV data)
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


import pandas as pd
import numpy as np
from scipy import signal
import argparse
from pathlib import Path

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




df = pd.read_csv('data/raw/logs/Flight1.csv')
df = add_euler_angles_to_dataframe(df, degrees=True)

output_path = ('output/rawWithEulerAndYawFix/Flight1.csv')
df.to_csv(output_path, index=False)

lowcut=0.3
highcut=10
fs=327.7
order=2

#unwrap yaw to avoid steps
df['yaw']=unwrap_yaw_relative(df['yaw'], degrees=True)

df['pitch'].values
df['roll_filtered']=apply_bandpass_filter(df['roll'].values, lowcut, highcut, fs, order)
df['pitch_filtered']=apply_bandpass_filter(df['pitch'].values, lowcut, highcut, fs, order)
df['yaw_filtered']=apply_bandpass_filter(df['yaw'].values, lowcut, highcut, fs, order)
df.to_csv('output/rawWithEulerAndFiltAndYawFix/Flight1.csv', index=False)

import matplotlib.pyplot as plt
# Create figure with 3 subplots (3 rows, 1 column)
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

time=df['system_time_s'].values
# Plot 1 (top)
axes[0].plot(time, df['roll'], label='Original')
axes[0].plot(time, df['roll_filtered'], label='Filtered')
axes[0].set_ylabel('Roll')
axes[0].set_title('Plot 1')
axes[0].grid(True)

# Plot 2 (middle)
axes[1].plot(time, df['pitch'], label='Original')
axes[1].plot(time, df['pitch_filtered'], label='Filtered')
axes[1].set_ylabel('Pitch')
axes[1].set_title('Plot 2')
axes[1].grid(True)

# Plot 3 (bottom)
axes[2].plot(time, df['yaw'], label='Original')
axes[2].plot(time, df['yaw_filtered'], label='Filtered')
axes[2].set_ylabel('Yaw')
axes[2].set_title('Plot 3')
axes[2].grid(True)

# Adjust spacing between subplots
plt.tight_layout()

# Show or save
plt.show()
# plt.savefig('my_plot.png')



