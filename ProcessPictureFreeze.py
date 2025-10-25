import numpy as np
import cv2
import pandas as pd
import math
import os
import glob
from tqdm import tqdm


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


# Configuration
FREEZE_THRESHOLD = 0.5  # Lower = stricter freeze detection
FILE_SIZE_TOLERANCE = 0.02  # 2% tolerance for file size comparison

# Path to Flight1 directory
flight1_dir = 'data/images/Flight1'
output_dir = 'output/rotated_freeze/Flight1'  # Include Flight1 subdirectory for compatibility

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Get all jpg files (automatically sorted)
frame_paths = sorted(glob.glob(os.path.join(flight1_dir, '*.jpg')))

# Read the CSV file
df = pd.read_csv('data/labels/Flight1.csv')

# Tracking state for frozen frame detection
last_unfrozen_path = None
last_unfrozen_frame = None
last_unfrozen_rotated = None
frozen_count = 0
total_count = 0

print(f"Processing {len(frame_paths)} frames with freeze detection...")
print(f"Freeze threshold: {FREEZE_THRESHOLD}")
print(f"Output directory: {output_dir}")
print("-" * 80)

for frame_path in tqdm(frame_paths, desc="Processing frames"):
    # Extract just the filename
    filename = os.path.basename(frame_path)

    # Extract the numerical frame number
    frame = int(os.path.splitext(filename)[0])

    total_count += 1
    is_frozen = False

    # Get IMU data for this frame
    roll, pitch, yaw = quaternion_to_euler(
        df.iloc[frame, 2], df.iloc[frame, 3],
        df.iloc[frame, 4], df.iloc[frame, 5],
        degrees=True
    )

    # Load current image
    image = cv2.imread(frame_path)

    # Check if this frame is frozen (compare to last unfrozen frame)
    if last_unfrozen_path is not None:
        # Quick file size check first (much faster than pixel comparison)
        size1 = os.path.getsize(last_unfrozen_path)
        size2 = os.path.getsize(frame_path)
        size_diff = abs(size1 - size2) / max(size1, size2) if max(size1, size2) > 0 else 1.0

        if size_diff < FILE_SIZE_TOLERANCE:
            # File sizes similar - do pixel comparison
            is_frozen = detect_frozen_frame(last_unfrozen_frame, image, FREEZE_THRESHOLD)

    if is_frozen:
        # Frame is frozen - copy the last unfrozen rotated image
        rotated_original = last_unfrozen_rotated.copy()
        frozen_count += 1
    else:
        # Frame is not frozen - process normally
        rotated_original = rotate_image(
            image,
            -roll,
            output_size=(math.ceil(1350*1.2), math.ceil(1080*1.2))
        )

        # Update last unfrozen references
        last_unfrozen_path = frame_path
        last_unfrozen_frame = image.copy()
        last_unfrozen_rotated = rotated_original.copy()

    # Save the result
    cv2.imwrite(os.path.join(output_dir, filename), rotated_original)

print("-" * 80)
print(f"Processing complete!")
print(f"Total frames: {total_count}")
print(f"Frozen frames: {frozen_count} ({100.0*frozen_count/total_count:.2f}%)")
print(f"Unfrozen frames: {total_count - frozen_count} ({100.0*(total_count-frozen_count)/total_count:.2f}%)")
print(f"Output saved to: {output_dir}")
