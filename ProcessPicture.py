import numpy as np


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


import cv2
import numpy as np


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


import pandas as pd
import math
import os
import glob

# Path to Flight1 directory
flight1_dir = 'data/images/Flight1'

# Get all jpg files (automatically sorted)
frame_paths = sorted(glob.glob(os.path.join(flight1_dir, '*.jpg')))

# Read the CSV file
df = pd.read_csv('data/labels/Flight1.csv')

for frame_path in frame_paths:
    # Extract just the filename
    filename = os.path.basename(frame_path)

    # Extract the numerical frame number
    frame = int(os.path.splitext(filename)[0])

    print(f"Processing frame {frame}: {filename}")

    # Example: Read cell at row 0, column 0
    cell_value = df.iloc[frame, 2]
    # print(f"Cell value: {cell_value}")

    roll, pitch, yaw = quaternion_to_euler(df.iloc[frame, 2], df.iloc[frame, 3], df.iloc[frame, 4], df.iloc[frame, 5], degrees=True)
    # print(f"Euler Roll: {roll}")

    # Load an image
    image = cv2.imread(f'data/images/Flight1/{filename}')
    # print(f"Original size: {image.shape[1]}x{image.shape[0]}")

    rotated_original = rotate_image(image, -roll,output_size=(math.ceil(1350*1.2),math.ceil(1080*1.2)))
    cv2.imwrite(f'output/rotated/{filename}', rotated_original)


