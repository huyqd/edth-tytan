#!/usr/bin/env python3
"""
Script to create video/GIF from image sequences in output/fusion directories.
"""

import cv2
import os
import glob
import json
import re
from pathlib import Path
import argparse

def create_video_from_images(image_dir, output_path, fps=30):
    """
    Create a video from a sequence of images using OpenCV.

    Args:
        image_dir (str): Directory containing the images
        output_path (str): Output video file path
        fps (int): Frames per second for the video
    """
    # Get all jpg files and sort them
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    if not image_files:
        print(f"No JPG files found in {image_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Create video writer with H.264 codec for better browser compatibility
    # Try H.264 codecs in order of preference
    codecs_to_try = [
        ('avc1', 'H.264 (avc1)'),  # H.264 - best browser support
        ('H264', 'H.264 (H264)'),  # Alternative H.264 identifier
        ('X264', 'H.264 (X264)'),  # Another H.264 variant
        ('mp4v', 'MPEG-4 (mp4v)')  # Fallback to original
    ]

    video_writer = None
    codec_used = None

    for codec, codec_name in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        test_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if test_writer.isOpened():
            video_writer = test_writer
            codec_used = codec_name
            break
        test_writer.release()

    if video_writer is None:
        print(f"ERROR: Could not initialize video writer with any codec")
        return

    print(f"Creating video: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Codec: {codec_used}")

    for i, image_file in enumerate(image_files):
        if i % 100 == 0:
            print(f"Processing frame {i+1}/{len(image_files)}")

        image = cv2.imread(image_file)
        video_writer.write(image)

    video_writer.release()
    print(f"Video saved to: {output_path}")

def create_gif_from_images(image_dir, output_path, fps=10, max_frames=None):
    """
    Create a GIF from a sequence of images using imageio.

    Args:
        image_dir (str): Directory containing the images
        output_path (str): Output GIF file path
        fps (int): Frames per second for the GIF
        max_frames (int): Maximum number of frames to include (for testing)
    """
    import imageio

    # Get all jpg files and sort them
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    if not image_files:
        print(f"No JPG files found in {image_dir}")
        return

    if max_frames:
        image_files = image_files[:max_frames]

    print(f"Creating GIF with {len(image_files)} frames")

    images = []
    for i, image_file in enumerate(image_files):
        if i % 50 == 0:
            print(f"Loading frame {i+1}/{len(image_files)}")
        images.append(imageio.imread(image_file))

    print(f"Saving GIF to: {output_path}")
    imageio.mimsave(output_path, images, fps=fps)
    print(f"GIF saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Create video/GIF from image sequences. Supports filtering frames by data split (data/data_split.json)."
    )
    parser.add_argument("--input-dir", help="Path to images directory or parent directory containing flight subfolders. (Deprecated: prefer --model)")
    parser.add_argument("--model", help="Model name to use for input images. Use 'original' to read from data/images/<flight>. Otherwise images are read from output/<model>/<flight>.")
    parser.add_argument("flight", nargs="?", help="Flight directory name (e.g., Flight1). If not provided and --input-dir points to a flight folder, the flight name is inferred from the input path")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4", help="Output format (default: mp4)")
    parser.add_argument("--max-frames", type=int, help="Maximum frames for GIF (for testing)")
    parser.add_argument("--split", choices=["train", "val", "test"], help="If provided, filter frames using data/data_split.json for the given split")
    parser.add_argument("--data-split", default="data/data_split.json", help="Path to data split JSON (default: data/data_split.json)")
    parser.add_argument("--ext", default="jpg", help="Image file extension to look for (default: jpg)")

    args = parser.parse_args()

    # Determine input image directory and flight
    image_dir = None
    flight_name = args.flight

    if args.model:
        # Prefer --model when provided
        model = args.model
        if not flight_name:
            parser.error("when using --model you must provide the positional flight name (e.g. Flight1)")

        if model.lower() == "original":
            # Read from data/images/<flight>
            image_dir = Path("data") / "images" / flight_name
            src_label = "original"
        else:
            # Read from output/<model>/<flight>
            image_dir = Path(args.output_dir) / model / flight_name
            src_label = model
    elif args.input_dir:
        # Backwards-compatible behavior: allow passing an explicit directory
        image_dir = Path(args.input_dir)
        # if input_dir points to parent with flight subfolders and flight provided, join them
        if flight_name and (image_dir / flight_name).exists():
            image_dir = image_dir / flight_name
        # infer flight name if not provided
        flight_name = flight_name or Path(image_dir).name

        # decide a source label for clearer output naming (e.g., original, fusion)
        image_dir_str = str(image_dir)
        if "data/images" in image_dir_str or os.path.sep + "images" + os.path.sep in image_dir_str:
            src_label = "original"
        else:
            parts = Path(image_dir).parts
            if "output" in parts:
                out_idx = parts.index("output")
                chosen = None
                for p in parts[out_idx + 1 :]:
                    if p and p != "videos":
                        chosen = p
                        break
                if chosen:
                    src_label = chosen
                else:
                    src_label = Path(image_dir).name
            else:
                src_label = Path(image_dir).name
    else:
        # Default legacy behavior: require flight and read from output/fusion/<flight>
        if not flight_name:
            parser.error("either provide --model (preferred) or a positional flight name")
        image_dir = Path(args.output_dir) / "fusion" / flight_name
        src_label = "fusion"

    # infer flight name if not provided already
    flight_name = flight_name or Path(image_dir).name

    output_dir = Path(args.output_dir) / "videos" / src_label
    output_dir.mkdir(parents=True, exist_ok=True)

    # handle split filtering
    frame_filter = None
    if args.split:
        split_path = Path(args.data_split)
        if not split_path.exists():
            print(f"Warning: data split file not found at {split_path}; skipping split filtering")
        else:
            try:
                with open(split_path, "r") as f:
                    ds = json.load(f)
                splits = ds.get("splits", {})
                split_map = splits.get(args.split, {})
                # Expect split_map to be mapping from flight name to list of frame indices
                if flight_name not in split_map:
                    print(f"Warning: flight {flight_name} not found in split '{args.split}'; no filtering applied")
                else:
                    indices = set(split_map[flight_name])
                    frame_filter = indices
                    print(f"Filtering {len(indices)} frames for flight {flight_name} from split '{args.split}'")
            except Exception as e:
                print(f"Failed to read/parse split file: {e}; skipping split filtering")

    # prepare output path
    safe_flight = flight_name.replace("/", "_")
    # choose filename base (do not append special suffixes like '_fusion')
    filename_base = f"{safe_flight}"

    if args.format == "mp4":
        output_path = output_dir / f"{filename_base}.mp4"
        _create_or_filter_and_write(
            image_dir, str(output_path), args.fps, args.ext, frame_filter, create_video_from_images
        )
    else:  # gif
        output_path = output_dir / f"{filename_base}.gif"
        _create_or_filter_and_write(
            image_dir, str(output_path), args.fps, args.ext, frame_filter, lambda d, o, f: create_gif_from_images(d, o, args.fps, args.max_frames)
        )


def _parse_frame_index_from_filename(fname: str) -> int | None:
    """Attempt to parse a frame index from a filename by extracting the last sequence of digits.

    Returns None if no digits found.
    """
    base = os.path.basename(fname)
    m = re.findall(r"(\d+)", base)
    if not m:
        return None
    # choose the last group of digits (usually the frame counter)
    return int(m[-1])


def _create_or_filter_and_write(image_dir, output_path, fps, ext, frame_filter, writer_fn):
    """Collect image files from image_dir, apply optional frame_filter (set of indices),
    write using writer_fn(directory, output_path, fps).

    If frame_filter is provided, this function will create a temporary directory with
    symlinks or copies of the filtered frames to preserve ordering and then call writer_fn
    on that directory.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"Image directory does not exist: {image_dir}")
        return

    pattern = f"*.{ext.lstrip('.')}"
    all_images = sorted([str(p) for p in image_dir.glob(pattern)])
    if not all_images:
        print(f"No images found with pattern {pattern} in {image_dir}")
        return

    if frame_filter is None:
        # no filtering, call writer directly
        writer_fn(str(image_dir), output_path, fps)
        return

    # Need to filter images by parsed frame indices
    filtered = []
    for img in all_images:
        idx = _parse_frame_index_from_filename(img)
        if idx is None:
            # if filename doesn't contain digits, skip it when filtering
            continue
        if idx in frame_filter:
            filtered.append(img)

    if not filtered:
        print("No frames matched the provided split filter.")
        return

    # create temporary directory of filtered frames preserving order
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, src in enumerate(filtered):
            # create zero-padded names to preserve ordering
            dst_name = f"frame_{i:08d}.{ext.lstrip('.') }"
            dst = os.path.join(tmpdir, dst_name)
            try:
                os.symlink(os.path.abspath(src), dst)
            except Exception:
                # fallback to copy
                from shutil import copy2

                copy2(src, dst)
        # call writer on temporary dir
        writer_fn(tmpdir, output_path, fps)

if __name__ == "__main__":
    main()