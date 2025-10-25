"""
Evaluation pipeline for video stabilization algorithms.

This script compares original raw images with stabilized images and computes
quantitative metrics to assess stabilization quality.

Metrics computed:
- Inter-frame Stability Score (lower is better) - measures smoothness
- Average Inter-frame Difference (lower is better) - pixel-level stability
- PSNR between consecutive frames (higher is better)
- Distortion Score - amount of content loss due to warping
- Cropping Ratio - percentage of valid pixels after stabilization

Usage:
    python src/evaluate.py --model baseline
    python src/evaluate.py --model mymodel --data-dir data/images --output-dir output
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import warnings


def compute_interframe_diff(frame1, frame2):
    """Compute mean absolute difference between consecutive frames."""
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    diff = np.mean(np.abs(frame1.astype(np.float32) - frame2.astype(np.float32)))
    return diff


def compute_psnr(frame1, frame2):
    """Compute Peak Signal-to-Noise Ratio between frames."""
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    mse = np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_optical_flow_smoothness(frame1, frame2):
    """
    Compute optical flow magnitude as a measure of motion.
    Lower values indicate more stable video.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2

    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    # Compute optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Compute magnitude of flow vectors
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    avg_magnitude = np.mean(magnitude)
    std_magnitude = np.std(magnitude)

    return avg_magnitude, std_magnitude


def compute_sharpness(frame):
    """Compute image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    return sharpness


def compute_largest_inscribed_rect(frame, threshold=5):
    """
    Find the largest inscribed rectangle with no black borders.
    This is a more accurate cropping ratio metric than simple pixel counting.

    Returns:
        cropping_ratio: ratio of largest valid rectangle to original frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # Create binary mask of valid pixels
    mask = (gray > threshold).astype(np.uint8)

    # Find the bounding box of valid pixels
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 0.0

    x, y, w, h = cv2.boundingRect(coords)

    # Area of largest valid rectangle
    valid_area = w * h
    total_area = frame.shape[0] * frame.shape[1]

    return valid_area / total_area


def compute_cropping_ratio(frame):
    """
    Estimate cropping ratio by detecting black/invalid borders.
    Returns the ratio of valid pixels.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # Count non-zero pixels as valid (assuming black borders from warping)
    valid_pixels = np.count_nonzero(gray > 5)  # threshold to avoid near-black pixels
    total_pixels = gray.shape[0] * gray.shape[1]

    return valid_pixels / total_pixels


def compute_stability_score_fft(translations, rotations, fps=30.0, low_freq_range=(0.5, 2.0)):
    """
    Compute stability score using FFT analysis of motion parameters.

    Args:
        translations: List of (dx, dy) tuples for each frame
        rotations: List of rotation angles (radians) for each frame
        fps: Frame rate of the video
        low_freq_range: (min_hz, max_hz) defining smooth intentional motion

    Returns:
        stability_score: Ratio of low-frequency energy to total energy (higher is better)
    """
    if len(translations) < 10:  # Need sufficient frames for meaningful FFT
        return None

    # Extract motion components
    dx = np.array([t[0] for t in translations])
    dy = np.array([t[1] for t in translations])
    da = np.array(rotations)

    # Compute FFT for each component
    fft_dx = np.fft.fft(dx)
    fft_dy = np.fft.fft(dy)
    fft_da = np.fft.fft(da)

    # Compute power spectrum
    power_dx = np.abs(fft_dx) ** 2
    power_dy = np.abs(fft_dy) ** 2
    power_da = np.abs(fft_da) ** 2

    # Frequency bins
    n = len(dx)
    freqs = np.fft.fftfreq(n, d=1.0/fps)

    # Only use positive frequencies
    positive_freqs = freqs[:n//2]
    power_dx = power_dx[:n//2]
    power_dy = power_dy[:n//2]
    power_da = power_da[:n//2]

    # Define low-frequency band (smooth intentional motion)
    low_freq_mask = (positive_freqs >= low_freq_range[0]) & (positive_freqs <= low_freq_range[1])

    # Calculate energy in low-frequency band
    low_freq_energy = (
        np.sum(power_dx[low_freq_mask]) +
        np.sum(power_dy[low_freq_mask]) +
        np.sum(power_da[low_freq_mask])
    )

    # Total energy
    total_energy = np.sum(power_dx) + np.sum(power_dy) + np.sum(power_da)

    if total_energy == 0:
        return None

    # Stability score: higher means more energy in low frequencies (smooth motion)
    stability_score = low_freq_energy / total_energy

    return stability_score


def compute_distortion_score(transform_matrix):
    """
    Compute distortion score from transformation matrix.
    Measures non-rigid warping (stretching/shearing).

    Args:
        transform_matrix: 2x3 affine or 3x3 homography matrix

    Returns:
        distortion: Ratio of max to min eigenvalue (1.0 = perfectly rigid)
    """
    if transform_matrix is None:
        return None

    # Extract affine component (top-left 2x2)
    if transform_matrix.shape == (3, 3):
        A = transform_matrix[:2, :2]
    elif transform_matrix.shape == (2, 3):
        A = transform_matrix[:2, :2]
    else:
        return None

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues = np.abs(eigenvalues)

    if np.min(eigenvalues) == 0:
        return None

    # Distortion is ratio of max to min eigenvalue
    distortion = np.max(eigenvalues) / np.min(eigenvalues)

    return distortion


def load_frames_from_dir(directory, extension=None):
    """
    Load all frames from a directory.
    Returns sorted list of (frame_path, frame_array) tuples.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    # Get all image files
    if extension:
        frame_files = sorted(dir_path.glob(f"*.{extension}"))
    else:
        # Try common image extensions
        frame_files = sorted(
            list(dir_path.glob("*.png")) +
            list(dir_path.glob("*.jpg")) +
            list(dir_path.glob("*.jpeg"))
        )

    frames = []
    for frame_path in frame_files:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            frames.append((str(frame_path), frame))

    return frames


def load_transform_data(transform_path):
    """
    Load transformation data if available.

    Expected format (JSON):
    {
        "frames": [0, 1, 2, ...],
        "translations": [[dx0, dy0], [dx1, dy1], ...],
        "rotations": [da0, da1, ...],
        "scales": [s0, s1, ...],
        "transforms": [...]  # Optional: full transformation matrices
    }

    Or NPZ format with same keys.

    Returns:
        dict with transformation data, or None if not available
    """
    transform_path = Path(transform_path)

    # Try JSON first
    json_path = transform_path.with_suffix('.json')
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)

    # Try NPZ
    npz_path = transform_path.with_suffix('.npz')
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        return {k: data[k].tolist() if isinstance(data[k], np.ndarray) else data[k] for k in data.keys()}

    return None


def evaluate_sequence(original_frames, stabilized_frames, flight_name, transform_data=None, fps=30.0):
    """
    Evaluate a single flight sequence.

    Args:
        original_frames: List of (path, frame) tuples for original video
        stabilized_frames: List of (path, frame) tuples for stabilized video
        flight_name: Name of the flight being evaluated
        transform_data: Optional dict with transformation parameters
        fps: Frame rate for FFT-based stability score

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "flight_name": flight_name,
        "num_frames_original": len(original_frames),
        "num_frames_stabilized": len(stabilized_frames),
    }

    if len(original_frames) == 0 or len(stabilized_frames) == 0:
        print(f"Warning: No frames found for {flight_name}")
        return metrics

    # Compute inter-frame metrics for original video
    orig_diffs = []
    orig_flow_mags = []
    orig_flow_stds = []

    for i in range(len(original_frames) - 1):
        _, frame1 = original_frames[i]
        _, frame2 = original_frames[i + 1]

        diff = compute_interframe_diff(frame1, frame2)
        orig_diffs.append(diff)

        flow_mag, flow_std = compute_optical_flow_smoothness(frame1, frame2)
        orig_flow_mags.append(flow_mag)
        orig_flow_stds.append(flow_std)

    # Compute inter-frame metrics for stabilized video
    stab_diffs = []
    stab_flow_mags = []
    stab_flow_stds = []
    stab_psnrs = []
    stab_sharpness = []
    stab_cropping_ratios = []

    for i in range(len(stabilized_frames) - 1):
        _, frame1 = stabilized_frames[i]
        _, frame2 = stabilized_frames[i + 1]

        diff = compute_interframe_diff(frame1, frame2)
        stab_diffs.append(diff)

        psnr = compute_psnr(frame1, frame2)
        if not np.isinf(psnr):
            stab_psnrs.append(psnr)

        flow_mag, flow_std = compute_optical_flow_smoothness(frame1, frame2)
        stab_flow_mags.append(flow_mag)
        stab_flow_stds.append(flow_std)

        sharpness = compute_sharpness(frame1)
        stab_sharpness.append(sharpness)

        # Use improved cropping ratio (largest inscribed rectangle)
        crop_ratio = compute_largest_inscribed_rect(frame1)
        stab_cropping_ratios.append(crop_ratio)

    # Aggregate metrics
    metrics.update({
        # Original video metrics
        "original_avg_interframe_diff": np.mean(orig_diffs) if orig_diffs else 0,
        "original_std_interframe_diff": np.std(orig_diffs) if orig_diffs else 0,
        "original_avg_flow_magnitude": np.mean(orig_flow_mags) if orig_flow_mags else 0,
        "original_std_flow_magnitude": np.std(orig_flow_mags) if orig_flow_mags else 0,

        # Stabilized video metrics
        "stabilized_avg_interframe_diff": np.mean(stab_diffs) if stab_diffs else 0,
        "stabilized_std_interframe_diff": np.std(stab_diffs) if stab_diffs else 0,
        "stabilized_avg_flow_magnitude": np.mean(stab_flow_mags) if stab_flow_mags else 0,
        "stabilized_std_flow_magnitude": np.std(stab_flow_mags) if stab_flow_mags else 0,
        "stabilized_avg_psnr": np.mean(stab_psnrs) if stab_psnrs else 0,
        "stabilized_avg_sharpness": np.mean(stab_sharpness) if stab_sharpness else 0,
        "stabilized_avg_cropping_ratio": np.mean(stab_cropping_ratios) if stab_cropping_ratios else 0,

        # Improvement metrics (lower is better for stability)
        "improvement_interframe_diff": (
            (np.mean(orig_diffs) - np.mean(stab_diffs)) / np.mean(orig_diffs) * 100
            if orig_diffs and stab_diffs and np.mean(orig_diffs) > 0 else 0
        ),
        "improvement_flow_magnitude": (
            (np.mean(orig_flow_mags) - np.mean(stab_flow_mags)) / np.mean(orig_flow_mags) * 100
            if orig_flow_mags and stab_flow_mags and np.mean(orig_flow_mags) > 0 else 0
        ),
    })

    # Advanced metrics (require transformation data)
    if transform_data is not None:
        # Compute FFT-based stability score
        if 'translations' in transform_data and 'rotations' in transform_data:
            translations = transform_data['translations']
            rotations = transform_data['rotations']
            stability_score = compute_stability_score_fft(translations, rotations, fps)
            metrics['stability_score_fft'] = stability_score if stability_score is not None else 0
        else:
            metrics['stability_score_fft'] = None

        # Compute distortion score
        if 'transforms' in transform_data:
            distortions = []
            for transform in transform_data['transforms']:
                if transform is not None:
                    transform_matrix = np.array(transform)
                    distortion = compute_distortion_score(transform_matrix)
                    if distortion is not None:
                        distortions.append(distortion)
            metrics['avg_distortion_score'] = np.mean(distortions) if distortions else None
        else:
            metrics['avg_distortion_score'] = None
    else:
        metrics['stability_score_fft'] = None
        metrics['avg_distortion_score'] = None

    return metrics


def evaluate_model(model_name, data_dir, output_dir):
    """
    Evaluate a stabilization model.

    Args:
        model_name: Name of the model (folder name under output/)
        data_dir: Path to original data directory (e.g., data/images)
        output_dir: Path to output directory (e.g., output)

    Returns:
        Dictionary containing evaluation results for all flights
    """
    data_path = Path(data_dir)
    model_output_path = Path(output_dir) / model_name

    if not model_output_path.exists():
        raise ValueError(f"Model output directory not found: {model_output_path}")

    print(f"\nEvaluating model: {model_name}")
    print(f"Data directory: {data_path}")
    print(f"Model output: {model_output_path}")
    print("-" * 80)

    # Find all flight directories
    flight_dirs = [d for d in model_output_path.iterdir() if d.is_dir()]

    if not flight_dirs:
        print(f"No flight directories found in {model_output_path}")
        return {}

    all_metrics = []

    for flight_dir in tqdm(flight_dirs, desc="Evaluating flights"):
        flight_name = flight_dir.name

        # Load original frames
        orig_flight_dir = data_path / flight_name
        if not orig_flight_dir.exists():
            print(f"Warning: Original data not found for {flight_name}, skipping...")
            continue

        print(f"\n  Processing {flight_name}...")
        original_frames = load_frames_from_dir(orig_flight_dir)
        stabilized_frames = load_frames_from_dir(flight_dir)

        # Try to load transformation data
        transform_path = flight_dir / flight_name  # e.g., output/baseline/Flight1/Flight1.json
        transform_data = load_transform_data(transform_path)
        if transform_data is None:
            # Try alternative location: output/baseline/Flight1_transforms.json
            transform_path = flight_dir.parent / f"{flight_name}_transforms"
            transform_data = load_transform_data(transform_path)

        if transform_data is not None:
            print(f"    Found transformation data")

        # Evaluate this flight
        metrics = evaluate_sequence(original_frames, stabilized_frames, flight_name, transform_data)
        all_metrics.append(metrics)

        # Print summary for this flight
        print(f"    Original frames: {metrics['num_frames_original']}")
        print(f"    Stabilized frames: {metrics['num_frames_stabilized']}")
        print(f"    Interframe diff improvement: {metrics['improvement_interframe_diff']:.2f}%")
        print(f"    Flow magnitude improvement: {metrics['improvement_flow_magnitude']:.2f}%")
        print(f"    Avg PSNR: {metrics['stabilized_avg_psnr']:.2f} dB")
        print(f"    Avg cropping ratio: {metrics['stabilized_avg_cropping_ratio']:.2%}")
        if metrics.get('stability_score_fft') is not None:
            print(f"    Stability score (FFT): {metrics['stability_score_fft']:.4f}")
        if metrics.get('avg_distortion_score') is not None:
            print(f"    Avg distortion score: {metrics['avg_distortion_score']:.4f}")

    # Compute aggregate statistics across all flights
    if all_metrics:
        agg_metrics = {
            "avg_improvement_interframe_diff": np.mean([m["improvement_interframe_diff"] for m in all_metrics]),
            "avg_improvement_flow_magnitude": np.mean([m["improvement_flow_magnitude"] for m in all_metrics]),
            "avg_psnr": np.mean([m["stabilized_avg_psnr"] for m in all_metrics if m["stabilized_avg_psnr"] > 0]),
            "avg_sharpness": np.mean([m["stabilized_avg_sharpness"] for m in all_metrics if m["stabilized_avg_sharpness"] > 0]),
            "avg_cropping_ratio": np.mean([m["stabilized_avg_cropping_ratio"] for m in all_metrics]),
        }

        # Add advanced metrics if available
        stability_scores = [m["stability_score_fft"] for m in all_metrics if m.get("stability_score_fft") is not None]
        if stability_scores:
            agg_metrics["avg_stability_score_fft"] = np.mean(stability_scores)

        distortion_scores = [m["avg_distortion_score"] for m in all_metrics if m.get("avg_distortion_score") is not None]
        if distortion_scores:
            agg_metrics["avg_distortion_score"] = np.mean(distortion_scores)

        aggregate = {
            "model_name": model_name,
            "num_flights": len(all_metrics),
            "per_flight_metrics": all_metrics,
            "aggregate_metrics": agg_metrics
        }
    else:
        aggregate = {
            "model_name": model_name,
            "num_flights": 0,
            "per_flight_metrics": [],
            "aggregate_metrics": {}
        }

    return aggregate


def print_summary(results):
    """Print a summary of evaluation results."""
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY: {results['model_name']}")
    print("=" * 80)

    if results['num_flights'] == 0:
        print("No flights evaluated.")
        return

    agg = results['aggregate_metrics']

    print(f"\nNumber of flights evaluated: {results['num_flights']}")
    print("\nAggregate Metrics (across all flights):")
    print(f"  Interframe Difference Improvement:  {agg['avg_improvement_interframe_diff']:>8.2f}%")
    print(f"  Flow Magnitude Improvement:         {agg['avg_improvement_flow_magnitude']:>8.2f}%")
    print(f"  Average PSNR (consecutive frames):  {agg['avg_psnr']:>8.2f} dB")
    print(f"  Average Sharpness:                  {agg['avg_sharpness']:>8.2f}")
    print(f"  Average Cropping Ratio:             {agg['avg_cropping_ratio']:>8.2%}")

    # Print advanced metrics if available
    if 'avg_stability_score_fft' in agg:
        print(f"  Stability Score (FFT):              {agg['avg_stability_score_fft']:>8.4f}")
    if 'avg_distortion_score' in agg:
        print(f"  Average Distortion Score:           {agg['avg_distortion_score']:>8.4f}")

    print("\nPer-Flight Results:")
    print(f"{'Flight':<15} {'Frames':<8} {'Diff Improv':<12} {'Flow Improv':<12} {'PSNR':<10} {'Crop Ratio':<12}")
    print("-" * 80)

    for m in results['per_flight_metrics']:
        print(
            f"{m['flight_name']:<15} "
            f"{m['num_frames_stabilized']:<8} "
            f"{m['improvement_interframe_diff']:>10.2f}%  "
            f"{m['improvement_flow_magnitude']:>10.2f}%  "
            f"{m['stabilized_avg_psnr']:>8.2f}  "
            f"{m['stabilized_avg_cropping_ratio']:>10.2%}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate video stabilization quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/evaluate.py --model baseline
  python src/evaluate.py --model mymodel --data-dir data/images --output-dir output
  python src/evaluate.py --model baseline --save-json results.json
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (folder name under output directory)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/images",
        help="Path to original data directory (default: data/images)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Path to output directory containing model results (default: output)"
    )

    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Save detailed results to JSON file (optional)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default='data/data_split.json',
        help="Path to data split JSON file (default: data/data_split.json, for reporting split info). Set to empty string to disable."
    )

    args = parser.parse_args()

    # Get repository root
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / args.data_dir
    output_dir = repo_root / args.output_dir

    # Load split info if provided
    split_info = None
    if args.split and args.split.strip():  # Check if split is not empty string
        split_path = repo_root / args.split
        if split_path.exists():
            with open(split_path, 'r') as f:
                split_info = json.load(f)
            print(f"Split file loaded: {split_path}")
            print(f"Split strategy: {split_info['metadata']['strategy']}")
            print(f"Split ratios: Train={split_info['metadata']['ratios']['train']:.1%}, "
                  f"Val={split_info['metadata']['ratios']['val']:.1%}, "
                  f"Test={split_info['metadata']['ratios']['test']:.1%}")
            print("-" * 80)
        else:
            print(f"Note: Split file not found at {split_path}, proceeding without split info...")
            print("-" * 80)

    # Evaluate model
    results = evaluate_model(args.model, str(data_dir), str(output_dir))

    # Add split info to results if available
    if split_info:
        results['split_info'] = split_info['metadata']

    # Print summary
    print_summary(results)

    # Save to JSON if requested
    if args.save_json:
        json_path = repo_root / args.save_json
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {json_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
