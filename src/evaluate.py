"""
Evaluation pipeline for video stabilization algorithms.

This script compares original raw images with stabilized images and computes
quantitative metrics to assess stabilization quality.

Metrics computed for ALL frames (original and stabilized):
- Inter-frame Difference (lower is better) - pixel-level stability
- Optical Flow Magnitude (lower is better) - measures motion smoothness
- PSNR between consecutive frames (higher is better) - frame similarity
- Sharpness (higher is better) - image clarity via Laplacian variance
- Cropping Ratio - percentage of valid pixels after stabilization
- Distortion Score (optional) - amount of content loss due to warping

Performance:
- Uses parallel processing to compute metrics faster (default: auto-detect CPU count)
- For sequences with >50 frame pairs, metrics are computed in parallel batches
- Control parallelization with --workers flag (e.g., --workers 4)

When evaluating a model:
- Computes the same metrics for both original and stabilized frames
- Calculates improvement percentages over original
- Uses pre-computed original metrics if available (from --original-metrics file)
- This avoids redundant computation when evaluating multiple models

Usage:
    # First, compute original video metrics (recommended, done once)
    python src/evaluate.py --model original --split-set test
    
    # Then evaluate models (will use cached original metrics)
    python src/evaluate.py --model baseline
    python src/evaluate.py --model mymodel --data-dir data/images --output-dir output
    
    # Control parallel workers
    python src/evaluate.py --model baseline --workers 8
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial


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


def _compute_frame_pair_metrics(frame_pair):
    """
    Compute all metrics for a single frame pair.
    This function is designed to be called in parallel.
    
    Args:
        frame_pair: tuple of (frame1, frame2)
    
    Returns:
        dict with all computed metrics for this frame pair
    """
    frame1, frame2 = frame_pair
    
    # Inter-frame difference
    diff = compute_interframe_diff(frame1, frame2)
    
    # PSNR
    psnr = compute_psnr(frame1, frame2)
    if np.isinf(psnr):
        psnr = None
    
    # Optical flow
    flow_mag, flow_std = compute_optical_flow_smoothness(frame1, frame2)
    
    # Sharpness
    sharpness = compute_sharpness(frame1)
    
    # Cropping ratio
    crop_ratio = compute_largest_inscribed_rect(frame1)
    
    return {
        'diff': diff,
        'psnr': psnr,
        'flow_mag': flow_mag,
        'flow_std': flow_std,
        'sharpness': sharpness,
        'crop_ratio': crop_ratio
    }


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


def compute_stability_score_simple(interframe_diffs, flow_magnitudes):
    """
    Compute a simple stability score based on the consistency of motion.
    Lower variance in inter-frame differences and flow indicates better stability.
    
    Args:
        interframe_diffs: List of inter-frame difference values
        flow_magnitudes: List of optical flow magnitude values
    
    Returns:
        stability_score: Normalized stability score (0-100, higher is better)
    """
    if len(interframe_diffs) < 2 or len(flow_magnitudes) < 2:
        return None
    
    # Coefficient of variation (CV) = std / mean
    # Lower CV means more stable (less relative variation)
    diff_mean = np.mean(interframe_diffs)
    diff_std = np.std(interframe_diffs)
    diff_cv = diff_std / diff_mean if diff_mean > 0 else 0
    
    flow_mean = np.mean(flow_magnitudes)
    flow_std = np.std(flow_magnitudes)
    flow_cv = flow_std / flow_mean if flow_mean > 0 else 0
    
    # Combined coefficient of variation (average)
    combined_cv = (diff_cv + flow_cv) / 2.0
    
    # Convert to stability score (0-100, higher is better)
    # CV typically ranges from 0 (perfectly stable) to 2+ (very unstable)
    # We map this inversely to a 0-100 scale
    stability_score = max(0, min(100, 100 * (1 - combined_cv / 2.0)))
    
    return stability_score


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


def load_frames_from_dir(directory, extension=None, frame_names=None, frame_indices=None, desc=None):
    """
    Load frames from a directory.

    Args:
        directory: Path to directory containing frames
        extension: Optional file extension filter
        frame_names: Optional list of frame filenames to load (optimization)
        frame_indices: Optional list of frame indices to load (will try common extensions)
        desc: Optional description for progress bar

    Returns:
        Sorted list of (frame_path, frame_array) tuples.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    # If specific frame indices provided, convert to frame names and load
    if frame_indices is not None:
        frames = []
        for frame_idx in tqdm(sorted(frame_indices), desc=desc or "Loading frames", leave=False, disable=len(frame_indices) < 50):
            # Try different extensions
            for ext in ['.png', '.jpg', '.jpeg']:
                frame_path = dir_path / f"{frame_idx:08d}{ext}"
                if frame_path.exists():
                    frame = cv2.imread(str(frame_path))
                    if frame is not None:
                        frames.append((str(frame_path), frame))
                    break
        return frames

    # If specific frame names provided, load only those
    if frame_names is not None:
        frames = []
        frame_list = sorted(frame_names)
        for frame_name in tqdm(frame_list, desc=desc or "Loading frames", leave=False, disable=len(frame_list) < 50):
            frame_path = dir_path / frame_name
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    frames.append((str(frame_path), frame))
        return frames

    # Otherwise load all frames (original behavior)
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
    for frame_path in tqdm(frame_files, desc=desc or "Loading frames", leave=False, disable=len(frame_files) < 50):
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


def load_original_metrics(original_metrics_path):
    """
    Load pre-computed original video metrics.

    Args:
        original_metrics_path: Path to original_metrics.json

    Returns:
        dict: {flight_name: metrics_dict} or None if not available
    """
    if not Path(original_metrics_path).exists():
        return None

    try:
        with open(original_metrics_path, 'r') as f:
            data = json.load(f)

        # Convert list of per-flight metrics to dict keyed by flight name
        metrics_dict = {}
        for flight_metrics in data.get('per_flight_metrics', []):
            flight_name = flight_metrics['flight_name']
            metrics_dict[flight_name] = flight_metrics

        return metrics_dict
    except Exception as e:
        print(f"Warning: Error loading original metrics: {e}")
        return None


def save_original_metrics(metrics_list, output_path, split_info=None):
    """
    Save original video metrics to JSON.

    Args:
        metrics_list: List of per-flight metrics dicts
        output_path: Path to save JSON file
        split_info: Optional split metadata
    """
    results = {
        "per_flight_metrics": metrics_list,
    }

    if split_info:
        results['split_info'] = split_info

    # Compute aggregate statistics
    if metrics_list:
        results["aggregate_metrics"] = {
            "num_flights": len(metrics_list),
            "total_frames": sum(m["num_frames_original"] for m in metrics_list),
            "avg_interframe_diff": float(np.mean([m["original_avg_interframe_diff"] for m in metrics_list if m.get("original_avg_interframe_diff", 0) > 0])),
            "avg_flow_magnitude": float(np.mean([m["original_avg_flow_magnitude"] for m in metrics_list if m.get("original_avg_flow_magnitude", 0) > 0])),
        }

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def compute_frame_sequence_metrics(frames, prefix="", desc="Computing metrics", num_workers=None):
    """
    Compute metrics for a sequence of frames using parallel processing.

    Args:
        frames: List of (path, frame) tuples
        prefix: Prefix for metric keys (e.g., "original_" or "stabilized_")
        desc: Description for progress bar
        num_workers: Number of parallel workers (None = use cpu_count)

    Returns:
        Dictionary of metrics with the specified prefix
    """
    metrics = {}

    if len(frames) < 2:
        return metrics

    # Prepare frame pairs for parallel processing
    frame_pairs = [(frames[i][1], frames[i+1][1]) for i in range(len(frames) - 1)]
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), len(frame_pairs))
    
    # Use parallel processing for large datasets, sequential for small ones
    if len(frame_pairs) > 50 and num_workers > 1:
        # Parallel processing
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(_compute_frame_pair_metrics, frame_pairs, chunksize=10),
                total=len(frame_pairs),
                desc=desc,
                leave=False
            ))
    else:
        # Sequential processing for small datasets (overhead not worth it)
        results = []
        for pair in tqdm(frame_pairs, desc=desc, leave=False):
            results.append(_compute_frame_pair_metrics(pair))
    
    # Aggregate results
    diffs = [r['diff'] for r in results]
    psnrs = [r['psnr'] for r in results if r['psnr'] is not None]
    flow_mags = [r['flow_mag'] for r in results]
    flow_stds = [r['flow_std'] for r in results]
    sharpness_values = [r['sharpness'] for r in results]
    cropping_ratios = [r['crop_ratio'] for r in results]

    # Compute simple stability score
    stability_score = compute_stability_score_simple(diffs, flow_mags)

    # Aggregate metrics
    metrics.update({
        f"{prefix}avg_interframe_diff": float(np.mean(diffs)) if diffs else 0.0,
        f"{prefix}std_interframe_diff": float(np.std(diffs)) if diffs else 0.0,
        f"{prefix}avg_flow_magnitude": float(np.mean(flow_mags)) if flow_mags else 0.0,
        f"{prefix}std_flow_magnitude": float(np.std(flow_mags)) if flow_mags else 0.0,
        f"{prefix}avg_psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        f"{prefix}avg_sharpness": float(np.mean(sharpness_values)) if sharpness_values else 0.0,
        f"{prefix}avg_cropping_ratio": float(np.mean(cropping_ratios)) if cropping_ratios else 0.0,
        f"{prefix}stability_score": float(stability_score) if stability_score is not None else None,
    })

    return metrics


def evaluate_original_only(original_frames, flight_name, num_workers=None):
    """
    Compute metrics for original frames only (for --model original).

    Args:
        original_frames: List of (path, frame) tuples for original video
        flight_name: Name of the flight being evaluated
        num_workers: Number of parallel workers for metric computation

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "flight_name": flight_name,
        "num_frames_original": len(original_frames),
    }

    if len(original_frames) < 2:
        print(f"Warning: Not enough frames found for {flight_name}")
        return metrics

    # Compute metrics using unified function
    print(f"    Computing metrics...")
    frame_metrics = compute_frame_sequence_metrics(
        original_frames, 
        prefix="original_", 
        desc="    Metrics",
        num_workers=num_workers
    )
    metrics.update(frame_metrics)

    return metrics


def evaluate_sequence(original_frames, stabilized_frames, flight_name, transform_data=None, fps=30.0, precomputed_original_metrics=None, num_workers=None):
    """
    Evaluate a single flight sequence.

    Args:
        original_frames: List of (path, frame) tuples for original video
        stabilized_frames: List of (path, frame) tuples for stabilized video
        flight_name: Name of the flight being evaluated
        transform_data: Optional dict with transformation parameters
        fps: Frame rate for FFT-based stability score
        precomputed_original_metrics: Optional pre-computed metrics for original frames
        num_workers: Number of parallel workers for metric computation

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

    # Use pre-computed original metrics if available
    if precomputed_original_metrics:
        print("    Using cached original metrics")
        # Extract the relevant metrics from precomputed data
        for key in ['original_avg_interframe_diff', 'original_std_interframe_diff',
                    'original_avg_flow_magnitude', 'original_std_flow_magnitude',
                    'original_avg_psnr', 'original_avg_sharpness', 'original_avg_cropping_ratio']:
            if key in precomputed_original_metrics:
                metrics[key] = precomputed_original_metrics[key]
    else:
        # Compute metrics for original frames using unified function
        print("    Computing metrics for original frames...")
        original_metrics = compute_frame_sequence_metrics(
            original_frames,
            prefix="original_",
            desc="    Original metrics",
            num_workers=num_workers
        )
        metrics.update(original_metrics)

    # Compute metrics for stabilized frames using unified function
    print("    Computing metrics for stabilized frames...")
    stabilized_metrics = compute_frame_sequence_metrics(
        stabilized_frames,
        prefix="stabilized_",
        desc="    Stabilized metrics",
        num_workers=num_workers
    )
    metrics.update(stabilized_metrics)

    # Compute improvement metrics
    orig_diff = metrics.get('original_avg_interframe_diff', 0)
    stab_diff = metrics.get('stabilized_avg_interframe_diff', 0)
    orig_flow = metrics.get('original_avg_flow_magnitude', 0)
    stab_flow = metrics.get('stabilized_avg_flow_magnitude', 0)
    orig_stability = metrics.get('original_stability_score')
    stab_stability = metrics.get('stabilized_stability_score')

    metrics.update({
        # Improvement metrics (lower is better for stability)
        "improvement_interframe_diff": float(
            (orig_diff - stab_diff) / orig_diff * 100 if orig_diff > 0 else 0.0
        ),
        "improvement_flow_magnitude": float(
            (orig_flow - stab_flow) / orig_flow * 100 if orig_flow > 0 else 0.0
        ),
    })
    
    # Stability score improvement (higher is better, so reverse calculation)
    if orig_stability is not None and stab_stability is not None:
        metrics["improvement_stability_score"] = float(stab_stability - orig_stability)
    else:
        metrics["improvement_stability_score"] = None

    # Advanced metrics (require transformation data)
    if transform_data is not None:
        # Compute FFT-based stability score
        if 'translations' in transform_data and 'rotations' in transform_data:
            translations = transform_data['translations']
            rotations = transform_data['rotations']
            stability_score = compute_stability_score_fft(translations, rotations, fps)
            metrics['stability_score_fft'] = float(stability_score) if stability_score is not None else None
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
            metrics['avg_distortion_score'] = float(np.mean(distortions)) if distortions else None
        else:
            metrics['avg_distortion_score'] = None
    else:
        metrics['stability_score_fft'] = None
        metrics['avg_distortion_score'] = None

    return metrics


def evaluate_original(data_dir, split_info=None, split_set=None, num_workers=None):
    """
    Evaluate original (unstabilized) video frames.

    Args:
        data_dir: Path to original data directory (e.g., data/images)
        split_info: Optional split configuration
        split_set: Optional split set to process ('train', 'val', 'test')
        num_workers: Number of parallel workers for metric computation

    Returns:
        Dictionary containing evaluation results for all flights
    """
    data_path = Path(data_dir)

    print(f"\nComputing original video metrics")
    print(f"Data directory: {data_path}")
    print("-" * 80)

    # Determine which flights to process and which frame indices
    flights_to_process = {}  # {flight_name: (flight_dir, frame_indices or None)}

    if split_info and split_set:
        print(f"Processing split set: {split_set}")
        split_data = split_info['splits'][split_set]
        for flight_name, frame_indices in split_data.items():
            flight_dir = data_path / flight_name
            if flight_dir.exists() and flight_dir.is_dir() and frame_indices:
                flights_to_process[flight_name] = (flight_dir, frame_indices)
    else:
        print("Processing all flights")
        for flight_dir in sorted(data_path.iterdir()):
            if flight_dir.is_dir():
                flights_to_process[flight_dir.name] = (flight_dir, None)  # None = all frames

    if not flights_to_process:
        print("No flights found to process.")
        return {}

    all_metrics = []

    for flight_name, (flight_dir, frame_indices) in tqdm(list(flights_to_process.items()), desc="Processing flights"):
        print(f"\n  Processing {flight_name}...")
        print(f"    Loading frames...")

        # Load frames (either specific indices or all)
        original_frames = load_frames_from_dir(flight_dir, frame_indices=frame_indices, desc="    Loading")

        if not original_frames:
            print(f"    Warning: No frames found for {flight_name}")
            continue

        # Compute metrics
        metrics = evaluate_original_only(original_frames, flight_name, num_workers=num_workers)
        all_metrics.append(metrics)

        print(f"    ✓ {metrics['num_frames_original']} frames | "
              f"Diff: {metrics['original_avg_interframe_diff']:.2f} | "
              f"Flow: {metrics['original_avg_flow_magnitude']:.2f}")

    # Compile results
    results = {
        "model_name": "original",
        "num_flights": len(all_metrics),
        "per_flight_metrics": all_metrics,
    }

    # Compute aggregate statistics
    if all_metrics:
        agg_stats = {
            "num_flights": len(all_metrics),
            "total_frames": sum(m["num_frames_original"] for m in all_metrics),
            "avg_interframe_diff": float(np.mean([m["original_avg_interframe_diff"] for m in all_metrics if m.get("original_avg_interframe_diff", 0) > 0])),
            "avg_flow_magnitude": float(np.mean([m["original_avg_flow_magnitude"] for m in all_metrics if m.get("original_avg_flow_magnitude", 0) > 0])),
        }
        
        # Add optional metrics if available
        psnr_values = [m["original_avg_psnr"] for m in all_metrics if m.get("original_avg_psnr", 0) > 0]
        if psnr_values:
            agg_stats["avg_psnr"] = float(np.mean(psnr_values))
            
        sharpness_values = [m["original_avg_sharpness"] for m in all_metrics if m.get("original_avg_sharpness", 0) > 0]
        if sharpness_values:
            agg_stats["avg_sharpness"] = float(np.mean(sharpness_values))
            
        cropping_values = [m["original_avg_cropping_ratio"] for m in all_metrics if m.get("original_avg_cropping_ratio", 0) > 0]
        if cropping_values:
            agg_stats["avg_cropping_ratio"] = float(np.mean(cropping_values))
        
        results["aggregate_metrics"] = agg_stats

    return results


def evaluate_model(model_name, data_dir, output_dir, original_metrics_path=None, num_workers=None):
    """
    Evaluate a stabilization model.

    Args:
        model_name: Name of the model (folder name under output/)
        data_dir: Path to original data directory (e.g., data/images)
        output_dir: Path to output directory (e.g., output)
        original_metrics_path: Optional path to pre-computed original metrics JSON
        num_workers: Number of parallel workers for metric computation

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

    # Try to load pre-computed original metrics
    original_metrics_dict = None
    if original_metrics_path:
        original_metrics_dict = load_original_metrics(original_metrics_path)
        if original_metrics_dict:
            print(f"Loaded pre-computed original metrics from: {original_metrics_path}")
            print(f"  (Will skip computing original metrics for flights with cached data)")
        else:
            print(f"Note: Could not load original metrics from {original_metrics_path}")
            print(f"  (Will compute original metrics on-the-fly)")

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

        # OPTIMIZATION: First load stabilized frames to see which ones exist
        print(f"    Loading stabilized frames...")
        stabilized_frames = load_frames_from_dir(flight_dir, desc="    Stabilized")

        # Extract frame filenames from stabilized paths and map extensions
        stab_frame_names = []
        for stab_path, _ in stabilized_frames:
            stab_name = Path(stab_path).name
            # Try to find corresponding original (might have different extension)
            base_name = Path(stab_name).stem
            # Check for .png first (most common in original data)
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_name = base_name + ext
                if (orig_flight_dir / potential_name).exists():
                    stab_frame_names.append(potential_name)
                    break

        # OPTIMIZATION: Only load original frames that have stabilized counterparts
        print(f"    Loading original frames...")
        original_frames = load_frames_from_dir(orig_flight_dir, frame_names=stab_frame_names, desc="    Original")

        # Try to load transformation data
        transform_path = flight_dir / flight_name  # e.g., output/baseline/Flight1/Flight1.json
        transform_data = load_transform_data(transform_path)
        if transform_data is None:
            # Try alternative location: output/baseline/Flight1_transforms.json
            transform_path = flight_dir.parent / f"{flight_name}_transforms"
            transform_data = load_transform_data(transform_path)

        if transform_data is not None:
            print(f"    Found transformation data")

        # Get pre-computed original metrics for this flight if available
        precomputed_metrics = None
        if original_metrics_dict and flight_name in original_metrics_dict:
            precomputed_metrics = original_metrics_dict[flight_name]

        # Evaluate this flight
        metrics = evaluate_sequence(original_frames, stabilized_frames, flight_name, transform_data, precomputed_original_metrics=precomputed_metrics, num_workers=num_workers)
        all_metrics.append(metrics)

        # Print concise summary for this flight
        print(f"    ✓ {metrics['num_frames_stabilized']} frames | "
              f"Diff: {metrics['improvement_interframe_diff']:.1f}% | "
              f"Flow: {metrics['improvement_flow_magnitude']:.1f}% | "
              f"PSNR: {metrics['stabilized_avg_psnr']:.1f}dB")

    # Compute aggregate statistics across all flights
    if all_metrics:
        agg_metrics = {
            "avg_improvement_interframe_diff": float(np.mean([m["improvement_interframe_diff"] for m in all_metrics])),
            "avg_improvement_flow_magnitude": float(np.mean([m["improvement_flow_magnitude"] for m in all_metrics])),
            "avg_psnr": float(np.mean([m["stabilized_avg_psnr"] for m in all_metrics if m["stabilized_avg_psnr"] > 0])),
            "avg_sharpness": float(np.mean([m["stabilized_avg_sharpness"] for m in all_metrics if m["stabilized_avg_sharpness"] > 0])),
            "avg_cropping_ratio": float(np.mean([m["stabilized_avg_cropping_ratio"] for m in all_metrics])),
        }

        # Add advanced metrics if available
        stability_scores = [m["stability_score_fft"] for m in all_metrics if m.get("stability_score_fft") is not None]
        if stability_scores:
            agg_metrics["avg_stability_score_fft"] = float(np.mean(stability_scores))

        distortion_scores = [m["avg_distortion_score"] for m in all_metrics if m.get("avg_distortion_score") is not None]
        if distortion_scores:
            agg_metrics["avg_distortion_score"] = float(np.mean(distortion_scores))

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
    all_metrics = results['per_flight_metrics']

    print(f"\nNumber of flights evaluated: {results['num_flights']}")

    # Check if this is original-only evaluation
    is_original_only = results['model_name'] == 'original'

    if is_original_only:
        # Original-only metrics display
        print("\n" + "-" * 80)
        print("ORIGINAL VIDEO METRICS (across all flights):")
        print("-" * 80)
        print(f"  Total frames: {agg['total_frames']}")
        print(f"  Avg inter-frame difference: {agg['avg_interframe_diff']:.2f}")
        print(f"  Avg optical flow magnitude: {agg['avg_flow_magnitude']:.2f}")
        
        # Show additional metrics if available
        if 'avg_psnr' in agg and agg['avg_psnr'] > 0:
            print(f"  Avg PSNR (consecutive frames): {agg['avg_psnr']:.2f} dB")
        if 'avg_sharpness' in agg and agg['avg_sharpness'] > 0:
            print(f"  Avg sharpness: {agg['avg_sharpness']:.2f}")
        if 'avg_cropping_ratio' in agg:
            print(f"  Avg cropping ratio: {agg['avg_cropping_ratio']:.2%}")
        
        # Show stability score if available
        avg_stability_scores = [m.get('original_stability_score') for m in all_metrics if m.get('original_stability_score') is not None]
        if avg_stability_scores:
            avg_stability = np.mean(avg_stability_scores)
            print(f"  Avg stability score: {avg_stability:.2f}/100")

        print("\n" + "-" * 80)
        print("PER-FLIGHT RESULTS:")
        print("-" * 80)
        print(f"{'Flight':<15} {'Frames':<8} {'Diff':<11} {'Flow':<11} {'PSNR':<10} {'Stability':<11}")
        print("-" * 80)

        for m in all_metrics:
            psnr_val = m.get('original_avg_psnr', 0)
            stability_val = m.get('original_stability_score', 0)
            print(
                f"{m['flight_name']:<15} "
                f"{m['num_frames_original']:<8} "
                f"{m['original_avg_interframe_diff']:>10.2f} "
                f"{m['original_avg_flow_magnitude']:>10.2f} "
                f"{psnr_val:>9.2f} "
                f"{stability_val:>10.2f}"
            )
    else:
        # Model evaluation with comparison to original
        # Compute raw aggregate metrics from per-flight data
        avg_orig_diff = np.mean([m['original_avg_interframe_diff'] for m in all_metrics if m.get('original_avg_interframe_diff', 0) > 0])
        avg_orig_flow = np.mean([m['original_avg_flow_magnitude'] for m in all_metrics if m.get('original_avg_flow_magnitude', 0) > 0])
        avg_stab_diff = np.mean([m['stabilized_avg_interframe_diff'] for m in all_metrics if m.get('stabilized_avg_interframe_diff', 0) > 0])
        avg_stab_flow = np.mean([m['stabilized_avg_flow_magnitude'] for m in all_metrics if m.get('stabilized_avg_flow_magnitude', 0) > 0])

        print("\n" + "-" * 80)
        print("RAW METRICS (across all flights):")
        print("-" * 80)
        print(f"  {'Metric':<35} {'Original':>12} {'Stabilized':>12} {'Improvement':>12}")
        print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}")
        print(f"  {'Inter-frame Difference':<35} {avg_orig_diff:>12.2f} {avg_stab_diff:>12.2f} {agg['avg_improvement_interframe_diff']:>11.2f}%")
        print(f"  {'Optical Flow Magnitude':<35} {avg_orig_flow:>12.2f} {avg_stab_flow:>12.2f} {agg['avg_improvement_flow_magnitude']:>11.2f}%")
        
        # Show stability score comparison if available
        orig_stability_scores = [m.get('original_stability_score') for m in all_metrics if m.get('original_stability_score') is not None]
        stab_stability_scores = [m.get('stabilized_stability_score') for m in all_metrics if m.get('stabilized_stability_score') is not None]
        if orig_stability_scores and stab_stability_scores:
            avg_orig_stability = np.mean(orig_stability_scores)
            avg_stab_stability = np.mean(stab_stability_scores)
            stability_improvement = avg_stab_stability - avg_orig_stability
            print(f"  {'Stability Score (0-100)':<35} {avg_orig_stability:>12.2f} {avg_stab_stability:>12.2f} {stability_improvement:>11.2f}pts")

        print("\n" + "-" * 80)
        print("STABILIZED VIDEO QUALITY METRICS:")
        print("-" * 80)
        print(f"  Average PSNR (consecutive frames):  {agg['avg_psnr']:>8.2f} dB")
        print(f"  Average Sharpness:                  {agg['avg_sharpness']:>8.2f}")
        print(f"  Average Cropping Ratio:             {agg['avg_cropping_ratio']:>8.2%}")

        # Print stability scores
        print("\n" + "-" * 80)
        print("STABILITY METRICS:")
        print("-" * 80)
        
        # Simple stability score (always available)
        orig_stability_scores = [m.get('original_stability_score') for m in all_metrics if m.get('original_stability_score') is not None]
        stab_stability_scores = [m.get('stabilized_stability_score') for m in all_metrics if m.get('stabilized_stability_score') is not None]
        if orig_stability_scores and stab_stability_scores:
            avg_orig_stability = np.mean(orig_stability_scores)
            avg_stab_stability = np.mean(stab_stability_scores)
            stability_improvement = avg_stab_stability - avg_orig_stability
            print(f"  Simple Stability Score (0-100):")
            print(f"    Original:    {avg_orig_stability:>6.2f}")
            print(f"    Stabilized:  {avg_stab_stability:>6.2f}")
            print(f"    Improvement: {stability_improvement:>+6.2f} pts")
        
        # FFT-based stability score (when transformation data available)
        if 'avg_stability_score_fft' in agg:
            print(f"\n  FFT-based Stability Score:          {agg['avg_stability_score_fft']:>8.4f}")
            print(f"    (Ratio of low-freq to total energy, higher is better)")
        
        # Distortion score
        if 'avg_distortion_score' in agg:
            print(f"\n  Average Distortion Score:           {agg['avg_distortion_score']:>8.4f}")
            print(f"    (Non-rigid warping, closer to 1.0 is better)")

        print("\n" + "-" * 80)
        print("PER-FLIGHT RESULTS:")
        print("-" * 80)
        print(f"{'Flight':<15} {'Frames':<8} {'Improv%':<9} {'PSNR':<10} {'Stability':<12} {'StabFFT':<10} {'Crop%':<8}")
        print("-" * 80)

        for m in all_metrics:
            stability_val = m.get('stabilized_stability_score', 0)
            fft_val = m.get('stability_score_fft')
            fft_str = f"{fft_val:.4f}" if fft_val is not None else "N/A"
            print(
                f"{m['flight_name']:<15} "
                f"{m['num_frames_stabilized']:<8} "
                f"{m.get('improvement_interframe_diff', 0):>8.1f}% "
                f"{m.get('stabilized_avg_psnr', 0):>9.2f} "
                f"{stability_val:>11.2f} "
                f"{fft_str:>9} "
                f"{m.get('stabilized_avg_cropping_ratio', 0):>7.1%}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate video stabilization quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute original video metrics first (recommended)
  python src/evaluate.py --model original --split-set test

  # Evaluate a stabilization model (will use cached original metrics if available)
  python src/evaluate.py --model baseline
  python src/evaluate.py --model mymodel --data-dir data/images --output-dir output
  python src/evaluate.py --model baseline --save-json results.json
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (folder name under output directory), or 'original' to compute original video metrics"
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
        help="Save detailed results to custom JSON file path (default: auto-saves to output/{model}/evaluation_results.json)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default='data/data_split.json',
        help="Path to data split JSON file (default: data/data_split.json, for reporting split info). Set to empty string to disable."
    )

    parser.add_argument(
        "--split-set",
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help="Which split set to process when --model original (default: test). Ignored for other models."
    )

    parser.add_argument(
        "--original-metrics",
        type=str,
        default='output/original_metrics.json',
        help="Path to pre-computed original metrics JSON (default: output/original_metrics.json)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for metric computation (default: auto-detect CPU count)"
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

    # Check if evaluating original or a model
    if args.model == "original":
        # Compute original video metrics
        results = evaluate_original(str(data_dir), split_info, args.split_set if split_info else None, num_workers=args.workers)

        # Add split info to results if available
        if split_info:
            results['split_info'] = {**split_info['metadata'], 'split_set': args.split_set}
    else:
        # Evaluate stabilization model
        original_metrics_path = repo_root / args.original_metrics
        results = evaluate_model(args.model, str(data_dir), str(output_dir), original_metrics_path, num_workers=args.workers)

        # Add split info to results if available
        if split_info:
            results['split_info'] = split_info['metadata']

    # Print summary
    print_summary(results)

    # Save to JSON (always save by default)
    if args.save_json:
        # Custom path specified
        json_path = repo_root / args.save_json
    elif args.model == "original":
        # Default for original: save to data/original_metrics.json
        json_path = repo_root / args.original_metrics
    else:
        # Default for models: save to output/{model_name}/evaluation_results.json
        json_path = output_dir / args.model / "evaluation_results.json"

    # Ensure parent directory exists
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {json_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
