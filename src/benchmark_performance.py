"""
Performance Benchmark Script for Video Stabilization Models

This script tests different configurations to find optimal settings for real-time
stabilization on UAV video streams.
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import json
from model import BaselineModel, FusionModel


def generate_synthetic_frames(n_frames=10, height=720, width=1280):
    """Generate synthetic test frames."""
    frames = []
    for i in range(n_frames):
        # Create frame with some texture
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        # Add some structure (circles and lines for feature detection)
        cv2.circle(frame, (width//2, height//2), 100, (255, 255, 255), 2)
        cv2.line(frame, (0, 0), (width, height), (255, 255, 255), 2)
        cv2.line(frame, (width, 0), (0, height), (255, 255, 255), 2)
        frames.append(frame)
    return frames


def generate_synthetic_sensor_data(n_frames=10):
    """Generate synthetic IMU sensor data."""
    sensor_data = []
    for i in range(n_frames):
        # Generate quaternion (identity + small noise)
        qw = 0.999 + np.random.randn() * 0.001
        qx = np.random.randn() * 0.01
        qy = np.random.randn() * 0.01
        qz = np.random.randn() * 0.01
        # Normalize
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        sensor_data.append({
            'qw': qw/norm,
            'qx': qx/norm,
            'qy': qy/norm,
            'qz': qz/norm,
            'wx_radDs': np.random.randn() * 0.1,
            'wy_radDs': np.random.randn() * 0.1,
            'wz_radDs': np.random.randn() * 0.1,
            'ax_mDs2': np.random.randn() * 0.5,
            'ay_mDs2': np.random.randn() * 0.5,
            'az_mDs2': 9.8 + np.random.randn() * 0.1,
        })
    return sensor_data


def load_real_frames(data_root, n_frames=10):
    """Load real frames from data directory."""
    data_path = Path(data_root) / "images"

    # Find first available flight
    flights = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not flights:
        return None

    flight_dir = flights[0]
    frame_files = sorted(flight_dir.glob("*.png"))[:n_frames]

    if len(frame_files) < n_frames:
        return None

    frames = []
    for frame_file in frame_files:
        img = cv2.imread(str(frame_file))
        if img is not None:
            frames.append(img)

    return frames if len(frames) == n_frames else None


def benchmark_model(model, frames, sensor_data=None, n_iterations=10, warmup=2):
    """
    Benchmark a stabilization model.

    Args:
        model: Stabilization model instance
        frames: List of test frames
        sensor_data: Optional sensor data
        n_iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        dict with timing statistics
    """
    timings = []

    # Warmup
    for _ in range(warmup):
        model.stabilize_frames(frames, sensor_data=sensor_data)

    # Benchmark
    for _ in range(n_iterations):
        t_start = time.perf_counter()
        result = model.stabilize_frames(frames, sensor_data=sensor_data)
        t_end = time.perf_counter()
        timings.append((t_end - t_start) * 1000)  # Convert to ms

    timings = np.array(timings)
    n_frames_processed = len(frames) - 1  # Exclude reference frame

    return {
        'total_ms': {
            'mean': float(np.mean(timings)),
            'median': float(np.median(timings)),
            'min': float(np.min(timings)),
            'max': float(np.max(timings)),
            'std': float(np.std(timings)),
        },
        'per_frame_ms': {
            'mean': float(np.mean(timings) / n_frames_processed),
            'median': float(np.median(timings) / n_frames_processed),
        },
        'fps': float(1000.0 / (np.mean(timings) / n_frames_processed)),
        'realtime_30fps': bool(np.mean(timings) / n_frames_processed < 33.33),
        'realtime_60fps': bool(np.mean(timings) / n_frames_processed < 16.67),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark video stabilization models for real-time performance"
    )

    parser.add_argument(
        '--data-root',
        type=str,
        default='data',
        help='Path to data directory (default: data)'
    )

    parser.add_argument(
        '--use-real-data',
        action='store_true',
        help='Use real frames instead of synthetic data'
    )

    parser.add_argument(
        '--n-frames',
        type=int,
        default=10,
        help='Number of frames per window (default: 10)'
    )

    parser.add_argument(
        '--n-iterations',
        type=int,
        default=20,
        help='Number of benchmark iterations (default: 20)'
    )

    parser.add_argument(
        '--resolution',
        type=str,
        choices=['480p', '720p', '1080p', 'custom'],
        default='720p',
        help='Frame resolution for synthetic data (default: 720p)'
    )

    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Frame height for custom resolution (default: 720)'
    )

    parser.add_argument(
        '--width',
        type=int,
        default=1280,
        help='Frame width for custom resolution (default: 1280)'
    )

    parser.add_argument(
        '--output-json',
        type=str,
        default='benchmark_results.json',
        help='Output JSON file for results (default: benchmark_results.json)'
    )

    args = parser.parse_args()

    # Set resolution
    resolutions = {
        '480p': (480, 640),
        '720p': (720, 1280),
        '1080p': (1080, 1920),
        'custom': (args.height, args.width)
    }
    height, width = resolutions[args.resolution]

    print("="*70)
    print("VIDEO STABILIZATION PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"Resolution:   {width}x{height}")
    print(f"Window size:  {args.n_frames} frames")
    print(f"Iterations:   {args.n_iterations}")
    print("="*70 + "\n")

    # Load or generate test data
    if args.use_real_data:
        print("Loading real frames...")
        frames = load_real_frames(args.data_root, args.n_frames)
        if frames is None:
            print("Failed to load real frames, falling back to synthetic data")
            frames = generate_synthetic_frames(args.n_frames, height, width)
            data_type = "synthetic"
        else:
            print(f"Loaded {len(frames)} real frames")
            data_type = "real"
            # Update dimensions from actual frames
            height, width = frames[0].shape[:2]
    else:
        print("Generating synthetic frames...")
        frames = generate_synthetic_frames(args.n_frames, height, width)
        data_type = "synthetic"

    sensor_data = generate_synthetic_sensor_data(args.n_frames)

    # Test configurations
    configs = [
        # Baseline configurations
        {'name': 'Baseline (500 features)', 'model': 'baseline', 'max_features': 500, 'use_sensor': False},
        {'name': 'Baseline (1000 features)', 'model': 'baseline', 'max_features': 1000, 'use_sensor': False},
        {'name': 'Baseline (2000 features)', 'model': 'baseline', 'max_features': 2000, 'use_sensor': False},

        # Fusion configurations
        {'name': 'Fusion (500 features)', 'model': 'fusion', 'max_features': 500, 'use_sensor': True, 'imu_weight': 0.7},
        {'name': 'Fusion (1000 features)', 'model': 'fusion', 'max_features': 1000, 'use_sensor': True, 'imu_weight': 0.7},
        {'name': 'Fusion (2000 features)', 'model': 'fusion', 'max_features': 2000, 'use_sensor': True, 'imu_weight': 0.7},
    ]

    results = {
        'metadata': {
            'resolution': f"{width}x{height}",
            'n_frames': args.n_frames,
            'n_iterations': args.n_iterations,
            'data_type': data_type
        },
        'configs': []
    }

    # Run benchmarks
    for config in configs:
        print(f"\nBenchmarking: {config['name']}")
        print("-" * 70)

        # Initialize model
        if config['model'] == 'baseline':
            model = BaselineModel(max_features=config['max_features'])
        elif config['model'] == 'fusion':
            model = FusionModel(
                max_features=config['max_features'],
                imu_weight=config.get('imu_weight', 0.7),
                enable_profiling=False
            )

        # Run benchmark
        sensor = sensor_data if config['use_sensor'] else None
        stats = benchmark_model(model, frames, sensor, n_iterations=args.n_iterations)

        # Print results
        print(f"  Total time:        {stats['total_ms']['mean']:6.2f} ± {stats['total_ms']['std']:5.2f} ms")
        print(f"  Per-frame time:    {stats['per_frame_ms']['mean']:6.2f} ms")
        print(f"  Throughput:        {stats['fps']:6.1f} FPS")
        print(f"  Real-time 30 FPS:  {'✓ YES' if stats['realtime_30fps'] else '✗ NO'}")
        print(f"  Real-time 60 FPS:  {'✓ YES' if stats['realtime_60fps'] else '✗ NO'}")

        # Store results
        config_result = {
            'config': config,
            'performance': stats
        }
        results['configs'].append(config_result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Configuration':<35} {'FPS':>8} {'Per-frame':>12} {'RT-30fps':>10}")
    print("-"*70)

    for result in results['configs']:
        name = result['config']['name']
        fps = result['performance']['fps']
        per_frame = result['performance']['per_frame_ms']['mean']
        rt_status = "✓" if result['performance']['realtime_30fps'] else "✗"
        print(f"{name:<35} {fps:>8.1f} {per_frame:>10.2f} ms {rt_status:>10}")

    print("="*70)

    # Find fastest configuration
    fastest = min(results['configs'], key=lambda x: x['performance']['per_frame_ms']['mean'])
    print(f"\nFastest configuration: {fastest['config']['name']}")
    print(f"  Per-frame: {fastest['performance']['per_frame_ms']['mean']:.2f} ms")
    print(f"  Throughput: {fastest['performance']['fps']:.1f} FPS")

    # Save results to JSON
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_json}")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR REAL-TIME UAV STABILIZATION")
    print("="*70)

    rt_configs = [r for r in results['configs'] if r['performance']['realtime_30fps']]
    if rt_configs:
        print("\nConfigurations capable of 30 FPS:")
        for r in rt_configs:
            print(f"  • {r['config']['name']}")
            print(f"    → {r['performance']['fps']:.1f} FPS ({r['performance']['per_frame_ms']['mean']:.2f} ms/frame)")
    else:
        print("\nNo configurations achieved 30 FPS real-time performance.")
        print("Consider:")
        print("  • Reducing max_features to 500 or less")
        print("  • Reducing frame resolution")
        print("  • Using GPU acceleration (requires implementation)")

    print("\nFor best quality, use higher max_features (1000-2000)")
    print("For best speed, use lower max_features (500 or less)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
