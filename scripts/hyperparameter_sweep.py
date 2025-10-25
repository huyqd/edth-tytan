"""
Hyperparameter sweep script for video stabilization.

This script runs a grid search over key hyperparameters:
1. EMA alpha (quaternion smoothing)
2. Window/stride combinations
3. Freeze threshold (for freeze-aware stabilization)

Usage:
    # Run full sweep on validation set
    python scripts/hyperparameter_sweep.py --split-set val

    # Run quick sweep (fewer combinations)
    python scripts/hyperparameter_sweep.py --split-set val --quick

    # Dry run (print configs without running)
    python scripts/hyperparameter_sweep.py --dry-run

    # Test specific models
    python scripts/hyperparameter_sweep.py --models ema ema_optical --split-set val
"""

import argparse
import subprocess
import os
import json
import time
from pathlib import Path
from itertools import product


# Hyperparameter configurations
ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
ALPHA_VALUES_QUICK = [0.2, 0.3, 0.4]

WINDOW_STRIDE_CONFIGS = [
    (5, 3),
    (10, 5),
    (15, 7),
    (20, 10),
]
WINDOW_STRIDE_CONFIGS_QUICK = [
    (5, 3),
    (10, 5),
    (15, 7),
]

FREEZE_THRESHOLDS = [0.3, 0.5, 0.7, 1.0]
FREEZE_THRESHOLDS_QUICK = [0.5, 0.7]


def run_command(cmd, dry_run=False):
    """Run a shell command and return success status."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {' '.join(cmd)}")
    if dry_run:
        return True

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False


def sweep_ema_alpha(split_set='val', quick=False, dry_run=False):
    """
    Sweep over EMA alpha values (IMU-only stabilization).

    Tests different smoothing strengths to find optimal jitter removal
    vs motion preservation trade-off.
    """
    print("\n" + "="*80)
    print("SWEEP 1: EMA Alpha (IMU-only stabilization)")
    print("="*80)

    alphas = ALPHA_VALUES_QUICK if quick else ALPHA_VALUES
    results = []

    for alpha in alphas:
        output_name = f"imu_ema_alpha{int(alpha*10):02d}"

        print(f"\n--- Testing alpha={alpha} ---")

        # Run inference
        cmd = [
            "uv", "run", "python", "src/inference_imu_ema.py",
            "--split-set", split_set,
            "--output-name", output_name,
            "--alpha", str(alpha)
        ]

        success = run_command(cmd, dry_run)
        if not success and not dry_run:
            print(f"Inference failed for alpha={alpha}, skipping evaluation")
            continue

        # Run evaluation
        cmd_eval = [
            "uv", "run", "python", "src/evaluate.py",
            "--model", output_name,
            "--split-set", split_set
        ]

        eval_success = run_command(cmd_eval, dry_run)

        results.append({
            "alpha": alpha,
            "output_name": output_name,
            "success": success and eval_success
        })

    return results


def sweep_ema_alpha_optical(split_set='val', quick=False, dry_run=False):
    """
    Sweep over EMA alpha values (IMU + optical flow stabilization).

    Tests EMA smoothing combined with optical flow refinement.
    """
    print("\n" + "="*80)
    print("SWEEP 2: EMA Alpha (IMU + Optical Flow)")
    print("="*80)

    alphas = ALPHA_VALUES_QUICK if quick else ALPHA_VALUES
    results = []

    for alpha in alphas:
        output_name = f"imu_ema_optical_alpha{int(alpha*10):02d}"

        print(f"\n--- Testing alpha={alpha} (with optical flow) ---")

        # Run inference
        cmd = [
            "uv", "run", "python", "src/inference_imu_ema_optical.py",
            "--split-set", split_set,
            "--output-name", output_name,
            "--alpha", str(alpha)
        ]

        success = run_command(cmd, dry_run)
        if not success and not dry_run:
            print(f"Inference failed for alpha={alpha}, skipping evaluation")
            continue

        # Run evaluation
        cmd_eval = [
            "uv", "run", "python", "src/evaluate.py",
            "--model", output_name,
            "--split-set", split_set
        ]

        eval_success = run_command(cmd_eval, dry_run)

        results.append({
            "alpha": alpha,
            "output_name": output_name,
            "success": success and eval_success
        })

    return results


def sweep_window_stride(split_set='val', alpha=0.3, quick=False, dry_run=False):
    """
    Sweep over window/stride combinations.

    Tests different temporal context sizes to find optimal stability
    vs responsiveness trade-off.

    Note: This requires modifying the inference scripts to accept
    window/stride as command-line arguments. Currently uses code modification.
    """
    print("\n" + "="*80)
    print("SWEEP 3: Window/Stride Combinations")
    print("="*80)
    print("NOTE: This sweep requires modifying window/stride in inference scripts.")
    print("      Current implementation uses fixed window=10, stride=5.")
    print("      To enable this sweep, add --window and --stride arguments to:")
    print("      - src/inference_imu_ema.py")
    print("      - src/inference_imu_ema_optical.py")
    print("="*80)

    configs = WINDOW_STRIDE_CONFIGS_QUICK if quick else WINDOW_STRIDE_CONFIGS
    results = []

    for window, stride in configs:
        output_name = f"imu_ema_w{window}_s{stride}_alpha{int(alpha*10):02d}"

        print(f"\n--- Testing window={window}, stride={stride} ---")
        print("    [SKIPPED - requires script modification]")

        results.append({
            "window": window,
            "stride": stride,
            "alpha": alpha,
            "output_name": output_name,
            "success": False,
            "skipped": True,
            "reason": "Requires adding --window and --stride CLI arguments"
        })

    return results


def sweep_freeze_threshold(split_set='val', alpha=0.3, quick=False, dry_run=False):
    """
    Sweep over freeze detection thresholds.

    Tests different sensitivity levels for detecting frozen frames
    (camera stopped updating).
    """
    print("\n" + "="*80)
    print("SWEEP 4: Freeze Detection Threshold")
    print("="*80)

    thresholds = FREEZE_THRESHOLDS_QUICK if quick else FREEZE_THRESHOLDS
    results = []

    for threshold in thresholds:
        output_name = f"imu_ema_freeze_t{int(threshold*10):02d}_alpha{int(alpha*10):02d}"

        print(f"\n--- Testing freeze_threshold={threshold} ---")

        # Run inference
        cmd = [
            "uv", "run", "python", "src/inference_imu_ema_freeze.py",
            "--split-set", split_set,
            "--output-name", output_name,
            "--alpha", str(alpha),
            "--freeze-threshold", str(threshold)
        ]

        success = run_command(cmd, dry_run)
        if not success and not dry_run:
            print(f"Inference failed for threshold={threshold}, skipping evaluation")
            continue

        # Run evaluation
        cmd_eval = [
            "uv", "run", "python", "src/evaluate.py",
            "--model", output_name,
            "--split-set", split_set
        ]

        eval_success = run_command(cmd_eval, dry_run)

        results.append({
            "freeze_threshold": threshold,
            "alpha": alpha,
            "output_name": output_name,
            "success": success and eval_success
        })

    return results


def save_sweep_results(results, output_file):
    """Save sweep results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSweep results saved to: {output_path}")


def print_summary(all_results):
    """Print summary of sweep results."""
    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)

    total_runs = sum(len(results) for results in all_results.values())
    successful_runs = sum(
        sum(1 for r in results if r.get('success', False))
        for results in all_results.values()
    )
    skipped_runs = sum(
        sum(1 for r in results if r.get('skipped', False))
        for results in all_results.values()
    )

    print(f"Total configurations: {total_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Skipped runs: {skipped_runs}")
    print(f"Failed runs: {total_runs - successful_runs - skipped_runs}")

    print("\n" + "-"*80)
    print("Output directories created:")
    for sweep_name, results in all_results.items():
        print(f"\n{sweep_name}:")
        for r in results:
            status = "[OK]" if r.get('success') else ("[SKIP]" if r.get('skipped') else "[FAIL]")
            print(f"  {status} output/{r['output_name']}/")

    print("\n" + "-"*80)
    print("Next steps:")
    print("1. Check evaluation results in output/*/evaluation_results.json")
    print("2. Compare metrics across configurations")
    print("3. Identify best hyperparameters")
    print("4. Run best config on test set")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for video stabilization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full sweep on validation set
  python scripts/hyperparameter_sweep.py --split-set val

  # Quick sweep with fewer combinations
  python scripts/hyperparameter_sweep.py --split-set val --quick

  # Dry run (print commands without executing)
  python scripts/hyperparameter_sweep.py --dry-run

  # Test only specific sweeps
  python scripts/hyperparameter_sweep.py --models ema freeze --split-set val

  # Use custom alpha for window/stride and freeze sweeps
  python scripts/hyperparameter_sweep.py --models window freeze --alpha 0.25
        """
    )

    parser.add_argument(
        '--split-set',
        type=str,
        choices=['train', 'val', 'test'],
        default='val',
        help='Which split to process (default: val)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick sweep with fewer hyperparameter values'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing them'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=['ema', 'ema_optical', 'window', 'freeze', 'all'],
        default=['all'],
        help='Which models to sweep (default: all)'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.3,
        help='Alpha value for window/stride and freeze sweeps (default: 0.3)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output/sweep_results.json',
        help='Output file for sweep results (default: output/sweep_results.json)'
    )

    args = parser.parse_args()

    # Expand 'all' to all model types
    if 'all' in args.models:
        models = ['ema', 'ema_optical', 'freeze']
    else:
        models = args.models

    print("="*80)
    print("HYPERPARAMETER SWEEP")
    print("="*80)
    print(f"Split set: {args.split_set}")
    print(f"Quick mode: {args.quick}")
    print(f"Dry run: {args.dry_run}")
    print(f"Models: {', '.join(models)}")
    print(f"Alpha (for window/freeze): {args.alpha}")
    print("="*80)

    start_time = time.time()
    all_results = {}

    # Run selected sweeps
    if 'ema' in models:
        all_results['ema_alpha'] = sweep_ema_alpha(
            args.split_set, args.quick, args.dry_run
        )

    if 'ema_optical' in models:
        all_results['ema_alpha_optical'] = sweep_ema_alpha_optical(
            args.split_set, args.quick, args.dry_run
        )

    if 'window' in models:
        all_results['window_stride'] = sweep_window_stride(
            args.split_set, args.alpha, args.quick, args.dry_run
        )

    if 'freeze' in models:
        all_results['freeze_threshold'] = sweep_freeze_threshold(
            args.split_set, args.alpha, args.quick, args.dry_run
        )

    # Save results
    if not args.dry_run:
        sweep_metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "split_set": args.split_set,
            "quick": args.quick,
            "models": models,
            "alpha_base": args.alpha,
            "results": all_results
        }
        save_sweep_results(sweep_metadata, args.output)

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n\nTotal sweep time: {elapsed/60:.1f} minutes")
    print_summary(all_results)


if __name__ == "__main__":
    main()
