# Hyperparameter Sweep Scripts

This directory contains scripts for systematically testing different hyperparameter configurations to optimize video stabilization performance.

## Overview

The hyperparameter sweep tests three key parameters:
1. **EMA Alpha** - Controls quaternion smoothing strength (0.1-0.5)
2. **Window/Stride** - Controls temporal context size (currently requires code modification)
3. **Freeze Threshold** - Controls frozen frame detection sensitivity (0.3-1.0)

## Quick Start

### 1. Full Sweep (Recommended for first run)

```bash
# Run all sweeps on validation set
python scripts/hyperparameter_sweep.py --split-set val
```

This will test:
- 5 alpha values for IMU-only stabilization (0.1, 0.2, 0.3, 0.4, 0.5)
- 5 alpha values for IMU+optical stabilization
- 4 freeze threshold values (0.3, 0.5, 0.7, 1.0)

**Expected runtime**: 2-6 hours (depends on dataset size)

### 2. Quick Sweep (Faster testing)

```bash
# Run reduced set of configurations
python scripts/hyperparameter_sweep.py --split-set val --quick
```

This tests:
- 3 alpha values (0.2, 0.3, 0.4)
- 2 freeze thresholds (0.5, 0.7)

**Expected runtime**: 1-2 hours

### 3. Dry Run (Test without executing)

```bash
# See what commands would be run
python scripts/hyperparameter_sweep.py --dry-run
```

## Selective Sweeps

Test only specific hyperparameters:

```bash
# Test only EMA alpha (IMU-only)
python scripts/hyperparameter_sweep.py --models ema --split-set val

# Test only EMA alpha (IMU + optical flow)
python scripts/hyperparameter_sweep.py --models ema_optical --split-set val

# Test only freeze threshold
python scripts/hyperparameter_sweep.py --models freeze --split-set val

# Test multiple specific models
python scripts/hyperparameter_sweep.py --models ema freeze --split-set val --quick
```

## Understanding the Parameters

### EMA Alpha (Quaternion Smoothing)

Controls the trade-off between smoothness and responsiveness:

- **0.1** - Heavy smoothing
  - ✅ Maximum jitter removal
  - ❌ May blur intentional camera motion
  - **Use when**: Video has severe jitter, slow motion

- **0.3** - Balanced (default)
  - ✅ Good jitter removal
  - ✅ Preserves most intentional motion
  - **Use when**: General-purpose stabilization

- **0.5** - Light smoothing
  - ✅ Preserves camera motion
  - ❌ Less jitter removal
  - **Use when**: Fast maneuvers, need responsiveness

### Freeze Threshold

Controls frozen frame detection sensitivity (mean pixel difference threshold):

- **0.3** - Very sensitive
  - Detects subtle frame freezing
  - May false-positive on static scenes

- **0.5** - Balanced (default)
  - Good for typical camera freeze detection

- **0.7-1.0** - Less sensitive
  - Only detects obvious freezes
  - Use if false positives occur

### Window/Stride (Currently requires code modification)

Controls temporal context:

- **(5, 3)** - Small window
  - Fast response, less stable
  - Lower latency

- **(10, 5)** - Default
  - Balanced stability/latency

- **(15, 7)** or **(20, 10)** - Large window
  - More stable, higher latency
  - Better for slow motion

**Note**: Window/stride sweep is currently marked as "SKIPPED" because the inference scripts use hardcoded values. To enable:
1. Add `--window` and `--stride` arguments to inference scripts
2. Modify sweep script to uncomment window/stride testing

## Output Structure

Results are saved to:

```
output/
├── imu_ema_alpha01/           # Alpha=0.1, IMU-only
│   ├── Flight1/
│   │   └── evaluation_results.json
│   └── Flight2/
├── imu_ema_alpha03/           # Alpha=0.3, IMU-only
├── imu_ema_optical_alpha02/   # Alpha=0.2, IMU+optical
├── imu_ema_freeze_t05_alpha03/  # Freeze threshold=0.5
└── sweep_results.json         # Consolidated sweep metadata
```

## Analyzing Results

### 1. Check individual model results

```bash
# View results for a specific configuration
cat output/imu_ema_alpha03/evaluation_results.json
```

Key metrics to look at:
- **stability_score** (↓ lower is better) - Inter-frame motion
- **optical_flow_magnitude** (↓ lower is better) - Residual motion
- **psnr** (↑ higher is better) - Frame quality
- **cropping_ratio** (↑ higher is better) - Retained FOV

### 2. Compare across configurations

```python
# Compare metrics programmatically
import json
from pathlib import Path

models = ['imu_ema_alpha01', 'imu_ema_alpha03', 'imu_ema_alpha05']
for model in models:
    results_path = Path(f'output/{model}/evaluation_results.json')
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
            print(f"{model}: stability={data['aggregate']['stability_score']:.4f}")
```

### 3. Visualize in Gradio app

```bash
# Launch interactive comparison app
python app.py
```

Select different models from the dropdown to compare side-by-side.

## Best Practices

### 1. Start with validation set
Always test on `--split-set val` first to avoid overfitting to test data.

### 2. Use quick mode for iteration
Use `--quick` flag when experimenting to get faster feedback.

### 3. Run full sweep before final testing
Once you've narrowed down promising configurations, run full sweep to confirm.

### 4. Test best config on test set
```bash
# After identifying best alpha (e.g., 0.3)
python src/inference_imu_ema.py --split-set test --output-name imu_ema_alpha03_final --alpha 0.3
python src/evaluate.py --model imu_ema_alpha03_final --split-set test
```

## Troubleshooting

### Sweep fails on specific configurations

Check the console output - failed runs are marked with ✗. Common issues:
- Missing data files
- Insufficient memory
- Invalid parameter values

### All configurations perform similarly

This might indicate:
- Parameter range is too narrow → Try wider range
- Metric isn't sensitive to this parameter → Focus on other parameters
- Dataset is too small → Try on full dataset (remove `--split-set`)

### Out of disk space

Each configuration saves stabilized frames. To save space:
- Use `--quick` mode
- Delete intermediate outputs after evaluation
- Only keep best-performing configurations

## Advanced Usage

### Custom alpha range

Edit `scripts/hyperparameter_sweep.py`:

```python
# Add custom alpha values
ALPHA_VALUES = [0.05, 0.15, 0.25, 0.35, 0.45]
```

### Custom freeze threshold range

```python
# Add custom freeze thresholds
FREEZE_THRESHOLDS = [0.2, 0.4, 0.6, 0.8]
```

### Parallel execution

For faster sweeps, run multiple configurations in parallel (requires multiple terminals):

```bash
# Terminal 1
python scripts/hyperparameter_sweep.py --models ema --split-set val

# Terminal 2
python scripts/hyperparameter_sweep.py --models ema_optical --split-set val

# Terminal 3
python scripts/hyperparameter_sweep.py --models freeze --split-set val
```

## Expected Results

Based on the NEXT_STEPS document, expected improvements:

- **EMA smoothing**: 15-25% improvement in stability metrics
- **Optimal alpha**: Likely between 0.2-0.4 for most use cases
- **Freeze detection**: Modest improvement if frozen frames are present in data

## Next Steps After Sweep

1. **Identify best configuration** from validation results
2. **Run on test set** with best config
3. **Document findings** - which parameters work best and why
4. **Consider Phase 2 improvements**:
   - Savitzky-Golay filtering
   - Dense optical flow
   - Adaptive windowing

## Questions or Issues?

See the main [README.md](../README.md) or check [NEXT_STEPS_AFTER_MVP.md](../NEXT_STEPS_AFTER_MVP.md) for more context on the stabilization algorithm.
