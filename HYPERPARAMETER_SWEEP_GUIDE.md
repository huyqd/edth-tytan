# Hyperparameter Sweep Quick Reference

## What Was Implemented

Created an automated hyperparameter sweep system to optimize video stabilization performance by testing:

1. **EMA Alpha** (0.1-0.5) - Quaternion smoothing strength
2. **Freeze Threshold** (0.3-1.0) - Frozen frame detection sensitivity
3. **Window/Stride** - Noted as future work (requires adding CLI args to inference scripts)

## Files Created

- [scripts/hyperparameter_sweep.py](scripts/hyperparameter_sweep.py) - Main sweep script
- [scripts/README.md](scripts/README.md) - Detailed documentation and usage guide

## Quick Commands

### Run Full Sweep (Recommended)
```bash
# Test all configurations on validation set (2-6 hours)
python scripts/hyperparameter_sweep.py --split-set val
```

### Run Quick Sweep (For Testing)
```bash
# Reduced configurations for faster iteration (1-2 hours)
python scripts/hyperparameter_sweep.py --split-set val --quick
```

### Test Specific Parameters
```bash
# Only test EMA alpha values
python scripts/hyperparameter_sweep.py --models ema --split-set val

# Only test EMA + optical flow
python scripts/hyperparameter_sweep.py --models ema_optical --split-set val

# Only test freeze detection
python scripts/hyperparameter_sweep.py --models freeze --split-set val

# Test multiple
python scripts/hyperparameter_sweep.py --models ema freeze --split-set val --quick
```

### Dry Run (Preview commands)
```bash
# See what will be executed without running
python scripts/hyperparameter_sweep.py --dry-run --quick
```

## What Gets Tested

### Quick Mode (--quick flag)
- **EMA Alpha**: 0.2, 0.3, 0.4 (3 values)
- **Freeze Threshold**: 0.5, 0.7 (2 values)
- **Total**: 8 configurations (3 EMA + 3 EMA_optical + 2 freeze)

### Full Mode (default)
- **EMA Alpha**: 0.1, 0.2, 0.3, 0.4, 0.5 (5 values)
- **Freeze Threshold**: 0.3, 0.5, 0.7, 1.0 (4 values)
- **Total**: 14 configurations (5 EMA + 5 EMA_optical + 4 freeze)

## Output Structure

Results saved to:
```
output/
├── imu_ema_alpha01/              # Alpha=0.1, IMU-only
│   ├── Flight1/
│   │   ├── 00000000.jpg          # Stabilized frames
│   │   └── Flight1.json          # Transformation data
│   └── evaluation_results.json   # Metrics
├── imu_ema_alpha02/              # Alpha=0.2, IMU-only
├── imu_ema_alpha03/              # Alpha=0.3, IMU-only
├── imu_ema_optical_alpha02/      # Alpha=0.2, IMU+optical
├── imu_ema_freeze_t05_alpha03/   # Freeze threshold=0.5
└── sweep_results.json            # Metadata (timestamp, configs)
```

## Analyzing Results

### 1. View Individual Results
```bash
# Check metrics for specific configuration
cat output/imu_ema_alpha03/evaluation_results.json
```

### 2. Compare Across Configs
```python
import json
from pathlib import Path

# Compare stability scores
models = ['imu_ema_alpha02', 'imu_ema_alpha03', 'imu_ema_alpha04']
for model in models:
    path = Path(f'output/{model}/evaluation_results.json')
    if path.exists():
        data = json.load(open(path))
        stability = data['aggregate']['stability_score']
        print(f"{model}: {stability:.4f}")
```

### 3. Use Gradio App
```bash
python app.py
# Select different models from dropdown to compare
```

## Key Metrics to Watch

- **stability_score** (↓ lower = better) - Inter-frame motion
- **optical_flow_magnitude** (↓ lower = better) - Residual motion
- **psnr** (↑ higher = better) - Frame quality
- **cropping_ratio** (↑ higher = better) - FOV retained

## Expected Improvements

According to [NEXT_STEPS_AFTER_MVP.md](NEXT_STEPS_AFTER_MVP.md):
- **EMA smoothing**: 15-25% improvement in stability
- **Hyperparameter tuning**: 10-20% additional improvement
- **Combined**: 20-35% total improvement expected

## Workflow Recommendations

### Phase 1: Initial Sweep
```bash
# 1. Run quick sweep on validation set
python scripts/hyperparameter_sweep.py --split-set val --quick

# 2. Identify top 2-3 configurations
# (check evaluation_results.json files)

# 3. Run full sweep for top configs
python scripts/hyperparameter_sweep.py --split-set val --models ema
```

### Phase 2: Test Best Config
```bash
# Run best configuration on test set
python src/inference_imu_ema.py \
    --split-set test \
    --output-name imu_ema_alpha03_final \
    --alpha 0.3

# Evaluate
python src/evaluate.py \
    --model imu_ema_alpha03_final \
    --split-set test
```

### Phase 3: Production
```bash
# Run on full dataset (no split)
python src/inference_imu_ema.py \
    --split "" \
    --output-name production_stabilized \
    --alpha 0.3
```

## Troubleshooting

### Sweep takes too long
- Use `--quick` flag
- Test only specific models: `--models ema`
- Reduce dataset size

### Out of disk space
- Delete intermediate outputs: `rm -rf output/imu_ema_alpha*/Flight*/`
- Keep only evaluation_results.json files
- Run on smaller subset

### All configs perform similarly
- Try wider alpha range (edit ALPHA_VALUES in script)
- Check if dataset has significant jitter/motion
- Verify metrics are being calculated correctly

### Script errors
- Check Python version (requires 3.7+)
- Verify dependencies: `uv sync` or `pip install -e .`
- Check data split exists: `ls data/data_split.json`

## Next Steps

After finding optimal hyperparameters:

1. **Document findings** - Which alpha/threshold works best
2. **Update defaults** in inference scripts
3. **Consider Phase 2 improvements**:
   - Savitzky-Golay filtering (better smoothing)
   - Dense optical flow (better on thermal)
   - Adaptive windowing (motion-dependent)

See [NEXT_STEPS_AFTER_MVP.md](NEXT_STEPS_AFTER_MVP.md) for more advanced improvements.

## Implementation Notes

### What's Working
- ✅ EMA alpha sweep (IMU-only and IMU+optical)
- ✅ Freeze threshold sweep
- ✅ Automatic evaluation after inference
- ✅ Results consolidation in JSON
- ✅ Dry-run mode for testing

### What's Pending
- ⏸️ Window/stride sweep (requires adding CLI arguments to inference scripts)
  - To enable: Add `--window` and `--stride` args to:
    - `src/inference_imu_ema.py`
    - `src/inference_imu_ema_optical.py`
    - `src/inference_imu_ema_freeze.py`

### Design Decisions

1. **Why separate sweeps for each model type?**
   - Different models have different parameter spaces
   - Allows selective testing (faster iteration)
   - Easier to parallelize

2. **Why validation set by default?**
   - Prevents overfitting to test set
   - Faster than full dataset
   - Industry best practice

3. **Why quick mode?**
   - Rapid prototyping and testing
   - Verify sweep works before long runs
   - Good enough for identifying trends

## References

- Main documentation: [scripts/README.md](scripts/README.md)
- Algorithm roadmap: [NEXT_STEPS_AFTER_MVP.md](NEXT_STEPS_AFTER_MVP.md)
- Project overview: [CLAUDE.md](CLAUDE.md)
- Evaluation metrics: [README_evaluate.md](README_evaluate.md)
