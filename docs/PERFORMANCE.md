# Performance Optimization Guide

This guide helps you optimize video stabilization for real-time UAV applications.

## Quick Start

### Run Performance Benchmark

Test different configurations to find optimal settings for your hardware:

```bash
# Benchmark with synthetic data (fast)
uv run python src/benchmark_performance.py

# Benchmark with real data
uv run python src/benchmark_performance.py --use-real-data --data-root data

# Custom resolution and iterations
uv run python src/benchmark_performance.py --resolution 1080p --n-iterations 50

# Save results to custom location
uv run python src/benchmark_performance.py --output-json my_benchmark.json
```

### Enable Profiling in Inference

```bash
# Run inference with detailed performance profiling
uv run python src/inference.py --model fusion --profile --split-set test

# Try faster configuration (fewer features)
uv run python src/inference.py --model fusion --profile --max-features 500
```

## Performance Targets

### Real-Time Requirements

| Target FPS | Max Time per Frame | Use Case |
|------------|-------------------|----------|
| 30 FPS     | 33.33 ms         | Standard UAV video |
| 60 FPS     | 16.67 ms         | High-speed UAV video |
| 120 FPS    | 8.33 ms          | Racing drones |

### Typical Performance

On modern hardware (2020+ CPU), expect:

| Configuration | Resolution | Per-Frame Time | FPS | Real-time 30 FPS? |
|--------------|------------|----------------|-----|-------------------|
| Baseline (500 features) | 720p | ~15-20 ms | 50-67 | ✓ YES |
| Baseline (2000 features) | 720p | ~40-60 ms | 17-25 | ✗ NO |
| Fusion (500 features) | 720p | ~18-25 ms | 40-56 | ✓ YES |
| Fusion (2000 features) | 720p | ~45-70 ms | 14-22 | ✗ NO |

**Note**: These are approximate values. Run `benchmark_performance.py` for accurate measurements on your hardware.

## Optimization Strategies

### 1. Reduce Feature Count

**Impact**: Significant speedup (2-3x faster)
**Trade-off**: Slightly reduced stabilization quality

```bash
# Fast mode (500 features)
uv run python src/inference.py --model fusion --max-features 500

# Balanced mode (1000 features)
uv run python src/inference.py --model fusion --max-features 1000

# Quality mode (2000 features, default)
uv run python src/inference.py --model fusion --max-features 2000
```

**Recommendation**: Start with 1000 features for good balance.

### 2. Reduce Frame Resolution

**Impact**: Quadratic speedup (4x faster for 2x reduction)
**Trade-off**: Lower output quality

**Preprocessing options**:
- Resize frames before stabilization
- Process at lower resolution, then upscale
- Use ROI (region of interest) cropping

### 3. Choose the Right Model

| Model | Speed | Quality | When to Use |
|-------|-------|---------|-------------|
| Baseline | Fast | Good | Smooth, slow motion scenes |
| Fusion | Medium | Better | Fast rotations, vibrations |

### 4. Batch Processing

For offline processing, increase window size and stride in inference.py:

```python
# In inference.py:
window, stride = 20, 10  # Larger window for better context
```

**Note**: This doesn't improve per-frame speed but can improve overall throughput.

## Performance Analysis Tools

### 1. Built-in Profiling

Enable detailed timing breakdown:

```bash
uv run python src/inference.py --model fusion --profile --split-set test
```

Output shows:
- Grayscale conversion time
- Transform estimation time (per frame)
- Warping time (per frame)
- Total time
- Throughput (FPS)
- Real-time capability indicator

### 2. Benchmark Script

Comprehensive testing across configurations:

```bash
# Full benchmark
uv run python src/benchmark_performance.py --n-iterations 50

# Quick test
uv run python src/benchmark_performance.py --n-iterations 10
```

Outputs:
- Per-configuration timing statistics
- FPS measurements
- Real-time capability indicators
- Recommendations
- JSON file with detailed results

### 3. Performance Summary

At the end of inference with `--profile`, you'll see:

```
============================================================
PERFORMANCE SUMMARY
============================================================

Warping Performance (per frame):
  Mean:       18.45ms
  Median:     17.89ms
  Min:        15.23ms
  Max:        25.67ms
  P95:        22.34ms
  P99:        24.12ms

Total Processing Time (per window):
  Mean:      184.50ms
  Median:    178.90ms

Throughput:
  Average:    54.2 FPS
  Real-time capable (30 FPS): ✓ YES
============================================================
```

## Bottleneck Analysis

### Typical Time Distribution

For Fusion model (2000 features, 720p):

```
Grayscale conversion:   ~2-5 ms   (5%)
Transform estimation:   ~30-50 ms (70%)
  └─ Feature detection: ~15-20 ms
  └─ Optical flow:      ~10-20 ms
  └─ RANSAC fitting:    ~5-10 ms
  └─ IMU processing:    ~0.5-1 ms
Warping:               ~10-15 ms (25%)
```

**Key insight**: Optical flow (transform estimation) is the bottleneck.

### Optimization Priorities

1. **Reduce max_features** → Directly reduces optical flow time
2. **Use baseline model** → Skips IMU processing (minimal gain)
3. **Reduce resolution** → Reduces both optical flow and warping time

## Hardware Considerations

### CPU Optimization

- OpenCV uses multiple threads by default
- Set `OMP_NUM_THREADS` or `cv2.setNumThreads()` to control parallelism
- Modern CPUs (4+ cores) will see ~2-3x speedup from threading

### GPU Acceleration (Future)

The current implementation uses CPU only. GPU acceleration could provide:
- 5-10x speedup for optical flow (using cv2.cuda)
- 2-3x speedup for warping (using GPU-accelerated warpAffine)

**Implementation status**: Not yet implemented

## Recommended Configurations

### For Real-Time 30 FPS (Standard UAV)

```bash
# Baseline model, fast mode
uv run python src/inference.py \
  --model baseline \
  --max-features 800 \
  --split-set test

# Fusion model, fast mode
uv run python src/inference.py \
  --model fusion \
  --max-features 700 \
  --split-set test
```

### For Best Quality (Offline Processing)

```bash
# Fusion model, quality mode
uv run python src/inference.py \
  --model fusion \
  --max-features 2000 \
  --split-set test
```

### For High-Speed UAV (60 FPS)

```bash
# Minimal features, baseline model
uv run python src/inference.py \
  --model baseline \
  --max-features 300 \
  --split-set test
```

## Measuring Your Performance

### Step 1: Run Benchmark

```bash
uv run python src/benchmark_performance.py \
  --use-real-data \
  --data-root data \
  --n-iterations 20
```

### Step 2: Identify Fastest Real-Time Config

Look for configurations marked with ✓ in "Real-time 30 FPS" column.

### Step 3: Test on Full Dataset

```bash
# Use the configuration identified in benchmark
uv run python src/inference.py \
  --model <baseline|fusion> \
  --max-features <value from benchmark> \
  --profile \
  --split-set test
```

### Step 4: Evaluate Quality

```bash
# Check if quality is acceptable
uv run python src/evaluate.py --model <output_name>

# Compare with baseline
uv run python src/evaluate.py --model baseline
```

## Troubleshooting

### Issue: Performance Slower Than Expected

**Check**:
1. Run benchmark to verify hardware capability
2. Check CPU utilization (should be high)
3. Verify OpenCV is using threading: `cv2.getNumThreads()`
4. Check if other processes are consuming CPU

**Solutions**:
- Close other applications
- Reduce max_features
- Use baseline model instead of fusion

### Issue: Can't Achieve 30 FPS

**Options**:
1. Reduce max_features below 500
2. Reduce frame resolution (preprocess frames)
3. Process every other frame (temporal subsampling)
4. Consider GPU acceleration (requires implementation)

### Issue: Quality Degradation at Low Feature Count

**Strategies**:
- Use fusion model (IMU helps compensate)
- Increase IMU weight in fusion model
- Apply temporal smoothing to transformations
- Use higher quality preprocessing (denoising, sharpening)

## Advanced: Custom Optimization

### Modify Feature Detection Parameters

In `baseline.py` or `fusion.py`:

```python
# Faster but lower quality
p0 = cv2.goodFeaturesToTrack(
    img1_gray,
    maxCorners=max_features,
    qualityLevel=0.02,  # Lower = faster (default: 0.01)
    minDistance=10,      # Higher = fewer features (default: 7)
    blockSize=5          # Smaller = faster (default: 7)
)
```

### Modify Optical Flow Parameters

```python
p1, st, err = cv2.calcOpticalFlowPyrLK(
    img1_gray, img2_gray, p0, None,
    winSize=(15, 15),    # Smaller = faster (default: 21x21)
    maxLevel=2,          # Fewer levels = faster (default: 3)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02)
)
```

### Enable GPU Acceleration (If Available)

```python
# Check if CUDA is available
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # Use GPU-accelerated optical flow
    # (Requires cv2 built with CUDA support)
    pass
```

## Conclusion

For typical UAV stabilization:
- **Start with**: Fusion model, 1000 features
- **If too slow**: Reduce to 500-700 features
- **If still too slow**: Switch to baseline model
- **Always**: Run benchmark first to know your hardware limits

Monitor performance with `--profile` flag and adjust accordingly.
