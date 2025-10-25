# Changes Summary

## Overview

Updated the codebase to support sensor data splitting and generalized model inference. The changes enable:
1. Splitting sensor data (CSV files) alongside image data
2. A model interface that allows any stabilization model to use both images and sensor data
3. Generalized inference pipeline that works with any model implementing the interface

## Changes Made

### 1. Data Splitting (`src/split_data.py`)

**New Features:**
- `--split-csv` flag: Creates split-specific CSV files for sensor data
- `--labels-dir` argument: Specifies the labels directory (default: `data/labels`)
- `save_split_csv_files()`: New function that filters CSV rows based on frame_id and saves to split directories

**Output Structure:**
```
data/
├── data_split.json          # Contains split configuration + labels_split_dir metadata
└── labels_split/            # New directory created with --split-csv
    ├── train/
    │   ├── Flight1.csv
    │   ├── Flight2.csv
    │   └── Flight3.csv
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

**Usage:**
```bash
# Create split with CSV files
python src/split_data.py --strategy temporal --ratios 0.8 0.1 0.1 --split-csv

# Create split without CSV files (original behavior)
python src/split_data.py --strategy temporal --ratios 0.8 0.1 0.1
```

### 2. Model Interface (`src/model/`)

**New Files:**
- `src/model/base.py`: Defines `StabilizationModel` abstract base class
- `src/model/__init__.py`: Exports model classes

**Updated Files:**
- `src/model/baseline.py`: Added `BaselineModel` class that implements the interface

**Interface Definition:**
```python
class StabilizationModel(ABC):
    @abstractmethod
    def stabilize_frames(
        self,
        frames: List[np.ndarray],
        sensor_data: Optional[List[Dict]] = None,
        ref_idx: Optional[int] = None
    ) -> Dict:
        """
        Returns dict with keys:
        - "warped": list of stabilized frames
        - "orig": original frames
        - "scales": list of scales
        - "translations": list of (tx, ty) tuples
        - "rotations": list of rotation angles
        - "transforms": list of 3x3 transformation matrices
        - "inliers": list of inlier masks (if applicable)
        - "ref_idx": reference frame index
        """
        pass
```

**Sensor Data Format:**
Each sensor data dict contains:
- `qw, qx, qy, qz`: Quaternion orientation
- `wx_radDs, wy_radDs, wz_radDs`: Angular rates (rad/s)
- `ax_mDs2, ay_mDs2, az_mDs2`: Accelerations (m/s²)
- `timestamp`: Frame timestamp
- Additional fields from CSV

### 3. Generalized Inference (`src/inference.py`)

**New Features:**
- `--model` argument: Select model (default: `baseline`)
- `--use-sensor-data` flag: Load and pass sensor data to model
- `load_sensor_data_cache()`: Loads CSV files into memory for fast lookup
- `get_sensor_data_for_frames()`: Retrieves sensor data for specific frames

**Key Changes:**
- Model instantiation based on `--model` argument
- Sensor data loading with automatic split-aware CSV file selection
- Model interface usage: `model.stabilize_frames(frames, sensor_data, ref_idx)`

**Usage:**
```bash
# Run inference with baseline (no sensor data)
python src/inference.py --split-set test

# Run inference with sensor data
python src/inference.py --split-set test --use-sensor-data

# Run inference with custom model (when available)
python src/inference.py --split-set test --model your_model --use-sensor-data
```

## Creating a New Model

To create a new model that uses sensor data:

1. Create a new file in `src/model/`, e.g., `src/model/your_model.py`

2. Implement the `StabilizationModel` interface:

```python
from .base import StabilizationModel
from typing import Dict, List, Optional
import numpy as np

class YourModel(StabilizationModel):
    def __init__(self, **kwargs):
        # Initialize your model
        pass

    def stabilize_frames(
        self,
        frames: List[np.ndarray],
        sensor_data: Optional[List[Dict]] = None,
        ref_idx: Optional[int] = None
    ) -> Dict:
        # Your stabilization logic here
        # Access sensor data: sensor_data[i]['qw'], sensor_data[i]['wx_radDs'], etc.

        # Return dict with required keys
        return {
            "warped": warped_frames,
            "orig": frames,
            "scales": scales,
            "translations": translations,
            "rotations": rotations,
            "transforms": transform_matrices,
            "inliers": inliers,
            "ref_idx": ref_idx
        }
```

3. Register your model in `src/model/__init__.py`:

```python
from .your_model import YourModel
__all__ = ['StabilizationModel', 'BaselineModel', 'YourModel']
```

4. Add model selection in `src/inference.py`:

```python
# In main(), after argument parsing:
if args.model == 'baseline':
    model = BaselineModel()
elif args.model == 'your_model':
    model = YourModel()
else:
    raise ValueError(f"Unknown model: {args.model}")
```

5. Update the `--model` choices in the argument parser:

```python
parser.add_argument(
    '--model',
    type=str,
    default='baseline',
    choices=['baseline', 'your_model'],
    help='Model to use for stabilization'
)
```

## Example Workflow

### With Data Splits (Recommended)

```bash
# 1. Create data split with sensor CSV files
python src/split_data.py --strategy temporal --ratios 0.8 0.1 0.1 --split-csv

# 2. Run inference on test set with sensor data
python src/inference.py --split-set test --use-sensor-data --output-name baseline_with_imu

# 3. Evaluate results
python src/evaluate.py --model original --split-set test  # Compute once
python src/evaluate.py --model baseline_with_imu

# 4. Visualize
python app.py
```

### Without Data Splits

```bash
# 1. Run inference on all data
python src/inference.py --split "" --use-sensor-data

# 2. Evaluate
python src/evaluate.py --model original --split ""
python src/evaluate.py --model baseline --split ""
```

## Backward Compatibility

All changes are backward compatible:
- `split_data.py` still creates `data_split.json` without `--split-csv`
- `inference.py` works without `--use-sensor-data` (baseline doesn't use it)
- Existing `stabilize_frames()` function still available for direct use
- Models can ignore sensor_data if not needed (like BaselineModel)

## Testing

The implementation has been tested:
- ✓ Data splitting with CSV files
- ✓ Model interface with and without sensor data
- ✓ Callable model interface
- ✓ Sensor data loading from split directories
- ✓ Backward compatibility with existing code
