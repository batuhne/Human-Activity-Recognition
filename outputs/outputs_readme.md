# Outputs Directory Guide

This directory contains all generated outputs from data preprocessing and analysis.

## Folder Structure

```
outputs/
├── processed/
│   ├── feature_groups.json         # Feature categorization by sensor type and domain
│   ├── ucihar_70_10_20_std.npz    # Standardized 70/10/20 train/val/test split
│   └── ucihar_80_10_10_std.npz    # Standardized 80/10/10 train/val/test split
├── 3d_scatter_plot.png            # 3D visualization of activity clusters
├── algorithm_comparison.png        # Performance comparison across algorithms
├── confusion_matrix.png            # Model confusion matrix heatmap
├── data_splitting.png              # Train/validation/test distribution visualization
└── outputs_readme.md               # This file
```

## Files Description

### Processed Data

#### 1) `processed/ucihar_70_10_20_std.npz`
- **Purpose:** Z-score normalized data split into **70% train / 10% validation / 20% test**
- **Contents:**
  - `X_train, X_val, X_test` → Feature matrices of shape (num_samples, 561)
  - `y_train, y_val, y_test` → Activity labels (string names)
  - `feature_names` → List of all 561 feature column names
- **When to use:** Standard experimental setup; commonly used ratio in research papers

#### 2) `processed/ucihar_80_10_10_std.npz`
- **Purpose:** Alternative split with **80% train / 10% validation / 10% test**
- **Contents:** Same structure as above
- **When to use:** When you need more training data or want to compare split ratios

#### 3) `processed/feature_groups.json`
- **Purpose:** Categorization of 561 features by sensor type and signal domain
- **Key groups:**
  - `time_all` → All time-domain features (265 features)
  - `freq_all` → All frequency-domain features (289 features)
  - `acc_all` → All accelerometer-derived features (345 features)
  - `gyro_all` → All gyroscope-derived features (213 features)
  - `time_body_acc_xyz`, `freq_body_gyro_xyz`, etc. → Specific sensor-axis combinations
- **When to use:** For sensor-specific experiments (e.g., accelerometer-only, time-domain-only analysis)

## Usage Examples

### Loading Processed Data

```python
import numpy as np

# Load the 70/10/20 split
data = np.load('outputs/processed/ucihar_70_10_20_std.npz', allow_pickle=True)

X_train = data['X_train']  # Shape: (7209, 561)
y_train = data['y_train']  # Shape: (7209,)
X_val = data['X_val']      # Shape: (1029, 561)
y_val = data['y_val']      # Shape: (1029,)
X_test = data['X_test']    # Shape: (2061, 561)
y_test = data['y_test']    # Shape: (2061,)

feature_names = data['feature_names']  # List of 561 feature names
```

### Loading Feature Groups

```python
import json

with open('outputs/processed/feature_groups.json', 'r') as f:
    groups = json.load(f)

# Access specific feature groups
acc_features = groups['acc_all']      # All accelerometer features
gyro_features = groups['gyro_all']    # All gyroscope features
time_features = groups['time_all']    # All time-domain features
freq_features = groups['freq_all']    # All frequency-domain features
```

## Notes

- All data files use **StandardScaler** normalization (zero mean, unit variance)
- Stratified splitting ensures balanced class distribution across all splits
- Random seed is fixed to **42** for reproducibility
- Feature names are cleaned versions from the original UCI HAR Dataset
