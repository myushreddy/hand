# Week 4 Results Reference Guide

## Quick Access to Generated Files

### 📊 Best Performing Model (v3 - Full Dataset)

**Use this for production or further analysis**

```python
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
with open('models/rf_model_ga_week4_full.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Load selected feature names
with open('results/metrics/ga_week4_full_selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

# Load feature selection binary vector
best_chromosome = np.load('results/metrics/ga_week4_full_best_chromosome.npy')

print(f"Model: {rf_model}")
print(f"Features: {len(selected_features)} selected")
print(f"Feature names: {selected_features[:5]}...")  # First 5
```

### 📈 Performance Metrics

```json
{
  "type": "GA_WEEK4_FULL_DATASET",
  "dataset_size": 28752,
  "train_size": 23001,
  "test_size": 5751,
  "final_accuracy": 0.999478,
  "final_precision": 1.0,
  "final_recall": 0.994709,
  "final_f1": 0.997347,
  "final_fpr": 0.0,
  "n_features_selected": 63,
  "n_features_total": 500,
  "feature_reduction_percent": 87.4,
  "generations": 32,
  "ga_runtime_sec": 853.8
}

File: results/metrics/ga_week4_full_metrics.json
```

### 🧬 Feature Selection Vector

```
File: results/metrics/ga_week4_full_best_chromosome.npy
Type: numpy binary array (500,)
Shape: (500,)
Values: 0 or 1 (0=not selected, 1=selected)
Count: 63 ones, 437 zeros

Example:
  chromosome = np.load('results/metrics/ga_week4_full_best_chromosome.npy')
  selected_indices = np.where(chromosome == 1)[0]
  # Returns: array([ 5, 23, 45, ..., 482]) (indices of selected features)
```

### 📋 Selected Feature Names

```
File: results/metrics/ga_week4_full_selected_features.txt
Type: Plain text list (one feature per line)
Format:
  Feature_5
  Feature_23
  Feature_45
  ...
  Feature_482
Count: 63 lines

Pickle alternative:
  with open('results/metrics/ga_week4_full_selected_features.pkl', 'rb') as f:
      features_list = pickle.load(f)
  # Returns: ['Feature_5', 'Feature_23', ..., 'Feature_482']
```

### 📊 Convergence History

```json
{
  "gen": [0, 1, 2, ..., 31],
  "best": [0.9535, 0.9535, ..., 0.9535],
  "avg": [0.8692, 0.8417, ..., 0.8390],
  "nfeat": [63, 63, ..., 63],
  "navg": [84, 82, ..., 79],
  "pm": [0.275, 0.275, ..., 0.275]
}

File: results/metrics/ga_week4_full_history.json
Keys:
  gen:   Generation number [0-31]
  best:  Best fitness in generation
  avg:   Average population fitness
  nfeat: Number of features in best solution
  navg:  Average features in population
  pm:    Average mutation probability
```

### 🖼️ Convergence Visualization

```
File: results/plots/ga_week4_full_convergence.png
Size: 14" x 10" (1400x1000 pixels at 300 dpi)

Subplots:
  [1,1] Fitness convergence (best vs average)
  [1,2] Feature count evolution (best vs average)
  [2,1] Mutation probability by generation
  [2,2] Summary statistics box
```

---

## Comparison: v2 vs v3

### v2: Improved RAM (5k sample subset)
| Metric | Value |
|--------|-------|
| Dataset | 5,000 samples (4k train, 1k test) |
| Features | 44 selected |
| Accuracy | 99.90% |
| Runtime | 109 seconds |
| Files | `*_` (not v3) |

**Files:**
- `results/metrics/ga_week4_selected_features.pkl` (44 features)
- `results/metrics/ga_week4_metrics.json`
- `models/rf_model_ga_week4.pkl`

### v3: Full Dataset (28.7k samples) ⭐ USE THIS
| Metric | Value |
|--------|-------|
| Dataset | 28,752 samples (23k train, 5.7k test) |
| Features | 63 selected |
| Accuracy | 99.95% |
| Runtime | 854 seconds |
| Files | `*_full_` |

**Files:**
- `results/metrics/ga_week4_full_selected_features.pkl` (63 features)
- `results/metrics/ga_week4_full_metrics.json`
- `models/rf_model_ga_week4_full.pkl`

**Recommendation:** Use v3 for all downstream analysis (Week 5+)

---

## How to Use Selected Features

### Option 1: Load from Pickle
```python
import pickle

with open('results/metrics/ga_week4_full_selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

print(f"Selected {len(selected_features)} features:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i}. {feat}")
```

### Option 2: Load from Text File
```python
with open('results/metrics/ga_week4_full_selected_features.txt', 'r') as f:
    selected_features = [line.strip() for line in f]

print(f"Selected {len(selected_features)} features")
```

### Option 3: Use with Pandas
```python
import pandas as pd

# Read full dataset
df = pd.read_csv('data/processed/dataset_with_labels_full.csv')

# Load selected features
with open('results/metrics/ga_week4_full_selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

# Keep only selected features
X = df[selected_features]
y = df['CLASS']

print(f"Reduced dataset: {X.shape[0]} samples x {X.shape[1]} features")
```

### Option 4: Make Predictions
```python
import pickle
import pandas as pd

# Load model
with open('models/rf_model_ga_week4_full.pkl', 'rb') as f:
    rf = pickle.load(f)

# Load features
with open('results/metrics/ga_week4_full_selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

# Load new data
df_new = pd.read_csv('path/to/new_data.csv')

# Make predictions on selected features only
predictions = rf.predict(df_new[selected_features])
probabilities = rf.predict_proba(df_new[selected_features])

print(f"Predictions: {predictions}")
print(f"Confidence: {probabilities[:, 1]}")
```

---

## Performance Breakdown

### Test Set (5,751 samples)
```
Accuracy:   0.999478 (99.9478%)
Precision:  1.000000 (100%)
Recall:     0.994709 (99.47%)
F1:         0.997347 (99.73%)

Confusion Matrix:
  True Positives:  564
  True Negatives:  5,184
  False Positives: 0
  False Negatives: 3

Interpretation:
  ✓ Perfect precision: No false alarms
  ✓ High recall: Catches 99.47% of malware
  ✓ Only 3 misclassified out of 567 positive samples
  ✓ Suitable for security applications
```

### Failure Analysis (3 False Negatives)
```
Total malware samples: 567
Misclassified as benign: 3
Misclassification rate: 0.53%

Question: Which malware samples are these?
Answer: Unknown without detailed error analysis (Week 5 task)

Hypothesis:
  • Edge cases or unusual malware variants
  • Samples with atypical feature combinations
  • Potential novel malware families not in training set
```

---

## Parameter Reference

### GA Configuration
```
Algorithm:               Genetic Algorithm
Fitness function:        Accuracy - Linear Penalty
Population size:         20
Training samples:        23,001
Test samples:            5,751
Total dataset:           28,752
Feature pool:            500 (top MI features)
Random state:            42 (reproducible)
```

### RAM Parameters
```
P_max:                   0.5  (worst performers: 50% mutation)
P_min:                   0.05 (best performers: 5% mutation)
Feature bounds:          10-120 selected features
Elitism:                 Keep top 2 solutions
Penalty target:          40 features
Penalty coefficient:     0.002 (linear)
Stagnation threshold:    15 generations
Early stopping:          Gen > 30 AND no improvement for 15 gens
```

### Random Forest Configuration
```
Algorithm:               Random Forest Classifier
Number of trees:        30
Random state:           42 (reproducible)
Parallel jobs:          1 (serial, memory efficient)
Split criterion:        Gini
Max features:           sqrt(n_features) (auto)
```

---

## Reproducibility

### Full Reproduction Command
```bash
python scripts/week4_ga_ram_full_dataset.py
```

### Parameters to Vary
```python
# In script, modify these to experiment:
POPULATION_SIZE = 20         # Try 10, 30, 50
TOURNAMENT_SIZE = 3          # Try 2, 4, 5
N_GENERATIONS = 100          # Try 50, 200
P_MAX = 0.5                  # Try 0.3, 0.7
P_MIN = 0.05                 # Try 0.01, 0.1
TARGET_FEATURES = 40         # Try 30, 50, 60
RANDOM_STATE = 42            # Change for different runs
N_JOBS = 1                   # Change to -1 for parallel
RF_N_TREES = 30              # Try 50, 100
```

---

## Next Steps (Week 5)

### 1. Load Week 4 Results
```python
import pickle
import numpy as np

# Load everything
with open('models/rf_model_ga_week4_full.pkl', 'rb') as f:
    model = pickle.load(f)
with open('results/metrics/ga_week4_full_selected_features.pkl', 'rb') as f:
    features = pickle.load(f)
best_chromosome = np.load('results/metrics/ga_week4_full_best_chromosome.npy')

print(f"Ready for Week 5 with {len(features)} features and {model}")
```

### 2. Prepare for SHAP Analysis
- Install: `pip install shap`
- Create: `scripts/week5_shap_analysis.py`
- Analyze: Feature importance, SHAP values, interactions

### 3. Prepare for Cross-Validation
- Split data into k-folds
- Run GA on each fold
- Compare feature consistency
- Generate stability metrics

---

## Troubleshooting

### Problem: Model file not found
```python
# Check files exist
import os
assert os.path.exists('models/rf_model_ga_week4_full.pkl'), "Model not found"
assert os.path.exists('results/metrics/ga_week4_full_selected_features.pkl'), "Features not found"
```

### Problem: Incompatible Python version
```
Tested with: Python 3.12
Required packages:
  - pandas >= 1.3
  - numpy >= 1.21
  - scikit-learn >= 1.0
  - matplotlib >= 3.4
  
Check: python --version
```

### Problem: Out of memory
```python
# If you get MemoryError, modify:
FEATURE_POOL_SIZE = 250    # Reduce feature pool
N_JOBS = 1                 # Ensure serial (not parallel)
# Skip plotting for large datasets
```

---

## Summary

**Week 4 delivered:**
- ✅ Feature reduction: 500 → 63 features (87.4%)
- ✅ Test accuracy: 99.9478%
- ✅ Perfect precision: 100% (zero false alarms)
- ✅ High recall: 99.47% (catches almost all malware)
- ✅ Production-ready model with 63-feature set
- ✅ Comprehensive metrics and visualizations

**Status:** Ready for Week 5 (SHAP Interpretability)

**Estimated Week 5 Duration:** 2-3 hours

