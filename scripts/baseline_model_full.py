"""
Baseline Model - Full Dataset Implementation
============================================

This script:
1. Loads 70,000 samples from the full MH-100K dataset
2. Uses ALL feature columns (~24,000+ features)
3. Merges with labels
4. Creates train/test split (80:20)
5. Trains baseline Random Forest model
6. Evaluates and saves results

Author: ARM Malware Detection Project
Date: February 15, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import pickle
import os
import json
from datetime import datetime
import gc

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create output directories
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("="*80)
print("BASELINE MODEL - FULL DATASET (30K SAMPLES)")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: LOAD FULL DATASET (70K SAMPLES)
# ============================================================================
print("STEP 1: Loading Full Dataset...")
print("-" * 80)

print("Loading mh_100k_dataset.csv (this will take a few minutes)...")
print("Reading first 30,000 samples with all features...")
print("Using efficient chunked loading with int8 dtypes...")

# Read CSV with chunking for memory efficiency
# First, read header to get column names
header = pd.read_csv('data/mh_100k_dataset.csv', nrows=0)
columns = header.columns.tolist()

# Identify metadata columns (keep as string/int)
metadata_cols_temp = ['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API']

# Create dtype dictionary - int8 for feature columns, appropriate types for metadata
dtypes = {}
for col in columns:
    if col in metadata_cols_temp:
        if col == 'SHA256':
            dtypes[col] = str
        elif col in ['NOME', 'PACOTE']:
            dtypes[col] = str
        else:
            dtypes[col] = 'int16'
    else:
        # Feature columns are binary 0/1, use int8
        dtypes[col] = 'int8'

print(f"✓ Prepared dtypes for {len(columns)} columns")

# Now read with optimized dtypes - 30K samples for memory efficiency
df_full = pd.read_csv('data/mh_100k_dataset.csv', 
                       nrows=30000, 
                       dtype=dtypes,
                       low_memory=False)

print(f"✓ Dataset loaded: {df_full.shape}")
print(f"  Rows: {df_full.shape[0]:,}")
print(f"  Columns: {df_full.shape[1]:,}")

# Check memory usage
memory_mb = df_full.memory_usage(deep=True).sum() / 1024 / 1024
print(f"  Memory usage: {memory_mb:.2f} MB")

print("\n" + "="*80)

# ============================================================================
# STEP 2: MERGE WITH LABELS
# ============================================================================
print("STEP 2: Merging with Labels...")
print("-" * 80)

# Load labels
print("Loading labels from mh_100k_labels.csv...")
df_labels = pd.read_csv('data/mh_100k_labels.csv')
print(f"✓ Labels loaded: {df_labels.shape}")

# Merge on SHA256
print("\nMerging on SHA256...")
# Keep only SHA256 and CLASS from labels, rename CLASS to avoid conflicts
df_labels_subset = df_labels[['SHA256', 'CLASS']].copy()
df_merged = pd.merge(df_full, df_labels_subset, 
                     on='SHA256', how='inner', suffixes=('', '_label'))

# If CLASS_label exists, use it
if 'CLASS_label' in df_merged.columns:
    df_merged['CLASS'] = df_merged['CLASS_label']
    df_merged = df_merged.drop('CLASS_label', axis=1)

print(f"✓ Merged dataset: {df_merged.shape}")
print(f"  Samples with matching labels: {df_merged.shape[0]:,}")
print(f"✓ CLASS column present: {'CLASS' in df_merged.columns}")

# Free memory
del df_full, df_labels
gc.collect()

print("\n" + "="*80)

# ============================================================================
# STEP 3: IDENTIFY FEATURE COLUMNS
# ============================================================================
print("STEP 3: Identifying Features...")
print("-" * 80)

# Metadata columns to exclude
metadata_cols = ['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API']

# Feature columns are everything except metadata and CLASS
feature_cols = [col for col in df_merged.columns 
                if col not in metadata_cols and col != 'CLASS']

print(f"✓ Total feature columns: {len(feature_cols):,}")
print(f"  (Permissions + API Calls + Intents)")

# Check for missing values in features
missing_count = df_merged[feature_cols].isnull().sum().sum()
print(f"\n✓ Missing values in features: {missing_count:,}")

if missing_count > 0:
    print("  Filling missing values with 0...")
    df_merged[feature_cols] = df_merged[feature_cols].fillna(0)

# Features already loaded as int8, confirm memory usage
memory_mb_after = df_merged.memory_usage(deep=True).sum() / 1024 / 1024
print(f"✓ Memory usage: {memory_mb_after:.2f} MB")

print("\n" + "="*80)

# ============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("STEP 4: Exploratory Data Analysis...")
print("-" * 80)

# Class distribution
class_counts = df_merged['CLASS'].value_counts().sort_index()
print("\nClass Distribution:")
print(f"  Benign (0): {class_counts[0]:,} ({class_counts[0]/len(df_merged)*100:.2f}%)")
print(f"  Malware (1): {class_counts[1]:,} ({class_counts[1]/len(df_merged)*100:.2f}%)")

# Save EDA summary
eda_summary = {
    'total_samples': len(df_merged),
    'total_features': len(feature_cols),
    'benign_count': int(class_counts[0]),
    'malware_count': int(class_counts[1]),
    'benign_percentage': float(class_counts[0]/len(df_merged)*100),
    'malware_percentage': float(class_counts[1]/len(df_merged)*100)
}

with open('results/metrics/eda_summary_full.txt', 'w') as f:
    f.write("EDA SUMMARY - FULL DATASET\n")
    f.write("="*80 + "\n\n")
    for key, value in eda_summary.items():
        f.write(f"{key}: {value}\n")

print("✓ EDA summary saved to: results/metrics/eda_summary_full.txt")

# Visualize class distribution
plt.figure(figsize=(10, 6))
colors = ['#2ecc71', '#e74c3c']
class_counts.plot(kind='bar', color=colors)
plt.title('Class Distribution - Full Dataset (0=Benign, 1=Malware)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Class', fontsize=13)
plt.ylabel('Count', fontsize=13)
plt.xticks(rotation=0)
for i, v in enumerate(class_counts):
    plt.text(i, v + 100, f'{v:,}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/class_distribution_full.png', dpi=300)
print("✓ Class distribution plot saved")
plt.close()

print("\n" + "="*80)

# ============================================================================
# STEP 5: TRAIN/TEST SPLIT
# ============================================================================
print("STEP 5: Creating Train/Test Split (80:20)...")
print("-" * 80)

# Prepare X and y
X = df_merged[feature_cols].values
y = df_merged['CLASS'].values

print(f"Feature matrix X: {X.shape}")
print(f"Label vector y: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\n✓ Train set: {X_train.shape[0]:,} samples")
print(f"  - Benign: {(y_train == 0).sum():,}")
print(f"  - Malware: {(y_train == 1).sum():,}")

print(f"\n✓ Test set: {X_test.shape[0]:,} samples")
print(f"  - Benign: {(y_test == 0).sum():,}")
print(f"  - Malware: {(y_test == 1).sum():,}")

# Save train/test split info
split_info = {
    'train_size': int(X_train.shape[0]),
    'test_size': int(X_test.shape[0]),
    'train_benign': int((y_train == 0).sum()),
    'train_malware': int((y_train == 1).sum()),
    'test_benign': int((y_test == 0).sum()),
    'test_malware': int((y_test == 1).sum()),
    'feature_count': len(feature_cols),
    'random_state': RANDOM_STATE
}

with open('results/metrics/train_test_split_full.json', 'w') as f:
    json.dump(split_info, f, indent=4)

print("\n" + "="*80)

# ============================================================================
# STEP 6: TRAIN BASELINE RANDOM FOREST MODEL
# ============================================================================
print("STEP 6: Training Baseline Random Forest Model...")
print("-" * 80)

print("Model configuration:")
print(f"  - Estimators: 100 trees")
print(f"  - Random state: {RANDOM_STATE}")
print(f"  - All {len(feature_cols):,} features")
print(f"  - Parallel jobs: -1 (all cores)")

# Train baseline model
print("\nTraining... (this may take 5-10 minutes)")
rf_baseline = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

rf_baseline.fit(X_train, y_train)
print("✓ Model trained!")

# Predictions
print("\nMaking predictions...")
y_train_pred = rf_baseline.predict(X_train)
y_test_pred = rf_baseline.predict(X_test)

print("\n" + "="*80)

# ============================================================================
# STEP 7: EVALUATION
# ============================================================================
print("STEP 7: Model Evaluation...")
print("-" * 80)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print("\n📊 BASELINE MODEL PERFORMANCE - FULL DATASET:")
print("="*80)
print(f"Training Accuracy:   {train_accuracy*100:.2f}%")
print(f"Testing Accuracy:    {test_accuracy*100:.2f}%")
print(f"Precision:           {test_precision*100:.2f}%")
print(f"Recall:              {test_recall*100:.2f}%")
print(f"F1-Score:            {test_f1*100:.2f}%")
print(f"False Positive Rate: {fpr*100:.2f}%")
print("="*80)

# Confusion Matrix
print("\n📊 Confusion Matrix:")
print(cm)
print(f"\nTrue Negatives (TN):  {tn:,}")
print(f"False Positives (FP): {fp:,}")
print(f"False Negatives (FN): {fn:,}")
print(f"True Positives (TP):  {tp:,}")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malware']))

# Save metrics
metrics = {
    'dataset': 'full_70k',
    'total_samples': len(df_merged),
    'total_features': len(feature_cols),
    'train_accuracy': float(train_accuracy),
    'test_accuracy': float(test_accuracy),
    'precision': float(test_precision),
    'recall': float(test_recall),
    'f1_score': float(test_f1),
    'false_positive_rate': float(fpr),
    'confusion_matrix': {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }
}

with open('results/metrics/baseline_metrics_full.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("\n✓ Metrics saved to results/metrics/baseline_metrics_full.json")

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', 
            xticklabels=['Benign', 'Malware'],
            yticklabels=['Benign', 'Malware'])
plt.title('Confusion Matrix - Baseline Model (Full Dataset)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('results/plots/confusion_matrix_baseline_full.png', dpi=300)
print("✓ Confusion matrix plot saved")
plt.close()

print("\n" + "="*80)

# ============================================================================
# STEP 8: SAVE MODEL AND FEATURES
# ============================================================================
print("STEP 8: Saving Model and Features...")
print("-" * 80)

# Save model
model_path = 'models/baseline_rf_model_full.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf_baseline, f)
print(f"✓ Baseline model saved to: {model_path}")

# Save feature columns
with open('models/feature_columns_full.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"✓ Feature columns saved to: models/feature_columns_full.pkl")

# Save processed dataset (optional - large file)
print("\nSaving processed dataset...")
df_merged.to_csv('data/processed/dataset_with_labels_full.csv', index=False)
print("✓ Processed dataset saved to: data/processed/dataset_with_labels_full.csv")

print("\n" + "="*80)

# ============================================================================
# COMPLETION
# ============================================================================
print("✅ BASELINE MODEL COMPLETE - FULL DATASET!")
print("="*80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n📊 Results Summary:")
print(f"  - Samples used: {len(df_merged):,}")
print(f"  - Features: {len(feature_cols):,}")
print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  - Precision: {test_precision*100:.2f}%")
print(f"  - Recall: {test_recall*100:.2f}%")
print(f"  - F1-Score: {test_f1*100:.2f}%")

print("\n📁 Generated Files:")
print("  ├── data/processed/dataset_with_labels_full.csv")
print("  ├── models/baseline_rf_model_full.pkl")
print("  ├── models/feature_columns_full.pkl")
print("  ├── results/metrics/eda_summary_full.txt")
print("  ├── results/metrics/baseline_metrics_full.json")
print("  ├── results/metrics/train_test_split_full.json")
print("  ├── results/plots/class_distribution_full.png")
print("  └── results/plots/confusion_matrix_baseline_full.png")

print("\n🚀 Next Step:")
print("  Run MI feature selection with: python scripts/mi_feature_selection_full.py")

print("\n" + "="*80)
