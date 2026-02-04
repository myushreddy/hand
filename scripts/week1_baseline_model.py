"""
Week 1: Data Preparation and Baseline Model
============================================

This script:
1. Loads data_sample_25k.csv
2. Merges with labels from mh_100k_labels.csv
3. Performs EDA and preprocessing
4. Creates train/test split (80:20)
5. Trains baseline Random Forest model
6. Saves results and metrics

Author: ARM Malware Detection Project
Date: February 1, 2026 (Recreated after accidental deletion)
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
from datetime import datetime

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create output directories
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("="*80)
print("WEEK 1: DATA PREPARATION & BASELINE MODEL")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("STEP 1: Loading Data...")
print("-" * 80)

# Load the 25k feature dataset
print("Loading data_sample_25k.csv...")
df_features = pd.read_csv('data/data_sample_25k.csv')
print(f"✓ Features loaded: {df_features.shape}")
print(f"  Columns: {df_features.shape[1]}")
print(f"  Rows: {df_features.shape[0]}")

# Load labels
print("\nLoading mh_100k_labels.csv...")
df_labels = pd.read_csv('data/mh_100k_labels.csv')
print(f"✓ Labels loaded: {df_labels.shape}")

# Take only first 25k labels to match our sample
df_labels_25k = df_labels.head(25000).copy()
print(f"✓ Using first 25k labels: {df_labels_25k.shape}")

print("\n" + "="*80)

# ============================================================================
# STEP 2: MERGE DATA WITH LABELS
# ============================================================================
print("STEP 2: Merging Features with Labels...")
print("-" * 80)

# Check if SHA256 exists in both
if 'SHA256' in df_features.columns and 'SHA256' in df_labels_25k.columns:
    print("Merging on SHA256...")
    df_merged = pd.merge(df_features, df_labels_25k[['SHA256', 'CLASS']], 
                         on='SHA256', how='inner')
else:
    # If no SHA256 or merge fails, assume rows are aligned
    print("No SHA256 column found or merge failed. Assuming aligned rows...")
    df_merged = df_features.copy()
    df_merged['CLASS'] = df_labels_25k['CLASS'].values[:len(df_features)]

print(f"✓ Merged dataset: {df_merged.shape}")
print(f"  Features: {df_merged.shape[1] - 1} (excluding CLASS label)")
print(f"  Samples: {df_merged.shape[0]}")

# Save merged dataset
merged_path = 'data/processed/dataset_with_labels.csv'
df_merged.to_csv(merged_path, index=False)
print(f"✓ Saved merged dataset to: {merged_path}")

print("\n" + "="*80)

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("STEP 3: Exploratory Data Analysis (EDA)...")
print("-" * 80)

# Check for CLASS column
if 'CLASS' not in df_merged.columns:
    print("ERROR: CLASS column not found!")
    print("Available columns:", df_merged.columns.tolist())
    exit(1)

# Basic statistics
print("\n📊 Dataset Overview:")
print(f"  Total samples: {len(df_merged)}")
print(f"  Total features: {df_merged.shape[1] - 1}")

# Class distribution
print("\n📊 Class Distribution:")
class_counts = df_merged['CLASS'].value_counts()
print(class_counts)
print(f"\n  Benign (0): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df_merged)*100:.2f}%)")
print(f"  Malware (1): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df_merged)*100:.2f}%)")

# Missing values
print("\n📊 Missing Values:")
missing = df_merged.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values found!")
else:
    print(f"  ⚠ Found {missing.sum()} missing values:")
    print(missing[missing > 0])

# Data types
print("\n📊 Data Types:")
print(df_merged.dtypes.value_counts())

# Identify feature columns (exclude metadata)
metadata_cols = ['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'CLASS']
feature_cols = [col for col in df_merged.columns if col not in metadata_cols]
print(f"\n📊 Feature Columns: {len(feature_cols)}")

# Check if features are binary
print("\n📊 Feature Value Analysis:")
for col in feature_cols[:10]:  # Check first 10 features
    unique_vals = df_merged[col].unique()
    print(f"  {col}: {unique_vals}")

# Save EDA summary
eda_summary = {
    'total_samples': len(df_merged),
    'total_features': len(feature_cols),
    'benign_count': int(class_counts.get(0, 0)),
    'malware_count': int(class_counts.get(1, 0)),
    'missing_values': int(missing.sum()),
    'feature_columns': feature_cols
}

with open('results/metrics/eda_summary.txt', 'w') as f:
    f.write("EDA SUMMARY\n")
    f.write("="*80 + "\n\n")
    for key, value in eda_summary.items():
        if key != 'feature_columns':
            f.write(f"{key}: {value}\n")
    f.write(f"\nFeature columns: {len(feature_cols)} features\n")

print("✓ EDA summary saved to results/metrics/eda_summary.txt")

# Visualizations
print("\n📊 Creating Visualizations...")

# 1. Class distribution plot
plt.figure(figsize=(10, 6))
class_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Class Distribution (0=Benign, 1=Malware)', fontsize=14, fontweight='bold')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
for i, v in enumerate(class_counts):
    plt.text(i, v + 100, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('results/plots/class_distribution.png', dpi=300)
print("  ✓ Saved: results/plots/class_distribution.png")
plt.close()

# 2. Feature sparsity (for first 20 features)
plt.figure(figsize=(14, 6))
feature_sums = df_merged[feature_cols[:20]].sum().sort_values(ascending=False)
feature_sums.plot(kind='bar', color='steelblue')
plt.title('Feature Frequency (Top 20 Features)', fontsize=14, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Count (Presence)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/plots/feature_frequency.png', dpi=300)
print("  ✓ Saved: results/plots/feature_frequency.png")
plt.close()

print("\n" + "="*80)

# ============================================================================
# STEP 4: DATA PREPROCESSING
# ============================================================================
print("STEP 4: Data Preprocessing...")
print("-" * 80)

# Handle missing values (if any)
if df_merged[feature_cols].isnull().sum().sum() > 0:
    print("Filling missing values with 0...")
    df_merged[feature_cols] = df_merged[feature_cols].fillna(0)
    print("✓ Missing values handled")
else:
    print("✓ No missing values to handle")

# Verify data types (features should be numeric)
print("\nConverting features to numeric...")
for col in feature_cols:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0)
print("✓ All features are numeric")

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

print(f"\n✓ Train set: {X_train.shape[0]} samples")
print(f"  - Benign: {(y_train == 0).sum()}")
print(f"  - Malware: {(y_train == 1).sum()}")

print(f"\n✓ Test set: {X_test.shape[0]} samples")
print(f"  - Benign: {(y_test == 0).sum()}")
print(f"  - Malware: {(y_test == 1).sum()}")

# Save train/test split info
split_info = {
    'train_size': X_train.shape[0],
    'test_size': X_test.shape[0],
    'train_benign': int((y_train == 0).sum()),
    'train_malware': int((y_train == 1).sum()),
    'test_benign': int((y_test == 0).sum()),
    'test_malware': int((y_test == 1).sum()),
    'feature_count': len(feature_cols),
    'random_state': RANDOM_STATE
}

import json
with open('results/metrics/train_test_split.json', 'w') as f:
    json.dump(split_info, f, indent=4)

print("\n" + "="*80)

# ============================================================================
# STEP 6: BASELINE RANDOM FOREST MODEL
# ============================================================================
print("STEP 6: Training Baseline Random Forest Model...")
print("-" * 80)

print("Model configuration:")
print("  - Estimators: 100 trees")
print("  - Random state: 42")
print(f"  - All {len(feature_cols)} features")

# Train baseline model
print("\nTraining...")
rf_baseline = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)

rf_baseline.fit(X_train, y_train)
print("✓ Model trained!")

# Predictions
print("\nMaking predictions...")
y_train_pred = rf_baseline.predict(X_train)
y_test_pred = rf_baseline.predict(X_test)

# ============================================================================
# STEP 7: EVALUATION
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Model Evaluation...")
print("-" * 80)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

# Calculate False Positive Rate
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print("\n📊 BASELINE MODEL PERFORMANCE:")
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
print(f"\nTrue Negatives (TN):  {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP):  {tp}")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malware']))

# Save metrics
metrics = {
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

with open('results/metrics/baseline_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("\n✓ Metrics saved to results/metrics/baseline_metrics.json")

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Malware'],
            yticklabels=['Benign', 'Malware'])
plt.title('Confusion Matrix - Baseline Model', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('results/plots/confusion_matrix_baseline.png', dpi=300)
print("✓ Confusion matrix plot saved: results/plots/confusion_matrix_baseline.png")
plt.close()

# Feature Importance
print("\n📊 Top 10 Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_baseline.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))

feature_importance.to_csv('results/metrics/feature_importance_baseline.csv', index=False)
print("\n✓ Feature importance saved: results/metrics/feature_importance_baseline.csv")

# Plot top 20 features
plt.figure(figsize=(12, 6))
top_20 = feature_importance.head(20)
plt.barh(range(len(top_20)), top_20['importance'].values)
plt.yticks(range(len(top_20)), top_20['feature'].values)
plt.xlabel('Importance', fontsize=12)
plt.title('Top 20 Feature Importance - Baseline Model', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('results/plots/feature_importance_baseline.png', dpi=300)
print("✓ Feature importance plot saved: results/plots/feature_importance_baseline.png")
plt.close()

# ============================================================================
# STEP 8: SAVE MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 8: Saving Model...")
print("-" * 80)

model_path = 'models/baseline_rf_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf_baseline, f)

print(f"✓ Baseline model saved to: {model_path}")

# Save feature columns for future use
with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print(f"✓ Feature columns saved to: models/feature_columns.pkl")

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "="*80)
print("✅ WEEK 1 COMPLETE!")
print("="*80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n📁 Generated Files:")
print("  ├── data/processed/dataset_with_labels.csv")
print("  ├── models/baseline_rf_model.pkl")
print("  ├── models/feature_columns.pkl")
print("  ├── results/metrics/eda_summary.txt")
print("  ├── results/metrics/baseline_metrics.json")
print("  ├── results/metrics/feature_importance_baseline.csv")
print("  ├── results/plots/class_distribution.png")
print("  ├── results/plots/feature_frequency.png")
print("  ├── results/plots/confusion_matrix_baseline.png")
print("  └── results/plots/feature_importance_baseline.png")

print("\n📊 Baseline Results Summary:")
print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  - Precision: {test_precision*100:.2f}%")
print(f"  - Recall: {test_recall*100:.2f}%")
print(f"  - F1-Score: {test_f1*100:.2f}%")

print("\n🎯 Next Steps (Week 2):")
print("  1. Implement Mutual Information feature selection")
print("  2. Select top 50-80 features")
print("  3. Retrain model with selected features")
print("  4. Compare with baseline performance")

print("\n" + "="*80)
