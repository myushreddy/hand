"""
Diagnostic Script - Investigate 100% Accuracy Issue
===================================================

This script investigates why MI-155 model achieved 100% accuracy.
Checks for:
1. Data loading consistency
2. Train/test split differences
3. Potential data leakage
4. Feature selection issues
"""

import pandas as pd
import numpy as np
import json

print("="*80)
print("DIAGNOSTIC: Investigating 100% Accuracy Issue")
print("="*80)
print()

# ============================================================================
# 1. CHECK DATASET LOADING
# ============================================================================
print("1. CHECKING DATASET CONSISTENCY")
print("-" * 80)

print("Loading dataset_with_labels_full.csv...")
df = pd.read_csv('data/processed/dataset_with_labels_full.csv', low_memory=False)
print(f"✓ Total rows: {len(df):,}")
print(f"✓ Total columns: {df.shape[1]}")
print()

print("Class Distribution:")
class_dist = df['CLASS'].value_counts().sort_index()
print(f"  Benign (0): {class_dist.get(0.0, 0):,} ({class_dist.get(0.0, 0)/len(df)*100:.2f}%)")
print(f"  Malware (1): {class_dist.get(1.0, 0):,} ({class_dist.get(1.0, 0)/len(df)*100:.2f}%)")
print()

# Check for duplicates
print("Checking for duplicate rows...")
duplicates = df.duplicated(subset=[col for col in df.columns if col != 'SHA256']).sum()
print(f"  Duplicate rows: {duplicates}")
print()

# Check for missing values
print("Checking for missing values...")
missing = df.isnull().sum().sum()
print(f"  Total missing values: {missing}")
print()

# ============================================================================
# 2. COMPARE WITH BASELINE METRICS
# ============================================================================
print("2. COMPARING WITH BASELINE MODEL RESULTS")
print("-" * 80)

with open('results/metrics/baseline_metrics_full.json', 'r') as f:
    baseline = json.load(f)

with open('results/metrics/mi155_metrics.json', 'r') as f:
    mi155 = json.load(f)

print("Baseline Model:")
print(f"  Total samples: {baseline['total_samples']:,}")
print(f"  Features: {baseline['total_features']:,}")
print(f"  Accuracy: {baseline['test_accuracy']*100:.2f}%")
print(f"  Recall: {baseline['recall']*100:.2f}%")
print(f"  TP: {baseline['confusion_matrix']['tp']}, FN: {baseline['confusion_matrix']['fn']}")
print()

print("MI-155 Model:")
print(f"  Total samples: {mi155['samples']:,}")
print(f"  Features: {mi155['selected_features']}")
print(f"  Accuracy: {mi155['test_accuracy']*100:.2f}%")
print(f"  Recall: {mi155['recall']*100:.2f}%")
print(f"  TP: {mi155['confusion_matrix']['tp']}, FN: {mi155['confusion_matrix']['fn']}")
print()

print("⚠️ DISCREPANCY DETECTED:")
sample_diff = baseline['total_samples'] - mi155['samples']
print(f"  Sample count difference: {sample_diff:,} samples")
if sample_diff != 0:
    print(f"  ❌ Models used DIFFERENT datasets!")
    print(f"     Baseline: {baseline['total_samples']:,} samples")
    print(f"     MI-155:   {mi155['samples']:,} samples")
print()

# ============================================================================
# 3. CHECK TRAIN/TEST SPLIT REPRODUCIBILITY
# ============================================================================
print("3. TESTING TRAIN/TEST SPLIT REPRODUCIBILITY")
print("-" * 80)

from sklearn.model_selection import train_test_split

metadata_cols = ['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'CLASS']
feature_cols = [col for col in df.columns if col not in metadata_cols]

X = df[feature_cols].values
y = df['CLASS'].values

# Split 1
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split 2 (should be identical)
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Split 1 - Train: {len(y_train1):,}, Test: {len(y_test1):,}")
print(f"Split 2 - Train: {len(y_train2):,}, Test: {len(y_test2):,}")

# Check if splits are identical
identical = np.array_equal(y_test1, y_test2)
print(f"Splits identical: {'✓ YES' if identical else '✗ NO (PROBLEM!)'}")

# Count malware in test sets
malware_test1 = (y_test1 == 1).sum()
malware_test2 = (y_test2 == 1).sum()
print(f"Malware in test set: {malware_test1:,}")
print()

# ============================================================================
# 4. CHECK FOR DATA LEAKAGE
# ============================================================================
print("4. CHECKING FOR POTENTIAL DATA LEAKAGE")
print("-" * 80)

# Load MI selected features
import pickle
with open('models/mi_selected_features_155.pkl', 'rb') as f:
    selected_features = pickle.load(f)

print(f"Selected 155 features loaded")
print()

# Check if any metadata leaked
suspicious_features = []
for feat in selected_features:
    if 'CLASS' in feat.upper() or 'LABEL' in feat.upper() or 'MALWARE' in feat.upper():
        suspicious_features.append(feat)

if suspicious_features:
    print("⚠️ SUSPICIOUS FEATURES FOUND:")
    for feat in suspicious_features:
        print(f"  - {feat}")
else:
    print("✓ No obvious label leakage in feature names")
print()

# Check MI scores for abnormally high values
mi_scores_df = pd.read_csv('results/metrics/mi_scores_full_dataset.csv')
top_10_mi = mi_scores_df.head(10)

print("Top 10 MI Scores:")
for idx, row in top_10_mi.iterrows():
    print(f"  {row['feature'][:50]:50s} - {row['mi_score']:.6f}")
print()

# Check for perfect correlation
if top_10_mi['mi_score'].iloc[0] > 0.9:
    print("⚠️ WARNING: Very high MI scores detected (>0.9)")
    print("   This suggests potential data leakage or perfect correlation")
else:
    print("✓ MI scores look reasonable")
print()

# ============================================================================
# 5. RECOMMENDATIONS
# ============================================================================
print("5. DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
print("="*80)

issues_found = []

if sample_diff != 0:
    issues_found.append(f"Different sample counts ({sample_diff:,} difference)")

if mi155['test_accuracy'] >= 1.0:
    issues_found.append("Perfect 100% accuracy (suspicious)")

if mi155['recall'] >= 1.0 and mi155['precision'] >= 1.0:
    issues_found.append("Perfect recall AND precision (very suspicious)")

if issues_found:
    print("⚠️ ISSUES FOUND:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
    print()
    print("RECOMMENDATIONS:")
    print("  1. Ensure both models use EXACT same dataset")
    print("  2. Use same train/test split (save indices and reuse)")
    print("  3. Test with more features (200, 300, 500) to check overfitting")
    print("  4. Perform 5-fold cross-validation")
    print("  5. Test on completely held-out validation set")
else:
    print("✓ No obvious issues detected")
    print("  However, 100% accuracy is still unusual - recommend cross-validation")

print()
print("="*80)
