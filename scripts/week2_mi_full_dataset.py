"""
Week 2: Mutual Information Feature Selection - FULL DATASET
============================================================

This script implements MI feature selection for the full dataset:
1. Loads dataset_with_labels_full.csv (29,915 samples, 24,836 features)
2. Calculates Mutual Information (MI) scores for all 24,836 features
3. Selects top 155 features (as per ARM paper)
4. Trains Random Forest with selected 155 features
5. Compares performance with baseline (all features)
6. Saves results, visualizations, and selected feature list

Goal: Reduce from 24,836 features → 155 features while maintaining performance

Author: ARM Malware Detection Project
Date: February 17, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix)
import pickle
import json
from datetime import datetime
import gc

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("WEEK 2: MUTUAL INFORMATION FEATURE SELECTION - FULL DATASET")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%H:%M:%S')}")
print()

# ============================================================================
# STEP 1: LOAD FULL DATASET WITH VALIDATION
# ============================================================================
print("STEP 1: Loading full dataset...")
print("-" * 80)

df = pd.read_csv('data/processed/dataset_with_labels_full.csv', low_memory=False)
print(f"✓ Initial load: {df.shape[0]:,} samples × {df.shape[1]:,} columns")

# Validate against baseline to ensure consistency
EXPECTED_SAMPLES = 29915
if df.shape[0] != EXPECTED_SAMPLES:
    print(f"⚠️ WARNING: Expected {EXPECTED_SAMPLES:,} samples, got {df.shape[0]:,}")
    print(f"   Difference: {abs(df.shape[0] - EXPECTED_SAMPLES):,} samples")
    
    # Check for NaN values
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"   Found {nan_count:,} NaN values - dropping rows with NaN...")
        df = df.dropna()
        print(f"   After dropna: {df.shape[0]:,} samples")

# CRITICAL: Ensure we have exactly the same samples as baseline
if df.shape[0] != EXPECTED_SAMPLES:
    raise ValueError(f"Dataset size mismatch! Expected {EXPECTED_SAMPLES:,}, got {df.shape[0]:,}")

print(f"✓ Validated: {df.shape[0]:,} samples (matches baseline)")

# Separate features and labels
metadata_cols = ['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'CLASS']
feature_cols = [col for col in df.columns if col not in metadata_cols]
print(f"✓ Feature columns: {len(feature_cols):,}")
print(f"✓ Class distribution: {df['CLASS'].value_counts().to_dict()}")
print()

X = df[feature_cols].values
y = df['CLASS'].values

# ============================================================================
# STEP 2: TRAIN/TEST SPLIT (Same as baseline for fair comparison)
# ============================================================================
print("STEP 2: Creating train/test split...")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"✓ Train: {X_train.shape[0]:,} samples")
print(f"✓ Test:  {X_test.shape[0]:,} samples")
print()

# ============================================================================
# STEP 3: COMPUTE MUTUAL INFORMATION SCORES
# ============================================================================
print("STEP 3: Computing Mutual Information scores for all features...")
print("-" * 80)
print("⏳ This will take several minutes for 24,836 features...")
print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE, 
                                 discrete_features=True, n_neighbors=3)

print(f"✓ MI computation complete: {datetime.now().strftime('%H:%M:%S')}")
print(f"✓ Computed MI for {len(mi_scores):,} features")
print()

# Create MI scores dataframe
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False).reset_index(drop=True)

print("Top 10 features by MI score:")
print(mi_df.head(10).to_string(index=False))
print()

# Save MI scores
mi_df.to_csv('results/metrics/mi_scores_full_dataset.csv', index=False)
print("✓ Saved: results/metrics/mi_scores_full_dataset.csv")
print()

# ============================================================================
# STEP 4: SELECT TOP 155 FEATURES (ARM Paper)
# ============================================================================
print("STEP 4: Selecting top 155 features (as per ARM paper)...")
print("-" * 80)

K = 155
top_features = mi_df.head(K)['feature'].tolist()
print(f"✓ Selected top {K} features")
print(f"✓ MI score range: {mi_df.iloc[0]['mi_score']:.6f} → {mi_df.iloc[K-1]['mi_score']:.6f}")
print()

# Save selected features
with open('models/mi_selected_features_155.pkl', 'wb') as f:
    pickle.dump(top_features, f)

with open('results/metrics/mi_selected_features_155.txt', 'w') as f:
    f.write(f"Top {K} Features Selected by Mutual Information\n")
    f.write("="*80 + "\n\n")
    for i, feat in enumerate(top_features, 1):
        mi_score = mi_df[mi_df['feature'] == feat]['mi_score'].values[0]
        f.write(f"{i:3d}. {feat:50s} (MI: {mi_score:.6f})\n")

print("✓ Saved: models/mi_selected_features_155.pkl")
print("✓ Saved: results/metrics/mi_selected_features_155.txt")
print()

# Get feature indices
feature_indices = [feature_cols.index(feat) for feat in top_features]
X_train_mi = X_train[:, feature_indices]
X_test_mi = X_test[:, feature_indices]

print(f"✓ Reduced training set: {X_train_mi.shape}")
print(f"✓ Reduced test set: {X_test_mi.shape}")
print()

# ============================================================================
# STEP 5: TRAIN RANDOM FOREST WITH 155 FEATURES
# ============================================================================
print("STEP 5: Training Random Forest with 155 MI-selected features...")
print("-" * 80)

rf_mi = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, 
                               n_jobs=-1, verbose=0)
rf_mi.fit(X_train_mi, y_train)
print("✓ Training complete!")
print()

# ============================================================================
# STEP 6: EVALUATE MODEL
# ============================================================================
print("STEP 6: Evaluating model performance...")
print("-" * 80)

y_pred_train = rf_mi.predict(X_train_mi)
y_pred_test = rf_mi.predict(X_test_mi)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn)

print("RESULTS - 155 MI-SELECTED FEATURES:")
print("="*80)
print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy:  {test_acc*100:.2f}%")
print(f"Precision:      {precision*100:.2f}%")
print(f"Recall:         {recall*100:.2f}%")
print(f"F1-Score:       {f1*100:.2f}%")
print(f"FPR:            {fpr*100:.2f}%")
print()
print("Confusion Matrix:")
print(f"  TN: {tn:,}  FP: {fp:,}")
print(f"  FN: {fn:,}  TP: {tp:,}")
print("="*80)
print()

# Save model
with open('models/rf_model_mi155.pkl', 'wb') as f:
    pickle.dump(rf_mi, f)
print("✓ Saved: models/rf_model_mi155.pkl")

# Save metrics
metrics = {
    'dataset': 'full_30k_mi155',
    'feature_selection': 'Mutual Information',
    'total_features': len(feature_cols),
    'selected_features': K,
    'samples': df.shape[0],
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'fpr': float(fpr),
    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
}

with open('results/metrics/mi155_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("✓ Saved: results/metrics/mi155_metrics.json")
print()

# ============================================================================
# STEP 7: COMPARISON WITH BASELINE
# ============================================================================
print("STEP 7: Comparing with baseline (all features)...")
print("-" * 80)

# Load baseline metrics
with open('results/metrics/baseline_metrics_full.json', 'r') as f:
    baseline = json.load(f)

print("PERFORMANCE COMPARISON:")
print("="*80)
print(f"{'Metric':<20} {'Baseline (24,836)':<20} {'MI-155':<20} {'Difference':<15}")
print("-"*80)
print(f"{'Accuracy':<20} {baseline['test_accuracy']*100:>18.2f}% {test_acc*100:>18.2f}% {(test_acc-baseline['test_accuracy'])*100:>13.2f}%")
print(f"{'Recall':<20} {baseline['recall']*100:>18.2f}% {recall*100:>18.2f}% {(recall-baseline['recall'])*100:>13.2f}%")
print(f"{'Precision':<20} {baseline['precision']*100:>18.2f}% {precision*100:>18.2f}% {(precision-baseline['precision'])*100:>13.2f}%")
print(f"{'F1-Score':<20} {baseline['f1_score']*100:>18.2f}% {f1*100:>18.2f}% {(f1-baseline['f1_score'])*100:>13.2f}%")
print(f"{'FPR':<20} {baseline['false_positive_rate']*100:>18.2f}% {fpr*100:>18.2f}% {(fpr-baseline['false_positive_rate'])*100:>13.2f}%")
print("-"*80)
print(f"{'Features':<20} {baseline['total_features']:>18,} {K:>18,} {K-baseline['total_features']:>13,}")
print(f"{'Reduction':<20} {'':>18} {'':>18} {((baseline['total_features']-K)/baseline['total_features'])*100:>12.1f}%")
print("="*80)
print()

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================
print("STEP 8: Creating visualizations...")
print("-" * 80)

# Plot 1: Top 30 MI scores
plt.figure(figsize=(12, 8))
top_30 = mi_df.head(30)
plt.barh(range(30), top_30['mi_score'].values, color='steelblue')
plt.yticks(range(30), top_30['feature'].values, fontsize=8)
plt.xlabel('Mutual Information Score', fontsize=12)
plt.title('Top 30 Features by Mutual Information Score', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('results/plots/mi_top30_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: results/plots/mi_top30_features.png")

# Plot 2: MI score distribution
plt.figure(figsize=(10, 6))
plt.hist(mi_scores, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
plt.axvline(mi_df.iloc[K-1]['mi_score'], color='red', linestyle='--', 
            linewidth=2, label=f'Top {K} threshold')
plt.xlabel('MI Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Mutual Information Scores', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('results/plots/mi_score_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: results/plots/mi_score_distribution.png")

# Plot 3: Confusion Matrix
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Benign', 'Malware'], 
            yticklabels=['Benign', 'Malware'], ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Confusion Matrix - MI-155 Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/confusion_matrix_mi155.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: results/plots/confusion_matrix_mi155.png")

# Plot 4: Performance Comparison
metrics_names = ['Accuracy', 'Recall', 'Precision', 'F1-Score']
baseline_values = [baseline['test_accuracy']*100, baseline['recall']*100, 
                   baseline['precision']*100, baseline['f1_score']*100]
mi_values = [test_acc*100, recall*100, precision*100, f1*100]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline (24,836 features)', 
               color='steelblue')
bars2 = ax.bar(x + width/2, mi_values, width, label='MI-155 features', 
               color='coral')

ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Performance Comparison: Baseline vs MI-155', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/plots/performance_comparison_mi155.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: results/plots/performance_comparison_mi155.png")

# ============================================================================
# SUMMARY
# ============================================================================
print()
print("="*80)
print("WEEK 2 COMPLETE!")
print("="*80)
print(f"End Time: {datetime.now().strftime('%H:%M:%S')}")
print()
print("DELIVERABLES:")
print("  ✓ MI scores for all 24,836 features")
print("  ✓ Top 155 features selected")
print("  ✓ Model trained with MI-selected features")
print("  ✓ Performance comparison completed")
print("  ✓ Visualizations generated")
print()
print("FILES SAVED:")
print("  • results/metrics/mi_scores_full_dataset.csv")
print("  • results/metrics/mi_selected_features_155.txt")
print("  • results/metrics/mi155_metrics.json")
print("  • models/mi_selected_features_155.pkl")
print("  • models/rf_model_mi155.pkl")
print("  • results/plots/mi_top30_features.png")
print("  • results/plots/mi_score_distribution.png")
print("  • results/plots/confusion_matrix_mi155.png")
print("  • results/plots/performance_comparison_mi155.png")
print()
print("FEATURE REDUCTION:")
print(f"  24,836 features → 155 features (99.38% reduction)")
print()
print("Ready for Week 3: GA-RAM Implementation!")
print("="*80)
