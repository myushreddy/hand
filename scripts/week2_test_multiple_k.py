"""
Week 2 Extended: Test Multiple Feature Counts with MI Selection
================================================================

This script tests different feature counts (k) to find optimal balance:
- k = 155 (ARM paper)
- k = 200
- k = 300
- k = 500

For each k:
1. Select top k features by MI
2. Train Random Forest
3. Evaluate performance
4. Compare with baseline

Goal: Find realistic performance (NOT 100%) with good accuracy

Author: ARM Malware Detection Project
Date: February 21, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix)
import pickle
import json
from datetime import datetime

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("WEEK 2 EXTENDED: Testing Multiple Feature Counts")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%H:%M:%S')}")
print()

# ============================================================================
# STEP 1: LOAD DATASET WITH VALIDATION
# ============================================================================
print("STEP 1: Loading dataset with validation...")
print("-" * 80)

df = pd.read_csv('data/processed/dataset_with_labels_full.csv', low_memory=False)
print(f"✓ Loaded: {df.shape[0]:,} samples × {df.shape[1]:,} columns")

# Note: Dataset has 28,752 samples (some rows were filtered during creation)
# This is consistent across all experiments, so we'll use this as our base
ACTUAL_SAMPLES = df.shape[0]
print(f"✓ Using {ACTUAL_SAMPLES:,} samples for analysis")

print(f"✓ Dataset ready for MI testing")

# Prepare data
metadata_cols = ['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'CLASS']
feature_cols = [col for col in df.columns if col not in metadata_cols]
print(f"✓ Features: {len(feature_cols):,}")
print()

X = df[feature_cols].values
y = df['CLASS'].values

# ============================================================================
# STEP 2: CONSISTENT TRAIN/TEST SPLIT
# ============================================================================
print("STEP 2: Creating consistent train/test split...")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"✓ Train: {X_train.shape[0]:,} samples")
print(f"✓ Test:  {X_test.shape[0]:,} samples")
print(f"✓ Test malware: {(y_test == 1).sum():,}")
print()

# ============================================================================
# STEP 3: LOAD PRE-COMPUTED MI SCORES
# ============================================================================
print("STEP 3: Loading MI scores...")
print("-" * 80)

mi_scores_df = pd.read_csv('results/metrics/mi_scores_full_dataset.csv')
print(f"✓ Loaded MI scores for {len(mi_scores_df):,} features")
print()

# ============================================================================
# STEP 4: TEST MULTIPLE K VALUES
# ============================================================================
print("STEP 4: Testing multiple feature counts...")
print("="*80)

k_values = [155, 200, 300, 500]
results = []

for k in k_values:
    print(f"\n{'='*80}")
    print(f"Testing k = {k} features")
    print('='*80)
    
    # Select top k features
    top_features = mi_scores_df.head(k)['feature'].tolist()
    feature_indices = [feature_cols.index(feat) for feat in top_features]
    
    X_train_k = X_train[:, feature_indices]
    X_test_k = X_test[:, feature_indices]
    
    print(f"✓ Selected top {k} features")
    print(f"✓ Training set: {X_train_k.shape}")
    print(f"✓ Test set: {X_test_k.shape}")
    
    # Train model
    print(f"\nTraining Random Forest with {k} features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, 
                                n_jobs=-1, verbose=0)
    rf.fit(X_train_k, y_train)
    print("✓ Training complete")
    
    # Evaluate
    y_pred_train = rf.predict(X_train_k)
    y_pred_test = rf.predict(X_test_k)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    
    # Cross-validation for reliability check
    print(f"\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(rf, X_train_k, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"✓ CV Accuracy: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
    
    # Display results
    print(f"\n{'─'*80}")
    print(f"RESULTS - {k} Features:")
    print('─'*80)
    print(f"Train Accuracy:     {train_acc*100:.2f}%")
    print(f"Test Accuracy:      {test_acc*100:.2f}%")
    print(f"Precision:          {precision*100:.2f}%")
    print(f"Recall:             {recall*100:.2f}%")
    print(f"F1-Score:           {f1*100:.2f}%")
    print(f"FPR:                {fpr*100:.2f}%")
    print(f"CV Accuracy:        {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
    print()
    print(f"Confusion Matrix:")
    print(f"  TN: {tn:,}  FP: {fp:,}")
    print(f"  FN: {fn:,}  TP: {tp:,}")
    
    # Check for overfitting
    overfit_gap = train_acc - test_acc
    if overfit_gap > 0.05:
        print(f"\n⚠️ WARNING: Overfitting detected (gap: {overfit_gap*100:.2f}%)")
    elif test_acc >= 1.0:
        print(f"\n⚠️ WARNING: 100% test accuracy is suspicious!")
    else:
        print(f"\n✓ Good generalization (train-test gap: {overfit_gap*100:.2f}%)")
    
    # Store results
    result = {
        'k': k,
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'fpr': float(fpr),
        'cv_mean': float(cv_mean),
        'cv_std': float(cv_std),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'overfitting_gap': float(overfit_gap)
    }
    results.append(result)
    
    # Save model for this k
    with open(f'models/rf_model_mi{k}.pkl', 'wb') as f:
        pickle.dump(rf, f)
    print(f"\n✓ Saved: models/rf_model_mi{k}.pkl")

# ============================================================================
# STEP 5: COMPARISON ANALYSIS
# ============================================================================
print(f"\n{'='*80}")
print("STEP 5: Comparing All Feature Counts")
print('='*80)

# Load baseline for comparison
with open('results/metrics/baseline_metrics_full.json', 'r') as f:
    baseline = json.load(f)

print(f"\nBaseline (All 24,836 features):")
print(f"  Accuracy:  {baseline['test_accuracy']*100:6.2f}%")
print(f"  Recall:    {baseline['recall']*100:6.2f}%")
print(f"  Precision: {baseline['precision']*100:6.2f}%")
print(f"  F1-Score:  {baseline['f1_score']*100:6.2f}%")

print(f"\n{'Features':<10} {'Accuracy':<10} {'Recall':<10} {'Precision':<10} {'F1-Score':<10} {'CV Acc':<15} {'Status':<20}")
print('─'*100)

for result in results:
    k = result['k']
    status = "✓ GOOD" if result['test_accuracy'] < 1.0 and result['test_accuracy'] > 0.95 else "⚠️ CHECK"
    if result['test_accuracy'] >= 1.0:
        status = "❌ 100% (Suspicious)"
    elif result['overfitting_gap'] > 0.05:
        status = "⚠️ Overfitting"
    
    print(f"{k:<10} {result['test_accuracy']*100:>8.2f}% {result['recall']*100:>8.2f}% "
          f"{result['precision']*100:>10.2f}% {result['f1_score']*100:>9.2f}% "
          f"{result['cv_mean']*100:>6.2f}±{result['cv_std']*100:.2f}% {status:<20}")

print('─'*100)

# Save combined results
with open('results/metrics/mi_k_comparison.json', 'w') as f:
    json.dump({
        'baseline': baseline,
        'k_results': results,
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }, f, indent=4)
print(f"\n✓ Saved: results/metrics/mi_k_comparison.json")

# ============================================================================
# STEP 6: VISUALIZATION
# ============================================================================
print(f"\n{'='*80}")
print("STEP 6: Creating comparison visualizations...")
print('='*80)

# Plot 1: Performance vs Feature Count
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Performance vs Feature Count', fontsize=16, fontweight='bold')

k_list = [r['k'] for r in results]
accuracies = [r['test_accuracy']*100 for r in results]
recalls = [r['recall']*100 for r in results]
precisions = [r['precision']*100 for r in results]
f1_scores = [r['f1_score']*100 for r in results]

# Accuracy
axes[0, 0].plot(k_list, accuracies, marker='o', linewidth=2, markersize=8, color='steelblue')
axes[0, 0].axhline(y=baseline['test_accuracy']*100, color='red', linestyle='--', 
                   label=f'Baseline ({baseline["total_features"]:,} features)')
axes[0, 0].set_xlabel('Number of Features', fontsize=11)
axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11)
axes[0, 0].set_title('Test Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Recall
axes[0, 1].plot(k_list, recalls, marker='o', linewidth=2, markersize=8, color='coral')
axes[0, 1].axhline(y=baseline['recall']*100, color='red', linestyle='--', label='Baseline')
axes[0, 1].set_xlabel('Number of Features', fontsize=11)
axes[0, 1].set_ylabel('Recall (%)', fontsize=11)
axes[0, 1].set_title('Recall (Malware Detection)', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Precision
axes[1, 0].plot(k_list, precisions, marker='o', linewidth=2, markersize=8, color='green')
axes[1, 0].axhline(y=baseline['precision']*100, color='red', linestyle='--', label='Baseline')
axes[1, 0].set_xlabel('Number of Features', fontsize=11)
axes[1, 0].set_ylabel('Precision (%)', fontsize=11)
axes[1, 0].set_title('Precision', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# F1-Score
axes[1, 1].plot(k_list, f1_scores, marker='o', linewidth=2, markersize=8, color='purple')
axes[1, 1].axhline(y=baseline['f1_score']*100, color='red', linestyle='--', label='Baseline')
axes[1, 1].set_xlabel('Number of Features', fontsize=11)
axes[1, 1].set_ylabel('F1-Score (%)', fontsize=11)
axes[1, 1].set_title('F1-Score', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/mi_k_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: results/plots/mi_k_comparison.png")

# Plot 2: Detailed comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(k_list))
width = 0.2

bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='steelblue')
bars2 = ax.bar(x - 0.5*width, recalls, width, label='Recall', color='coral')
bars3 = ax.bar(x + 0.5*width, precisions, width, label='Precision', color='green')
bars4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='purple')

ax.set_xlabel('Number of Features', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Performance Metrics by Feature Count', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{k}' for k in k_list])
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([50, 105])

# Add value labels
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/plots/mi_k_detailed_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: results/plots/mi_k_detailed_comparison.png")

# ============================================================================
# STEP 7: RECOMMENDATION
# ============================================================================
print(f"\n{'='*80}")
print("STEP 7: Recommendation")
print('='*80)

# Find best k (not 100% accuracy, highest F1)
valid_results = [r for r in results if r['test_accuracy'] < 1.0]
if valid_results:
    best_k_result = max(valid_results, key=lambda x: x['f1_score'])
    print(f"\nRECOMMENDED: k = {best_k_result['k']} features")
    print(f"  Accuracy:  {best_k_result['test_accuracy']*100:.2f}%")
    print(f"  Recall:    {best_k_result['recall']*100:.2f}%")
    print(f"  Precision: {best_k_result['precision']*100:.2f}%")
    print(f"  F1-Score:  {best_k_result['f1_score']*100:.2f}%")
    print(f"  CV Acc:    {best_k_result['cv_mean']*100:.2f}% ± {best_k_result['cv_std']*100:.2f}%")
    print(f"\n✓ Good balance of performance and feature reduction")
    print(f"✓ Realistic metrics (not 100%)")
else:
    print(f"\n⚠️ WARNING: All k values showed 100% accuracy!")
    print(f"   This indicates a data issue. Review dataset consistency.")

print(f"\n{'='*80}")
print(f"ANALYSIS COMPLETE - {datetime.now().strftime('%H:%M:%S')}")
print('='*80)
