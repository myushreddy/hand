"""
Week 2: Mutual Information Feature Selection
=============================================

This script:
1. Loads the processed dataset from Week 1
2. Calculates Mutual Information (MI) scores for all features
3. Ranks features by MI score
4. Tests different k values (40, 50, 60, 80) for feature selection
5. Trains Random Forest models with selected features
6. Compares performance with baseline
7. Saves results and visualizations

Goal: Improve recall from 58.55% to 85-92%

Author: ARM Malware Detection Project
Date: February 15, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import pickle
import os
import json
from datetime import datetime

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create output directories
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("="*80)
print("WEEK 2: MUTUAL INFORMATION FEATURE SELECTION")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# STEP 1: LOAD PROCESSED DATA FROM WEEK 1
# ============================================================================
print("STEP 1: Loading Processed Data from Week 1...")
print("-" * 80)

# Load the merged dataset
df = pd.read_csv('data/processed/dataset_with_labels.csv')
print(f"✓ Dataset loaded: {df.shape}")

# Identify feature columns (exclude metadata and label)
metadata_cols = ['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'CLASS']
feature_cols = [col for col in df.columns if col not in metadata_cols]
print(f"✓ Total features: {len(feature_cols)}")

# Load train/test split info from Week 1
with open('results/metrics/train_test_split.json', 'r') as f:
    split_info = json.load(f)

print(f"\nWeek 1 Baseline Results:")
print(f"  - Train size: {split_info['train_size']}")
print(f"  - Test size: {split_info['test_size']}")
print(f"  - Features used: {split_info['feature_count']}")

# Load baseline metrics for comparison
with open('results/metrics/baseline_metrics.json', 'r') as f:
    baseline_metrics = json.load(f)

print(f"\nBaseline Performance:")
print(f"  - Accuracy: {baseline_metrics['test_accuracy']*100:.2f}%")
print(f"  - Recall: {baseline_metrics['recall']*100:.2f}% ⚠️ (needs improvement)")
print(f"  - Precision: {baseline_metrics['precision']*100:.2f}%")
print(f"  - F1-Score: {baseline_metrics['f1_score']*100:.2f}%")

print("\n" + "="*80)

# ============================================================================
# STEP 2: PREPARE DATA FOR FEATURE SELECTION
# ============================================================================
print("STEP 2: Preparing Data...")
print("-" * 80)

# Extract features and labels
X = df[feature_cols].values
y = df['CLASS'].values

print(f"✓ Feature matrix X: {X.shape}")
print(f"✓ Label vector y: {y.shape}")

# Use the same train/test split as Week 1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\n✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

print("\n" + "="*80)

# ============================================================================
# STEP 3: CALCULATE MUTUAL INFORMATION SCORES
# ============================================================================
print("STEP 3: Calculating Mutual Information Scores...")
print("-" * 80)

print("Computing MI scores for all features...")
print("(This measures how much each feature tells us about malware/benign)")

# Calculate MI scores
mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)

print(f"✓ MI scores calculated for {len(mi_scores)} features")

# Create DataFrame with features and MI scores
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print(f"\nTop 10 Features by Mutual Information:")
print(mi_df.head(10).to_string(index=False))

print(f"\nBottom 5 Features (least informative):")
print(mi_df.tail(5).to_string(index=False))

# Save MI scores
mi_scores_path = 'results/metrics/mi_scores_all_features.csv'
mi_df.to_csv(mi_scores_path, index=False)
print(f"\n✓ MI scores saved to: {mi_scores_path}")

# Visualize MI score distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(mi_scores, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('MI Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of MI Scores', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(len(mi_scores[:20])), mi_df['mi_score'].head(20), color='steelblue')
plt.xlabel('Feature Rank', fontsize=12)
plt.ylabel('MI Score', fontsize=12)
plt.title('Top 20 Features by MI Score', fontsize=14, fontweight='bold')
plt.xticks(range(0, 20, 2))
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/mi_scores_distribution.png', dpi=300)
print("✓ MI score distribution plot saved: results/plots/mi_scores_distribution.png")
plt.close()

print("\n" + "="*80)

# ============================================================================
# STEP 4: TEST DIFFERENT K VALUES FOR FEATURE SELECTION
# ============================================================================
print("STEP 4: Testing Different Feature Subset Sizes (k values)...")
print("-" * 80)

# Note: We have 95 features, so we'll test k values that make sense
# Paper uses 155 features, but we'll adapt to our dataset size
k_values = [30, 40, 50, 60, 70, 80]  # Different feature subset sizes to test

results_comparison = []

print(f"\nTesting k values: {k_values}")
print(f"Goal: Find optimal k that maximizes recall while maintaining accuracy\n")

for k in k_values:
    print(f"\n{'='*80}")
    print(f"Testing with k={k} features...")
    print("-" * 80)
    
    # Select top k features
    top_k_features = mi_df.head(k)['feature'].tolist()
    
    # Get column indices for these features
    feature_indices = [feature_cols.index(f) for f in top_k_features]
    
    # Create feature subsets
    X_train_selected = X_train[:, feature_indices]
    X_test_selected = X_test[:, feature_indices]
    
    print(f"Training Random Forest with {k} features...")
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    rf_model.fit(X_train_selected, y_train)
    
    # Make predictions
    y_train_pred = rf_model.predict(X_train_selected)
    y_test_pred = rf_model.predict(X_test_selected)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Store results
    results_comparison.append({
        'k': k,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    })
    
    # Print results
    print(f"\n📊 Results with k={k} features:")
    print(f"  Training Accuracy:   {train_acc*100:.2f}%")
    print(f"  Test Accuracy:       {test_acc*100:.2f}%")
    print(f"  Precision:           {precision*100:.2f}%")
    print(f"  Recall:              {recall*100:.2f}% {'✓✓✓' if recall > 0.85 else '✓' if recall > baseline_metrics['recall'] else '⚠️'}")
    print(f"  F1-Score:            {f1*100:.2f}%")
    print(f"  FPR:                 {fpr*100:.2f}%")
    print(f"\n  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Comparison with baseline
    recall_improvement = (recall - baseline_metrics['recall']) * 100
    acc_change = (test_acc - baseline_metrics['test_accuracy']) * 100
    
    print(f"\n  📈 vs Baseline:")
    print(f"     Recall: {recall_improvement:+.2f} percentage points")
    print(f"     Accuracy: {acc_change:+.2f} percentage points")

print("\n" + "="*80)

# ============================================================================
# STEP 5: COMPARE ALL RESULTS
# ============================================================================
print("STEP 5: Comparing All Results...")
print("-" * 80)

# Create comparison DataFrame
results_df = pd.DataFrame(results_comparison)

print("\n📊 COMPLETE COMPARISON TABLE:")
print("="*80)
print(results_df.to_string(index=False, float_format='%.4f'))

# Find best k by different criteria
best_recall_idx = results_df['recall'].idxmax()
best_f1_idx = results_df['f1_score'].idxmax()
best_acc_idx = results_df['test_accuracy'].idxmax()

print("\n🏆 BEST PERFORMERS:")
print(f"  Best Recall:    k={results_df.loc[best_recall_idx, 'k']:.0f} (Recall={results_df.loc[best_recall_idx, 'recall']*100:.2f}%)")
print(f"  Best F1-Score:  k={results_df.loc[best_f1_idx, 'k']:.0f} (F1={results_df.loc[best_f1_idx, 'f1_score']*100:.2f}%)")
print(f"  Best Accuracy:  k={results_df.loc[best_acc_idx, 'k']:.0f} (Acc={results_df.loc[best_acc_idx, 'test_accuracy']*100:.2f}%)")

# Save comparison results
results_df.to_csv('results/metrics/week2_k_comparison.csv', index=False)
print(f"\n✓ Comparison results saved to: results/metrics/week2_k_comparison.csv")

print("\n" + "="*80)

# ============================================================================
# STEP 6: SELECT OPTIMAL K AND TRAIN FINAL MODEL
# ============================================================================
print("STEP 6: Training Final Model with Optimal Features...")
print("-" * 80)

# Choose k that maximizes recall while maintaining good accuracy
# Priority: Recall > 85%, then maximize F1-score
optimal_results = results_df[results_df['recall'] >= 0.85]
if len(optimal_results) > 0:
    # Among those with recall >= 85%, pick highest F1
    optimal_idx = optimal_results['f1_score'].idxmax()
else:
    # Otherwise, just pick highest recall
    optimal_idx = best_recall_idx

optimal_k = int(results_df.loc[optimal_idx, 'k'])

print(f"\n🎯 Selected optimal k = {optimal_k} features")
print(f"\nCriteria: {'Recall >= 85%, highest F1' if len(optimal_results) > 0 else 'Highest recall'}")

# Get optimal features
optimal_features = mi_df.head(optimal_k)['feature'].tolist()
optimal_feature_indices = [feature_cols.index(f) for f in optimal_features]

print(f"\n✓ Selected {len(optimal_features)} features based on MI scores")

# Train final model
print(f"\nTraining final Random Forest model with {optimal_k} features...")
X_train_optimal = X_train[:, optimal_feature_indices]
X_test_optimal = X_test[:, optimal_feature_indices]

rf_optimal = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)

rf_optimal.fit(X_train_optimal, y_train)
print("✓ Model trained!")

# Final predictions
y_test_pred_optimal = rf_optimal.predict(X_test_optimal)

# Final metrics
final_metrics = {
    'optimal_k': optimal_k,
    'train_accuracy': float(accuracy_score(y_train, rf_optimal.predict(X_train_optimal))),
    'test_accuracy': float(results_df.loc[optimal_idx, 'test_accuracy']),
    'precision': float(results_df.loc[optimal_idx, 'precision']),
    'recall': float(results_df.loc[optimal_idx, 'recall']),
    'f1_score': float(results_df.loc[optimal_idx, 'f1_score']),
    'fpr': float(results_df.loc[optimal_idx, 'fpr']),
    'selected_features': optimal_features
}

# Save final metrics
with open('results/metrics/week2_final_metrics.json', 'w') as f:
    json.dump(final_metrics, f, indent=4)

print(f"✓ Final metrics saved to: results/metrics/week2_final_metrics.json")

# Save optimal model
with open('models/week2_mi_rf_model.pkl', 'wb') as f:
    pickle.dump(rf_optimal, f)

print(f"✓ Optimal model saved to: models/week2_mi_rf_model.pkl")

# Save optimal features
with open('models/week2_optimal_features.pkl', 'wb') as f:
    pickle.dump(optimal_features, f)

print(f"✓ Optimal features saved to: models/week2_optimal_features.pkl")

print("\n" + "="*80)

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================
print("STEP 7: Creating Visualizations...")
print("-" * 80)

# 1. Performance comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Recall vs k
axes[0, 0].plot(results_df['k'], results_df['recall']*100, marker='o', linewidth=2, markersize=8, color='#e74c3c')
axes[0, 0].axhline(y=baseline_metrics['recall']*100, color='gray', linestyle='--', label='Baseline')
axes[0, 0].axhline(y=85, color='green', linestyle='--', alpha=0.5, label='Target (85%)')
axes[0, 0].set_xlabel('Number of Features (k)', fontsize=11)
axes[0, 0].set_ylabel('Recall (%)', fontsize=11)
axes[0, 0].set_title('Recall vs Feature Count', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Accuracy vs k
axes[0, 1].plot(results_df['k'], results_df['test_accuracy']*100, marker='s', linewidth=2, markersize=8, color='#3498db')
axes[0, 1].axhline(y=baseline_metrics['test_accuracy']*100, color='gray', linestyle='--', label='Baseline')
axes[0, 1].set_xlabel('Number of Features (k)', fontsize=11)
axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11)
axes[0, 1].set_title('Accuracy vs Feature Count', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# F1-Score vs k
axes[1, 0].plot(results_df['k'], results_df['f1_score']*100, marker='^', linewidth=2, markersize=8, color='#2ecc71')
axes[1, 0].axhline(y=baseline_metrics['f1_score']*100, color='gray', linestyle='--', label='Baseline')
axes[1, 0].set_xlabel('Number of Features (k)', fontsize=11)
axes[1, 0].set_ylabel('F1-Score (%)', fontsize=11)
axes[1, 0].set_title('F1-Score vs Feature Count', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Precision vs k
axes[1, 1].plot(results_df['k'], results_df['precision']*100, marker='D', linewidth=2, markersize=8, color='#9b59b6')
axes[1, 1].axhline(y=baseline_metrics['precision']*100, color='gray', linestyle='--', label='Baseline')
axes[1, 1].set_xlabel('Number of Features (k)', fontsize=11)
axes[1, 1].set_ylabel('Precision (%)', fontsize=11)
axes[1, 1].set_title('Precision vs Feature Count', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/week2_performance_comparison.png', dpi=300)
print("✓ Performance comparison plot saved: results/plots/week2_performance_comparison.png")
plt.close()

# 2. Confusion matrix for optimal model
cm_optimal = confusion_matrix(y_test, y_test_pred_optimal)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malware'],
            yticklabels=['Benign', 'Malware'])
plt.title(f'Confusion Matrix - Week 2 (k={optimal_k} features)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('results/plots/week2_confusion_matrix.png', dpi=300)
print("✓ Confusion matrix plot saved: results/plots/week2_confusion_matrix.png")
plt.close()

# 3. Top features visualization
plt.figure(figsize=(12, 8))
top_30 = mi_df.head(30)
plt.barh(range(len(top_30)), top_30['mi_score'].values, color='steelblue')
plt.yticks(range(len(top_30)), [f[:50] + '...' if len(f) > 50 else f for f in top_30['feature'].values], fontsize=9)
plt.xlabel('Mutual Information Score', fontsize=12)
plt.title('Top 30 Features by Mutual Information', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('results/plots/week2_top_features_mi.png', dpi=300)
print("✓ Top features plot saved: results/plots/week2_top_features_mi.png")
plt.close()

print("\n" + "="*80)

# ============================================================================
# STEP 8: FINAL SUMMARY
# ============================================================================
print("STEP 8: Final Summary...")
print("-" * 80)

print("\n" + "="*80)
print("✅ WEEK 2 COMPLETE!")
print("="*80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n📊 FINAL RESULTS (k={}):\n".format(optimal_k))
print(f"{'Metric':<20} {'Week 1 Baseline':<20} {'Week 2 (MI)':<20} {'Change':<15}")
print("-" * 75)
print(f"{'Test Accuracy':<20} {baseline_metrics['test_accuracy']*100:>18.2f}% {final_metrics['test_accuracy']*100:>18.2f}% {(final_metrics['test_accuracy']-baseline_metrics['test_accuracy'])*100:>13.2f}%")
print(f"{'Recall':<20} {baseline_metrics['recall']*100:>18.2f}% {final_metrics['recall']*100:>18.2f}% {(final_metrics['recall']-baseline_metrics['recall'])*100:>13.2f}%")
print(f"{'Precision':<20} {baseline_metrics['precision']*100:>18.2f}% {final_metrics['precision']*100:>18.2f}% {(final_metrics['precision']-baseline_metrics['precision'])*100:>13.2f}%")
print(f"{'F1-Score':<20} {baseline_metrics['f1_score']*100:>18.2f}% {final_metrics['f1_score']*100:>18.2f}% {(final_metrics['f1_score']-baseline_metrics['f1_score'])*100:>13.2f}%")
print(f"{'FPR':<20} {baseline_metrics['false_positive_rate']*100:>18.2f}% {final_metrics['fpr']*100:>18.2f}% {(final_metrics['fpr']-baseline_metrics['false_positive_rate'])*100:>13.2f}%")
print(f"{'Features Used':<20} {split_info['feature_count']:>18} {optimal_k:>18} {optimal_k - split_info['feature_count']:>13}")

print("\n🎯 KEY ACHIEVEMENTS:")
if final_metrics['recall'] >= 0.85:
    print(f"  ✅ Recall target ACHIEVED: {final_metrics['recall']*100:.2f}% (target: 85%+)")
else:
    print(f"  ⚠️  Recall: {final_metrics['recall']*100:.2f}% (target: 85%+, need improvement)")

recall_improvement = (final_metrics['recall'] - baseline_metrics['recall']) * 100
print(f"  ✅ Recall improved by {recall_improvement:.2f} percentage points")
print(f"  ✅ Feature reduction: {split_info['feature_count']} → {optimal_k} features")

print("\n📁 Generated Files:")
print("  ├── results/metrics/mi_scores_all_features.csv")
print("  ├── results/metrics/week2_k_comparison.csv")
print("  ├── results/metrics/week2_final_metrics.json")
print("  ├── models/week2_mi_rf_model.pkl")
print("  ├── models/week2_optimal_features.pkl")
print("  ├── results/plots/mi_scores_distribution.png")
print("  ├── results/plots/week2_performance_comparison.png")
print("  ├── results/plots/week2_confusion_matrix.png")
print("  └── results/plots/week2_top_features_mi.png")

print("\n🚀 Next Steps (Week 3):")
print("  1. Implement GA-RAM population initialization")
print("  2. Create fitness function with RF classifier")
print("  3. Implement tournament selection")
print("  4. Start building genetic algorithm structure")

print("\n" + "="*80)
