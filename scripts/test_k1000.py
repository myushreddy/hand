"""
Quick test: k=1000 features vs k=500 baseline
Simple comparison to verify accuracy impact
"""

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
import json

print("="*70)
print("TESTING k=1000 vs k=500")
print("="*70)

# Load dataset with optimized dtypes
print("\n[1/4] Loading dataset (optimized)...")
print("  Loading 1.4GB CSV with int8 optimization...")

# First get column names to set dtypes
df_peek = pd.read_csv('data/processed/dataset_with_labels_full.csv', nrows=1)
int_cols = [col for col in df_peek.columns if col not in ['SHA256', 'NOME', 'PACOTE', 'API']]

# Create dtype dict - use int8 for binary features (saves memory, faster loading)
dtype_dict = {col: 'int8' for col in int_cols}
dtype_dict['SHA256'] = str
dtype_dict['NOME'] = str
dtype_dict['PACOTE'] = str

# Load with optimized dtypes
df = pd.read_csv('data/processed/dataset_with_labels_full.csv', dtype=dtype_dict, low_memory=False)
                 
print(f"✓ Loaded: {len(df):,} samples × {len(df.columns):,} columns")

# Separate features and labels  
X = df.drop(['label', 'SHA256', 'NOME', 'PACOTE', 'API'], axis=1, errors='ignore')
y = df['label']

# Load train/test split info
with open('results/metrics/train_test_split_full.json', 'r') as f:
    split_info = json.load(f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_info['test_size'], 
    random_state=split_info['random_state'], 
    stratify=y
)

print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# Load MI scores
print("\n[2/4] Loading MI scores...")
mi_df = pd.read_csv('results/metrics/mi_scores_full_dataset.csv')
mi_df_sorted = mi_df.sort_values('mi_score', ascending=False)

print(f"✓ MI scores loaded for {len(mi_df_sorted):,} features")
print(f"\n  MI Score at key ranks:")
print(f"    Rank 500:  {mi_df_sorted.iloc[499]['mi_score']:.6f}")
print(f"    Rank 1000: {mi_df_sorted.iloc[999]['mi_score']:.6f}")
print(f"    → Drop of {(1 - mi_df_sorted.iloc[999]['mi_score']/mi_df_sorted.iloc[499]['mi_score'])*100:.1f}% from 500 to 1000")

# Test k=1000
print("\n" + "="*70)
print("TESTING k=1000")
print("="*70)

start_time = time.time()

# Select top 1000 features
top_1000_features = mi_df_sorted.head(1000)['feature'].tolist()
available_features = [f for f in top_1000_features if f in X_train.columns]

print(f"\n✓ Using {len(available_features)} features")

X_train_1000 = X_train[available_features]
X_test_1000 = X_test[available_features]

# Train Random Forest
print(f"\nTraining Random Forest (100 trees)...")
train_start = time.time()

rf_1000 = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_1000.fit(X_train_1000, y_train)

train_time = time.time() - train_start
print(f"✓ Training complete: {train_time:.1f}s")

# Evaluate
print(f"\nEvaluating performance...")
y_train_pred = rf_1000.predict(X_train_1000)
y_test_pred = rf_1000.predict(X_test_1000)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
fpr = fp / (fp + tn)

# Cross-validation
print(f"\nRunning 5-fold cross-validation...")
cv_start = time.time()
cv_scores = cross_val_score(rf_1000, X_train_1000, y_train, cv=5, scoring='accuracy', n_jobs=-1)
cv_time = time.time() - cv_start

total_time = time.time() - start_time

# Display results
print(f"\n{'─'*70}")
print(f"RESULTS (k=1000):")
print(f"{'─'*70}")
print(f"  Train Accuracy:  {train_acc*100:.2f}%")
print(f"  Test Accuracy:   {test_acc*100:.2f}%")
print(f"  Precision:       {precision*100:.2f}%")
print(f"  Recall:          {recall*100:.2f}%")
print(f"  F1 Score:        {f1*100:.2f}%")
print(f"  FPR:             {fpr*100:.4f}%")
print(f"  CV Score:        {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"{'─'*70}")
print(f"  Confusion Matrix:")
print(f"    TN: {tn:,} | FP: {fp}")
print(f"    FN: {fn} | TP: {tp:,}")
print(f"{'─'*70}")
print(f"  ⏱️  Timing:")
print(f"    Training:   {train_time:.1f}s")
print(f"    CV (5-fold): {cv_time:.1f}s")
print(f"    TOTAL:      {total_time:.1f}s")
print(f"{'─'*70}")

# Check for overfitting
overfit_gap = train_acc - test_acc
if overfit_gap > 0.05:
    print(f"\n⚠️  WARNING: Possible overfitting (train-test gap: {overfit_gap*100:.2f}%)")
elif test_acc == 1.0:
    print(f"\n⚠️  WARNING: 100% test accuracy is suspicious!")
else:
    print(f"\n✓ Good generalization (train-test gap: {overfit_gap*100:.2f}%)")

# Compare with k=500
print("\n" + "="*70)
print("COMPARISON: k=500 vs k=1000")
print("="*70)

# Load k=500 results
with open('results/metrics/mi_k_comparison.json', 'r') as f:
    comparison_data = json.load(f)
    k500_results = comparison_data['k_results'][3]  # k=500 is 4th result

print(f"\n{'Metric':<20} {'k=500':<15} {'k=1000':<15} {'Difference':<15}")
print("─"*70)
print(f"{'Test Accuracy':<20} {k500_results['test_accuracy']*100:>7.2f}%      {test_acc*100:>7.2f}%      {(test_acc-k500_results['test_accuracy'])*100:>+6.2f}%")
print(f"{'Recall':<20} {k500_results['recall']*100:>7.2f}%      {recall*100:>7.2f}%      {(recall-k500_results['recall'])*100:>+6.2f}%")
print(f"{'Precision':<20} {k500_results['precision']*100:>7.2f}%      {precision*100:>7.2f}%      {(precision-k500_results['precision'])*100:>+6.2f}%")
print(f"{'F1 Score':<20} {k500_results['f1_score']*100:>7.2f}%      {f1*100:>7.2f}%      {(f1-k500_results['f1_score'])*100:>+6.2f}%")
print(f"{'CV Mean':<20} {k500_results['cv_mean']*100:>7.2f}%      {cv_scores.mean()*100:>7.2f}%      {(cv_scores.mean()-k500_results['cv_mean'])*100:>+6.2f}%")
print(f"{'CV Std':<20} {k500_results['cv_std']*100:>7.2f}%      {cv_scores.std()*100:>7.2f}%      {(cv_scores.std()-k500_results['cv_std'])*100:>+6.2f}%")

k500_cm = k500_results['confusion_matrix']
print(f"\n{'Confusion Matrix':<20} {'k=500':<15} {'k=1000':<15}")
print("─"*70)
print(f"{'False Positives':<20} {k500_cm['fp']:<15} {fp:<15} ({fp-k500_cm['fp']:+d})")
print(f"{'False Negatives':<20} {k500_cm['fn']:<15} {fn:<15} ({fn-k500_cm['fn']:+d})")
print(f"{'Total Errors':<20} {k500_cm['fp']+k500_cm['fn']:<15} {fp+fn:<15} ({(fp+fn)-(k500_cm['fp']+k500_cm['fn']):+d})")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

total_errors_500 = k500_cm['fp'] + k500_cm['fn']
total_errors_1000 = fp + fn
error_diff = total_errors_1000 - total_errors_500
acc_diff = test_acc - k500_results['test_accuracy']

if abs(acc_diff) < 0.001:
    print(f"\n✓ k=1000 performs IDENTICALLY to k=500")
    print(f"  → Adding 500 weaker features (MI: 0.0233-0.0366) had NO impact")
    print(f"  → k=500 is optimal - more features = wasted computation")
elif error_diff < 0:
    print(f"\n✓ k=1000 performs SLIGHTLY BETTER than k=500")
    print(f"  → {abs(error_diff)} fewer errors with k=1000")
    print(f"  → Accuracy improved by {acc_diff*100:.2f}%")
    print(f"  → But is the gain worth 2x training time?")
elif error_diff > 0:
    print(f"\n⚠️  k=1000 performs SLIGHTLY WORSE than k=500")
    print(f"  → {error_diff} more errors with k=1000")
    print(f"  → Accuracy dropped by {abs(acc_diff)*100:.2f}%")
    print(f"  → Weaker features (501-1000) added noise, not signal")
    print(f"  → Confirms k=500 is optimal")

print(f"\n{'Feature Count':<15} {'Training Time':<15} {'Accuracy':<12} {'Total Errors'}")
print("─"*70)
print(f"{'k=500':<15} {'~90s (baseline)':<15} {k500_results['test_accuracy']*100:>6.2f}%     {total_errors_500}")
print(f"{'k=1000':<15} {f'{train_time:.0f}s (+{train_time-90:.0f}s)':<15} {test_acc*100:>6.2f}%     {total_errors_1000}")

print("\n✓ Analysis complete!")
