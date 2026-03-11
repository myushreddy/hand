"""
Test larger k values (1000, 2000, 5000) to compare with k=500
Shows timing and performance impact of including weaker MI features
"""

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import json

print("="*70)
print("TESTING LARGER K VALUES: Impact of Weaker MI Features")
print("="*70)

# Load dataset
print("\n[1/5] Loading dataset...")
EXPECTED_SAMPLES = 28752
df = pd.read_csv('data/processed/dataset_with_labels_full.csv')
actual_samples = len(df)

if actual_samples != EXPECTED_SAMPLES:
    print(f"⚠️  Warning: Expected {EXPECTED_SAMPLES} samples, found {actual_samples}")
    print(f"    Using {actual_samples} samples for analysis")

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Load train/test split info
with open('results/metrics/train_test_split_full.json', 'r') as f:
    split_info = json.load(f)
    test_size = split_info['test_size']
    random_state = split_info['random_state']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

print(f"✓ Dataset loaded: {len(X_train)} train, {len(X_test)} test")
print(f"  Total features available: {X.shape[1]}")

# Load MI scores
print("\n[2/5] Loading MI scores...")
mi_df = pd.read_csv('results/metrics/mi_scores_full_dataset.csv')
mi_df_sorted = mi_df.sort_values('mi_score', ascending=False)

print(f"✓ MI scores loaded for {len(mi_df_sorted)} features")
print(f"\n  MI Score Distribution:")
print(f"    Top 1:    {mi_df_sorted.iloc[0]['mi_score']:.6f}")
print(f"    Rank 155: {mi_df_sorted.iloc[154]['mi_score']:.6f}")
print(f"    Rank 500: {mi_df_sorted.iloc[499]['mi_score']:.6f}")
print(f"    Rank 1000: {mi_df_sorted.iloc[999]['mi_score']:.6f}")
print(f"    Rank 2000: {mi_df_sorted.iloc[1999]['mi_score']:.6f}")
print(f"    Rank 5000: {mi_df_sorted.iloc[4999]['mi_score']:.6f}")
print(f"    Bottom:   {mi_df_sorted.iloc[-1]['mi_score']:.6f}")

# Test different k values
k_values = [500, 1000, 2000, 5000]
results = []

print("\n[3/5] Testing different k values...")
print("-"*70)

for k in k_values:
    print(f"\n{'='*70}")
    print(f"TESTING k={k} features")
    print(f"{'='*70}")
    
    # Select top k features
    start_time = time.time()
    top_k_features = mi_df_sorted.head(k)['feature'].tolist()
    
    # Get feature columns (some might be missing)
    available_features = [f for f in top_k_features if f in X_train.columns]
    missing_count = k - len(available_features)
    
    if missing_count > 0:
        print(f"⚠️  Warning: {missing_count} features not found in dataset")
    
    print(f"Using {len(available_features)} features")
    
    X_train_k = X_train[available_features]
    X_test_k = X_test[available_features]
    
    selection_time = time.time() - start_time
    print(f"  Feature selection: {selection_time:.2f}s")
    
    # Train Random Forest
    print(f"\n  Training Random Forest (100 trees)...")
    train_start = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train_k, y_train)
    
    train_time = time.time() - train_start
    print(f"  ✓ Training complete: {train_time:.2f}s")
    
    # Evaluate
    print(f"\n  Evaluating performance...")
    eval_start = time.time()
    
    y_train_pred = rf.predict(X_train_k)
    y_test_pred = rf.predict(X_test_k)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    fpr = fp / (fp + tn)
    
    eval_time = time.time() - eval_start
    
    # Cross-validation (only on training set)
    print(f"  Running 5-fold cross-validation...")
    cv_start = time.time()
    cv_scores = cross_val_score(rf, X_train_k, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    cv_time = time.time() - cv_start
    
    total_time = time.time() - start_time
    
    # Display results
    print(f"\n  RESULTS (k={k}):")
    print(f"  {'─'*66}")
    print(f"    Train Accuracy: {train_acc*100:.2f}%")
    print(f"    Test Accuracy:  {test_acc*100:.2f}%")
    print(f"    Precision:      {precision*100:.2f}%")
    print(f"    Recall:         {recall*100:.2f}%")
    print(f"    F1 Score:       {f1*100:.2f}%")
    print(f"    FPR:            {fpr*100:.4f}%")
    print(f"    CV Score:       {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"  {'─'*66}")
    print(f"    Confusion Matrix: TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}")
    print(f"  {'─'*66}")
    print(f"    ⏱️  TIMING:")
    print(f"      Selection:  {selection_time:.2f}s")
    print(f"      Training:   {train_time:.2f}s")
    print(f"      Evaluation: {eval_time:.2f}s")
    print(f"      CV (5-fold): {cv_time:.2f}s")
    print(f"      TOTAL:      {total_time:.2f}s")
    
    # Check for overfitting
    overfit_gap = train_acc - test_acc
    if overfit_gap > 0.05:
        print(f"\n  ⚠️  WARNING: Possible overfitting (train-test gap: {overfit_gap*100:.2f}%)")
    elif test_acc == 1.0:
        print(f"\n  ⚠️  WARNING: 100% test accuracy is suspicious!")
    else:
        print(f"\n  ✓ Good generalization (train-test gap: {overfit_gap*100:.2f}%)")
    
    results.append({
        'k': k,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'timing': {
            'selection': selection_time,
            'training': train_time,
            'evaluation': eval_time,
            'cv': cv_time,
            'total': total_time
        },
        'overfitting_gap': overfit_gap
    })

# Summary comparison
print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)
print(f"\n{'k':<8} {'Test Acc':<10} {'Recall':<10} {'F1':<10} {'FP':<6} {'FN':<6} {'Total Time':<12}")
print("-"*70)
for r in results:
    cm = r['confusion_matrix']
    print(f"{r['k']:<8} {r['test_accuracy']*100:>7.2f}%  {r['recall']*100:>7.2f}%  {r['f1_score']*100:>7.2f}%  {cm['fp']:<6} {cm['fn']:<6} {r['timing']['total']:>10.2f}s")

# Save results
print("\n[4/5] Saving results...")
output_metrics = {
    'test_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'samples': actual_samples,
    'k_results': results
}

with open('results/metrics/larger_k_comparison.json', 'w') as f:
    json.dump(output_metrics, f, indent=4)

print("✓ Results saved to: results/metrics/larger_k_comparison.json")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nKEY INSIGHTS:")
print("1. MI Score Drop-off:")
print(f"   - Top features (rank 1-500): MI scores 0.0366-0.3222")
print(f"   - Mid features (501-1000): MI scores 0.0233-0.0366")
print(f"   - Weak features (1001-2000): MI scores 0.0101-0.0233")
print(f"   - Very weak (2001-5000): MI scores 0.0007-0.0101")
print("\n2. Diminishing Returns:")
print("   - Compare test accuracy across k values above")
print("   - More features ≠ better performance beyond optimal k")
print("\n3. Computational Cost:")
print("   - Training time increases with k (see 'Total Time' column)")
print("   - Memory usage scales proportionally")
print("\n4. Overfitting Risk:")
print("   - Larger k may add noisy features that hurt generalization")
print("   - Check train-test gaps and CV variance")

print("\n✓ Testing complete!")
