"""
Ultra-fast k=1000 test - loads data once, trains k=1000 only
"""
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
import json

print("="*70)
print("QUICK TEST: k=1000")
print("="*70)

start_total = time.time()

# Load dataset efficiently with chunks to avoid memory issues
print("\n[1/3] Loading dataset...")
load_start = time.time()

# Use int8 for binary features to speed up
chunks = []
chunk_size = 10000
for chunk in pd.read_csv('data/processed/dataset_with_labels_full.csv', chunksize=chunk_size):
    chunks.append(chunk)
df =pd.concat(chunks, ignore_index=True)

load_time = time.time() - load_start
print(f"✓ Loaded in {load_time:.1f}s: {len(df):,} samples")

# Prepare data
X = df.drop(['CLASS', 'SHA256', 'NOME', 'PACOTE', 'API'], axis=1, errors='ignore')
y = df['CLASS']

# Use same split as before
split_info = json.load(open('results/metrics/train_test_split_full.json'))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_info['test_size'], 
    random_state=split_info['random_state'], stratify=y
)

print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# Load MI scores and select top 1000
print("\n[2/3] Selecting top 1000 features by MI...")
mi_df = pd.read_csv('results/metrics/mi_scores_full_dataset.csv')
mi_df_sorted = mi_df.sort_values('mi_score', ascending=False)

top_1000 = mi_df_sorted.head(1000)['feature'].tolist()
available = [f for f in top_1000 if f in X_train.columns]

X_train_1000 = X_train[available]
X_test_1000 = X_test[available]

print(f"✓ Using {len(available)} features")
print(f"  MI score range: {mi_df_sorted.iloc[0]['mi_score']:.4f} to {mi_df_sorted.iloc[999]['mi_score']:.4f}")

# Train and evaluate
print("\n[3/3] Training Random Forest...")
train_start = time.time()

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_1000, y_train)

train_time = time.time() - train_start

y_pred = rf.predict(X_test_1000)
test_acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"✓ Training complete in {train_time:.1f}s")

# Quick 3-fold CV (faster than 5-fold)
print(f"\nRunning 3-fold CV...")
cv_start = time.time()
cv_scores = cross_val_score(rf, X_train_1000, y_train, cv=3, scoring='accuracy', n_jobs=-1)
cv_time = time.time() - cv_start

total_time = time.time() - start_total

# Results
print("\n" + "="*70)
print(f"RESULTS (k=1000 features)")
print("="*70)
print(f"  Test Accuracy:   {test_acc*100:.2f}%")
print(f"  Precision:       {precision*100:.2f}%")
print(f"  Recall:          {recall*100:.2f}%")
print(f"  F1 Score:        {f1*100:.2f}%")
print(f"  CV (3-fold):     {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print("-"*70)
print(f"  Confusion Matrix:")
print(f"    True Negatives:  {tn:,}")
print(f"    False Positives: {fp}")
print(f"    False Negatives: {fn}")
print(f"    True Positives:  {tp:,}")
print(f"    Total Errors:    {fp + fn}")
print("-"*70)
print(f"  Timing:")
print(f"    Data loading:  {load_time:.1f}s")
print(f"    Training:      {train_time:.1f}s")
print(f"    CV (3-fold):   {cv_time:.1f}s")
print(f"    TOTAL:         {total_time:.1f}s")
print("="*70)

# Compare with k=500
comparison_data = json.load(open('results/metrics/mi_k_comparison.json'))
k500 = comparison_data['k_results'][3]  # k=500

print("\nCOMPARISON vs k=500:")
print("-"*70)
print(f"{'Metric':<20} {'k=500':<12} {'k=1000':<12} {'Difference'}")
print("-"*70)
print(f"{'Test Accuracy':<20} {k500['test_accuracy']*100:>7.2f}%    {test_acc*100:>7.2f}%    {(test_acc-k500['test_accuracy'])*100:+.2f}%")
print(f"{'Recall':<20} {k500['recall']*100:>7.2f}%    {recall*100:>7.2f}%    {(recall-k500['recall'])*100:+.2f}%")
print(f"{'Precision':<20} {k500['precision']*100:>7.2f}%    {precision*100:>7.2f}%    {(precision-k500['precision'])*100:+.2f}%")
print(f"{'F1 Score':<20} {k500['f1_score']*100:>7.2f}%    {f1*100:>7.2f}%    {(f1-k500['f1_score'])*100:+.2f}%")
print(f"{'Total Errors':<20} {k500['confusion_matrix']['fp']+k500['confusion_matrix']['fn']:<12} {fp+fn:<12} {(fp+fn)-(k500['confusion_matrix']['fp']+k500['confusion_matrix']['fn']):+d}")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)

error_diff = (fp + fn) - (k500['confusion_matrix']['fp'] + k500['confusion_matrix']['fn'])
acc_diff = test_acc - k500['test_accuracy']

if abs(error_diff) == 0:
    print("✓ k=1000 performs IDENTICALLY to k=500")
    print("  → Adding 500 weaker features had NO impact on accuracy")
    print("  → k=500 is optimal - more features = wasted time")
elif error_diff < 0:
    print(f"✓ k=1000 is SLIGHTLY BETTER ({abs(error_diff)} fewer errors)")
    print(f"  → But is {abs(acc_diff)*100:.2f}% improvement worth 2x longer training?")
else:
    print(f"⚠️  k=1000 is SLIGHTLY WORSE ({error_diff} more errors)")
    print(f"  → Accuracy dropped by {abs(acc_diff)*100:.2f}%")
    print("  → Weaker features (rank 501-1000) added noise")

print(f"\n✓ Test complete in {total_time:.0f} seconds")
