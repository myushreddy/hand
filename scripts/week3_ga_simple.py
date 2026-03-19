"""
Week 3: GA-RAM (Memory Optimized Sample)

For systems with limited RAM:
- Load smaller sample to demonstrate GA
- 5000 train + 1250 test samples
- 500 features
- Same GA framework for Week 4 extension
"""

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("WEEK 3: GA-RAM (Memory Optimized - Sample)")
print("="*80)

# CONFIG
POPULATION_SIZE = 20
TOURNAMENT_SIZE = 3
N_GENERATIONS = 40
RANDOM_STATE = 42
N_JOBS = 1
RF_N_TREES = 30
FEATURE_POOL_SIZE = 500

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\nSTEP 1: Loading data...")
load_start = time.time()

# Load MI scores
print("  [1/3] MI scores...")
mi_df = pd.read_csv('results/metrics/mi_scores_full_dataset.csv')
mi_sorted = mi_df.sort_values('mi_score', ascending=False)
top_features = mi_sorted.head(FEATURE_POOL_SIZE)['feature'].tolist()

# Load only selected columns and a sample
print("  [2/3] Loading dataset sample (5000+1250)...")
cols = top_features + ['CLASS']

# Use nrows to get smaller dataset
df = pd.read_csv('data/processed/dataset_with_labels_full.csv',
                  usecols=cols, nrows=6250, low_memory=False)

X = df.drop('CLASS', axis=1).head(5000)  # Take first 5000
y = df['CLASS'].head(5000)

print(f"  Data: {len(X)} x {X.shape[1]}")

# Split
print("  [3/3] Train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

load_time = time.time() - load_start
print(f"  Load time: {load_time:.1f}s")

feature_names = list(X.columns)
n_features = len(feature_names)

# ============================================================================
# GA FUNCTIONS
# ============================================================================

def init_population(pop_size, n_feat):
    pop = []
    for _ in range(pop_size):
        act = np.random.uniform(0.1, 0.3)
        chrom = np.random.binomial(1, act, n_feat)
        n_sel = np.sum(chrom)
        while n_sel < 10 or n_sel > int(0.3 * n_feat):
            act = np.random.uniform(0.1, 0.3)
            chrom = np.random.binomial(1, act, n_feat)
            n_sel = np.sum(chrom)
        pop.append(chrom)
    return pop

def eval_fitness(pop, X_tr, y_tr, X_te, y_te, feat_names):
    fit = []
    for chrom in pop:
        sel_idx = np.where(chrom == 1)[0]
        if len(sel_idx) == 0:
            fit.append(0.0)
            continue
        try:
            sel_feats = [feat_names[i] for i in sel_idx]
            rf = RandomForestClassifier(
                n_estimators=RF_N_TREES,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS
            )
            rf.fit(X_tr[sel_feats], y_tr)
            acc = accuracy_score(y_te, rf.predict(X_te[sel_feats]))
            
            # Penalty for >50 features
            if len(sel_idx) > 50:
                pen = ((len(sel_idx) - 50) / 100.0) ** 2 * 0.2
            else:
                pen = 0.0
            
            fit.append(max(0, acc - pen))
        except:
            fit.append(0.0)
    return np.array(fit)

def select_parents(pop, fit):
    sel = []
    n = len(pop)
    for _ in range(n):
        idxs = np.random.choice(n, TOURNAMENT_SIZE, replace=False)
        best = idxs[np.argmax(fit[idxs])]
        sel.append(pop[best].copy())
    return sel

def breed(p1, p2):
    n = len(p1)
    pt1 = np.random.randint(1, n - 1)
    pt2 = np.random.randint(pt1 + 1, n)
    
    c1 = np.concatenate([p1[:pt1], p2[pt1:pt2], p1[pt2:]]).astype(int)
    c2 = np.concatenate([p2[:pt1], p1[pt1:pt2], p2[pt2:]]).astype(int)
    
    for c in [c1, c2]:
        n_sel = np.sum(c)
        if n_sel > 150:
            idxs = np.where(c == 1)[0]
            to_rm = np.random.choice(idxs, n_sel - 150, replace=False)
            c[to_rm] = 0
    
    return c1, c2

# ============================================================================
# RUN GA
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Running GA")
print("="*80)

pop = init_population(POPULATION_SIZE, n_features)
n_sel_list = [np.sum(c) for c in pop]
print(f"Initial: avg={np.mean(n_sel_list):.0f}, range={np.min(n_sel_list)}-{np.max(n_sel_list)}")

fit = eval_fitness(pop, X_train, y_train, X_test, y_test, feature_names)
print(f"Fitness: best={np.max(fit):.4f}, avg={np.mean(fit):.4f}")

hist = {'gen': [], 'best': [], 'avg': [], 'nfeat': [], 'navg': []}

print(f"\n{'Gen':<4} {'Best':<8} {'Avg':<8} {'Feat':<6}")
print("-" * 30)

ga_start = time.time()
best_ever = 0
best_chrom = None
early_stop = 0

for gen in range(N_GENERATIONS):
    best_idx = np.argmax(fit)
    best_f = fit[best_idx]
    best_c = pop[best_idx]
    nf = np.sum(best_c)
    
    hist['gen'].append(gen)
    hist['best'].append(float(best_f))
    hist['avg'].append(float(np.mean(fit)))
    hist['nfeat'].append(int(nf))
    hist['navg'].append(float(np.mean([np.sum(c) for c in pop])))
    
    if best_f > best_ever:
        best_ever = best_f
        best_chrom = best_c.copy()
    
    print(f"{gen:<4} {best_f:<8.4f} {np.mean(fit):<8.4f} {nf:<6}")
    
    # Early stop
    if np.var(fit) < 1e-6:
        early_stop += 1
        if early_stop >= 2:
            print("  Early stop!")
            break
    else:
        early_stop = 0
    
    # GA step
    parents = select_parents(pop, fit)
    pop = []
    for i in range(0, len(parents) - 1, 2):
        c1, c2 = breed(parents[i], parents[i+1])
        pop.append(c1)
        pop.append(c2)
    if len(parents) % 2:
        pop.append(parents[-1].copy())
    
    pop = pop[:POPULATION_SIZE]
    fit = eval_fitness(pop, X_train, y_train, X_test, y_test, feature_names)

ga_time = time.time() - ga_start

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Results")
print("="*80)

nf = np.sum(best_chrom)
sel_idx = np.where(best_chrom == 1)[0]
sel_names = [feature_names[i] for i in sel_idx]

print(f"\nBest GA solution:")
print(f"  Fitness: {best_ever:.4f}")
print(f"  Features: {nf}/{n_features}")

# Final model
print(f"\nTraining final model...")
rf = RandomForestClassifier(
    n_estimators=RF_N_TREES, random_state=RANDOM_STATE, n_jobs=N_JOBS
)
rf.fit(X_train[sel_names], y_train)
pred = rf.predict(X_test[sel_names])

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\nPerformance:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  FPR:       {fpr:.6f}")

# ============================================================================
# SAVE
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Saving")
print("="*80)

# Features
with open('results/metrics/ga_week3_selected_features.pkl', 'wb') as f:
    pickle.dump(sel_names, f)
with open('results/metrics/ga_week3_selected_features.txt', 'w') as f:
    for n in sel_names:
        f.write(f"{n}\n")
print(f"  Saved features ({len(sel_names)})")

# Model
np.save('results/metrics/ga_week3_best_chromosome.npy', best_chrom)
with open('models/rf_model_ga_week3.pkl', 'wb') as f:
    pickle.dump(rf, f)
print(f"  Saved model")

# Metrics
metrics = {
    'type': 'GA_WEEK3_SAMPLE',
    'sample_size': int(len(X_train)),
    'generations': len(hist['gen']),
    'best_fitness_ga': float(best_ever),
    'final_accuracy': float(acc),
    'final_precision': float(prec),
    'final_recall': float(rec),
    'final_f1': float(f1),
    'final_fpr': float(fpr),
    'n_features_selected': int(nf),
    'n_features_total': int(n_features),
    'feature_reduction_percent': float((1 - nf/n_features)*100),
    'ga_runtime_sec': float(ga_time),
    'timestamp': datetime.now().isoformat()
}

with open('results/metrics/ga_week3_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved metrics")

# History
with open('results/metrics/ga_week3_history.json', 'w') as f:
    json.dump(hist, f, indent=2)
print(f"  Saved history")

# ============================================================================
# PLOT
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Plotting")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GA Week 3: Feature Reduction via Genetic Algorithm', fontsize=16, fontweight='bold')

# Fitness
ax = axes[0, 0]
ax.plot(hist['gen'], hist['best'], 'g-', linewidth=2, label='Best')
ax.plot(hist['gen'], hist['avg'], 'b--', linewidth=1.5, label='Average')
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness (Accuracy - Penalty)')
ax.set_title('Fitness Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# Features
ax = axes[0, 1]
ax.plot(hist['gen'], hist['nfeat'], 'g-', linewidth=2, label='Best')
ax.plot(hist['gen'], hist['navg'], 'b--', linewidth=1.5, label='Average')
ax.axhline(y=50, color='r', linestyle=':', linewidth=2, alpha=0.7, label='Target (50)')
ax.set_xlabel('Generation')
ax.set_ylabel('Number of Features')
ax.set_title('Feature Count Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# Metrics
ax = axes[1, 0]
names = ['Accuracy', 'Precision', 'Recall', 'F1']
vals = [acc, prec, rec, f1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax.bar(names, vals, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Score')
ax.set_title('Final Model Performance')
ax.set_ylim([0.8, 1.01])
for bar, val in zip(bars, vals):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Summary
ax = axes[1, 1]
ax.axis('off')
txt = f"""WEEK 3 RESULTS - Feature Selection

GA Configuration:
  Population: {POPULATION_SIZE}
  Generations: {len(hist['gen'])}
  RF Trees: {RF_N_TREES}

Best Solution:
  Fitness: {best_ever:.4f}
  Features: {nf}/{n_features}
  Reduction: {(1-nf/n_features)*100:.1f}%

Final Performance:
  Accuracy: {acc:.4f}
  Precision: {prec:.4f}
  Recall: {rec:.4f}
  F1-Score: {f1:.4f}

Runtime:
  GA: {ga_time:.1f}s
  Data: {load_time:.1f}s

Note: Using sample (5K train)
Next: Week 4 - RAM mutations
"""
ax.text(0.05, 0.95, txt, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/plots/ga_week3_convergence.png', dpi=300, bbox_inches='tight')
print("  Saved plot")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("WEEK 3 COMPLETE")
print("="*80)

print(f"""
SUMMARY:
  * GA framework: working
  * Sample size: {len(X_train)} train
  * Features: {n_features} -> {nf} ({(1-nf/n_features)*100:.1f}% reduction)
  * Accuracy: {acc:.4f}
  * Generations: {len(hist['gen'])}

OUTPUT:
  * models/rf_model_ga_week3.pkl
  * results/metrics/ga_week3_*.json
  * results/metrics/ga_week3_selected_features.txt
  * results/plots/ga_week3_convergence.png

NEXT WEEK:
  Add Rank-Based Adaptive Mutation (RAM) to eliminate
  low-fitness features and converge to smaller subset
""")

print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
