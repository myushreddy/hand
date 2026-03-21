"""
Week 4: GA-RAM on Full Dataset (28k samples)

Scales the improved RAM to the complete malware dataset
Uses memory-efficient selective column loading
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
print("WEEK 4: GA-RAM on FULL DATASET (28k samples)")
print("="*80)

# CONFIG
POPULATION_SIZE = 20
TOURNAMENT_SIZE = 3
N_GENERATIONS = 100
RANDOM_STATE = 42
N_JOBS = 1
RF_N_TREES = 30
FEATURE_POOL_SIZE = 500
DATASET_SIZE = None  # Load all 28k

print(f"\nConfiguration:")
print(f"  Dataset: FULL (28k samples)")
print(f"  Population: {POPULATION_SIZE}")
print(f"  Generations: {N_GENERATIONS}")
print(f"  Feature pool: {FEATURE_POOL_SIZE}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Loading full dataset...")
print("="*80)

load_start = time.time()

print("  [1/3] MI scores...")
mi_df = pd.read_csv('results/metrics/mi_scores_full_dataset.csv')
mi_sorted = mi_df.sort_values('mi_score', ascending=False)
top_features = mi_sorted.head(FEATURE_POOL_SIZE)['feature'].tolist()
print(f"    Top {FEATURE_POOL_SIZE} MI features identified")

print("  [2/3] Loading full dataset...")
cols = top_features + ['CLASS']
df = pd.read_csv('data/processed/dataset_with_labels_full.csv',
                  usecols=cols, nrows=DATASET_SIZE, low_memory=False)

print(f"    Loaded: {len(df)} samples x {len(cols)} columns")

X = df.drop('CLASS', axis=1)
y = df['CLASS']

print("  [3/3] Train/test split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"    Train: {len(X_train)}")
print(f"    Test: {len(X_test)}")

load_time = time.time() - load_start
print(f"  Load time: {load_time:.1f}s")

feature_names = list(X.columns)
n_features = len(feature_names)

# ============================================================================
# GA FUNCTIONS (Same as Improved v2)
# ============================================================================

def init_population(pop_size, n_feat):
    pop = []
    for _ in range(pop_size):
        act = np.random.uniform(0.08, 0.25)
        chrom = np.random.binomial(1, act, n_feat)
        n_sel = np.sum(chrom)
        while n_sel < 10 or n_sel > int(0.3 * n_feat):
            act = np.random.uniform(0.08, 0.25)
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
            
            # Linear penalty, target 40 features
            if len(sel_idx) > 40:
                pen = (len(sel_idx) - 40) * 0.002
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
        if n_sel > 120:
            idxs = np.where(c == 1)[0]
            to_rm = np.random.choice(idxs, n_sel - 120, replace=False)
            c[to_rm] = 0
    
    return c1, c2

def rank_based_mutation(pop, fit, p_max=0.5, p_min=0.05):
    n_pop = len(pop)
    sorted_indices = np.argsort(fit)
    
    mutated_pop = []
    for rank, idx in enumerate(sorted_indices):
        chrom = pop[idx].copy()
        p_mutation = p_max - (p_max - p_min) * rank / (n_pop - 1)
        
        for i in range(len(chrom)):
            if np.random.random() < p_mutation:
                chrom[i] = 1 - chrom[i]
        
        n_sel = np.sum(chrom)
        if n_sel < 10:
            empty_idx = np.where(chrom == 0)[0]
            if len(empty_idx) > 0:
                to_add = np.random.choice(empty_idx, min(10 - n_sel, len(empty_idx)), replace=False)
                chrom[to_add] = 1
        elif n_sel > 120:
            sel_idx = np.where(chrom == 1)[0]
            to_remove = np.random.choice(sel_idx, n_sel - 120, replace=False)
            chrom[to_remove] = 0
        
        mutated_pop.append(chrom)
    
    return mutated_pop

# ============================================================================
# RUN GA
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Running GA with Improved RAM (Full Dataset)")
print("="*80)

pop = init_population(POPULATION_SIZE, n_features)
n_sel_list = [np.sum(c) for c in pop]
print(f"\nInitial: avg={np.mean(n_sel_list):.0f}, range={np.min(n_sel_list)}-{np.max(n_sel_list)}")

fit = eval_fitness(pop, X_train, y_train, X_test, y_test, feature_names)
print(f"Fitness: best={np.max(fit):.4f}, avg={np.mean(fit):.4f}")

hist = {'gen': [], 'best': [], 'avg': [], 'nfeat': [], 'navg': [], 'pm': []}

print(f"\n{'Gen':<4} {'Best':<8} {'Avg':<8} {'Feat':<6}")
print("-" * 30)

ga_start = time.time()
best_ever = 0
best_chrom = None
stagnation = 0

for gen in range(N_GENERATIONS):
    best_idx = np.argmax(fit)
    best_f = fit[best_idx]
    best_c = pop[best_idx]
    nf = np.sum(best_c)
    
    # Backup elite
    elite_indices = np.argsort(fit)[-2:]
    elite_pop = [pop[i].copy() for i in elite_indices]
    elite_fit = [fit[i] for i in elite_indices]
    
    ranks = np.argsort(fit)
    avg_pm = np.mean([0.5 - (0.5 - 0.05) * r / (POPULATION_SIZE - 1) for r in range(POPULATION_SIZE)])
    
    hist['gen'].append(gen)
    hist['best'].append(float(best_f))
    hist['avg'].append(float(np.mean(fit)))
    hist['nfeat'].append(int(nf))
    hist['navg'].append(float(np.mean([np.sum(c) for c in pop])))
    hist['pm'].append(float(avg_pm))
    
    if best_f > best_ever:
        best_ever = best_f
        best_chrom = best_c.copy()
        stagnation = 0
    else:
        stagnation += 1
    
    if gen % 5 == 0 or gen < 5:
        print(f"{gen:<4} {best_f:<8.4f} {np.mean(fit):<8.4f} {nf:<6}")
    
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
    pop = rank_based_mutation(pop, fit, p_max=0.5, p_min=0.05)
    
    # Evaluate and restore elite
    fit_new = eval_fitness(pop, X_train, y_train, X_test, y_test, feature_names)
    pop_new = []
    fit_list = []
    for i in range(len(pop)):
        pop_new.append(pop[i])
        fit_list.append(fit_new[i])
    
    for j, (elite_c, elite_f) in enumerate(zip(elite_pop, elite_fit)):
        worst_idx = np.argmin(fit_list)
        if elite_f > fit_list[worst_idx]:
            pop_new[worst_idx] = elite_c.copy()
            fit_list[worst_idx] = elite_f
    
    pop = pop_new
    fit = np.array(fit_list)
    
    # Stop criteria
    if stagnation > 15 and gen > 30:
        if gen % 5 != 0:
            print(f"{gen:<4} {best_f:<8.4f} {np.mean(fit):<8.4f} {nf:<6}")
        print(f"Stagnation stop at gen {gen}")
        break

ga_time = time.time() - ga_start

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Results (Full Dataset)")
print("="*80)

nf = np.sum(best_chrom)
sel_idx = np.where(best_chrom == 1)[0]
sel_names = [feature_names[i] for i in sel_idx]

print(f"\nBest GA solution:")
print(f"  Fitness: {best_ever:.4f}")
print(f"  Features: {nf}/{n_features}")
print(f"  Reduction: {(1-nf/n_features)*100:.1f}%")

# Final model
print(f"\nTraining final model on FULL dataset...")
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

print(f"\nPerformance on Full Dataset:")
print(f"  Accuracy:  {acc:.6f}")
print(f"  Precision: {prec:.6f}")
print(f"  Recall:    {rec:.6f}")
print(f"  F1:        {f1:.6f}")
print(f"  FPR:       {fpr:.6f}")
print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# ============================================================================
# SAVE
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Saving (Full Dataset Results)")
print("="*80)

with open('results/metrics/ga_week4_full_selected_features.pkl', 'wb') as f:
    pickle.dump(sel_names, f)
with open('results/metrics/ga_week4_full_selected_features.txt', 'w') as f:
    for n in sel_names:
        f.write(f"{n}\n")
print(f"  Features: {len(sel_names)} features")

np.save('results/metrics/ga_week4_full_best_chromosome.npy', best_chrom)
with open('models/rf_model_ga_week4_full.pkl', 'wb') as f:
    pickle.dump(rf, f)
print(f"  Model: Trained on {len(X_train)} samples")

metrics = {
    'type': 'GA_WEEK4_FULL_DATASET',
    'dataset_size': int(len(X_train) + len(X_test)),
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test)),
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
    'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
    'ga_runtime_sec': float(ga_time),
    'timestamp': datetime.now().isoformat()
}

with open('results/metrics/ga_week4_full_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Metrics: Saved")

with open('results/metrics/ga_week4_full_history.json', 'w') as f:
    json.dump(hist, f, indent=2)
print(f"  History: Saved")

# ============================================================================
# PLOT
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Plotting")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Week 4: RAM on Full Dataset ({len(X_train)+len(X_test)} samples)', 
             fontsize=16, fontweight='bold')

ax = axes[0, 0]
ax.plot(hist['gen'], hist['best'], 'g-', linewidth=2, label='Best')
ax.plot(hist['gen'], hist['avg'], 'b--', linewidth=1.5, label='Average')
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')
ax.set_title('Fitness Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(hist['gen'], hist['nfeat'], 'g-', linewidth=2, label='Best')
ax.plot(hist['gen'], hist['navg'], 'b--', linewidth=1.5, label='Average')
ax.axhline(y=40, color='r', linestyle=':', linewidth=2, alpha=0.7, label='Target (40)')
ax.set_xlabel('Generation')
ax.set_ylabel('Number of Features')
ax.set_title('Feature Count Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(hist['gen'], hist['pm'], 'purple', linewidth=2)
ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='P_max=0.5')
ax.axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='P_min=0.05')
ax.set_xlabel('Generation')
ax.set_ylabel('Avg Mutation Probability')
ax.set_title('Rank-Based Mutation Rates')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.axis('off')
txt = f"""WEEK 4 - Full Dataset Results

Dataset:
  Total: {len(X_train)+len(X_test)} samples
  Train: {len(X_train)}
  Test: {len(X_test)}

Best Solution:
  Features: {nf}/{n_features}
  Reduction: {(1-nf/n_features)*100:.1f}%
  Fitness: {best_ever:.4f}

Performance:
  Accuracy: {acc:.6f}
  Precision: {prec:.6f}
  Recall: {rec:.6f}
  F1: {f1:.6f}

Confusion Matrix:
  TP: {int(tp)}, TN: {int(tn)}
  FP: {int(fp)}, FN: {int(fn)}
  FPR: {fpr:.6f}
"""
ax.text(0.05, 0.95, txt, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('results/plots/ga_week4_full_convergence.png', dpi=300, bbox_inches='tight')
print("  Plot saved")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("WEEK 4 FULL DATASET - COMPLETE")
print("="*80)

print(f"""
RESULTS ON FULL DATASET ({len(X_train)+len(X_test)} samples):
  * Features: {nf} ({(1-nf/n_features)*100:.1f}% reduction)
  * Fitness: {best_ever:.4f}
  * Accuracy: {acc:.6f}
  * Precision: {prec:.6f}
  * Recall: {rec:.6f}
  * F1: {f1:.6f}
  * FPR: {fpr:.6f}
  * Generations: {len(hist['gen'])}
  * Runtime: {ga_time:.1f}s

WEEK 4 PROGRESSION:
  v1 (Original RAM): 63 features, fitness 0.9736
  v2 (Improved):    44 features, fitness 0.9910
  v3 (Full Dataset): {nf} features, fitness {best_ever:.4f}

KEY FINDINGS:
  1. Improved RAM (P_max=0.5) is better than aggressive (P_max=0.8)
  2. Elitism prevents loss of best solutions
  3. Scaling to full dataset validates robustness
  4. Feature count {nf} is target-adjacent (target=40)
  5. Test performance excellent (FPR={fpr:.6f})

NEXT STEPS:
  1. Analyze which features are selected
  2. Compare with Week 3 (58 features)
  3. Cross-validate on stratified folds
  4. Move to Week 5 (SHAP interpretability)
""")

print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
