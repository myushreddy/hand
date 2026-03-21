"""
Week 4: GA-RAM (Improved - Tuned Parameters)

Issues from first attempt:
1. Mutation too aggressive -> fitness collapsed at Gen 1
2. Stuck at poor local optimum (fitness 0.39 for 20+ generations)
3. Selected 63 instead of ~40 features

Fixes:
1. More conservative mutation rates (P_max reduced to 0.5)
2. Better fitness function tuning
3. Keep best solution from previous generation (elitism)
4. Allow more recovery time before early stopping
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
print("WEEK 4: GA-RAM (Improved - Tuned Parameters)")
print("="*80)

# CONFIG
POPULATION_SIZE = 20
TOURNAMENT_SIZE = 3
N_GENERATIONS = 100
RANDOM_STATE = 42
N_JOBS = 1
RF_N_TREES = 30
FEATURE_POOL_SIZE = 500

# RAM Parameters - TUNED
P_MAX = 0.5        # Max mutation for worst (reduced from 0.8)
P_MIN = 0.05       # Min mutation for best (reduced from 0.1)
ELITISM = True     # Keep best solutions
ELITISM_SIZE = 2   # Keep top 2
TARGET_FEATURES = 40  # Target feature count

print(f"\nConfiguration:")
print(f"  Population: {POPULATION_SIZE}")
print(f"  Generations: {N_GENERATIONS}")
print(f"  RAM: P_max={P_MAX}, P_min={P_MIN}")
print(f"  Elitism: {ELITISM} (keep top {ELITISM_SIZE})")
print(f"  Target features: {TARGET_FEATURES}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Loading data...")
print("="*80)

load_start = time.time()

print("  [1/3] MI scores...")
mi_df = pd.read_csv('results/metrics/mi_scores_full_dataset.csv')
mi_sorted = mi_df.sort_values('mi_score', ascending=False)
top_features = mi_sorted.head(FEATURE_POOL_SIZE)['feature'].tolist()

print("  [2/3] Loading dataset sample...")
cols = top_features + ['CLASS']
df = pd.read_csv('data/processed/dataset_with_labels_full.csv',
                  usecols=cols, nrows=6250, low_memory=False)

X = df.drop('CLASS', axis=1).head(5000)
y = df['CLASS'].head(5000)
print(f"  Data: {len(X)} x {X.shape[1]}")

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
        act = np.random.uniform(0.08, 0.25)  # Slightly more conservative
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
            
            # TUNED penalty: target 40 features with moderate penalty
            if len(sel_idx) > TARGET_FEATURES:
                # Linear penalty (not quadratic) for stability
                pen = (len(sel_idx) - TARGET_FEATURES) * 0.002
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

def rank_based_mutation(pop, fit, p_max=0.5, p_min=0.05):
    """Conservative rank-based mutation with feature bounds."""
    n_pop = len(pop)
    sorted_indices = np.argsort(fit)
    
    mutated_pop = []
    for rank, idx in enumerate(sorted_indices):
        chrom = pop[idx].copy()
        
        # More conservative mutation probability
        p_mutation = p_max - (p_max - p_min) * rank / (n_pop - 1)
        
        # Mutation: flip bits with probability p_mutation
        for i in range(len(chrom)):
            if np.random.random() < p_mutation:
                chrom[i] = 1 - chrom[i]
        
        # Enforce feature bounds (10-120 features)
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
# RUN GA WITH IMPROVED RAM
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Running GA with Improved RAM")
print("="*80)

pop = init_population(POPULATION_SIZE, n_features)
n_sel_list = [np.sum(c) for c in pop]
print(f"\nInitial: avg={np.mean(n_sel_list):.0f}, range={np.min(n_sel_list)}-{np.max(n_sel_list)}")

fit = eval_fitness(pop, X_train, y_train, X_test, y_test, feature_names)
print(f"Fitness: best={np.max(fit):.4f}, avg={np.mean(fit):.4f}")

hist = {'gen': [], 'best': [], 'avg': [], 'nfeat': [], 'navg': [], 'pm': []}

print(f"\n{'Gen':<4} {'Best':<8} {'Avg':<8} {'Feat':<6} {'P_avg':<8}")
print("-" * 40)

ga_start = time.time()
best_ever = 0
best_chrom = None
stagnation = 0

for gen in range(N_GENERATIONS):
    best_idx = np.argmax(fit)
    best_f = fit[best_idx]
    best_c = pop[best_idx]
    nf = np.sum(best_c)
    
    # Create elite backup before mutation
    if ELITISM:
        elite_indices = np.argsort(fit)[-ELITISM_SIZE:]
        elite_pop = [pop[i].copy() for i in elite_indices]
        elite_fit = [fit[i] for i in elite_indices]
    
    # Avg mutation probability
    ranks = np.argsort(fit)
    avg_pm = np.mean([P_MAX - (P_MAX - P_MIN) * r / (POPULATION_SIZE - 1) for r in range(POPULATION_SIZE)])
    
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
    
    print(f"{gen:<4} {best_f:<8.4f} {np.mean(fit):<8.4f} {nf:<6} {avg_pm:<8.4f}")
    
    # GA step: Selection -> Crossover -> Mutation
    parents = select_parents(pop, fit)
    
    pop = []
    for i in range(0, len(parents) - 1, 2):
        c1, c2 = breed(parents[i], parents[i+1])
        pop.append(c1)
        pop.append(c2)
    if len(parents) % 2:
        pop.append(parents[-1].copy())
    
    pop = pop[:POPULATION_SIZE]
    
    # Apply conservative mutation
    pop = rank_based_mutation(pop, fit, p_max=P_MAX, p_min=P_MIN)
    
    # ELITISM: Replace worst with elite if worse
    if ELITISM:
        fit_new = eval_fitness(pop, X_train, y_train, X_test, y_test, feature_names)
        pop_new = []
        fit_list = []
        for i in range(len(pop)):
            pop_new.append(pop[i])
            fit_list.append(fit_new[i])
        
        # Keep elite members if they're better
        for j, (elite_c, elite_f) in enumerate(zip(elite_pop, elite_fit)):
            worst_idx = np.argmin(fit_list)
            if elite_f > fit_list[worst_idx]:
                pop_new[worst_idx] = elite_c.copy()
                fit_list[worst_idx] = elite_f
        
        pop = pop_new
        fit = np.array(fit_list)
    else:
        fit = eval_fitness(pop, X_train, y_train, X_test, y_test, feature_names)
    
    # Stop if 15 generations of no improvement AND at least gen 30
    if stagnation > 15 and gen > 30:
        print(f"  Stagnation stop at gen {gen}")
        break

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

print(f"\nBest GA solution (improved RAM):")
print(f"  Fitness: {best_ever:.4f}")
print(f"  Features: {nf}/{n_features}")
print(f"  Reduction: {(1-nf/n_features)*100:.1f}%")

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

with open('results/metrics/ga_week4_selected_features.pkl', 'wb') as f:
    pickle.dump(sel_names, f)
with open('results/metrics/ga_week4_selected_features.txt', 'w') as f:
    for n in sel_names:
        f.write(f"{n}\n")
print(f"  Saved features ({len(sel_names)})")

np.save('results/metrics/ga_week4_best_chromosome.npy', best_chrom)
with open('models/rf_model_ga_week4.pkl', 'wb') as f:
    pickle.dump(rf, f)
print(f"  Saved model")

metrics = {
    'type': 'GA_WEEK4_RAM_IMPROVED',
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
    'p_max': float(P_MAX),
    'p_min': float(P_MIN),
    'target_features': int(TARGET_FEATURES),
    'timestamp': datetime.now().isoformat()
}

with open('results/metrics/ga_week4_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved metrics")

with open('results/metrics/ga_week4_history.json', 'w') as f:
    json.dump(hist, f, indent=2)
print(f"  Saved history")

# ============================================================================
# PLOT
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Plotting")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Week 4: Improved GA with Rank-Based Adaptive Mutation (RAM)', fontsize=16, fontweight='bold')

ax = axes[0, 0]
ax.plot(hist['gen'], hist['best'], 'g-', linewidth=2, label='Best')
ax.plot(hist['gen'], hist['avg'], 'b--', linewidth=1.5, label='Average')
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')
ax.set_title('Fitness Convergence (Improved RAM)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(hist['gen'], hist['nfeat'], 'g-', linewidth=2, label='Best')
ax.plot(hist['gen'], hist['navg'], 'b--', linewidth=1.5, label='Average')
ax.axhline(y=TARGET_FEATURES, color='r', linestyle=':', linewidth=2, alpha=0.7, label=f'Target ({TARGET_FEATURES})')
ax.set_xlabel('Generation')
ax.set_ylabel('Number of Features')
ax.set_title('Feature Count Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(hist['gen'], hist['pm'], 'purple', linewidth=2)
ax.axhline(y=P_MAX, color='r', linestyle='--', alpha=0.5, label=f'P_max={P_MAX}')
ax.axhline(y=P_MIN, color='g', linestyle='--', alpha=0.5, label=f'P_min={P_MIN}')
ax.set_xlabel('Generation')
ax.set_ylabel('Avg Mutation Probability')
ax.set_title('Rank-Based Mutation Rates')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.axis('off')
txt = f"""WEEK 4 - Improved RAM Results

Configuration:
  RAM: P_max={P_MAX}, P_min={P_MIN}
  Elitism: {ELITISM} (keep {ELITISM_SIZE})
  Target: {TARGET_FEATURES} features

Best Solution:
  Fitness: {best_ever:.4f}
  Features: {nf}/{n_features}
  Reduction: {(1-nf/n_features)*100:.1f}%

Performance:
  Accuracy: {acc:.4f}
  Precision: {prec:.4f}
  Recall: {rec:.4f}
  F1: {f1:.4f}

Vs Week 3:
  Week 3: 58 features
  Week 4: {nf} features
  Change: {nf-58:+d} features
"""
ax.text(0.05, 0.95, txt, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('results/plots/ga_week4_convergence.png', dpi=300, bbox_inches='tight')
print("  Saved plot")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("WEEK 4 COMPLETE")
print("="*80)

print(f"""
WEEK 4 RESULTS (Improved RAM):
  * Features: {nf} (target was {TARGET_FEATURES})
  * Reduction: {(1-nf/n_features)*100:.1f}%
  * Fitness: {best_ever:.4f}
  * Accuracy: {acc:.4f}
  * Generations: {len(hist['gen'])}
  * Runtime: {ga_time:.1f}s

KEY IMPROVEMENTS IN V2:
  1. Conservative mutation (P_max: 0.8 -> 0.5)
  2. Elitism: Keep top 2 performers
  3. Linear penalty (not quadratic)
  4. Better stagnation detection
  5. Feature bounds: 10-120 (not 10-150)

VS WEEK 3 (No Mutation):
  Week 3: 58 features, 16 generations, early stop
  Week 4: {nf} features, {len(hist['gen'])} generations, RAM exploration

NEXT STEPS:
  1. Run on full dataset (28k samples)
  2. Validate feature selections
  3. Analyze core malware-detecting features
  4. Compare with other methods
""")

print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
