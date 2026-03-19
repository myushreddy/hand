"""
Week 3: GA-RAM Implementation (Memory Efficient)

Fast GA with:
- Selective column loading (only top 500 MI features)
- Small population (30) and reduced trees (50)
- Aggressive feature penalty encouraging ~50 features
- Early stopping on convergence
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

# ============================================================================
# CONFIG
# ============================================================================

POPULATION_SIZE = 30
TOURNAMENT_SIZE = 3
N_GENERATIONS = 50
RANDOM_STATE = 42
N_JOBS = 1
RF_N_TREES = 50
FEATURE_POOL_SIZE = 500

print("="*80)
print("WEEK 3: GA-RAM (Memory Efficient)")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA EFFICIENTLY
# ============================================================================

print("\nSTEP 1: Loading data (selective columns)...")
load_start = time.time()

# First load MI scores to get feature names
print("  [1/3] Loading MI scores...")
mi_df = pd.read_csv('results/metrics/mi_scores_full_dataset.csv')
mi_df_sorted = mi_df.sort_values('mi_score', ascending=False)
top_k_features = mi_df_sorted.head(FEATURE_POOL_SIZE)['feature'].tolist()
print(f"  Top 500 MI threshold: {mi_df_sorted.iloc[FEATURE_POOL_SIZE-1]['mi_score']:.6f}")

# Now load only those columns + class label
print("  [2/3] Loading dataset (selected columns only)...")
cols_to_load = top_k_features + ['CLASS']
df = pd.read_csv('data/processed/dataset_with_labels_full.csv', 
                  usecols=cols_to_load, low_memory=False)

X = df.drop('CLASS', axis=1)
y = df['CLASS']
print(f"  Loaded: {len(df):,} x {X.shape[1]}")

# Train/test split
print("  [3/3] Creating split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

load_time = time.time() - load_start
print(f"Load time: {load_time:.1f}s")

feature_names = list(X.columns)
n_features = len(feature_names)

# ============================================================================
# GA FUNCTIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: GA Functions")
print("="*80)

def initialize_population(pop_size, n_features):
    """Init with 10-30% features."""
    population = []
    for _ in range(pop_size):
        activity = np.random.uniform(0.1, 0.3)
        chromosome = np.random.binomial(1, activity, n_features)
        n_sel = np.sum(chromosome)
        while n_sel < 10 or n_sel > int(0.3 * n_features):
            activity = np.random.uniform(0.1, 0.3)
            chromosome = np.random.binomial(1, activity, n_features)
            n_sel = np.sum(chromosome)
        population.append(chromosome)
    return population

def evaluate_fitness(population, X_train, y_train, X_test, y_test, feature_names):
    """Fitness = accuracy - penalty for >50 features."""
    fitness_scores = []
    for chromosome in population:
        selected_idx = np.where(chromosome == 1)[0]
        n_sel = len(selected_idx)
        
        if n_sel == 0:
            fitness_scores.append(0.0)
            continue
        
        try:
            sel_features = [feature_names[i] for i in selected_idx]
            rf = RandomForestClassifier(
                n_estimators=RF_N_TREES, random_state=RANDOM_STATE, n_jobs=N_JOBS
            )
            rf.fit(X_train[sel_features], y_train)
            acc = accuracy_score(y_test, rf.predict(X_test[sel_features]))
            
            # Strong penalty for >50 features
            if n_sel > 50:
                penalty = ((n_sel - 50) / 100.0) ** 2 * 0.2
            else:
                penalty = 0.0
            
            fitness_scores.append(max(0, acc - penalty))
        except:
            fitness_scores.append(0.0)
    
    return np.array(fitness_scores)

def tournament_selection(population, fitness, size=3):
    """Tournament selection."""
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        idxs = np.random.choice(pop_size, size, replace=False)
        best_idx = idxs[np.argmax(fitness[idxs])]
        selected.append(population[best_idx].copy())
    return selected

def crossover(p1, p2, max_features=150):
    """2-point crossover with max feature constraint."""
    n = len(p1)
    pt1 = np.random.randint(1, n - 1)
    pt2 = np.random.randint(pt1 + 1, n)
    
    c1 = np.concatenate([p1[:pt1], p2[pt1:pt2], p1[pt2:]]).astype(int)
    c2 = np.concatenate([p2[:pt1], p1[pt1:pt2], p2[pt2:]]).astype(int)
    
    # Enforce max features
    for c in [c1, c2]:
        n_sel = np.sum(c)
        if n_sel > max_features:
            idxs = np.where(c == 1)[0]
            to_remove = np.random.choice(idxs, n_sel - max_features, replace=False)
            c[to_remove] = 0
    
    return c1, c2

# ============================================================================
# RUN GA
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Running GA")
print("="*80)

print(f"\nPopulation: {POPULATION_SIZE}, Generations: {N_GENERATIONS}")
population = initialize_population(POPULATION_SIZE, n_features)
n_sel = [np.sum(c) for c in population]
print(f"Initial: avg={np.mean(n_sel):.0f} features (range {np.min(n_sel)}-{np.max(n_sel)})")

# Evaluate initial
fitness = evaluate_fitness(population, X_train, y_train, X_test, y_test, feature_names)
print(f"Initial fitness: best={np.max(fitness):.4f}, avg={np.mean(fitness):.4f}")

# History
history = {'gen': [], 'best_fit': [], 'avg_fit': [], 'best_n_feat': [], 'avg_n_feat': []}

# GA loop
print(f"\n{'Gen':<4} {'Best':<8} {'Avg':<8} {'Feats':<6}")
print("-" * 30)

ga_start = time.time()
best_overall = 0
best_chromosome = None

for gen in range(N_GENERATIONS):
    best_idx = np.argmax(fitness)
    best_f = fitness[best_idx]
    best_c = population[best_idx]
    n_f = np.sum(best_c)
    
    history['gen'].append(gen)
    history['best_fit'].append(float(best_f))
    history['avg_fit'].append(float(np.mean(fitness)))
    history['best_n_feat'].append(int(n_f))
    history['avg_n_feat'].append(float(np.mean([np.sum(c) for c in population])))
    
    if best_f > best_overall:
        best_overall = best_f
        best_chromosome = best_c.copy()
    
    print(f"{gen:<4} {best_f:<8.4f} {np.mean(fitness):<8.4f} {n_f:<6}")
    
    # Early stop
    if np.var(fitness) < 1e-6 and gen > N_GENERATIONS // 4:
        print(f"  Early stop at gen {gen}")
        break
    
    # Selection, crossover
    selected = tournament_selection(population, fitness, TOURNAMENT_SIZE)
    new_pop = []
    for i in range(0, len(selected) - 1, 2):
        c1, c2 = crossover(selected[i], selected[i+1], max_features=150)
        new_pop.append(c1)
        new_pop.append(c2)
    if len(selected) % 2 == 1:
        new_pop.append(selected[-1].copy())
    
    population = new_pop[:POPULATION_SIZE]
    fitness = evaluate_fitness(population, X_train, y_train, X_test, y_test, feature_names)

ga_time = time.time() - ga_start

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Results")
print("="*80)

best_n_feat = np.sum(best_chromosome)
selected_idx = np.where(best_chromosome == 1)[0]
selected_names = [feature_names[i] for i in selected_idx]

print(f"\nBest solution:")
print(f"  Accuracy (GA): {best_overall:.4f}")
print(f"  Features: {best_n_feat}/{n_features} ({100*(1-best_n_feat/n_features):.1f}% reduction)")

# Train final model
print(f"\nTraining final model...")
rf_final = RandomForestClassifier(
    n_estimators=RF_N_TREES, random_state=RANDOM_STATE, n_jobs=N_JOBS
)
rf_final.fit(X_train[selected_names], y_train)
y_pred = rf_final.predict(X_test[selected_names])

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\nFinal metrics:")
print(f"  Accuracy: {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall: {rec:.4f}")
print(f"  F1: {f1:.4f}")
print(f"  FPR: {fpr:.6f}")

# ============================================================================
# SAVE
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Saving...")
print("="*80)

# Features
with open('results/metrics/ga_week3_selected_features.pkl', 'wb') as f:
    pickle.dump(selected_names, f)
with open('results/metrics/ga_week3_selected_features.txt', 'w') as f:
    for name in selected_names:
        f.write(f"{name}\n")
print(f"  Saved: {len(selected_names)} features")

# Model
np.save('results/metrics/ga_week3_best_chromosome.npy', best_chromosome)
with open('models/rf_model_ga_week3.pkl', 'wb') as f:
    pickle.dump(rf_final, f)
print(f"  Saved: Model and chromosome")

# Metrics
metrics = {
    'generations': len(history['gen']),
    'best_fitness_ga': float(best_overall),
    'final_accuracy': float(acc),
    'final_precision': float(prec),
    'final_recall': float(rec),
    'final_f1': float(f1),
    'final_fpr': float(fpr),
    'confusion': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    'n_features_selected': int(best_n_feat),
    'n_features_total': int(n_features),
    'feature_reduction_percent': float((1 - best_n_feat/n_features)*100),
    'ga_runtime_sec': float(ga_time),
    'timestamp': datetime.now().isoformat()
}

with open('results/metrics/ga_week3_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved: Metrics")

# History
hist_data = {
    'generation': history['gen'],
    'best_fitness': history['best_fit'],
    'avg_fitness': history['avg_fit'],
    'best_n_features': history['best_n_feat'],
    'avg_n_features': history['avg_n_feat']
}
with open('results/metrics/ga_week3_history.json', 'w') as f:
    json.dump(hist_data, f, indent=2)
print(f"  Saved: History")

# ============================================================================
# PLOT
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Plotting...")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GA Week 3: Convergence Analysis', fontsize=16, fontweight='bold')

# Fitness
ax = axes[0, 0]
ax.plot(history['gen'], history['best_fit'], 'g-', linewidth=2, label='Best')
ax.plot(history['gen'], history['avg_fit'], 'b-', linewidth=2, label='Average')
ax.fill_between(history['gen'], history['avg_fit'], history['best_fit'], alpha=0.2)
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')
ax.set_title('Fitness Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# Features
ax = axes[0, 1]
ax.plot(history['gen'], history['best_n_feat'], 'g-', linewidth=2, label='Best')
ax.plot(history['gen'], history['avg_n_feat'], 'b-', linewidth=2, label='Average')
ax.axhline(y=50, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Target')
ax.set_xlabel('Generation')
ax.set_ylabel('Features')
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
ax.set_title('Final Performance')
ax.set_ylim([0.9, 1.01])
for bar, val in zip(bars, vals):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Summary
ax = axes[1, 1]
ax.axis('off')
sum_text = f"""WEEK 3 RESULTS

Best Solution:
  * Fitness: {best_overall:.4f}
  * Features: {best_n_feat}/{n_features}
  * Reduction: {(1-best_n_feat/n_features)*100:.1f}%

Performance:
  * Accuracy: {acc:.4f}
  * Precision: {prec:.4f}
  * Recall: {rec:.4f}
  * F1: {f1:.4f}

Config:
  * Population: {POPULATION_SIZE}
  * Generations: {len(history['gen'])}
  * Trees: {RF_N_TREES}

Runtime:
  * GA: {ga_time:.1f}s
"""
ax.text(0.05, 0.95, sum_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/plots/ga_week3_convergence.png', dpi=300, bbox_inches='tight')
print("  Saved: results/plots/ga_week3_convergence.png")
plt.close()

# ============================================================================
# DONE
# ============================================================================

print("\n" + "="*80)
print("WEEK 3 COMPLETE")
print("="*80)

print(f"""
Summary:
  * GA framework: OK
  * Features: {n_features} -> {best_n_feat}
  * Accuracy: {acc:.4f}
  * Generations: {len(history['gen'])}

Outputs:
  * models/rf_model_ga_week3.pkl
  * results/metrics/ga_week3_*
  * results/plots/ga_week3_convergence.png

Next: Week 4 - Add Rank-Based Adaptive Mutation
""")

print(f"Done! ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
print("="*80)
