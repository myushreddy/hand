"""
Week 3: GA-RAM Implementation (Optimized Version)

Goal: Implement genetic algorithm to evolve feature subsets
- Fast data loading with progress tracking
- Chromosome initialization (30 random feature subsets)
- Fitness function (Random Forest accuracy with penalty for feature count)
- Tournament selection
- 2-point crossover with max feature constraint
- Early stopping when convergence detected

Inputs: Top 500 features from Week 2 MI selection
Output: Basic GA working, convergence tracking
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

# ============================================================================
# CONFIGURATION
# ============================================================================

POPULATION_SIZE = 30          # Number of chromosomes (feature subsets) - small for memory
TOURNAMENT_SIZE = 3           # Tournament selection group size
N_GENERATIONS = 50            # Number of generations
RANDOM_STATE = 42             # For reproducibility
N_JOBS = 1                    # Single job to avoid parallel overhead
RF_N_TREES = 50               # Trees in Random Forest - reduced for speed

# Feature pool selection
USE_K500 = True               # True = use 500 features, False = use 155
FEATURE_POOL_SIZE = 500 if USE_K500 else 155

print("="*80)
print("WEEK 3: GA-RAM IMPLEMENTATION (Optimized)")
print("="*80)
print(f"\nConfiguration:")
print(f"  Population size: {POPULATION_SIZE}")
print(f"  Tournament size: {TOURNAMENT_SIZE}")
print(f"  Generations: {N_GENERATIONS}")
print(f"  Feature pool: k={FEATURE_POOL_SIZE}")
print(f"  Random Forest trees: {RF_N_TREES}")
print(f"  RF jobs (parallel): {N_JOBS}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Loading data and selected features")
print("="*80)

load_start = time.time()

# Load dataset
print("\n[1/4] Loading dataset...")
df_start = time.time()
df = pd.read_csv('data/processed/dataset_with_labels_full.csv', low_memory=False)
print(f"  * Loaded in {time.time() - df_start:.1f}s")

X = df.drop(['CLASS', 'SHA256', 'NOME', 'PACOTE', 'API'], axis=1, errors='ignore')
y = df['CLASS']

print(f"  * Dataset: {len(df):,} samples x {X.shape[1]:,} features")

# Load MI scores to identify top k features
print(f"\n[2/4] Loading MI scores to select top {FEATURE_POOL_SIZE} features...")
mi_df = pd.read_csv('results/metrics/mi_scores_full_dataset.csv')
mi_df_sorted = mi_df.sort_values('mi_score', ascending=False)

# Get top k feature names
top_k_features = mi_df_sorted.head(FEATURE_POOL_SIZE)['feature'].tolist()
available_features = [f for f in top_k_features if f in X.columns]

X_selected = X[available_features]
print(f"  * Using {len(available_features)} features (MI > {mi_df_sorted.iloc[FEATURE_POOL_SIZE-1]['mi_score']:.6f})")

# Create consistent train/test split
print(f"\n[3/4] Creating train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"  * Train: {len(X_train):,} samples")
print(f"  * Test:  {len(X_test):,} samples")
print(f"  * Malware in test: {int(y_test.sum()):,} samples")

load_time = time.time() - load_start
print(f"\n* Data loading complete: {load_time:.1f}s")

# ============================================================================
# STEP 2: GA FUNCTIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Defining GA Functions")
print("="*80)

def initialize_population(pop_size, n_features, feature_activity_range=(0.1, 0.3)):
    """Initialize population with 10-30% of features selected."""
    population = []
    
    for _ in range(pop_size):
        activity = np.random.uniform(feature_activity_range[0], feature_activity_range[1])
        chromosome = np.random.binomial(1, activity, n_features)
        
        # Ensure at least 10 features and at most 30% of features
        n_selected = np.sum(chromosome)
        while n_selected < 10 or n_selected > int(0.3 * n_features):
            activity = np.random.uniform(feature_activity_range[0], feature_activity_range[1])
            chromosome = np.random.binomial(1, activity, n_features)
            n_selected = np.sum(chromosome)
        
        population.append(chromosome)
    
    return population


def evaluate_fitness(population, X_train, y_train, X_test, y_test, feature_names):
    """Evaluate fitness for population with feature penalty."""
    fitness_scores = []
    trained_models = []
    
    for idx, chromosome in enumerate(population):
        selected_indices = np.where(chromosome == 1)[0]
        n_selected = len(selected_indices)
        
        if n_selected == 0:
            fitness_scores.append(0.0)
            trained_models.append(None)
            continue
        
        try:
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Train RF
            rf = RandomForestClassifier(
                n_estimators=RF_N_TREES,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                verbose=0
            )
            
            rf.fit(X_train[selected_features], y_train)
            y_pred = rf.predict(X_test[selected_features])
            accuracy = accuracy_score(y_test, y_pred)
            
            # AGGRESSIVE quadratic penalty: target ~50 features
            if n_selected > 50:
                excess = n_selected - 50
                feature_penalty = (excess / 100.0) ** 2 * 0.2
            else:
                feature_penalty = 0.0
            
            fitness = max(0, accuracy - feature_penalty)
            fitness_scores.append(fitness)
            trained_models.append(rf)
            
        except Exception as e:
            print(f"    Error evaluating chromosome {idx}: {str(e)[:50]}")
            fitness_scores.append(0.0)
            trained_models.append(None)
    
    return np.array(fitness_scores), trained_models


def tournament_selection(population, fitness_scores, tournament_size=3):
    """Select parents using tournament selection."""
    selected_parents =[]
    pop_size = len(population)
    
    for _ in range(pop_size):
        tournament_indices = np.random.choice(pop_size, tournament_size, replace=False)
        best_idx = tournament_indices[np.argmax(fitness_scores[tournament_indices])]
        selected_parents.append(population[best_idx].copy())
    
    return selected_parents


def two_point_crossover(parent1, parent2, max_features=150):
    """Perform 2-point crossover with max feature constraint."""
    n_genes = len(parent1)
    
    point1 = np.random.randint(1, n_genes - 1)
    point2 = np.random.randint(point1 + 1, n_genes)
    
    offspring1 = np.concatenate([
        parent1[:point1],
        parent2[point1:point2],
        parent1[point2:]
    ]).astype(int)
    
    offspring2 = np.concatenate([
        parent2[:point1],
        parent1[point1:point2],
        parent2[point2:]
    ]).astype(int)
    
    # Enforce max feature constraint
    for offspring in [offspring1, offspring2]:
        n_selected = np.sum(offspring)
        if n_selected > max_features:
            excess = n_selected - max_features
            selected_indices = np.where(offspring == 1)[0]
            to_remove = np.random.choice(selected_indices, excess, replace=False)
            offspring[to_remove] = 0
    
    return offspring1, offspring2


def create_new_generation(selected_parents, feature_names, X_train, y_train, X_test, y_test):
    """Create new generation through crossover."""
    new_population = []
    
    for i in range(0, len(selected_parents) - 1, 2):
        offspring1, offspring2 = two_point_crossover(
            selected_parents[i],
            selected_parents[i + 1],
            max_features=150
        )
        new_population.append(offspring1)
        new_population.append(offspring2)
    
    if len(selected_parents) % 2 == 1:
        new_population.append(selected_parents[-1].copy())
    
    new_population = new_population[:POPULATION_SIZE]
    
    fitness_scores, _ = evaluate_fitness(
        new_population, X_train, y_train, X_test, y_test, feature_names
    )
    
    return new_population, fitness_scores


# ============================================================================
# STEP 3: RUN GA
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Running Genetic Algorithm")
print("="*80)

feature_names = list(X_selected.columns)
n_features = len(feature_names)

print(f"\nInitializing population of {POPULATION_SIZE} chromosomes...")
population = initialize_population(POPULATION_SIZE, n_features)

initial_n_selected = [np.sum(c) for c in population]
print(f"  Average features: {np.mean(initial_n_selected):.0f} (range: {np.min(initial_n_selected)}-{np.max(initial_n_selected)})")

# Track history
history = {
    'generation': [],
    'best_fitness': [],
    'avg_fitness': [],
    'worst_fitness': [],
    'best_n_features': [],
    'avg_n_features': [],
    'best_chromosome': None,
    'best_fitness_value': 0
}

print(f"\nEvaluating initial population...")
fitness_scores, trained_models = evaluate_fitness(
    population, X_train, y_train, X_test, y_test, feature_names
)

print(f"  Best: {np.max(fitness_scores):.4f} | Avg: {np.mean(fitness_scores):.4f} | Worst: {np.min(fitness_scores):.4f}")

ga_start = time.time()

# Main GA loop
print(f"\n{'Gen':<5} {'Best':<8} {'Avg':<8} {'Worst':<8} {'Feat':<6} {'Time':<7}")
print("-" * 60)

early_stop_count = 0
for generation in range(N_GENERATIONS):
    gen_start = time.time()
    
    # Record history
    best_idx = np.argmax(fitness_scores)
    best_chromosome = population[best_idx]
    best_fitness = fitness_scores[best_idx]
    
    history['generation'].append(generation)
    history['best_fitness'].append(best_fitness)
    history['avg_fitness'].append(np.mean(fitness_scores))
    history['worst_fitness'].append(np.min(fitness_scores))
    history['best_n_features'].append(np.sum(best_chromosome))
    history['avg_n_features'].append(np.mean([np.sum(c) for c in population]))
    
    if best_fitness > history['best_fitness_value']:
        history['best_fitness_value'] = best_fitness
        history['best_chromosome'] = best_chromosome.copy()
    
    # Print progress
    gen_time = time.time() - gen_start
    print(f"{generation:<5} {best_fitness:<8.4f} {np.mean(fitness_scores):<8.4f} "
          f"{np.min(fitness_scores):<8.4f} {np.sum(best_chromosome):<6} {gen_time:<7.2f}s")
    
    # Early stopping: check for convergence
    fitness_variance = np.var(fitness_scores)
    if fitness_variance < 1e-6 and generation > N_GENERATIONS // 4:
        early_stop_count += 1
        if early_stop_count >= 3:
            print(f"\n  ! Early stopping at generation {generation}")
            print(f"    Population converged (variance < 1e-6)")
            break
    else:
        early_stop_count = 0
    
    # Selection
    selected_parents = tournament_selection(population, fitness_scores, TOURNAMENT_SIZE)
    
    # Crossover
    population, fitness_scores = create_new_generation(
        selected_parents, feature_names, X_train, y_train, X_test, y_test
    )

ga_time = time.time() - ga_start

# ============================================================================
# STEP 4: RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: GA Results")
print("="*80)

best_idx = len(history['best_fitness']) - 1
best_chromosome = history['best_chromosome']
best_fitness = history['best_fitness_value']
best_n_features = np.sum(best_chromosome)

print(f"\nBest Solution Found:")
print(f"  Generation: {best_idx}")
print(f"  Accuracy: {best_fitness:.4f} ({best_fitness*100:.2f}%)")
print(f"  Features: {best_n_features} out of {n_features}")
print(f"  Reduction: {(1 - best_n_features/n_features)*100:.1f}%")

# Get selected feature names
selected_indices = np.where(best_chromosome == 1)[0]
selected_feature_names = [feature_names[i] for i in selected_indices]

print(f"\nFirst 20 Selected Features:")
for i, feat in enumerate(selected_feature_names[:20], 1):
    print(f"  {i}. {feat}")

# Train final model
print(f"\nTraining final model...")
rf_final = RandomForestClassifier(
    n_estimators=RF_N_TREES,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)

rf_final.fit(X_train[selected_feature_names], y_train)
y_pred = rf_final.predict(X_test[selected_feature_names])

final_accuracy = accuracy_score(y_test, y_pred)
final_precision = precision_score(y_test, y_pred)
final_recall = recall_score(y_test, y_pred)
final_f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn)

print(f"\nFinal Model Performance:")
print(f"  Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"  Precision: {final_precision:.4f} ({final_precision*100:.2f}%)")
print(f"  Recall: {final_recall:.4f} ({final_recall*100:.2f}%)")
print(f"  F1-Score: {final_f1:.4f} ({final_f1*100:.2f}%)")
print(f"  FPR: {fpr:.4f} ({fpr*100:.4f}%)")
print(f"\nConfusion Matrix:")
print(f"  TN: {tn:,} | FP: {fp}")
print(f"  FN: {fn} | TP: {tp:,}")

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Saving Results")
print("="*80)

# Save selected features
print("\n[1/5] Saving selected features...")
with open('results/metrics/ga_week3_selected_features.pkl', 'wb') as f:
    pickle.dump(selected_feature_names, f)

with open('results/metrics/ga_week3_selected_features.txt', 'w') as f:
    for feat in selected_feature_names:
        f.write(f"{feat}\n")

print(f"  * {len(selected_feature_names)} features saved")

# Save best chromosome
print("\n[2/5] Saving best chromosome...")
np.save('results/metrics/ga_week3_best_chromosome.npy', best_chromosome)
print(f"  * Saved")

# Save model
print("\n[3/5] Saving trained model...")
with open('models/rf_model_ga_week3.pkl', 'wb') as f:
    pickle.dump(rf_final, f)
print(f"  * Saved")

# Save metrics
print("\n[4/5] Saving metrics...")
metrics = {
    'generations_run': len(history['generation']),
    'best_fitness': float(best_fitness),
    'final_accuracy': float(final_accuracy),
    'final_precision': float(final_precision),
    'final_recall': float(final_recall),
    'final_f1': float(final_f1),
    'final_fpr': float(fpr),
    'confusion_matrix': {
        'tn': int(tn), 'fp': int(fp),
        'fn': int(fn), 'tp': int(tp)
    },
    'n_features_selected': int(best_n_features),
    'n_features_total': int(n_features),
    'feature_reduction_percent': float((1 - best_n_features/n_features)*100),
    'ga_runtime_seconds': float(ga_time),
    'timestamp': datetime.now().isoformat()
}

with open('results/metrics/ga_week3_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"  * Saved")

# Save history
print("\n[5/5] Saving history...")
history_data = {
    'generation': history['generation'],
    'best_fitness': [float(x) for x in history['best_fitness']],
    'avg_fitness': [float(x) for x in history['avg_fitness']],
    'worst_fitness': [float(x) for x in history['worst_fitness']],
    'best_n_features': [int(x) for x in history['best_n_features']],
    'avg_n_features': [float(x) for x in history['avg_n_features']]
}

with open('results/metrics/ga_week3_history.json', 'w') as f:
    json.dump(history_data, f, indent=4)

print(f"  * Saved")

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Creating Visualizations")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GA Week 3: Basic GA Convergence (Optimized)', fontsize=16, fontweight='bold')

# Fitness convergence
ax = axes[0, 0]
ax.plot(history['generation'], history['best_fitness'], 'g-', linewidth=2, label='Best')
ax.plot(history['generation'], history['avg_fitness'], 'b-', linewidth=2, label='Average')
ax.plot(history['generation'], history['worst_fitness'], 'r--', linewidth=1.5, label='Worst')
ax.fill_between(history['generation'], history['worst_fitness'], history['best_fitness'], alpha=0.2)
ax.set_xlabel('Generation', fontsize=11)
ax.set_ylabel('Fitness (Accuracy)', fontsize=11)
ax.set_title('Fitness Convergence', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Feature count evolution
ax = axes[0, 1]
ax.plot(history['generation'], history['best_n_features'], 'g-', linewidth=2, label='Best')
ax.plot(history['generation'], history['avg_n_features'], 'b-', linewidth=2, label='Average')
ax.axhline(y=50, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Target (50)')
ax.set_xlabel('Generation', fontsize=11)
ax.set_ylabel('Number of Features', fontsize=11)
ax.set_title('Feature Count Evolution', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Performance metrics
ax = axes[1, 0]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [final_accuracy, final_precision, final_recall, final_f1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Final Model Performance', fontsize=12, fontweight='bold')
ax.set_ylim([0.90, 1.01])
for bar, val in zip(bars, metrics_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Summary
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""WEEK 3 RESULTS - Basic GA

Best Solution:
  * Generation: {min(best_idx, len(history['generation'])-1) if history['generation'] else 0}
  * Accuracy: {best_fitness:.4f}
  * Features: {best_n_features}/{n_features}
  * Reduction: {(1-best_n_features/n_features)*100:.1f}%

Performance:
  * Accuracy: {final_accuracy:.4f}
  * Precision: {final_precision:.4f}
  * Recall: {final_recall:.4f}
  * F1: {final_f1:.4f}

GA Config:
  * Population: {POPULATION_SIZE}
  * Generations: {len(history['generation'])}
  * RF Trees: {RF_N_TREES}
  
Runtime:
  * GA: {ga_time:.1f}s
  * Total: {time.time()-load_start:.1f}s
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/plots/ga_week3_convergence.png', dpi=300, bbox_inches='tight')
print(f"* Saved: results/plots/ga_week3_convergence.png")

plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("WEEK 3 COMPLETE")
print("="*80)

print(f"""
Summary:
  * GA framework implemented
  * Features reduced: {n_features} -> {best_n_features}
  * Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)
  * Generations: {len(history['generation'])}

Deliverables:
  * models/rf_model_ga_week3.pkl
  * results/metrics/ga_week3_selected_features.pkl
  * results/metrics/ga_week3_best_chromosome.npy
  * results/metrics/ga_week3_metrics.json
  * results/metrics/ga_week3_history.json
  * results/plots/ga_week3_convergence.png

Next: Week 4 - Add Rank-Based Adaptive Mutation (RAM)
""")

print(f"\nDone! ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
print("="*80)
