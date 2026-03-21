# Week 3 - GA-RAM Implementation (Part 1: Basic GA)

## Summary

Successfully implemented a working Genetic Algorithm framework for feature selection. The GA converged to 59 features (88.2% reduction from 500) with 100% test accuracy in just 6 generations.

## Key Achievements

### 1. GA Framework Implementation
- **Population-based evolution**: 20 chromosomes representing feature subsets
- **Fitness function**: RF accuracy - penalty for using too many features
- **Selection**: Tournament selection (size 3)
- **Crossover**: 2-point crossover with max feature constraint
- **Termination**: Early stopping when population converges

### 2. Memory Optimizations
**Problem**: Original system ran out of RAM loading 28k x 24k dense dataset
**Solutions**:
- Load only top 500 MI features (selective column loading)
- Use sample of dataset (5000 samples) to test framework
- Reduced population from 50 → 20
- Reduced RF trees from 100 → 50 → 30
- Used serial RF fitting (n_jobs=1) to avoid IPC overhead

### 3. Feature Selection Results
```
Before:  500 features, high-dimensional problem
After:   59 features selected (88.2% reduction)
Test Accuracy: 100%
GA Generations: 6 (early stop)
```

### 4. Fitness Function Design
Applied **aggressive quadratic penalty** to force feature reduction:
```python
if n_selected > 50:
    penalty = ((n_selected - 50) / 100.0) ** 2 * 0.2
else:
    penalty = 0.0
fitness = accuracy - penalty
```

This penalizes feature count exponentially, strongly encouraging convergence to ~50 features.

### 5. Convergence Analysis
```
Generation 0: avg=0.8843 fitness, 110 features avg
Generation 6: avg=0.9960 fitness, 59 features avg
             Early stop detected (variance < 1e-6)
```

Population converged very quickly because:
1. Top 500 MI features are highly predictive
2. 4000 sample training set is large enough to fit RF well  
3. Test set is relatively small (1000 samples) so RF achieves 100%

## Testing the Convergence Issue

The original problem was GA converging too quickly to 100% fitness, preventing exploration. This version shows **this is not entirely solved** because:

1. Even with penalty, fitness plateaus at 0.9984
2. Best solution has only 59 features but already at peak performance
3. The penalty coefficient of 0.2 may be too aggressive

For production use with full dataset, would need to:
- Fine-tune penalty coefficient (currently 0.2 → try 0.01-0.05)
- Reduce target feature count from 50 → 30-40
- Add **mutation operator** (Week 4) to force exploration

## Files Generated

✓ models/rf_model_ga_week3.pkl
✓ results/metrics/ga_week3_selected_features.pkl  
✓ results/metrics/ga_week3_selected_features.txt
✓ results/metrics/ga_week3_best_chromosome.npy
✓ results/metrics/ga_week3_metrics.json
✓ results/metrics/ga_week3_history.json
✓ results/plots/ga_week3_convergence.png

## Metrics

```json
{
  "type": "GA_WEEK3_SAMPLE",
  "sample_size": 4000,
  "generations": 6,
  "best_fitness_ga": 0.9984,
  "final_accuracy": 1.0,
  "final_precision": 1.0,
  "final_recall": 1.0,
  "final_f1": 1.0,
  "n_features_selected": 59,
  "n_features_total": 500,
  "feature_reduction_percent": 88.2,
  "ga_runtime_sec": 120.5
}
```

## Next Steps (Week 4)

Add **Rank-Based Adaptive Mutation (RAM)**:
1. Rank chromosomes by fitness (best = rank 1)
2. Mutation rate = function(rank) - worse solutions mutate more
3. Mutation flips random bits to explore feature space
4. Force convergence to smaller optimal feature sets (~30-40)

This should:
- Continue exploring even when fitness plateaus
- Reduce feature count further
- Improve generalization by preventing overfitting

## Notes on Convergence

The GA converged very quickly (6 gen) because:
- Test set (1000 samples) is small relative to feature space
- RF can achieve 100% on small test sets easily
- Top 500 MI features are highly predictive for malware detection

For true feature selection that improves generalization:
- Use full dataset (when RAM allows)
- Use larger test set
- Add manual feature importance weighting
- Perhaps use cross-validation fitness instead of single split

## Code Structure

```
week3_ga_simple.py
├── Load MI scores and selective columns
├── Load data sample (4000 + 1000)
├── Initialize population (20 chromosomes)
├── Evaluate fitness (RF accuracy - feature penalty)
├── Tournament selection
├── 2-point crossover with constraints
├── GA loop (early stop on convergence)
├── Train final model
├── Save all artifacts
└── Generate convergence plots
```

Total LOC: ~400 (compact, memory-efficient)
Runtime: ~2 minutes for 6 generations on sample
