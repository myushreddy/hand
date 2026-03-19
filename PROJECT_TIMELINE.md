# ARM Implementation Project Timeline
**Project:** Android Malware Detection using Adaptive Rank-Based Mutation (GA-RAM)  
**Duration:** January 24 - April 10, 2026 (11 weeks)  
**Dataset:** MH-100K Sample (25,000 apps for development, can scale to 100K)  
**Note:** Using 25k subset for faster prototyping; full dataset available for final evaluation

---

## Week 1: January 24-30, 2026 ✅ COMPLETE
**Focus:** Data Preparation & Baseline Model

### Tasks:
- [x] Load data_sample_25k.csv (25,000 samples × 100 features)
- [x] Merge with labels from mh_100k_labels.csv
- [x] Exploratory Data Analysis (EDA)
  - Class distribution: 88.6% benign, 11.4% malware
  - 95 feature columns (permissions + API calls + intents)
  - No missing values
- [x] Data preprocessing and validation
- [x] Train/test split (80:20) = 19,940 train / 4,985 test
- [x] Create baseline Random Forest model (100 trees, all features)
- [x] Document initial findings

### Deliverables:
- ✅ `scripts/week1_baseline_model.py` (complete script)
- ✅ `notebooks/01_Week1_EDA_Baseline.ipynb` (interactive notebook)
- ✅ `models/baseline_rf_model.pkl` (trained model)
- ✅ Baseline accuracy: **93.52%**
- ✅ `data/processed/dataset_with_labels.csv` (24,925 samples)
- ✅ Multiple visualizations and metrics

### Results:
- Test Accuracy: 93.52%
- Precision: 79.05%
- Recall: 58.55% (needs improvement ⚠️)
- F1-Score: 67.27%
- FPR: 1.99% ✓ (excellent!)

**Status:** ✅ COMPLETE - Ready for Week 2

---

## Week 2: January 31 - February 6, 2026 ✅ COMPLETE
**Focus:** Mutual Information (MI) Feature Selection

### Tasks:
- [x] Implement MI-based feature filtering
  - Calculate MI scores for all 24,837 features (full dataset!)
  - Rank features by relevance to CLASS label
  - Select top k features (experiment with k=155, 500, 1000)
- [x] Validate MI results
  - Compare feature importance metrics
  - Visualize feature distributions
- [x] Create feature subset for GA-RAM input
- [x] Retrain Random Forest with selected features (k=155)
- [x] Compare performance with baseline

### Deliverables:
- ✅ `mi_scores_full_dataset.csv` (MI scores for all features)
- ✅ Selected feature subsets (k=155, k=500, k=1000)
- ✅ Feature importance visualization
- ✅ Improved model with k=155 features

### Results:
- k=155 features: **95.28% accuracy** (baseline: 93.52%)
- k=500 features: ~96% accuracy
- Successfully filtered from 24,837 → 155 core features

**Status:** ✅ COMPLETE - Ready for Week 3

---

## Week 3: February 7-13, 2026 ✅ COMPLETE
**Focus:** GA-RAM Algorithm - Part 1 (Basic GA Framework)

### Tasks:
- [x] Implement population initialization (20 binary chromosomes)
- [x] Implement fitness function with feature penalty
- [x] Implement tournament selection mechanism
- [x] Implement 2-point crossover with constraints
- [x] Build GA loop with early stopping
- [x] Handle memory constraints with selective loading
- [x] Test on sample dataset

### Technical Challenges Solved:
**Problem 1:** System ran out of memory loading 28k × 24k dataset
- **Solution:** Load only top 500 MI features (selective column loading)

**Problem 2:** GA converged too quickly to 100% fitness
- **Solution:** Applied aggressive quadratic penalty for features > 50

**Problem 3:** Serial fitness evaluation too slow
- **Solution:** Reduced population (50→20), trees (100→30), used n_jobs=1

### Deliverables:
- ✅ `scripts/week3_ga_simple.py` (final working GA)
- ✅ `models/rf_model_ga_week3.pkl` (best feature subset model)
- ✅ `results/metrics/ga_week3_selected_features.pkl/.txt`
- ✅ `results/metrics/ga_week3_best_chromosome.npy`
- ✅ `results/metrics/ga_week3_metrics.json`
- ✅ `results/metrics/ga_week3_history.json`
- ✅ `results/plots/ga_week3_convergence.png`
- ✅ `WEEK3_SUMMARY.md` (detailed documentation)

### Results:
- **Feature reduction:** 500 → 59 features (88.2% reduction)
- **Convergence speed:** 6 generations (early stop)
- **Final accuracy:** 100% (on test set)
- **GA fitness:** 0.9984
- **Runtime:** ~85 seconds for full GA

### GA Parameters Used:
```
Population size: 20
Generations: 40 (stopped at 6)
Tournament size: 3
RF trees: 30
Feature pool: 500 (top MI)
Feature penalty: quadratic (target ~50)
```

**Status:** ✅ COMPLETE - Ready for Week 4

---

## Week 4: February 14-20, 2026 🔄 IN PROGRESS
**Focus:** GA-RAM Algorithm - Part 2 (Mutation & Full GA Loop)

### Tasks:
- [ ] Add mutation operator (flip bits in chromosomes)
  - Probability based on fitness rank
  - Lower rank = higher mutation rate
- [ ] Implement Rank-Based Adaptive Mutation (RAM)
  - Rank chromosomes by fitness (worst=1, best=n)
  - Mutation_rate[i] = p_max × (i-1)/(n-1)
  - Apply to low-fitness individuals
- [ ] Integrate mutation into GA loop
- [ ] Scale to full 500-feature dataset
- [ ] Target feature reduction: 500 → 30-40 features

### Expected Improvements Over Week 3:
- Week 3: 59 features, early convergence at gen 6
- Week 4: Target 30-40 features, explore more generations
- Better generalization through aggressive feature reduction

### Deliverables:
- [ ] Complete GA-RAM implementation with mutation
- [ ] Performance on full dataset
- [ ] Analysis of mutation effectiveness

### Key Challenges:
- Handle memory constraints on full dataset (use chunking or downsampling)
- Tune mutation rates for exploration vs exploitation
- Prevent mutation from degrading good solutions

**Status:** 🔄 IN PROGRESS

---

## Week 5: February 21-27, 2026
**Focus:** GA-RAM Training & Feature Selection Validation

### Tasks:
- [ ] Run full GA-RAM on complete dataset (28k samples × 500 features)
  - Monitor convergence
  - Track feature count reduction
  - Track fitness improvement
- [ ] Select optimal feature subset
- [ ] Train final Random Forest with selected features
- [ ] Evaluate on test set
  - Target accuracy: 95-97%
  - Compare against MI-only baseline
- [ ] Save final model and features

### Expected Results:
- ~30-40 optimal features
- Maintain or improve accuracy vs Week 3
- Better generalization

---

## Week 6: February 28 - March 6, 2026
**Focus:** White-Box Adversarial Attacks (FGSM & JSMA)

### Tasks:
- [ ] Implement FGSM attack
  - Binary, additive-only variant
  - Generate adversarial malware samples
- [ ] Implement JSMA attack
  - Saliency-map based approach
  - Target specific features
  - Generate adversarial samples
- [ ] Evaluate attacks on model
  - Measure evasion success rate
  - Evaluate detection accuracy
- [ ] Analyze attack effectiveness

### Deliverables:
- FGSM & JSMA implementations
- Adversarial samples
- Attack evaluation metrics

### Libraries:
- shap library

---

## Week 11: April 4-10, 2026
**Focus:** Final Evaluation, Documentation & Results

### Tasks:
- [ ] Compile all performance metrics
  - General malware: Accuracy, Precision, Recall, F1, FPR
  - Adversarial attacks: Performance for each attack type
  - Zero-day malware: All metrics
- [ ] Create comparison tables
  - Compare with ARM paper's results
  - Baseline vs MI vs GA-RAM comparison
- [ ] Generate visualizations
  - Accuracy comparison charts
  - Confusion matrices
  - ROC curves
  - GA-RAM convergence plots
- [ ] Write comprehensive documentation
  - README with usage instructions
  - Technical report explaining implementation
  - Results summary
- [ ] Code cleanup and commenting
- [ ] Prepare final presentation

### Deliverables:
- Complete performance report
- All visualizations and charts
- Updated README.md
- Technical documentation
- Clean, commented code
- Final project presentation

### Final Checklist:
- [ ] All target accuracies achieved
- [ ] All attack types implemented and tested
- [ ] SHAP explanations generated
- [ ] Code is clean and well-documented
- [ ] Results documented and analyzed

---

## Summary of Target Metrics (from ARM Paper)

### General Malware Detection:
- **Accuracy:** 98.6%
- **Precision:** 98.4%
- **Recall:** 98.8%
- **FPR:** 2.1%

### Adversarial Attacks Detection:
- **FGSM:** 92.3%
- **JSMA:** 93.4%
- **Salt-and-pepper:** 98.4%
- **Mimicry:** 96.5%
- **GAN:** 92.9%

### Zero-Day Malware:
- **Accuracy:** 94.1%
- **Precision:** 97.3%
- **Recall:** 90.8%
- **FPR:** 2.5%

**Note:** We're using 25k subset - targets may be slightly adjusted

---

## Key Milestones

| Date | Milestone | Status |
|------|-----------|--------|
| Jan 30 | Baseline model complete | ✅ Done |
| Feb 6 | MI feature selection completed | 🎯 Current |
| Feb 20 | Full GA-RAM algorithm implemented | ⏳ Upcoming |
| Feb 27 | General malware detection working (96%+ accuracy) | ⏳ Upcoming |
| Mar 6 | White-box attacks implemented and tested | ⏳ Upcoming |
| Mar 13 | Grey-box attacks implemented and tested | ⏳ Upcoming |
| Mar 20 | Black-box (GAN) attacks implemented and tested | ⏳ Upcoming |
| Mar 27 | Zero-day detection evaluated | ⏳ Upcoming |
| Apr 3 | SHAP explanations completed | ⏳ Upcoming |
| **Apr 10** | **Final documentation and results ready** | 🎯 **Deadline** |

---

## Dependencies & Prerequisites

### Python Libraries Required:
- ✅ pandas, numpy (installed)
- ✅ scikit-learn (installed)
- ✅ matplotlib, seaborn (installed)
- ⏳ shap (for Week 10)
- ⏳ tensorflow/pytorch (for GAN in Week 8)

### Computational Resources:
- GA-RAM training: May take 1-2 hours with 25k samples
- GAN training: GPU recommended (optional)
- Current setup: Windows 10, Python 3.12.10, 25k dataset

### Data Files (All Available):
- ✅ data_sample_25k.csv (25,000 samples)
- ✅ mh_100k_labels.csv (labels for all samples)
- ✅ mh_100k_dataset.csv (full 101K dataset - backup)
- ✅ mh_100k_vt_labels.csv (VirusTotal verification)

---

## Progress Tracking

**Current Week:** Week 2 (Feb 1-6, 2026)
**Current Status:** Week 1 complete, ready to start MI feature selection

**Completed:**
- ✅ Week 1: Baseline model (93.52% accuracy)
- ✅ Data preparation and EDA
- ✅ Train/test split
- ✅ Baseline Random Forest model trained
- ✅ Visualizations and metrics generated

**Next Immediate Tasks:**
1. Implement Mutual Information calculation
2. Select top 50-80 features
3. Retrain model with selected features
4. **Goal: Improve recall from 58.55% to 85%+**

---

## Notes
- Using 25k subset enables faster iteration
- Can scale to full 101K dataset for final evaluation
- Week 1 baseline provides solid foundation
- Primary focus in Week 2: **improve malware recall**
- Keep regular backups of code and results
- Git commits recommended after each week

**Last Updated:** February 1, 2026
**Status:** ✅ ON SCHEDULE - Week 1 Complete!
