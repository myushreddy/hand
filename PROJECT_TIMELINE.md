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

## Week 2: January 31 - February 6, 2026
**Focus:** Mutual Information (MI) Feature Selection

### Tasks:
- [ ] Implement MI-based feature filtering
  - Calculate MI scores for all 95 features
  - Rank features by relevance to CLASS label
  - Select top k features (experiment with k=40, 50, 60, 80)
- [ ] Validate MI results
  - Compare feature importance
  - Visualize feature distributions
- [ ] Create feature subset for GA-RAM input
- [ ] Retrain Random Forest with selected features
- [ ] Compare performance with baseline

### Deliverables:
- MI_scores_result.csv with all feature rankings
- Selected feature subset (50-80 features)
- Feature importance visualization
- Improved model (target: recall >85%)

### Expected Improvements:
- **Recall: 58.55% → 85-92%** (primary goal)
- Accuracy: maintain or improve (93.52% → 94-96%)
- F1-Score: 67.27% → 82-88%

**Dependencies:** Week 1 complete ✅

---

## Week 3: February 7-13, 2026
**Focus:** GA-RAM Algorithm - Part 1 (Core Components)

### Tasks:
- [ ] Implement population initialization (50 random binary feature subsets)
- [ ] Implement fitness function
  - Train Random Forest classifier
  - Compute accuracy for each feature subset
  - 80:20 train-test split
- [ ] Implement tournament selection mechanism
- [ ] Test basic GA loop structure

### Deliverables:
- Working population initialization
- Fitness evaluation function
- Tournament selection implementation
- Initial GA structure (without crossover/mutation)

### Key Parameters:
- Population size: 50
- Number of features: 50-80 (from MI selection)
- Fitness metric: Classification accuracy

---

## Week 4: February 14-20, 2026
**Focus:** GA-RAM Algorithm - Part 2 (Crossover & Mutation)

### Tasks:
- [ ] Implement 2-point crossover operator
  - Select two crossover points
  - Swap features between points
  - Generate offspring feature subsets
- [ ] Implement Rank-Based Adaptive Mutation (RAM)
  - Sort feature subsets by accuracy (ascending)
  - Assign ranks (worst = rank 1, best = rank n)
  - Calculate mutation probability: pᵢ = p_max × (i-1)/(n-1)
  - Apply mutation based on rank
- [ ] Test crossover and mutation operators independently
- [ ] Integrate into full GA-RAM loop

### Deliverables:
- 2-point crossover function
- Rank-based adaptive mutation function
- Complete GA-RAM algorithm

### Key Parameters:
- Crossover probability: 0.7
- Max mutation rate (p_max): 0.6
- Stopping threshold: 5 generations without improvement

---

## Week 5: February 21-27, 2026
**Focus:** GA-RAM Training & General Malware Detection

### Tasks:
- [ ] Run full GA-RAM on 25k dataset
  - Set generations (5-10 for testing, increase if needed)
  - Monitor convergence
  - Track best accuracy across generations
- [ ] Select optimal feature subset
- [ ] Train final Random Forest classifier with selected features
- [ ] Evaluate on general malware test set
  - Target: 96-98% accuracy
  - Compute Precision, Recall, F1-Score, FPR
- [ ] Save trained model and selected features

### Deliverables:
- Trained GA-RAM model
- Optimal feature subset (30-50 features)
- Performance metrics on general malware
- Model checkpoint file

### Success Criteria:
- Accuracy ≥ 96% on general malware
- Recall ≥ 90%

---

## Week 6: February 28 - March 6, 2026
**Focus:** White-Box Adversarial Attacks (FGSM & JSMA)

### Tasks:
- [ ] Implement FGSM attack
  - Binary, additive-only variant
  - Formula: x_adv = max(x, x + ε · sign(∇_x J(X, Y)))
  - Generate adversarial samples
- [ ] Implement JSMA attack
  - Binary variant (modify inactive features only)
  - Compute saliency map
  - Generate adversarial samples
- [ ] Test attacks on trained model
- [ ] Verify adversarial samples preserve malicious functionality
- [ ] Evaluate detection accuracy
  - Target: FGSM ≥ 90%, JSMA ≥ 90%

### Deliverables:
- FGSM adversarial sample generator
- JSMA adversarial sample generator
- Adversarial attack samples
- Detection performance metrics

---

## Week 7: March 7-13, 2026
**Focus:** Grey-Box Adversarial Attacks (Salt-and-Pepper & Mimicry)

### Tasks:
- [ ] Implement Salt-and-Pepper noise attack
  - Add random benign/irrelevant features
  - Formula: x_adv[i] = max(x[i], noise[i])
  - Generate adversarial samples
- [ ] Implement Mimicry attack
  - Modify malware to resemble benign apps
  - Mimicry × 30 approach
  - Generate mimicry samples
- [ ] Test on trained model
- [ ] Evaluate detection accuracy
  - Target: Salt-and-pepper ≥ 95%, Mimicry ≥ 93%

### Deliverables:
- Salt-and-pepper attack generator
- Mimicry attack generator
- Attack samples dataset
- Detection performance report

---

## Week 8: March 14-20, 2026
**Focus:** Black-Box Adversarial Attacks (GAN-based)

### Tasks:
- [ ] Design GAN architecture
  - Generator: Creates adversarial malware features
  - Discriminator: Distinguishes real vs generated malware
- [ ] Implement GAN training loop
  - Minimax game objective
  - Add Gaussian noise for stability
- [ ] Generate adversarial samples
- [ ] Validate generated samples
  - Ensure they preserve malicious behavior
  - Check feature distributions
- [ ] Evaluate detection accuracy
  - Target: ≥ 90%

### Deliverables:
- Trained GAN model
- GAN-generated adversarial samples
- Detection performance metrics

### Libraries:
- TensorFlow/PyTorch for GAN implementation

---

## Week 9: March 21-27, 2026
**Focus:** Zero-Day Malware Detection

### Tasks:
- [ ] Collect zero-day malware samples
  - Use latest malware (2023-2024)
  - Verify with VirusTotal
- [ ] Prepare test dataset (zero-day + benign)
- [ ] Evaluate trained model on zero-day samples
  - Target: 92-94% accuracy
  - Compute Precision, Recall
  - FPR target: <3%
- [ ] Analyze failure cases
- [ ] Document zero-day detection capabilities

### Deliverables:
- Zero-day malware test dataset
- Detection performance metrics
- Analysis report on detection vs new malware families

### Success Criteria:
- Accuracy ≥ 92% on unseen malware

---

## Week 10: March 28 - April 3, 2026
**Focus:** SHAP Explanations & Interpretability

### Tasks:
- [ ] Install and configure SHAP library
- [ ] Generate SHAP values for trained model
  - Compute feature importance scores
  - Identify top malware-indicating features
  - Identify top benign-indicating features
- [ ] Validate SHAP results
  - Compare with known malware behavior patterns
  - Verify feature semantics
- [ ] Create SHAP visualizations
  - Summary plots
  - Force plots for individual predictions
  - Dependence plots
- [ ] Write interpretability analysis

### Deliverables:
- SHAP values for all features
- Feature importance rankings
- Visualizations (plots and charts)
- Interpretability report

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
