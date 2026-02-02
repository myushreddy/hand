# Week 2 Quick Start Guide
## Mutual Information Feature Selection

**Goal:** Select top 50-80 features using Mutual Information to improve malware recall from 58.55% to 85%+

**Duration:** Feb 1-6, 2026 (6 days)

---

## Current Status (Week 1 Complete ✅)

**Baseline Model Performance:**
- Accuracy: 93.52%
- **Recall: 58.55%** ⚠️ (needs improvement - this is our main target!)
- FPR: 1.99% ✓ (excellent)

**Data Ready:**
- Training set: 19,940 samples
- Test set: 4,985 samples
- Features: 95 features (all currently used)

**Files Available:**
- `data/processed/dataset_with_labels.csv` - Merged data with labels
- `models/baseline_rf_model.pkl` - Baseline model
- `models/feature_columns.pkl` - List of 95 features

---

## Week 2 Objectives

### Primary Goal
**Improve malware recall from 58.55% to 85-92%** using Mutual Information feature selection

### What is Mutual Information?
- Measures how much knowing one variable reduces uncertainty about another
- MI(feature, CLASS) = how much does this feature help predict malware?
- Higher MI score = more relevant feature
- Will select top k features with highest MI scores

### Expected Results
- **Recall improvement** (main goal: detect more malware) from 58.55% → 85-92%
- Maintain or improve accuracy (93.52% → 94-96%)
- Reduced feature set = faster training
- Better generalization

---

## Implementation Steps

### Step 1: What We'll Do
1. Calculate MI score for each of the 95 features
2. Rank features by MI score (highest = most informative)
3. Try different k values (40, 50, 60, 80 features)
4. For each k:
   - Select top k features
   - Train Random Forest with only those features
   - Evaluate performance
5. Choose best k based on recall improvement
6. Save the best model

### Step 2: Code Overview
```python
from sklearn.feature_selection import mutual_info_classif

# Calculate MI scores for all features
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

# Create DataFrame and sort
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

# Select top k features
k = 50  # or 40, 60, 80
top_features = mi_df.head(k)['feature'].tolist()

# Train model with selected features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_selected, y_train)

# Evaluate
recall = recall_score(y_test, y_pred)
```

---

## Quick Start Commands

### Option 1: Run Complete Script (When Ready)
```bash
# Once we create the week2 script:
python scripts/week2_mutual_information.py
```

### Option 2: Use Interactive Notebook
```bash
# Open in VS Code:
# notebooks/02_Week2_Mutual_Information.ipynb
```

### Option 3: Quick Test (Python)
```python
import pandas as pd
import pickle
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/processed/dataset_with_labels.csv')

# Load feature columns
with open('models/feature_columns.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# Prepare data
X = df[feature_cols].values
y = df['CLASS'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate MI
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
print("Top 10 features by MI score:")
for i in range(10):
    print(f"{i+1}. {feature_cols[mi_scores.argsort()[-i-1]]}: {mi_scores[mi_scores.argsort()[-i-1]]:.4f}")
```

---

## Expected Improvements

| Metric | Baseline (Week 1) | Target (Week 2) | ARM Paper |
|--------|------------------|-----------------|-----------|
| Accuracy | 93.52% | 94-96% | 98.6% |
| **Recall** | **58.55%** | **85-92%** ⭐ | 98.8% |
| Precision | 79.05% | 80-85% | 98.4% |
| F1-Score | 67.27% | 82-88% | - |
| Features | 95 | 50-80 | 48 (after GA-RAM) |

**Key:** ⭐ = Primary improvement target

---

## Why This Will Work

### Theory
1. **Irrelevant features add noise** - baseline uses all 95 features
2. **Some features are redundant** - they don't help distinguish malware
3. **MI finds most informative features** - those that best predict malware vs benign
4. **Fewer, better features** → simpler model → better generalization

### Expected Changes
- **False Negatives will decrease** (currently 235 → target: <100)
  - This means we'll catch more malware!
- True Positives will increase (332 → ~450-500)
- Accuracy might slightly improve or stay the same
- Model will be faster (fewer features to process)

---

## Troubleshooting

### If MI calculation is slow:
```python
# Use fewer neighbors (faster but less accurate)
mi_scores = mutual_info_classif(X_train, y_train, random_state=42, n_neighbors=3)
```

### If recall is still low after MI:
1. Try different k values (more features)
2. Check class imbalance - may need SMOTE
3. Tune Random Forest hyperparameters
4. Verify feature preprocessing

### If accuracy drops:
- Selected k might be too low
- Try k=60 or k=80 instead
- Check if important features were excluded

---

## Deliverables for Week 2

By Feb 6, we should have:

- [ ] `scripts/week2_mutual_information.py` (complete implementation)
- [ ] `results/metrics/mi_scores.csv` (all 95 features with MI scores)
- [ ] `results/metrics/mi_model_comparison.json` (results for k=40,50,60,80)
- [ ] `results/plots/mi_scores_distribution.png` (visualization)
- [ ] `results/plots/mi_recall_comparison.png` (baseline vs MI)
- [ ] `models/rf_model_mi_XXfeatures.pkl` (best model)
- [ ] `models/mi_selected_features_XX.pkl` (selected feature list)
- [ ] `docs/WEEK2_COMPLETE.md` (summary)

---

## Daily Schedule (Suggested)

| Day | Date | Tasks | Hours |
|-----|------|-------|-------|
| **Day 1** | Feb 1 | MI implementation, test on small sample | 2-3h |
| **Day 2** | Feb 2 | Run full MI calculation, analyze top features | 2h |
| **Day 3** | Feb 3 | Test k=40,50,60,80, find best k | 2h |
| **Day 4** | Feb 4 | Visualizations and feature analysis | 1-2h |
| **Day 5** | Feb 5 | Documentation and comparison | 1h |
| **Day 6** | Feb 6 | Final review and prepare for Week 3 | 1h |

**Total:** ~9-11 hours spread over 6 days

---

## Success Criteria

✅ Week 2 is successful if:

1. **Recall improves to 85%+** (from 58.55%) - **MOST IMPORTANT**
2. Accuracy stays above 93%
3. Top features are semantically meaningful (make sense for malware detection)
4. MI scores saved and documented
5. Ready for Week 3 (GA-RAM implementation)

### Minimum Acceptable Results
- Recall ≥ 80% (vs 58.55% baseline)
- Accuracy ≥ 92% (vs 93.52% baseline)
- Feature count reduced to 50-80 (vs 95 baseline)

### Stretch Goals
- Recall ≥ 90%
- Accuracy ≥ 95%
- Identify key malware-indicative features

---

## Resources & References

### Documentation
- [Scikit-learn MI](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)
- [Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html)

### Our Files to Reference
- `docs/WEEK1_COMPLETE.md` - Baseline results
- `PROJECT_TIMELINE.md` - Overall plan
- `docs/timeline_ARM.txt` - Original timeline
- `data/processed/dataset_with_labels.csv` - Training data
- `models/baseline_rf_model.pkl` - Baseline model
- `models/feature_columns.pkl` - Feature names

### ARM Paper Notes
- Paper used 155 features after MI filtering
- We have 95 features, targeting 50-80 after MI
- This is proportional and appropriate for our 25k subset

---

## What Happens After Week 2?

**Week 3-4: GA-RAM Implementation**
- Genetic Algorithm framework
- Rank-Based Adaptive Mutation
- Optimize MI-selected features → final optimal set (30-50 features)
- Further improve recall and accuracy

**The Pipeline:**
```
Baseline (95 features, 58.55% recall)
    ↓
Week 2: MI Selection (50-80 features, 85-92% recall) ← YOU ARE HERE
    ↓
Week 3-4: GA-RAM (30-50 features, 95-98% recall)
    ↓
Week 5: Final Model Training
```

---

## Tips for Success

### 1. Start Simple
- Test MI on small subset first (5000 samples)
- Verify it works before running on full 20k training set

### 2. Document Everything
- Save MI scores to CSV
- Note which k value works best
- Screenshot interesting results

### 3. Compare with Baseline
- Always compare new results with Week 1 baseline
- Don't just look at accuracy - focus on **recall**

### 4. Understand the Features
- Look at top 10-20 features selected by MI
- Do they make sense for malware detection?
- Compare with ARM paper's top features

### 5. Be Patient
- MI calculation might take 30-60 minutes
- Model training with different k values: ~10 min each
- Total runtime: 1-2 hours for complete experimentation

---

## Questions to Ask Yourself

After Week 2, you should be able to answer:

1. ✅ Which features have highest MI scores?
2. ✅ What is the optimal k value (40, 50, 60, or 80)?
3. ✅ Did recall improve from 58.55%? By how much?
4. ✅ Are the selected features semantically meaningful?
5. ✅ Is the model faster with fewer features?
6. ✅ What are the top 10 malware-indicative features?

---

## Ready to Start?

### Next Steps:
1. Review this guide ✅
2. Check Week 1 results in `docs/WEEK1_COMPLETE.md` ✅
3. When ready, say: **"Create Week 2 MI implementation script"**
4. Or: **"Start Mutual Information feature selection"**

---

**Current Status:** ✅ Week 1 Complete, Ready for Week 2!  
**Timeline:** Feb 1-6, 2026 (6 days remaining)  
**Goal:** Improve recall 58.55% → 85-92% 🎯

**Let's improve that malware detection! 🚀**
