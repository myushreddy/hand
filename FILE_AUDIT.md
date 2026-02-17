# FILE AUDIT - What to Keep vs What to Remove

**Decision: Using ONLY `data/processed/dataset_with_labels_full.csv`**
- 30K samples, 24,836 features
- Best performance: 97.53% accuracy, 78.31% recall

---

## ✅ **KEEP THESE FILES** (Currently Needed)

### **DATA FILES**
- ✅ `data/processed/dataset_with_labels_full.csv` (1,367 MB) - **MAIN DATASET**
- ✅ `data/mh_100k_labels.csv` - Original labels (needed if regenerating)
- ✅ `data/mh_100k_dataset.csv` (5 GB) - Full source data (needed if regenerating)
- ✅ `data/mh_100k_dataset.csv.part001.rar` - Compressed backup
- ✅ `data/mh_100k_dataset.csv.part002.rar` - Compressed backup
- ✅ `data/data_README.md` - Dataset documentation

### **SCRIPTS**
- ✅ `scripts/baseline_model_full.py` - Trains baseline on full dataset (**USEFUL**)

### **MODELS**
- ✅ `models/baseline_rf_model_full.pkl` (19 MB) - Best baseline model
- ✅ `models/feature_columns_full.pkl` - Feature list for full dataset

### **METRICS & RESULTS**
- ✅ `results/metrics/baseline_metrics_full.json` - Best baseline results
- ✅ `results/metrics/train_test_split_full.json` - Train/test split info
- ✅ `results/metrics/eda_summary_full.txt` - Dataset summary

### **PLOTS**
- ✅ `results/plots/class_distribution_full.png` - Full dataset class distribution
- ✅ `results/plots/confusion_matrix_baseline_full.png` - Best model confusion matrix

### **DOCUMENTATION**
- ✅ `docs/MH-100K dataset_analysis.md` - Dataset documentation
- ✅ `docs/Reference_paper.md` - ARM algorithm paper reference
- ✅ `docs/timeline_ARM.txt` - Project timeline
- ✅ `README.md` - Project readme
- ✅ `LICENSE` - License file
- ✅ `PROJECT_TIMELINE.md` - Timeline documentation

### **CONFIGURATION**
- ✅ `.venv/` - Python virtual environment
- ✅ `.git/` - Git version control

---

## ❌ **REMOVE THESE FILES** (Not Needed - Old/Redundant)

### **OLD SMALL DATASET FILES (25K samples, 95 features)**
- ❌ `data/data_sample_25k.csv` (7.5 MB) - Old small sample
- ❌ `data/processed/dataset_with_labels.csv` (7.5 MB) - Old dataset with only 95 features
- ❌ `data/mh_100k_features_all.csv` - Redundant (already in mh_100k_dataset.csv)
- ❌ `data/mh_100k_features_classes.csv` - Redundant
- ❌ `data/mh_100k_labels.npy` - Redundant (have CSV version)
- ❌ `data/mh_100k_vt_labels.csv` - Not used

### **REDUCED DATASET FILES (30K samples, 8K features)**
- ❌ `data/dataset_30k_sample.csv` (460 MB) - Reduced features, worse performance
- ❌ `data/feature_list_30k_sample.txt` - Feature list for 8K dataset

### **OLD SCRIPTS (Week-based organization)**
- ❌ `scripts/week1_baseline_model.py` - Old 95-feature baseline
- ❌ `scripts/week2_mutual_information.py` - MI on old 95-feature dataset
- ❌ `scripts/extract_30k_8k_direct.py` - Creates 8K feature dataset (not using)
- ❌ `scripts/reduce_features_mi.py` - MI reduction (interrupted, not completed)
- ❌ `scripts/baseline_30k_8k.py` - Baseline for 8K features (not using)
- ❌ `scripts/mi_feature_selection_full.py` - If exists, was exploratory

### **OLD MODELS**
- ❌ `models/baseline_rf_model.pkl` - Old 95-feature model
- ❌ `models/week2_mi_rf_model.pkl` - Week 2 MI model (worse performance)
- ❌ `models/week2_optimal_features.pkl` - Week 2 features
- ❌ `models/features_30k_sample.pkl` - 8K feature list
- ❌ `models/feature_columns.pkl` - Old 95-feature list

### **OLD METRICS**
- ❌ `results/metrics/baseline_metrics.json` - Old 95-feature results
- ❌ `results/metrics/baseline_30k_8k.json` - 8K feature results
- ❌ `results/metrics/week2_final_metrics.json` - Week 2 results
- ❌ `results/metrics/week2_k_comparison.csv` - Week 2 MI comparison
- ❌ `results/metrics/train_test_split.json` - Old split info
- ❌ `results/metrics/eda_summary.txt` - Old dataset summary
- ❌ `results/metrics/feature_importance_baseline.csv` - Old feature importance
- ❌ `results/metrics/mi_scores_all_features.csv` - If from old dataset

### **OLD PLOTS**
- ❌ `results/plots/class_distribution.png` - Old dataset
- ❌ `results/plots/confusion_matrix_baseline.png` - Old model
- ❌ `results/plots/feature_frequency.png` - Old dataset
- ❌ `results/plots/feature_importance_baseline.png` - Old model
- ❌ `results/plots/mi_scores_distribution.png` - Old dataset
- ❌ `results/plots/week2_confusion_matrix.png` - Week 2 results
- ❌ `results/plots/week2_performance_comparison.png` - Week 2 comparison
- ❌ `results/plots/week2_top_features_mi.png` - Week 2 features

### **OLD DOCUMENTATION (Week-based)**
- ❌ `docs/WEEK1_COMPLETE.md` - Week 1 old approach
- ❌ `docs/WEEK2_QUICKSTART.md` - Week 2 old approach

### **OLD NOTEBOOKS**
- ❌ `notebooks/01_Week1_EDA_Baseline.ipynb` - Old 95-feature analysis

### **UTILITY SCRIPTS (Keep if useful)**
- ⚠️ `scripts/rar_compress.sh` - Utility (keep if needed)
- ⚠️ `scripts/rar_uncompress.sh` - Utility (keep if needed)

### **OTHER**
- ❌ `COPILOT_CHAT_HISTORY.md` - Old chat history
- ❌ `RECOVERY_COMPLETE.md` - Recovery notes

---

## 📊 **SUMMARY**

### **Files to Keep: 23**
- Data: 6 files (main dataset + sources + backups)
- Scripts: 1 file (baseline_model_full.py)
- Models: 2 files (best model + features)
- Metrics: 3 files (best results)
- Plots: 2 files (full dataset visualizations)
- Docs: 6 files (reference materials)
- Config: 3 items (.venv, .git, data_README)

### **Files to Remove: 48**
- Old datasets: 8 files (~530 MB)
- Old scripts: 6 files
- Old models: 5 files (~25 MB)
- Old metrics: 8 files
- Old plots: 8 files
- Old docs: 2 files
- Old notebooks: 1 file
- Other: 2 files

### **Space to Recover: ~555 MB**

---

## 🗑️ **SAFE REMOVAL COMMANDS**

```powershell
# Remove old datasets
Remove-Item "data\data_sample_25k.csv"
Remove-Item "data\dataset_30k_sample.csv"
Remove-Item "data\feature_list_30k_sample.txt"
Remove-Item "data\processed\dataset_with_labels.csv"
Remove-Item "data\mh_100k_features_all.csv"
Remove-Item "data\mh_100k_features_classes.csv"
Remove-Item "data\mh_100k_labels.npy"
Remove-Item "data\mh_100k_vt_labels.csv"

# Remove old scripts
Remove-Item "scripts\week1_baseline_model.py"
Remove-Item "scripts\week2_mutual_information.py"
Remove-Item "scripts\extract_30k_8k_direct.py"
Remove-Item "scripts\reduce_features_mi.py"
Remove-Item "scripts\baseline_30k_8k.py"

# Remove old models
Remove-Item "models\baseline_rf_model.pkl"
Remove-Item "models\week2_mi_rf_model.pkl"
Remove-Item "models\week2_optimal_features.pkl"
Remove-Item "models\features_30k_sample.pkl"
Remove-Item "models\feature_columns.pkl"

# Remove old metrics
Remove-Item "results\metrics\baseline_metrics.json"
Remove-Item "results\metrics\baseline_30k_8k.json"
Remove-Item "results\metrics\week2_final_metrics.json"
Remove-Item "results\metrics\week2_k_comparison.csv"
Remove-Item "results\metrics\train_test_split.json"
Remove-Item "results\metrics\eda_summary.txt"
Remove-Item "results\metrics\feature_importance_baseline.csv"

# Remove old plots
Remove-Item "results\plots\class_distribution.png"
Remove-Item "results\plots\confusion_matrix_baseline.png"
Remove-Item "results\plots\feature_frequency.png"
Remove-Item "results\plots\feature_importance_baseline.png"
Remove-Item "results\plots\mi_scores_distribution.png"
Remove-Item "results\plots\week2_confusion_matrix.png"
Remove-Item "results\plots\week2_performance_comparison.png"
Remove-Item "results\plots\week2_top_features_mi.png"

# Remove old docs
Remove-Item "docs\WEEK1_COMPLETE.md"
Remove-Item "docs\WEEK2_QUICKSTART.md"

# Remove old notebooks
Remove-Item "notebooks\01_Week1_EDA_Baseline.ipynb"

# Remove other
Remove-Item "COPILOT_CHAT_HISTORY.md"
Remove-Item "RECOVERY_COMPLETE.md"
```

---

**Ready to clean up? Run the commands above to remove all unnecessary files!**
