# Results Folder - Complete Documentation

This folder contains all experimental results, model performance metrics, feature sets, and comprehensive summaries for the Perovskite Bandgap Prediction project.

---

## Contents Overview

```
results/
├── all_models_summary.json                 # Original classification results (from pipeline)
├── all_models_summary_complete.json        # Complete results (both tasks, all models)
├── model_comparison.csv                    # Tabular comparison for analysis
├── model_comparison.png                    # Visual comparison plot
├── comprehensive_results_summary.md        # Detailed 9,000+ word analysis
└── feature_sets/
    ├── feature_importance_rankings.csv     # Complete feature rankings
    ├── feature_subset_F4.txt               # 4 features
    ├── feature_subset_F6.txt               # 6 features
    ├── feature_subset_F8.txt               # 8 features
    ├── feature_subset_F10.txt              # 10 features ⭐
    ├── feature_subset_F12.txt              # 12 features
    ├── feature_subset_F14.txt              # 14 features
    ├── feature_subset_F16.txt              # 16 features
    ├── feature_subset_F18.txt              # 18 features
    ├── feature_subset_F20.txt              # 20 features
    ├── feature_subset_F22.txt              # 22 features ⭐
    ├── feature_subset_F24.txt              # 24 features
    ├── feature_subset_F26.txt              # 26 features
    ├── feature_subset_F28.txt              # 28 features
    └── feature_subset_F30.txt              # 30 features
```

---

## File Descriptions

### Core Results Files

#### `all_models_summary.json`
**Original pipeline output** containing classification results from the initial run.

```json
{
  "F10": {
    "xgb_classification": {
      "Accuracy": 0.8936,
      "Precision": 0.8881,
      "Recall": 0.8936,
      "F1-Score": 0.8885
    },
    ...
  }
}
```

**Usage:** Reference for classification-only experiments

#### `all_models_summary_complete.json`
**Comprehensive results** including both regression and classification for F10 and F22.

```json
{
  "metadata": {
    "date": "2025-11-16",
    "dataset_size": 5776,
    "train_size": 4620,
    "test_size": 1156,
    "random_seed": 42
  },
  "F10": {
    "regression": { ... },
    "classification": { ... }
  },
  "F22": {
    "regression": { ... },
    "classification": { ... }
  }
}
```

**Usage:** Primary source for all paper tables and figures

#### `model_comparison.csv`
**Tabular format** for easy import into spreadsheets or statistical software.

```csv
Subset,Model,Accuracy,F1-Score,Precision,Recall
F10,XGB,0.8936,0.8885,0.8881,0.8936
F10,LGBM,0.8971,0.8908,0.8919,0.8971
...
```

**Usage:** Data analysis, plotting, statistical tests

#### `model_comparison.png`
**Visual comparison** of all models across both feature sets.

**Usage:** Figure 3 in paper (or supplementary figure)

#### `comprehensive_results_summary.md`
**Detailed analysis document** (9,000+ words) with:
- Complete performance tables
- Feature set descriptions
- Model insights and rankings
- Physical interpretation
- Recommendations
- Quick reference statistics

**Usage:** Primary reference for paper writing, presentations, documentation

---

## Feature Sets Directory

### Overview
Contains 11 feature subsets (F4-F30) generated via Recursive Feature Elimination (RFE) with cross-validation. Each file lists the selected features and their CV R² score.

### Selected Subsets

#### ⭐ F10 (Minimal Set) - `feature_subset_F10.txt`
**CV R² Score:** 0.7386  
**Test R² (LightGBM):** 0.8712  
**Test MAE (LightGBM):** 0.3934 eV

**Features (10):**
1. formation_energy_per_atom
2. energy_above_hull
3. MagpieData avg_dev NUnfilled
4. frac d valence electrons
5. avg_electron_affinity
6. MagpieData mean SpaceGroupNumber
7. MagpieData avg_dev Column
8. MagpieData mean MendeleevNumber
9. MagpieData maximum MeltingT
10. MagpieData mean Column

**Best For:** Simple deployment, interpretability, computational efficiency

#### ⭐ F22 (Expanded Set) - `feature_subset_F22.txt`
**CV R² Score:** 0.7620  
**Test R² (LightGBM):** 0.8836  
**Test MAE (LightGBM):** 0.3631 eV

**Features (22):** All F10 features plus 12 additional features including:
- energy_per_atom
- density
- MagpieData mean Electronegativity
- frac p valence electrons
- MagpieData mean MeltingT
- And 7 more electronic/structural features

**Best For:** Maximum prediction accuracy, comprehensive analysis

### Other Subsets
- **F4-F8:** Highly constrained (may underfit)
- **F12-F20:** Good balance between F10 and F22
- **F24-F30:** Marginal gains, risk of overfitting

---

## Performance Summary

### Best Models by Task

#### Regression (Bandgap Prediction)
| Feature Set | Best Model | R² | MAE (eV) | RMSE (eV) |
|-------------|------------|-----|----------|-----------|
| **F10** | LightGBM | 0.8712 | 0.3934 | 0.5933 |
| **F22** | LightGBM | 0.8836 | 0.3631 | 0.5639 |

#### Classification (Direct vs Indirect)
| Feature Set | Best Model | Accuracy | F1-Score |
|-------------|------------|----------|----------|
| **F10** | LightGBM | 89.71% | 0.8908 |
| **F22** | LightGBM | 91.18% | 0.9084 |

### Target Achievement
| Metric | Target | Best Achieved | Improvement |
|--------|--------|---------------|-------------|
| R² | ≥ 0.40 | 0.8836 (F22) | **2.2× better** |
| MAE (eV) | ≤ 0.45 | 0.3631 (F22) | **19% better** |
| Accuracy | ≥ 0.80 | 0.9118 (F22) | **14% better** |
| F1-Score | ≥ 0.80 | 0.9084 (F22) | **14% better** |

---

## Usage Examples

### Loading Results in Python

```python
import json
import pandas as pd

# Load complete results
with open('results/all_models_summary_complete.json') as f:
    results = json.load(f)

# Access F22 regression results
f22_regression = results['F22']['regression']
print(f"LightGBM R²: {f22_regression['lgbm']['R2']}")
print(f"LightGBM MAE: {f22_regression['lgbm']['MAE']} eV")

# Load comparison table
df = pd.read_csv('results/model_comparison.csv')
print(df.head())
```

### Loading Feature Subsets

```python
# Load F10 features
with open('results/feature_sets/feature_subset_F10.txt') as f:
    lines = f.readlines()
    # Skip comment lines starting with #
    f10_features = [line.strip() for line in lines 
                   if line.strip() and not line.startswith('#')]

print(f"F10 features ({len(f10_features)}):")
for feat in f10_features:
    print(f"  - {feat}")
```

---

## Related Files

### Data Files
- `data/processed/F10/` - Preprocessed F10 data
- `data/processed/F22/` - Preprocessed F22 data

### Model Files
- `models/F10/` - Trained F10 models (.pkl files)
- `models/F22/` - Trained F22 models (.pkl files)

### Figure Files
- `figures/F10/` - All F10 evaluation plots
- `figures/F22/` - All F22 evaluation plots
- `validation/F10/` - F10 validation plots
- `validation/F22/` - F22 validation plots

### Documentation
- `paper/results.md` - Results section with all values filled
- `paper/methods.md` - Methods section with actual parameters
- `paper/supplementary/supplementary_tables.md` - All supplementary tables
- `paper/FIGURE_TABLE_GUIDE.md` - Complete figure/table mapping
- `paper/PAPER_STATUS_COMPLETE.md` - Overall status report

---

## Key Findings

### Model Rankings

**Regression:**
1. LightGBM (Best overall: R² = 0.88, MAE = 0.36-0.39 eV)
2. XGBoost (Close second, best MAE on F22: 0.35 eV)
3. CatBoost (Solid performance, R² ~ 0.86-0.88)
4. Random Forest (Good baseline, R² ~ 0.84)
5. MLP (Weak on tabular data, R² = 0.53-0.74)

**Classification:**
1. LightGBM (Best: 89.7-91.2% accuracy)
2. XGBoost (Close: 89.4-91.0% accuracy)
3. CatBoost (Strong: 89.5-90.3% accuracy)
4. Random Forest (Good: 88.9-90.1% accuracy)
5. MLP (Acceptable: 86.3-86.7% accuracy)

### Feature Categories (by importance)

1. **Thermodynamic** (formation_energy, E_hull) - Strongest predictors
2. **Electronic** (d-electrons, unfilled orbitals) - Critical for bandgap
3. **Elemental** (electronegativity, electron affinity) - Important for bonding
4. **Structural** (space group, lattice) - Moderate influence
5. **Physical** (melting temp, density) - Supporting features

---

## Reproducibility

All results are fully reproducible with:
- **Random seed:** 42
- **Train/test split:** 80/20 (4,620 / 1,156)
- **Pipeline:** `python run_pipeline.py F10 F22`
- **Platform:** Windows 11, Python 3.11+, CPU training
- **Time:** ~2-3 hours for complete pipeline

---

## Citation

If you use these results, please cite:

```bibtex
@software{perovskite_bandgap_prediction_2025,
  author = {Aish Soni},
  title = {Perovskite Bandgap Prediction using Machine Learning},
  year = {2025},
  url = {https://github.com/AishSoni/Silica-Perovskite-Energy-Band-Gap-Prediction}
}
```

---

## Contact

**Author:** Aish Soni  
**GitHub:** [@AishSoni](https://github.com/AishSoni)  
**Repository:** [Silica-Perovskite-Energy-Band-Gap-Prediction](https://github.com/AishSoni/Silica-Perovskite-Energy-Band-Gap-Prediction)

For questions about the results or methodology, please open an issue on GitHub.

---

**Last Updated:** November 16, 2025  
**Version:** 1.0 - Complete Release  
**Status:** ✅ Ready for publication
