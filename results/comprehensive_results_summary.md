# Comprehensive Results Summary
## Perovskite Bandgap Prediction - F10 and F22 Feature Sets

**Date:** November 16, 2025  
**Dataset:** 5,776 double perovskites (ABC₂D₆)  
**Random Seed:** 42  
**Train/Test Split:** 80/20 (4,620 / 1,156 samples)

---

## Executive Summary

Both F10 (10 features) and F22 (22 features) substantially exceed all performance targets:

- **Regression:** R² up to **0.8836** (target: 0.40), MAE as low as **0.3631 eV** (target: 0.45 eV)
- **Classification:** Accuracy up to **91.18%** (target: 80%)
- **Key Finding:** Minimal F10 set achieves near-optimal performance with 55% fewer features

---

## 1. REGRESSION RESULTS (Bandgap Prediction)

### F10 Feature Set (10 features)

| Model | R² | MAE (eV) | RMSE (eV) | Rank |
|-------|-----|----------|-----------|------|
| **LightGBM** | **0.8712** | **0.3934** | **0.5933** | 🥇 1st |
| XGBoost | 0.8654 | 0.3662 | 0.6063 | 🥈 2nd |
| CatBoost | 0.8653 | 0.3948 | 0.6067 | 🥉 3rd |
| Random Forest | 0.8388 | 0.4738 | 0.6635 | 4th |
| MLP | 0.7428 | 0.6060 | 0.8382 | 5th |

**Best Model:** LightGBM  
**Performance vs. Target:** R² 2.18× better, MAE 13% better

### F22 Feature Set (22 features)

| Model | R² | MAE (eV) | RMSE (eV) | Rank |
|-------|-----|----------|-----------|------|
| **LightGBM** | **0.8836** | **0.3631** | **0.5639** | 🥇 1st |
| XGBoost | 0.8834 | 0.3483 | 0.5643 | 🥈 2nd |
| CatBoost | 0.8786 | 0.3716 | 0.5760 | 🥉 3rd |
| Random Forest | 0.8384 | 0.4831 | 0.6644 | 4th |
| MLP | 0.5308 | 0.7891 | 1.1321 | 5th |

**Best Model:** LightGBM (R²), XGBoost (MAE)  
**Performance vs. Target:** R² 2.21× better, MAE 19% better

### Regression Comparison: F10 vs F22

| Metric | F10 (LightGBM) | F22 (LightGBM) | F22 Improvement |
|--------|---------------|---------------|-----------------|
| R² | 0.8712 | 0.8836 | +1.4% |
| MAE (eV) | 0.3934 | 0.3631 | -7.7% (better) |
| RMSE (eV) | 0.5933 | 0.5639 | -5.0% (better) |
| Features | 10 | 22 | +120% |

**Insight:** F22 provides modest improvements (1-8%) at the cost of 2.2× more features. F10 offers excellent performance-to-complexity ratio.

---

## 2. CLASSIFICATION RESULTS (Bandgap Type: Direct vs Indirect)

### F10 Feature Set (10 features)

| Model | Accuracy | F1-Score | Precision | Recall | Rank |
|-------|----------|----------|-----------|--------|------|
| **LightGBM** | **0.8971** | **0.8908** | **0.8919** | **0.8971** | 🥇 1st |
| CatBoost | 0.8945 | 0.8887 | 0.8889 | 0.8945 | 🥈 2nd |
| XGBoost | 0.8936 | 0.8885 | 0.8881 | 0.8936 | 🥉 3rd |
| Random Forest | 0.8893 | 0.8774 | 0.8860 | 0.8893 | 4th |
| MLP | 0.8633 | 0.8435 | 0.8548 | 0.8633 | 5th |

**Best Model:** LightGBM  
**Performance vs. Target:** Accuracy 12.1% better, F1 11.4% better

### F22 Feature Set (22 features)

| Model | Accuracy | F1-Score | Precision | Recall | Rank |
|-------|----------|----------|-----------|--------|------|
| **LightGBM** | **0.9118** | **0.9084** | **0.9083** | **0.9118** | 🥇 1st |
| XGBoost | 0.9100 | 0.9062 | 0.9063 | 0.9100 | 🥈 2nd |
| CatBoost | 0.9031 | 0.8990 | 0.8987 | 0.9031 | 🥉 3rd |
| Random Forest | 0.9014 | 0.8923 | 0.8995 | 0.9014 | 4th |
| MLP | 0.8668 | 0.8517 | 0.8569 | 0.8668 | 5th |

**Best Model:** LightGBM  
**Performance vs. Target:** Accuracy 13.9% better, F1 13.6% better

### Classification Comparison: F10 vs F22

| Metric | F10 (LightGBM) | F22 (LightGBM) | F22 Improvement |
|--------|---------------|---------------|-----------------|
| Accuracy | 0.8971 | 0.9118 | +1.6% |
| F1-Score | 0.8908 | 0.9084 | +2.0% |
| Precision | 0.8919 | 0.9083 | +1.8% |
| Recall | 0.8971 | 0.9118 | +1.6% |

**Insight:** F22 provides consistent 1.6-2.0% improvements in classification metrics.

---

## 3. FEATURE SET DETAILS

### F10 Feature Set (10 features)
**CV R² Score:** 0.7386

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

**Category Breakdown:**
- Thermodynamic: 2 features (20%)
- Electronic: 2 features (20%)
- Elemental: 4 features (40%)
- Structural: 1 feature (10%)
- Physical: 1 feature (10%)

### F22 Feature Set (22 features)
**CV R² Score:** 0.7620

**Includes all F10 features plus:**
11. energy_per_atom
12. density
13. MagpieData mean Electronegativity
14. frac p valence electrons
15. MagpieData mean MeltingT
16. MagpieData mean NUnfilled
17. MagpieData avg_dev SpaceGroupNumber
18. gamma_dev_from_90
19. MagpieData avg_dev NValence
20. Eu
21. MagpieData maximum GSmagmom
22. avg_group

**Category Breakdown:**
- Thermodynamic: 3 features (13.6%)
- Electronic: 6 features (27.3%)
- Elemental: 9 features (40.9%)
- Structural: 3 features (13.6%)
- Physical: 1 feature (4.5%)

---

## 4. TARGET ACHIEVEMENT SUMMARY

### Regression Targets

| Metric | Target | F10 Best | F22 Best | Achievement |
|--------|--------|----------|----------|-------------|
| R² | ≥ 0.40 | 0.8712 | 0.8836 | ✅ 2.18-2.21× |
| MAE (eV) | ≤ 0.45 | 0.3934 | 0.3631 | ✅ 13-19% better |

### Classification Targets

| Metric | Target | F10 Best | F22 Best | Achievement |
|--------|--------|----------|----------|-------------|
| Accuracy | ≥ 0.80 | 0.8971 | 0.9118 | ✅ 12-14% better |
| F1-Score | ≥ 0.80 | 0.8908 | 0.9084 | ✅ 11-14% better |

**All targets exceeded by substantial margins.**

---

## 5. MODEL INSIGHTS

### Algorithm Performance Ranking

**Regression (averaged across F10 and F22):**
1. **LightGBM** - Consistent top performer (R² ~0.87-0.88)
2. **XGBoost** - Close second, best MAE on F22 (0.3483 eV)
3. **CatBoost** - Strong performance, stable across feature sets
4. **Random Forest** - Solid baseline, ~4% behind top models
5. **MLP** - Significantly weaker, struggles with tabular data

**Classification (averaged across F10 and F22):**
1. **LightGBM** - Best accuracy (89.7-91.2%)
2. **XGBoost** - Very close second (89.4-91.0%)
3. **CatBoost** - Consistent third place (89.5-90.3%)
4. **Random Forest** - Good baseline (88.9-90.1%)
5. **MLP** - Weakest performer (86.3-86.7%)

### Key Takeaways

✅ **Gradient boosting dominates** - LightGBM, XGBoost, CatBoost consistently outperform other algorithms  
✅ **Tree-based > Neural networks** - For this tabular data, tree ensembles are 10-40% better than MLP  
✅ **Minimal features sufficient** - F10 (10 features) achieves 98.6% of F22's regression R²  
✅ **Robust predictions** - Test performance meets or exceeds CV estimates  
✅ **Dual-task success** - Both regression and classification tasks solved effectively  

---

## 6. PHYSICAL INTERPRETATION

### Top Feature Categories (by importance)

1. **Thermodynamic Stability** (formation energy, E_hull)
   - Strong negative correlation with bandgap
   - More stable structures → larger bandgaps
   - Consistent with bonding theory

2. **Electronic Configuration** (d-electrons, unfilled orbitals)
   - Determines band structure complexity
   - d-orbital occupancy strongly influences gap magnitude
   - Direct correlation with bandgap type

3. **Elemental Properties** (electronegativity, electron affinity)
   - Governs ionic vs. covalent bonding character
   - Affects band edge positions
   - Critical for gap magnitude predictions

4. **Structural Features** (space group, lattice parameters)
   - Moderate influence on predictions
   - Less important than composition
   - Contributes to classification accuracy

---

## 7. DATASET STATISTICS

**Total Materials:** 5,776 double perovskites  
**Formula Pattern:** ABC₂D₆  
**Bandgap Range:** 0.000 - 8.547 eV  
**Class Distribution:**
- Indirect bandgap: 4,663 materials (80.7%)
- Direct bandgap: 1,113 materials (19.3%)

**Train Set:** 4,620 samples (80%)  
**Test Set:** 1,156 samples (20%)  
**Random Seed:** 42  
**Scaling:** RobustScaler  

---

## 8. COMPUTATIONAL DETAILS

**Platform:** Windows 11  
**Environment:** Python 3.11+ (perovskite venv)  
**Training Time:** ~2-3 hours for complete pipeline (both feature sets, SHAP analysis)  
**Libraries:**
- scikit-learn 1.3.0
- lightgbm 4.0.0
- xgboost 2.0.0
- catboost 1.2+
- shap 0.42.0
- matminer 0.9.0
- pymatgen 2023.9.0

---

## 9. OUTPUT FILES

### Trained Models
- `models/F10/*.pkl` - F10 trained models
- `models/F22/*.pkl` - F22 trained models

### Validation Plots
- `validation/F10/` - Data distribution plots (F10)
- `validation/F22/` - Data distribution plots (F22)

### Evaluation Figures
- `figures/F10/{model}/` - Parity plots, error histograms, feature importance (F10)
- `figures/F22/{model}/` - Parity plots, error histograms, feature importance (F22)
- `figures/F10/{model}/shap_*.png` - SHAP analysis (F10)
- `figures/F22/{model}/shap_*.png` - SHAP analysis (F22)

### Results
- `results/all_models_summary.json` - Complete metrics for all models
- `results/model_comparison.csv` - Tabular comparison
- `results/model_comparison.png` - Visual comparison
- `results/comprehensive_results_summary.md` - This document

---

## 10. RECOMMENDATIONS

### For Production Use:
- **Use F22 LightGBM** for highest accuracy (R² = 0.8836)
- **Use F10 LightGBM** for simpler deployment (R² = 0.8712, 10 features only)
- **Use F22 XGBoost** for lowest MAE (0.3483 eV)

### For Classification:
- **Use F22 LightGBM** for best accuracy (91.18%)
- **Use F10 LightGBM** for simpler model (89.71%, still excellent)

### For Materials Screening:
- Apply regression models to predict bandgap magnitude
- Apply classification models to identify direct bandgaps (better for optoelectronics)
- Filter candidates: 1.2-1.8 eV (solar cells), direct gap, E_hull < 0.2 eV/atom

---

## 11. CONCLUSIONS

1. **Mission Accomplished:** All performance targets substantially exceeded
2. **Feature Efficiency:** 10 features achieve 98.6% of 22-feature performance
3. **Model Consistency:** Gradient boosting (LightGBM, XGBoost, CatBoost) consistently outperform alternatives
4. **Physical Validity:** Feature importance aligns with solid-state physics principles
5. **Production Ready:** Models, metrics, and interpretability plots ready for publication

**Next Steps:** Experimental validation, GW-corrected DFT training data, graph neural networks for structure-aware learning

---

**Generated:** November 16, 2025  
**Pipeline Version:** run_pipeline.py (F10 + F22 mode)  
**Status:** ✅ COMPLETE - Ready for paper submission
