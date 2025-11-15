# Supplementary Materials

## Supplementary Tables and Information

---

## Table S1: Dataset Information

### Materials Project Query Parameters

| Parameter | Value |
|-----------|-------|
| Formula Pattern | ABC₂D₆ (double perovskites) |
| Number of Elements | 4-6 |
| Bandgap Range | 0.0 - 10.0 eV |
| Energy Above Hull | ≤ 0.2 eV/atom |
| Number of Sites | 1-200 |
| Query Date | October 11, 2025 |
| API Version | mp-api v0.41.0+ |

### Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total Materials Retrieved | 5,776 |
| Materials with Bandgap Data | 5,776 (100%) |
| Bandgap Range | 0.000 - 8.547 eV |
| Direct Bandgaps | 1,113 (19.3%) |
| Indirect Bandgaps | 4,663 (80.7%) |
| Median Bandgap | ~1.8 eV |
| Train Set Size | 4,620 (80%) |
| Test Set Size | 1,156 (20%) |

---

## Table S2: Complete Feature List

### Initial Feature Space (317 features)

Generated using matminer v0.9.0+ and pymatgen v2023.9.0+

#### Thermodynamic Features (3)
- formation_energy_per_atom
- energy_above_hull
- energy_per_atom

#### Structural Features (~15)
- lattice_a, lattice_b, lattice_c
- lattice_alpha, lattice_beta, lattice_gamma
- lattice_volume
- density
- nsites
- symmetry (space group number)
- Derived: b/a, c/a, alpha_dev_from_90, beta_dev_from_90, gamma_dev_from_90

#### Elemental Properties (MagpieData, ~128 features)
For each elemental property (Number, MendeleevNumber, AtomicWeight, MeltingT, Column, Row, CovalentRadius, Electronegativity, NsValence, NpValence, NdValence, NfValence, NValence, NsUnfilled, NpUnfilled, NdUnfilled, NfUnfilled, NUnfilled, GSvolume_pa, GSbandgap, GSmagmom, SpaceGroupNumber):
- minimum, maximum, range, mean, avg_dev, mode

#### Stoichiometric Features (~22)
- n_elements
- n_atoms
- avg_atomic_mass
- Fractional composition statistics

#### Compositional Features (~112)
- Element fractions for common elements
- avg_electronegativity, std_electronegativity, max_electronegativity, min_electronegativity, range_electronegativity
- avg_atomic_number, std_atomic_number
- avg_atomic_radius
- avg_ionization_energy
- avg_electron_affinity
- avg_group, avg_row
- avg_valence
- n_d_electrons
- frac_metal, frac_nonmetal
- frac s/p/d/f valence electrons

#### Derived Electronic Features (~37)
- Band structure descriptors
- Magnetic moment statistics
- Orbital occupancy statistics

### Selected Feature Subsets

#### F10 (10 features) - Minimal Set
**CV R² Score: 0.7386**

1. formation_energy_per_atom (Thermodynamic)
2. energy_above_hull (Thermodynamic)
3. MagpieData avg_dev NUnfilled (Electronic)
4. frac d valence electrons (Electronic)
5. avg_electron_affinity (Elemental)
6. MagpieData mean SpaceGroupNumber (Structural)
7. MagpieData avg_dev Column (Elemental)
8. MagpieData mean MendeleevNumber (Elemental)
9. MagpieData maximum MeltingT (Physical)
10. MagpieData mean Column (Elemental)

**Category Distribution:**
- Thermodynamic: 2 (20%)
- Electronic: 2 (20%)
- Elemental: 4 (40%)
- Structural: 1 (10%)
- Physical: 1 (10%)

#### F22 (22 features) - Expanded Set
**CV R² Score: 0.7620**

Includes all F10 features plus:

11. energy_per_atom (Thermodynamic)
12. density (Structural)
13. MagpieData mean Electronegativity (Elemental)
14. frac p valence electrons (Electronic)
15. MagpieData mean MeltingT (Physical)
16. MagpieData mean NUnfilled (Electronic)
17. MagpieData avg_dev SpaceGroupNumber (Structural)
18. gamma_dev_from_90 (Structural)
19. MagpieData avg_dev NValence (Electronic)
20. Eu (Thermodynamic - internal energy)
21. MagpieData maximum GSmagmom (Electronic)
22. avg_group (Elemental)

**Category Distribution:**
- Thermodynamic: 3 (13.6%)
- Electronic: 6 (27.3%)
- Elemental: 9 (40.9%)
- Structural: 3 (13.6%)
- Physical: 1 (4.5%)

---

## Table S3: Model Hyperparameters

### Regression Models

#### LightGBM Regressor
```python
{
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'num_leaves': 64,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}
```

#### XGBoost Regressor
```python
{
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'random_state': 42,
    'n_jobs': -1
}
```

#### CatBoost Regressor
```python
{
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_state': 42,
    'verbose': False
}
```

#### Random Forest Regressor
```python
{
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1
}
```

#### MLP Regressor
```python
{
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'learning_rate': 'adaptive',
    'max_iter': 500,
    'random_state': 42
}
```

### Classification Models

#### XGBoost Classifier
```python
{
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}
```

#### LightGBM Classifier
```python
{
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'num_leaves': 64,
    'max_depth': -1,
    'min_child_samples': 20,
    'random_state': 42,
    'n_jobs': -1
}
```

#### CatBoost Classifier
```python
{
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_state': 42,
    'verbose': False
}
```

#### Random Forest Classifier
```python
{
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2,
    'random_state': 42,
    'n_jobs': -1
}
```

#### MLP Classifier
```python
{
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 500,
    'random_state': 42
}
```

---

## Table S4: Software Environment

### System Information
- **Operating System:** Windows 11
- **Python Version:** 3.11+
- **Virtual Environment:** perovskite (venv)
- **Training Time:** ~2-3 hours (complete pipeline, both feature sets)

### Core Dependencies
```
mp-api==0.41.2
pandas==2.1.0
numpy==1.25.2
scikit-learn==1.3.0
lightgbm==4.0.0
xgboost==2.0.0
catboost==1.2.0
matplotlib==3.7.2
seaborn==0.12.2
shap==0.42.0
matminer==0.9.0
pymatgen==2023.9.25
python-dotenv==1.0.0
pyyaml==6.0.1
joblib==1.3.2
```

### Hardware
- **Processor:** Multi-core CPU (all cores utilized)
- **Memory:** Sufficient for dataset (< 8GB required)
- **Storage:** ~2GB for models, data, and figures

---

## Table S5: Complete Performance Results

### Regression Performance (Bandgap Prediction)

#### F10 Feature Set (10 features, Test Set)

| Model | R² | MAE (eV) | RMSE (eV) | Rank |
|-------|-----|----------|-----------|------|
| LightGBM | 0.8712 | 0.3934 | 0.5933 | 1st |
| XGBoost | 0.8654 | 0.3662 | 0.6063 | 2nd |
| CatBoost | 0.8653 | 0.3948 | 0.6067 | 3rd |
| Random Forest | 0.8388 | 0.4738 | 0.6635 | 4th |
| MLP | 0.7428 | 0.6060 | 0.8382 | 5th |

#### F22 Feature Set (22 features, Test Set)

| Model | R² | MAE (eV) | RMSE (eV) | Rank |
|-------|-----|----------|-----------|------|
| LightGBM | 0.8836 | 0.3631 | 0.5639 | 1st |
| XGBoost | 0.8834 | 0.3483 | 0.5643 | 2nd |
| CatBoost | 0.8786 | 0.3716 | 0.5760 | 3rd |
| Random Forest | 0.8384 | 0.4831 | 0.6644 | 4th |
| MLP | 0.5308 | 0.7891 | 1.1321 | 5th |

### Classification Performance (Bandgap Type: Direct vs Indirect)

#### F10 Feature Set (10 features, Test Set)

| Model | Accuracy | F1-Score | Precision | Recall | Rank |
|-------|----------|----------|-----------|--------|------|
| LightGBM | 0.8971 | 0.8908 | 0.8919 | 0.8971 | 1st |
| CatBoost | 0.8945 | 0.8887 | 0.8889 | 0.8945 | 2nd |
| XGBoost | 0.8936 | 0.8885 | 0.8881 | 0.8936 | 3rd |
| Random Forest | 0.8893 | 0.8774 | 0.8860 | 0.8893 | 4th |
| MLP | 0.8633 | 0.8435 | 0.8548 | 0.8633 | 5th |

#### F22 Feature Set (22 features, Test Set)

| Model | Accuracy | F1-Score | Precision | Recall | Rank |
|-------|----------|----------|-----------|--------|------|
| LightGBM | 0.9118 | 0.9084 | 0.9083 | 0.9118 | 1st |
| XGBoost | 0.9100 | 0.9062 | 0.9063 | 0.9100 | 2nd |
| CatBoost | 0.9031 | 0.8990 | 0.8987 | 0.9031 | 3rd |
| Random Forest | 0.9014 | 0.8923 | 0.8995 | 0.9014 | 4th |
| MLP | 0.8668 | 0.8517 | 0.8569 | 0.8668 | 5th |

---

## Figure Captions

### Main Figures

**Figure 1:** Distribution of bandgaps across 5,776 double perovskite materials showing bimodal distribution with peak around 1-2 eV.

**Figure 2:** Parity plots showing predicted vs. DFT-calculated bandgaps for (a) F10 LightGBM and (b) F22 LightGBM test sets. Dashed line indicates perfect prediction.

**Figure 3:** Model performance comparison across F10 and F22 feature sets for all five algorithms tested.

**Figure 4:** Error distribution histograms for (a) F10 and (b) F22 best models, showing near-zero mean error and tight distributions.

**Figure 5:** Absolute prediction error vs. true bandgap value, demonstrating increased error for high bandgap materials (>6 eV).

**Figure 6:** Confusion matrices for bandgap type classification on (a) F10 LightGBM and (b) F22 LightGBM test sets.

**Figure 7:** ROC curves for bandgap type classification showing high AUC scores for both feature sets.

**Figure 8:** Feature importance bar plots for (a) F10 and (b) F22, highlighting dominance of thermodynamic and electronic features.

**Figure 9:** SHAP summary plots showing global feature importance and value distributions for (a) F10 and (b) F22 LightGBM regression models.

**Figure 10:** SHAP dependence plots for top 3 features in F22: (a) formation_energy_per_atom, (b) MagpieData avg_dev NUnfilled, (c) frac d valence electrons.

---

## Supplementary Figures

Available in `figures/` directory:
- Validation plots (bandgap distribution, feature correlations, feature distributions)
- Per-model parity plots (all 5 models × 2 feature sets)
- Per-model error histograms and error vs. bandgap plots
- Per-model feature importance plots
- Per-model SHAP summary and bar plots
- Per-model confusion matrices and ROC curves (classification)

---

## Data Availability

- **Materials Project IDs:** Complete list available in `materials_data/` directory
- **Query Configuration:** `experiments/query_config.yaml`
- **Processed Features:** `data/processed/F10/` and `data/processed/F22/`
- **Trained Models:** `models/F10/` and `models/F22/`
- **Code Repository:** https://github.com/AishSoni/Silica-Perovskite-Energy-Band-Gap-Prediction

---

## Reproducibility Statement

All results are fully reproducible with random seed 42. Complete pipeline execution:

```bash
# Regression (default)
python run_pipeline.py F10 F22

# Classification
python run_pipeline.py --task classification F10 F22
```

Expected runtime: 2-3 hours on standard workstation with CPU training.

---

**Document Version:** 1.0  
**Last Updated:** November 16, 2025  
**Corresponds to:** Main manuscript results section
