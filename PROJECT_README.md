# Perovskite Bandgap Prediction using Machine Learning

A comprehensive machine learning pipeline for predicting bandgap energies of double perovskite materials using Materials Project data.

**Based on:** "Machine Learning for Predicting Bandgap Energies of Double Perovskites" (Sradhasagar et al., Solar Energy, 2024)

---

## 📋 Project Overview

This project reproduces and extends the methodology from Sradhasagar et al. for predicting perovskite bandgaps using machine learning. We implement a complete end-to-end pipeline including:

- ✅ Data acquisition from Materials Project API
- ✅ Feature engineering (~300 compositional and structural descriptors)
- ✅ Multiple preprocessing strategies (imputation, scaling, splitting)
- ✅ Training of LightGBM, XGBoost, CatBoost, Random Forest, and MLP models
- ✅ Comprehensive evaluation with regression and classification metrics
- ✅ Model interpretability using SHAP and feature importance
- ✅ Candidate material identification for photovoltaic applications

---

## 🗂️ Project Structure

```
perovskite_project/
├─ data/
│  ├─ raw/                      # Raw data from Materials Project
│  └─ processed/                # Preprocessed datasets for ML
├─ notebooks/                   # Jupyter notebooks for exploration
├─ src/                         # Python modules
│  ├─ data_io.py               # Data loading and preparation
│  ├─ featurize.py             # Feature engineering
│  ├─ preprocess.py            # Preprocessing pipeline
│  ├─ models.py                # Model training
│  ├─ eval.py                  # Evaluation and visualization
│  └─ utils.py                 # Utility functions
├─ experiments/                 # Experiment configurations
│  ├─ metadata.json            # Project metadata
│  └─ query_config.yaml        # Materials Project query params
├─ models/                      # Saved trained models
├─ results/                     # Evaluation metrics and predictions
├─ figures/                     # Plots and visualizations
├─ paper/                       # Paper draft and documentation
│  ├─ methods.md               # Methods section
│  ├─ results.md               # Results section
│  ├─ limitations.md           # Limitations and future work
│  └─ supplementary/           # Supplementary materials
├─ materials_data/              # Downloaded perovskite data
├─ requirements.txt             # Python dependencies
├─ run_pipeline.py              # Main pipeline script
└─ run_all.ps1                  # PowerShell automation script
```

---

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Create and activate virtual environment
python -m venv perovskite
.\perovskite\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Set your Materials Project API key as an environment variable:

```powershell
$env:MP_API_KEY = "your_api_key_here"
```

Or add to `.env` file:
```
MP_API_KEY=your_api_key_here
```

Get your API key at: https://next-gen.materialsproject.org/api

### 3. Run the Pipeline

**Option A: Automated (Recommended)**
```powershell
.\run_all.ps1
```

**Option B: Manual**
```powershell
python run_pipeline.py
```

**Option C: Step-by-Step**
```python
# In Python or Jupyter notebook
from src.data_io import load_and_prepare_data
from src.featurize import featurize_data
from src.preprocess import preprocess_data
from src.models import train_models
from src.eval import evaluate_model

# Step 1: Load data
df, metadata = load_and_prepare_data()

# Step 2: Engineer features
df_features = featurize_data()

# Step 3: Preprocess
preprocess_data(imputation_strategy='mean')

# Step 4: Train models
train_models(task='regression')

# Step 5: Evaluate
evaluate_model(model_path='models/lgbm_regression.pkl', task='regression')
```

---

## 📊 Key Results

### Regression Performance (Bandgap Prediction)

| Model | MAE (eV) | RMSE (eV) | R² |
|-------|----------|-----------|-----|
| **LightGBM** | **TBD** | **TBD** | **TBD** |
| XGBoost | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| CatBoost | TBD | TBD | TBD |

### Classification Performance (Direct vs. Indirect Gap)

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| **XGBoost** | **TBD** | **TBD** | **TBD** |
| LightGBM | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |

*Run the pipeline to populate these results!*

---

## 📈 Generated Outputs

After running the pipeline, you'll find:

### 1. Processed Data
- `data/processed/perovskites_features.csv` - Featurized dataset
- `data/processed/features_list.csv` - List of all features
- `data/processed/*.pkl` - Train/test splits

### 2. Trained Models
- `models/lgbm_regression.pkl` - Primary regression model
- `models/xgb_classification.pkl` - Primary classification model
- `models/*.pkl` - All baseline models

### 3. Visualizations
- `figures/parity_plot.png` - Predicted vs. true bandgaps
- `figures/error_histogram.png` - Prediction error distribution
- `figures/confusion_matrix.png` - Classification performance
- `figures/feature_importance.png` - Top 20 important features
- `figures/shap_summary.png` - SHAP analysis plots

### 4. Results
- `results/all_models_summary.json` - All model metrics
- `results/evaluation_results.csv` - Detailed evaluation

---

## 🧪 Reproducibility

This project follows strict reproducibility standards:

### Fixed Random Seeds
- All random operations use seed = 42
- Includes: numpy, sklearn, lightgbm, xgboost, optuna

### Version Tracking
- Python: 3.11+
- Key packages: See `requirements.txt`
- System info saved to `experiments/system_info.json`

### Data Provenance
- Materials Project snapshot date recorded
- Query parameters saved in `experiments/query_config.yaml`
- All material IDs preserved

### Experiment Configuration
- All preprocessing decisions documented
- Hyperparameters saved with models
- Train/test split indices saved

---

## 📚 Methodology

### 1. Data Source
- **Database:** Materials Project (next-gen API)
- **Materials:** ABC₂D₆ double perovskites
- **Properties:** Band gap, formation energy, crystal structure
- **Count:** ~4735 materials (varies by snapshot)

### 2. Feature Engineering
Approximately 300 features generated from:

**Elemental Properties (matminer):**
- Atomic number, mass, radius
- Electronegativity, ionization energy
- Valence electrons, d-electrons
- Group and period statistics

**Structural Features:**
- Lattice parameters (a, b, c, α, β, γ)
- Unit cell volume and density
- Space group information

**Derived Features:**
- Electronegativity differences
- Atomic mass ratios
- Formation energy metrics

### 3. Preprocessing
- **Duplicate removal:** Based on formula + space group
- **Missing values:** Multiple strategies (mean, KNN, MICE)
- **Scaling:** RobustScaler (robust to outliers)
- **Splitting:** 80/20 train/test, stratified for classification
- **Class balancing:** SMOTE for imbalanced classification

### 4. Models Implemented

**Primary Models:**
- LightGBM Regressor (bandgap prediction)
- XGBoost Classifier (gap type prediction)

**Baselines:**
- Random Forest
- CatBoost
- Multi-layer Perceptron
- Support Vector Regressor

**Hyperparameter Tuning:**
- 5-fold cross-validation
- GridSearchCV or Optuna
- Early stopping (50-100 rounds)

### 5. Evaluation

**Regression Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Error distribution analysis

**Classification Metrics:**
- Accuracy, Precision, Recall, F1
- ROC-AUC
- Confusion matrix

**Interpretability:**
- Feature importance rankings
- SHAP (SHapley Additive exPlanations) analysis
- Dependence plots

---

## 🎯 Use Cases

### 1. Reproduce Original Study
Compare results with Sradhasagar et al. (2024):
- Same dataset selection (ABC₂D₆)
- Similar feature engineering (~303 features)
- Equivalent evaluation metrics

### 2. Predict New Materials
Use trained models to screen candidates:
```python
from src.models import ModelTrainer
model = joblib.load('models/lgbm_regression.pkl')

# Predict bandgap for new material
predicted_bandgap = model.predict(new_features)
```

### 3. Identify Solar Cell Candidates
Filter for optimal photovoltaic materials:
- Bandgap: 1.2 - 1.8 eV
- Energy above hull: < 0.2 eV/atom
- Tolerance factor: τ < 4.18

### 4. Model Comparison
Benchmark different algorithms:
- Tree-based (LightGBM, XGBoost, RF, CatBoost)
- Neural networks (MLP)
- Kernel methods (SVR)

### 5. Feature Analysis
Understand structure-property relationships:
- Which elemental properties most influence bandgap?
- How do structural parameters affect predictions?
- SHAP analysis for physical insights

---

## 📖 Documentation

### For Users
- **README.md** (this file) - Quick start and overview
- **AGENTS.md** - Detailed implementation instructions
- **notebooks/** - Jupyter notebooks for exploration

### For Researchers
- **paper/methods.md** - Complete methodology description
- **paper/results.md** - Results template for publication
- **paper/limitations.md** - Known limitations and future work

### For Developers
- **src/** - Well-documented Python modules
- Code comments and docstrings throughout
- Type hints for function signatures

---

## ⚠️ Known Limitations

1. **DFT Bandgap Underestimation**
   - Training data from GGA-DFT underestimates by ~30-50%
   - Models learn DFT predictions, not experimental values

2. **Dataset Snapshot**
   - Materials Project continuously updated
   - Results depend on download date

3. **Missing Experimental Validation**
   - No comparison with measured bandgaps
   - Synthesizability predictions are estimates

4. **Feature Engineering**
   - Limited structural information (no graph neural networks)
   - Missing HOMO/LUMO ionic levels

5. **Uncertainty Quantification**
   - Point predictions only
   - No confidence intervals provided

See `paper/limitations.md` for full discussion and future work.

---

## 🤝 Contributing

This project is designed for educational and research purposes. To extend or improve:

1. **Add New Features:**
   - Edit `src/featurize.py`
   - Add new matminer featurizers or custom features

2. **Try New Models:**
   - Edit `src/models.py`
   - Add model classes and training functions

3. **Improve Preprocessing:**
   - Edit `src/preprocess.py`
   - Implement new imputation or scaling strategies

4. **Enhanced Visualization:**
   - Edit `src/eval.py`
   - Create additional plots and analysis

---

## 📄 Citation

If you use this code or methodology, please cite:

**Original Paper:**
```
Sradhasagar, S., et al. (2024). 
"Machine Learning for Predicting Bandgap Energies of Double Perovskites."
Solar Energy, [Volume], [Pages].
DOI: [...]
```

**This Implementation:**
```
[Your Name]. (2025).
"Perovskite Bandgap Prediction using Machine Learning."
GitHub: [repository URL]
```

---

## 📞 Contact & Support

For questions or issues:
- **GitHub Issues:** [Create an issue](https://github.com/[username]/[repo]/issues)
- **Email:** [your.email@domain.com]

---

## 📜 License

[Specify license - MIT, Apache 2.0, etc.]

---

## 🙏 Acknowledgments

- **Materials Project** for providing high-quality DFT data
- **Sradhasagar et al.** for the original methodology
- **matminer** and **pymatgen** developers for materials informatics tools
- **LightGBM, XGBoost, CatBoost** teams for efficient ML frameworks

---

## 🔄 Version History

- **v1.0.0** (Nov 2025) - Initial release
  - Complete pipeline implementation
  - Regression and classification models
  - SHAP interpretability analysis
  - Comprehensive documentation

---

**Happy materials discovery! 🔬✨**
