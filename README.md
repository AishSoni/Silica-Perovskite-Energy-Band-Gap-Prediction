# Perovskite Bandgap Prediction using Machine Learning

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Materials Project](https://img.shields.io/badge/data-Materials%20Project-green.svg)](https://materialsproject.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-accuracy machine learning pipeline for predicting electronic bandgaps and bandgap types of double perovskite materials using compositional and structural features from DFT calculations.

---

## 🎯 Key Features

- ✅ **Outstanding Performance**: R² = 0.88, MAE = 0.35 eV (2.2× better than targets)
- ✅ **Dual Tasks**: Bandgap regression + type classification (Direct vs Indirect)  
- ✅ **SHAP Analysis**: Explainable AI with feature importance visualization
- ✅ **Multiple Models**: LightGBM, XGBoost, Random Forest, CatBoost, MLP
- ✅ **Automated Pipeline**: End-to-end workflow from data download to evaluation
- ✅ **Production Ready**: Robust error handling, validation plots, comprehensive metrics

---

## 📊 Results Summary

### Regression (Bandgap Prediction)

| Feature Set | Best Model | R² | MAE (eV) | RMSE (eV) |
|-------------|------------|-----|----------|-----------|
| **F22** (22 features) | LightGBM | **0.8836** | **0.3631** | 0.5639 |
| **F10** (10 features) | LightGBM | 0.8712 | 0.3934 | 0.5933 |

**Target**: R² ≥ 0.40, MAE ≤ 0.45 eV  
**Achieved**: 2.2× better R², 23% lower MAE ✨

### Classification (Bandgap Type: Direct vs Indirect)

| Feature Set | Best Model | Accuracy | F1-Score | Precision | Recall |
|-------------|------------|----------|----------|-----------|--------|
| **F10** (10 features) | LightGBM | **0.8971** | **0.8908** | 0.8919 | 0.8971 |

**Target**: Accuracy ≥ 0.80, F1 ≥ 0.80  
**Achieved**: 12% above target ✨

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/AishSoni/Silica-Perovskite-Energy-Band-Gap-Prediction.git
cd Silica-Perovskite-Energy-Band-Gap-Prediction

# Create virtual environment
python -m venv perovskite
source perovskite/bin/activate  # On Windows: perovskite\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Access

Create a `.env` file with your Materials Project API key:

```env
MAPI_KEY=your_api_key_here
```

Get your free API key at: https://materialsproject.org/api

### 3. Run the Pipeline

#### Regression (Bandgap Prediction)

```bash
python run_pipeline.py F10          # Train with 10 features
python run_pipeline.py F10 F22      # Train with both feature sets
```

#### Classification (Bandgap Type Prediction)

```bash
python run_pipeline.py --task classification F10
```

#### Skip SHAP Analysis (Faster Execution)

```bash
python run_pipeline.py --no-shap F10
```

### 4. View Results

All outputs are automatically generated:

- **Validation plots**: `validation/{F10,F22}/`
- **Trained models**: `models/{F10,F22}/`
- **Evaluation figures**: `figures/{F10,F22}/`
- **SHAP analysis**: `figures/{F10,F22}/{model}/shap_*.png`
- **Results summary**: `results/all_models_summary.json`
- **Model comparison**: `results/model_comparison.png`

---

## 📁 Project Structure

```
perovskite_project/
├── data/                       # Raw and processed datasets
│   ├── raw/                    # Raw data from Materials Project
│   └── processed/              # Featurized and cleaned data
├── src/                        # Source code modules
│   ├── data_io.py             # Data loading and preparation
│   ├── featurize.py           # Feature engineering with Matminer
│   ├── preprocess.py          # Data preprocessing and scaling
│   ├── models.py              # Model training (regression/classification)
│   ├── eval.py                # Evaluation and SHAP analysis
│   └── utils.py               # Utility functions
├── experiments/                # Experiment configurations
│   ├── metadata.json          # System information
│   ├── query_config.yaml      # Data query parameters
│   └── system_info.json       # Pipeline run metadata
├── models/                     # Saved model artifacts (.pkl files)
├── results/                    # Predictions and metrics
├── figures/                    # Visualization outputs
├── paper/                      # Paper drafts and documentation
│   ├── methods.md             # Methodology description
│   ├── results.md             # Results and analysis
│   └── limitations.md         # Known limitations
├── run_pipeline.py            # Main pipeline script
├── download_data.py           # Data acquisition script
├── test_shap_classification.py # Example test script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🔬 Methodology

### Dataset

- **5,776 double perovskites** (ABC₂D₆ formula family)  
- Source: Materials Project next-gen API  
- DFT-calculated bandgaps using VASP (GGA/GGA+U, r2SCAN functional)
- Structure: Lattice parameters, space groups, density, volume
- Class distribution: 80.7% Indirect, 19.3% Direct

### Feature Engineering

**293 features** generated using Matminer:

- **ElementProperty** (magpie descriptors): 128 features
- **Stoichiometry**: 22 features  
- **ElementFraction**: 112 features  
- **Structural**: Lattice parameters, derived ratios, packing fraction
- **Compositional**: Electronegativity differences, valence electrons

### Feature Selection

- **F4 to F24**: Tested 11 feature subsets using RFE with cross-validation
- **F22**: Best performance (R²=0.7620 CV), 22 most important features
- **F10**: Simpler model (R²=0.7386 CV), good balance between accuracy and complexity

### Models

**Regression (Primary):**
- LightGBM (best: R²=0.8836)  
- XGBoost, Random Forest, CatBoost, MLP

**Classification (Primary):**
- XGBoost (Accuracy=0.8936)  
- LightGBM (best: Accuracy=0.8971)  
- Random Forest, CatBoost, MLP

### Evaluation

- **Train/Test Split**: 80/20 stratified split  
- **Scaling**: RobustScaler (handles outliers)
- **Metrics**: 
  - Regression: MAE, RMSE, R²
  - Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Explainability**: SHAP values for feature importance

---

## 📖 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Detailed setup and usage guide  
- **[paper/methods.md](paper/methods.md)** - Methodology documentation  
- **[paper/results.md](paper/results.md)** - Results and analysis  
- **[paper/limitations.md](paper/limitations.md)** - Known limitations and future work

---

## 🛠️ Dependencies

All required packages are in `requirements.txt`:

- **mp-api** (≥0.41.0) - Materials Project API client  
- **pandas**, **numpy** - Data manipulation  
- **scikit-learn** - ML algorithms and preprocessing  
- **lightgbm** (≥4.0.0) - Best performing model  
- **xgboost**, **catboost** - Gradient boosting models  
- **matplotlib**, **seaborn** - Visualization  
- **shap** - Explainability analysis  
- **matminer** (≥0.9.0) - Materials featurization  
- **pymatgen** (≥2023.9.0) - Materials analysis

---

## 📈 Usage Examples

### Basic Usage

```bash
# Default: F10 regression with SHAP
python run_pipeline.py

# Multiple feature sets
python run_pipeline.py F10 F22

# Classification task
python run_pipeline.py --task classification F10

# Faster (skip SHAP)
python run_pipeline.py --no-shap F22
```

### Help

```bash
python run_pipeline.py --help
```

---

## 🎯 Key Findings

1. **Feature Importance**: 
   - Top features: Electronegativity statistics, atomic radii, GSbandgap descriptors
   - SHAP analysis reveals complex feature interactions

2. **Performance**:
   - Best regression: F22 XGBoost (R²=0.8807, MAE=0.3454 eV)
   - Best classification: F10 LightGBM (Accuracy=89.71%)
   - Simpler F10 models nearly match F22 performance

3. **Validation**:
   - Error distribution centered near 0 eV
   - Most predictions within ±0.5 eV of DFT values
   - PV-relevant bandgap range (1.2-1.8 eV) well-represented

---

## 🔍 Future Work

- **Hyperparameter Optimization**: Optuna/GridSearch for even better performance  
- **Graph Neural Networks**: Structure-aware models (CGCNN, MEGNet)  
- **Active Learning**: Iterative model improvement with targeted experiments  
- **GW Corrections**: Train on GW-corrected bandgaps for higher accuracy  
- **Candidate Generation**: Predict properties of hypothetical perovskites

See [paper/limitations.md](paper/limitations.md) for detailed discussion.

---

## 📝 Citation

If you use this code or methodology, please cite:

```bibtex
@software{perovskite_bandgap_prediction_2024,
  author = {Aish Soni},
  title = {Perovskite Bandgap Prediction using Machine Learning},
  year = {2024},
  url = {https://github.com/AishSoni/Silica-Perovskite-Energy-Band-Gap-Prediction}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

**Aish Soni**  
GitHub: [@AishSoni](https://github.com/AishSoni)

For questions or issues, please open an issue on GitHub.

---

**Made with ❤️ for materials science research**
