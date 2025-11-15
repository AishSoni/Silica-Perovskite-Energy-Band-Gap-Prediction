# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Setup (One-time)
```powershell
# Clone or navigate to project directory
cd e:\Major_Project

# Create virtual environment (if not exists)
python -m venv perovskite

# Activate environment
.\perovskite\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API
Create a `.env` file or set environment variable:
```powershell
$env:MP_API_KEY = "your_materials_project_api_key"
```

Get your key at: https://next-gen.materialsproject.org/api

### Step 3: Run Pipeline
```
# Regression (default)
python run_pipeline.py F10                    # Train regression with F10
python run_pipeline.py F10 F22                # Train regression with both

# Classification
python run_pipeline.py --task classification F10      # Bandgap type prediction
python run_pipeline.py --task classification F10 F22  # Multiple feature sets

# Skip SHAP (faster)
python run_pipeline.py --no-shap F10

# Help
python run_pipeline.py --help
```

That's it! The pipeline will:
- ✅ Load existing perovskite data
- ✅ Engineer ~300 features
- ✅ Preprocess with multiple strategies
- ✅ Train LightGBM, XGBoost, and baseline models
- ✅ Generate evaluation metrics and visualizations

---

## 📂 Where to Find Results

After running:

| What | Where |
|------|-------|
| **Processed Data** | `data/processed/perovskites_features.csv` |
| **Trained Models** | `models/lgbm_regression.pkl` |
| **Metrics** | `results/all_models_summary.json` |
| **Plots** | `figures/*.png` |
| **Feature List** | `data/processed/features_list.csv` |

---

## 🎯 Common Tasks

### View Results
```python
import json
with open('results/all_models_summary.json') as f:
    results = json.load(f)
print(results)
```

### Load a Trained Model
```python
import joblib
model = joblib.load('models/lgbm_regression.pkl')
# Make predictions
predictions = model.predict(X_test)
```

### Check Feature Importance
```python
import pandas as pd
import joblib

model = joblib.load('models/lgbm_regression.pkl')
features = pd.read_csv('data/processed/features_list.csv')

importance = pd.DataFrame({
    'feature': features['feature_name'],
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(20))
```

### Run Specific Steps Only
```python
from src import featurize, preprocess, models

# Only featurize
featurize.featurize_data()

# Only preprocess
preprocess.preprocess_data(imputation_strategy='knn')

# Only train
models.train_models(task='regression')
```

---

## ⚠️ Troubleshooting

### Import Errors
```powershell
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Memory Errors
Reduce dataset size or use subsampling:
```python
df = df.sample(n=1000, random_state=42)  # Use 1000 samples
```

### SHAP Takes Too Long
Reduce samples in SHAP analysis:
```python
# In eval.py, line ~XXX
shap_values = explainer.shap_values(X_test[:100])  # Only 100 samples
```

### Missing Data Files
Ensure existing data is in `materials_data/`:
```powershell
ls materials_data/
# Should show: all_perovskite_complete_attributes.csv
```

---

## 📊 Quick Results Check

After pipeline completes, run:

```python
import pandas as pd
import json

# Load results
with open('results/all_models_summary.json') as f:
    results = json.load(f)

# Print regression results
for model_name, metrics in results.items():
    if 'Regression' in model_name:
        print(f"\n{model_name}:")
        print(f"  MAE: {metrics.get('MAE', 'N/A'):.3f} eV")
        print(f"  R²: {metrics.get('R²', 'N/A'):.3f}")
```

---

## 📖 Learn More

- **Full Documentation:** See `PROJECT_README.md`
- **Implementation Details:** See `IMPLEMENTATION_SUMMARY.md`
- **Paper Writing:** See `paper/methods.md` and `paper/results.md`
- **Limitations:** See `paper/limitations.md`

---

## 🆘 Need Help?

1. Check `PROJECT_README.md` for detailed instructions
2. Review error messages carefully
3. Verify Python version (3.11+)
4. Ensure all dependencies installed
5. Check that data files exist in `materials_data/`

---

**Happy modeling! 🔬✨**
