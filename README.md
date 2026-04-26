# Double Perovskite Band Gap Prediction

This repository contains a corrected machine-learning pipeline for predicting Materials Project DFT band gaps and direct/indirect gap labels for general double perovskites in the `ABC2D6` / `A2BB'X6` family.

Previous generated datasets, models, figures, feature subsets, and result summaries were removed because they were tied to an invalid data collection workflow. Current results are pending regeneration from the corrected pipeline.

## Corrected Workflow

The pipeline is now organized around one canonical configuration file:

- `experiments/query_config.yaml`

That config defines the dataset scope, Materials Project query ranges, validation rules, output paths, and modeling defaults.

The corrected workflow is:

1. Download broad Materials Project candidate records.
2. Validate candidates for double-perovskite chemistry, formula pattern, structure availability, hull-energy threshold, and target availability.
3. Featurize only validated materials.
4. Split train/test before imputation, scaling, SMOTE, feature selection, or model fitting.
5. Select features on training data only.
6. Train and evaluate models with leakage-free features.
7. Save manifests, split metadata, predictions, metrics, figures, and model metadata.

## Setup

```bash
python -m venv perovskite
source perovskite/bin/activate
pip install -r requirements.txt
```

Set your Materials Project API key:

```bash
export MP_API_KEY="your_materials_project_api_key"
```

`MAPI_KEY` is accepted as a fallback, but `MP_API_KEY` is preferred.

## Run The Corrected Pipeline

Download, validate, featurize, select features, train, and evaluate both tasks:

```bash
python run_pipeline.py --clean --download --validate --featurize --feature-selection --task both
```

If you already have fresh downloaded and validated artifacts:

```bash
python run_pipeline.py --task both --feature-selection
```

Run one task only:

```bash
python run_pipeline.py --task regression --feature-selection
python run_pipeline.py --task classification --feature-selection
```

## Active Artifacts

Generated artifacts are recreated by the pipeline and should not be treated as source files.

- Raw candidates: `data/raw/double_perovskites_raw.csv`
- Raw manifest: `data/raw/dataset_manifest.json`
- Validated materials: `data/processed/validated_materials.csv`
- Features: `data/processed/perovskites_features.csv`
- Validation report: `results/dataset_validation/`
- Feature subsets: `results/feature_sets/`
- Models: `models/`
- Predictions and metrics: `results/`
- Figures: `figures/`

## Important Notes

- Formula pattern alone is not considered proof of double-perovskite structure.
- Classification cannot use `band_gap` as a feature.
- Regression cannot use `is_gap_direct` as a feature.
- Imputation, scaling, SMOTE, and feature selection are fit only on training data or inside cross-validation folds.
- Reported metrics should be trusted only when they trace back to `data/raw/dataset_manifest.json` and `results/dataset_validation/validation_summary.json`.

See `PROJECT_ISSUES_AUDIT.md` for the issue audit and `CLEANUP_POLICY.md` for artifact cleanup rules.

