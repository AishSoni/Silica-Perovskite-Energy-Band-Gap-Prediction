# Quick Start

## 1. Install Dependencies

```bash
python -m venv perovskite
source perovskite/bin/activate
pip install -r requirements.txt
```

## 2. Configure Materials Project

```bash
export MP_API_KEY="your_materials_project_api_key"
```

## 3. Run The Full Corrected Pipeline

```bash
python run_pipeline.py --clean --download --validate --featurize --feature-selection --task both
```

This command:

- removes old generated outputs
- downloads candidate Materials Project records
- validates double-perovskite candidates
- generates features
- creates train-only feature subsets
- trains regression and classification models
- saves metrics, predictions, figures, and model metadata

## 4. Run Without Downloading

Use this only after fresh raw and validated artifacts already exist:

```bash
python run_pipeline.py --validate --featurize --feature-selection --task both
```

## 5. Expected Output Locations

- `data/raw/dataset_manifest.json`
- `data/processed/validated_materials.csv`
- `data/processed/perovskites_features.csv`
- `results/dataset_validation/validation_summary.json`
- `results/corrected_pipeline_results.json`
- `models/`
- `figures/`

No legacy metrics should be reused. If a result cannot be traced to the current dataset manifest and validation summary, treat it as pending regeneration.

