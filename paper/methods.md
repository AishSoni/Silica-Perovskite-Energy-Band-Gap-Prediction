# Methods

This methods draft describes the corrected pipeline. Numerical dataset sizes, feature counts, and model metrics are pending regeneration from the fixed workflow.

## Data Source

Candidate materials are downloaded from the Materials Project summary API using `download_data.py` and the canonical configuration in `experiments/query_config.yaml`.

The intended dataset scope is general `ABC2D6` / `A2BB'X6` double perovskites, including oxide and halide families. Candidate retrieval is intentionally broad; final acceptance is handled by `src/validate_dataset.py`.

## Dataset Validation

The validation gate checks:

- accepted anonymous formula patterns
- presence of configured X-site chemistry
- presence of configured A-site or B-site chemistry
- non-deprecated status
- structure availability
- band-gap target availability
- energy above hull within the configured threshold
- site-count range
- unique `material_id`

Accepted materials are saved to `data/processed/validated_materials.csv`. Validation counts and rejection reasons are saved under `results/dataset_validation/`.

## Feature Engineering

Features are generated from validated materials using `src/featurize.py`. The featurizer creates composition-based descriptors, selected matminer descriptors, and available scalar structural descriptors.

## Leakage Control

Train/test splitting happens before imputation, scaling, SMOTE, feature selection, or model fitting.

For regression, `is_gap_direct` is excluded from features. For classification, `band_gap` is excluded from features.

## Feature Selection

Feature selection is performed only on the training split. Cross-validation uses fold-local preprocessing through scikit-learn pipelines.

## Modeling

The pipeline supports regression for `band_gap` and classification for `is_gap_direct`. Model artifacts are saved with metadata linking them to the dataset manifest, split manifest, feature list, and configuration checksum.

## Evaluation

Regression metrics include MAE, RMSE, MSE, R2, median absolute error, and error summaries.

Classification metrics include accuracy, balanced accuracy, macro and weighted precision/recall/F1, class-specific direct/indirect metrics, ROC-AUC, and PR-AUC where probabilities are available.

