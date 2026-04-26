# Training Challenges And Solutions

This document will be regenerated after the corrected pipeline is run.

Known corrected-design safeguards:

- dataset validation before featurization
- split-before-preprocessing
- no `band_gap` feature for classification
- no `is_gap_direct` feature for regression
- train-only SMOTE
- train-only feature selection
- model metadata tied to dataset and split manifests

Legacy challenge descriptions that referenced invalid sample counts or corrupted generated outputs were removed.

