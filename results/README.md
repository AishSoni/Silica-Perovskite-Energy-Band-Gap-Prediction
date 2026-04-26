# Results Directory

This directory is reserved for regenerated outputs from the corrected pipeline.

Expected generated artifacts:

- `dataset_validation/validation_summary.json`
- `dataset_validation/rejected_materials.csv`
- `dataset_validation/accepted_material_ids.txt`
- `feature_sets/`
- `predictions/`
- `corrected_pipeline_results.json`

Legacy result summaries, plots, and feature subsets were removed. Do not manually add metrics here unless they are produced by `run_pipeline.py` from the current dataset manifest.

