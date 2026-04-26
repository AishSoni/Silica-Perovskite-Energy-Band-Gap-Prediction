"""
Canonical data loading helpers for the corrected pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .pipeline_config import load_config, path_from_config


METADATA_COLUMNS = {
    "material_id",
    "formula_pretty",
    "formula_anonymous",
    "elements",
    "symmetry_json",
    "structure_json",
    "composition",
    "crystal_system",
    "spacegroup_symbol",
    "deprecated",
    "theoretical",
}

TARGET_COLUMNS = {
    "regression": "band_gap",
    "classification": "is_gap_direct",
}

LEAKAGE_COLUMNS = {
    "regression": {"is_gap_direct"},
    "classification": {"band_gap"},
}


def target_column(task: str) -> str:
    """Return the target column for a task."""
    if task not in TARGET_COLUMNS:
        raise ValueError(f"Unknown task: {task}. Expected one of {sorted(TARGET_COLUMNS)}")
    return TARGET_COLUMNS[task]


def leakage_columns(task: str) -> set:
    """Return columns that must not be used as features for a task."""
    if task not in LEAKAGE_COLUMNS:
        raise ValueError(f"Unknown task: {task}. Expected one of {sorted(LEAKAGE_COLUMNS)}")
    return LEAKAGE_COLUMNS[task]


def load_validated_data(config_path: str | Path = "experiments/query_config.yaml") -> pd.DataFrame:
    """Load the canonical validated materials table."""
    config = load_config(config_path)
    path = path_from_config(config, "validated_csv")
    if not path.exists():
        raise FileNotFoundError(f"Validated data not found: {path}. Run src/validate_dataset.py first.")
    return pd.read_csv(path)


def load_features(
    config_path: str | Path = "experiments/query_config.yaml",
    features_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the canonical featurized dataset."""
    config = load_config(config_path)
    path = Path(features_path) if features_path else path_from_config(config, "features_csv")
    if not path.exists():
        raise FileNotFoundError(f"Featurized data not found: {path}. Run featurization first.")
    return pd.read_csv(path)


def select_feature_columns(
    df: pd.DataFrame,
    task: str,
    selected_features: Optional[List[str]] = None,
) -> List[str]:
    """Return numeric, leakage-free feature columns for a task."""
    target = target_column(task)
    excluded = set(METADATA_COLUMNS) | {target} | leakage_columns(task)

    if selected_features is not None:
        missing = [feature for feature in selected_features if feature not in df.columns]
        if missing:
            raise ValueError(f"Selected features missing from dataset: {missing}")
        candidates = selected_features
    else:
        candidates = [column for column in df.columns if column not in excluded]

    numeric_candidates = []
    for column in candidates:
        if column in excluded:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        if series.notna().any():
            numeric_candidates.append(column)

    return numeric_candidates


def prepare_training_frame(
    task: str,
    config_path: str | Path = "experiments/query_config.yaml",
    features_path: str | Path | None = None,
    selected_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Load features and return X, y, feature names, and metadata.

    No imputation, scaling, splitting, SMOTE, or feature selection happens here.
    """
    df = load_features(config_path=config_path, features_path=features_path)
    target = target_column(task)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in feature data.")

    feature_names = select_feature_columns(df, task=task, selected_features=selected_features)
    if not feature_names:
        raise ValueError("No numeric, leakage-free feature columns were found.")

    valid_mask = df[target].notna()
    X = df.loc[valid_mask, feature_names].apply(pd.to_numeric, errors="coerce")
    y = df.loc[valid_mask, target].copy()

    if task == "classification":
        y = y.astype(bool).astype(int)
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid_y = y.notna()
        X = X.loc[valid_y]
        y = y.loc[valid_y]

    metadata_cols = [column for column in ["material_id", "formula_pretty", "formula_anonymous"] if column in df.columns]
    metadata = df.loc[X.index, metadata_cols].copy() if metadata_cols else pd.DataFrame(index=X.index)

    return X, y, feature_names, metadata


def load_feature_subset(subset_name: str, features_dir: str | Path = "results/feature_sets") -> List[str]:
    """Load a generated feature subset file."""
    subset_path = Path(features_dir) / f"feature_subset_{subset_name}.txt"
    if not subset_path.exists():
        raise FileNotFoundError(f"Feature subset not found: {subset_path}")

    features = []
    for line in subset_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        features.append(line)
    return features


def dataset_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Return a small summary for logs and reports."""
    summary: Dict[str, object] = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
    }
    if "material_id" in df:
        summary["n_material_ids"] = int(df["material_id"].nunique())
    if "formula_pretty" in df:
        summary["n_formulas"] = int(df["formula_pretty"].nunique())
    if "band_gap" in df:
        band_gap = pd.to_numeric(df["band_gap"], errors="coerce")
        summary["band_gap_min"] = float(np.nanmin(band_gap)) if band_gap.notna().any() else None
        summary["band_gap_max"] = float(np.nanmax(band_gap)) if band_gap.notna().any() else None
    if "is_gap_direct" in df:
        summary["is_gap_direct_counts"] = df["is_gap_direct"].value_counts(dropna=False).to_dict()
    return summary


if __name__ == "__main__":
    data = load_validated_data()
    print(dataset_summary(data))

