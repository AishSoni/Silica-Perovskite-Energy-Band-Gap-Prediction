"""
Leakage-free preprocessing for the corrected ML pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from .data_io import leakage_columns


def assert_no_leakage_features(feature_names: List[str], task: str) -> None:
    """Raise if a task-specific leakage column is present."""
    forbidden = leakage_columns(task)
    overlap = forbidden & set(feature_names)
    if overlap:
        raise ValueError(f"Leakage columns found for {task}: {sorted(overlap)}")


def make_imputer(strategy: str):
    """Create an imputer from a config strategy."""
    if strategy in {"mean", "median"}:
        return SimpleImputer(strategy=strategy)
    if strategy == "zero":
        return SimpleImputer(strategy="constant", fill_value=0)
    if strategy == "knn":
        return KNNImputer(n_neighbors=5)
    raise ValueError(f"Unknown imputation strategy: {strategy}")


def make_scaler(scaler_type: Optional[str]):
    """Create a scaler from a config value."""
    if scaler_type in {None, "none", False}:
        return None
    if scaler_type == "robust":
        return RobustScaler()
    if scaler_type == "standard":
        return StandardScaler()
    raise ValueError(f"Unknown scaler type: {scaler_type}")


def _stratify_target(y: pd.Series, task: str):
    if task == "classification":
        counts = y.value_counts()
        return y if len(counts) > 1 and counts.min() >= 2 else None

    # Regression stratification by bins is useful only when every bin has enough samples.
    try:
        binned = pd.qcut(y, q=min(5, y.nunique()), labels=False, duplicates="drop")
        counts = pd.Series(binned).value_counts()
        return binned if len(counts) > 1 and counts.min() >= 2 else None
    except ValueError:
        return None


def _hash_indices(indices: List[Any]) -> str:
    payload = json.dumps([str(index) for index in indices], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    task: str,
    metadata: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    imputation_strategy: str = "median",
    scaler_type: Optional[str] = "robust",
    apply_smote: bool = False,
    output_dir: str | Path | None = None,
    run_name: str = "default",
) -> Dict[str, Any]:
    """
    Split first, then fit imputation/scaling on training data only.
    """
    assert_no_leakage_features(feature_names, task)

    X = X[feature_names].apply(pd.to_numeric, errors="coerce")
    y = y.loc[X.index]
    metadata = metadata.loc[X.index] if metadata is not None and not metadata.empty else pd.DataFrame(index=X.index)

    stratify = _stratify_target(y, task)
    split = train_test_split(
        X,
        y,
        metadata,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    X_train_raw, X_test_raw, y_train, y_test, metadata_train, metadata_test = split

    all_nan_cols = X_train_raw.columns[X_train_raw.isna().all()].tolist()
    if all_nan_cols:
        X_train_raw = X_train_raw.drop(columns=all_nan_cols)
        X_test_raw = X_test_raw.drop(columns=all_nan_cols)
        feature_names = [feature for feature in feature_names if feature not in all_nan_cols]

    imputer = make_imputer(imputation_strategy)
    X_train_arr = imputer.fit_transform(X_train_raw)
    X_test_arr = imputer.transform(X_test_raw)

    X_train = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test_raw.index)

    scaler = make_scaler(scaler_type)
    if scaler is not None:
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

    smote_applied = False
    if task == "classification" and apply_smote:
        counts = y_train.value_counts()
        if len(counts) > 1 and counts.min() >= 2 and counts.max() / counts.min() > 1.5:
            k_neighbors = min(5, int(counts.min()) - 1)
            if k_neighbors >= 1:
                smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                X_train = pd.DataFrame(X_resampled, columns=feature_names)
                y_train = pd.Series(y_resampled, name=y.name)
                smote_applied = True

    split_manifest = {
        "task": task,
        "run_name": run_name,
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "test_size": test_size,
        "random_state": random_state,
        "imputation_strategy": imputation_strategy,
        "scaler": scaler_type,
        "smote_requested": bool(apply_smote),
        "smote_applied": smote_applied,
        "train_index_checksum": _hash_indices(list(X_train_raw.index)),
        "test_index_checksum": _hash_indices(list(X_test_raw.index)),
    }

    result = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metadata_train": metadata_train,
        "metadata_test": metadata_test,
        "feature_names": feature_names,
        "imputer": imputer,
        "scaler": scaler,
        "split_manifest": split_manifest,
    }

    if output_dir is not None:
        save_preprocessed_data(result, output_dir=output_dir, run_name=run_name)

    return result


def save_preprocessed_data(result: Dict[str, Any], output_dir: str | Path, run_name: str) -> None:
    """Save split/preprocessing artifacts."""
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    for key in ["X_train", "X_test", "y_train", "y_test", "metadata_train", "metadata_test"]:
        joblib.dump(result[key], output_path / f"{key}.pkl")

    joblib.dump(result["feature_names"], output_path / "feature_names.pkl")
    joblib.dump(result["imputer"], output_path / "imputer.pkl")
    if result["scaler"] is not None:
        joblib.dump(result["scaler"], output_path / "scaler.pkl")

    with (output_path / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(result["split_manifest"], handle, indent=2, default=str)


if __name__ == "__main__":
    print("Use run_pipeline.py to prepare leakage-free train/test splits.")

