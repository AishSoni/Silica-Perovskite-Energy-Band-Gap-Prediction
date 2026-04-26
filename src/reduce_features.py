"""
Train-only feature selection for the corrected pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def _estimator(task: str, random_state: int):
    if task == "classification":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=random_state,
            verbose=-1,
            force_col_wise=True,
        )
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=random_state,
        verbose=-1,
        force_col_wise=True,
    )


def _cv(task: str, y: pd.Series, random_state: int):
    if task == "classification" and y.nunique() > 1 and y.value_counts().min() >= 3:
        return StratifiedKFold(n_splits=min(5, int(y.value_counts().min())), shuffle=True, random_state=random_state)
    return KFold(n_splits=min(5, max(2, len(y) // 5)), shuffle=True, random_state=random_state)


def filter_training_features(X_train: pd.DataFrame, missing_threshold: float = 0.10, corr_threshold: float = 0.92) -> pd.DataFrame:
    """Filter features using training data only."""
    missing_fraction = X_train.isna().mean()
    keep = missing_fraction[missing_fraction <= missing_threshold].index.tolist()
    X_filtered = X_train[keep].copy()

    imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_filtered),
        columns=X_filtered.columns,
        index=X_filtered.index,
    )

    selector = VarianceThreshold(threshold=0.0)
    selector.fit(imputed)
    keep = imputed.columns[selector.get_support()].tolist()
    X_filtered = X_filtered[keep]
    imputed = imputed[keep]

    corr = imputed.corr().abs()
    upper = corr.where(pd.DataFrame(True, index=corr.index, columns=corr.columns).values)
    to_drop = set()
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            if corr.loc[left, right] > corr_threshold:
                to_drop.add(right)

    return X_filtered[[column for column in X_filtered.columns if column not in to_drop]]


def rank_features(X_train: pd.DataFrame, y_train: pd.Series, task: str, random_state: int, top_n: int = 50) -> pd.DataFrame:
    """Rank features by a model fit on training data only."""
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_imputed = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imputed)

    model = _estimator(task, random_state)
    model.fit(X_scaled, y_train)

    rankings = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": getattr(model, "feature_importances_", [0] * X_train.shape[1]),
        }
    ).sort_values("importance", ascending=False)

    return rankings.head(min(top_n, len(rankings))).reset_index(drop=True)


def select_with_rfe(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_features: List[str],
    task: str,
    n_features: int,
    random_state: int,
) -> List[str]:
    """Run RFE on training data only."""
    candidate_features = candidate_features[:]
    n_features = min(n_features, len(candidate_features))
    X_candidates = X_train[candidate_features]

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X_candidates))

    selector = RFE(_estimator(task, random_state), n_features_to_select=n_features, step=1)
    selector.fit(X_scaled, y_train)

    return [feature for feature, keep in zip(candidate_features, selector.support_) if keep]


def evaluate_subset(X_train: pd.DataFrame, y_train: pd.Series, features: List[str], task: str, random_state: int) -> float:
    """Evaluate a feature subset with fold-local preprocessing."""
    scoring = "balanced_accuracy" if task == "classification" else "r2"
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", _estimator(task, random_state)),
        ]
    )
    scores = cross_val_score(
        pipeline,
        X_train[features],
        y_train,
        cv=_cv(task, y_train, random_state),
        scoring=scoring,
        n_jobs=-1,
    )
    return float(scores.mean())


def select_feature_subsets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: str,
    subset_sizes: Iterable[int],
    output_dir: str | Path = "results/feature_sets",
    random_state: int = 42,
    top_n: int = 50,
) -> Dict[str, Dict[str, object]]:
    """Create feature subsets using training data only."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X_filtered = filter_training_features(X_train)
    rankings = rank_features(X_filtered, y_train, task=task, random_state=random_state, top_n=top_n)
    rankings.to_csv(output_path / f"{task}_feature_importance_rankings.csv", index=False)

    candidates = rankings["feature"].tolist()
    subsets: Dict[str, Dict[str, object]] = {}

    for size in subset_sizes:
        if size > len(candidates):
            continue
        name = f"F{size}"
        features = select_with_rfe(X_filtered, y_train, candidates, task=task, n_features=size, random_state=random_state)
        score = evaluate_subset(X_filtered, y_train, features, task=task, random_state=random_state)
        subsets[name] = {"features": features, "cv_score": score}

        subset_file = output_path / f"{task}_feature_subset_{name}.txt"
        with subset_file.open("w", encoding="utf-8") as handle:
            handle.write(f"# {task} {name} Feature Subset ({len(features)} features)\n")
            handle.write(f"# Train-only CV score: {score:.6f}\n\n")
            for feature in features:
                handle.write(f"{feature}\n")

    return subsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature selection is run through run_pipeline.py.")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    print("Use run_pipeline.py --feature-selection to create train-only feature subsets.")

