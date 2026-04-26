"""
Corrected end-to-end pipeline for double-perovskite band-gap modeling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from download_data import run_download
from src.data_io import prepare_training_frame
from src.eval import evaluate_model
from src.featurize import featurize_data
from src.models import train_models
from src.pipeline_config import ensure_dir, load_config, path_from_config
from src.preprocess import split_preprocess_data
from src.reduce_features import select_feature_subsets
from src.utils import get_system_info, save_json, set_seeds
from src.validate_dataset import run_validation


GENERATED_PATTERNS = [
    "data/raw/*.csv",
    "data/raw/*.json",
    "data/processed/*.csv",
    "data/processed/*.pkl",
    "data/processed/*.json",
    "data/processed/features_list.csv",
    "results/*.json",
    "results/*.csv",
    "results/*.png",
    "results/feature_sets/*.txt",
    "results/feature_sets/*.csv",
    "results/dataset_validation/*.json",
    "results/dataset_validation/*.csv",
    "results/dataset_validation/*.txt",
    "figures/**/*.png",
    "validation/**/*.png",
    "models/**/*.pkl",
    "models/**/*.json",
]


def clean_generated_outputs() -> None:
    """Remove generated outputs while preserving .gitkeep placeholders."""
    for pattern in GENERATED_PATTERNS:
        for path in Path(".").glob(pattern):
            if path.name == ".gitkeep" or not path.is_file():
                continue
            path.unlink()


def _tasks(task_arg: str) -> List[str]:
    return ["regression", "classification"] if task_arg == "both" else [task_arg]


def _subset_sizes(config: Dict[str, Any]) -> List[int]:
    sizes = config.get("modeling", {}).get("feature_subset_sizes", [10, 22])
    return [int(size) for size in sizes]


def _run_name(task: str, subset_name: str) -> str:
    return f"{task}_{subset_name}"


def _save_predictions(
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metadata_test: pd.DataFrame,
    task: str,
    output_dir: Path,
) -> Path:
    predictions = metadata_test.reset_index(drop=True).copy()
    predictions["y_true"] = list(y_test.reset_index(drop=True))
    predictions["y_pred"] = list(model.predict(X_test))
    predictions["task"] = task
    predictions["model"] = model_name

    if task == "regression":
        predictions["error"] = predictions["y_pred"] - predictions["y_true"]
        predictions["absolute_error"] = predictions["error"].abs()
    elif hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] == 2:
            predictions["probability_direct"] = proba[:, 1]

    path = output_dir / f"{model_name}_predictions.csv"
    predictions.to_csv(path, index=False)
    return path


def _evaluate_trained_models(
    trained_models: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metadata_test: pd.DataFrame,
    feature_names: List[str],
    task: str,
    subset_name: str,
    config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    figures_root = ensure_dir(Path(config["paths"]["figures_dir"]) / task / subset_name)
    results_root = ensure_dir(Path(config["paths"]["results_dir"]) / "predictions" / task / subset_name)

    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    for model_name, model_info in trained_models.items():
        model = model_info["model"]
        model_figures = figures_root / model_name
        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            task=task,
            output_dir=str(model_figures),
            model_name=model_name,
            feature_names=feature_names,
        )
        prediction_path = _save_predictions(
            model_name=model_name,
            model=model,
            X_test=X_test,
            y_test=y_test,
            metadata_test=metadata_test,
            task=task,
            output_dir=results_root,
        )
        metrics["predictions_path"] = str(prediction_path)
        metrics_by_model[model_name] = metrics

    return metrics_by_model


def _dataset_manifest_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    manifest_path = path_from_config(config, "manifest")
    if not manifest_path.exists():
        return {"dataset_manifest": str(manifest_path), "dataset_manifest_available": False}
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_available": True,
        "dataset_name": manifest.get("dataset_name"),
        "dataset_family": manifest.get("dataset_family"),
        "config_checksum": manifest.get("config_checksum"),
        "material_id_checksum": manifest.get("material_id_checksum"),
    }


def run_task(task: str, config: Dict[str, Any], config_path: str, feature_selection: bool) -> Dict[str, Any]:
    """Run preprocessing, optional feature selection, training, and evaluation for one task."""
    modeling = config.get("modeling", {})
    X, y, feature_names, metadata = prepare_training_frame(task=task, config_path=config_path)

    base_split = split_preprocess_data(
        X=X,
        y=y,
        feature_names=feature_names,
        task=task,
        metadata=metadata,
        test_size=float(modeling.get("test_size", 0.2)),
        random_state=int(modeling.get("random_state", 42)),
        imputation_strategy=str(modeling.get("imputation_strategy", "median")),
        scaler_type=modeling.get("scaler", "robust"),
        apply_smote=bool(modeling.get("apply_smote", True)),
        output_dir=Path("data/processed/splits"),
        run_name=_run_name(task, "all_features"),
    )

    subsets: Dict[str, List[str]] = {"all_features": list(base_split["feature_names"])}
    feature_selection_scores: Dict[str, Any] = {}

    if feature_selection:
        selected = select_feature_subsets(
            X_train=base_split["X_train"],
            y_train=base_split["y_train"],
            task=task,
            subset_sizes=_subset_sizes(config),
            output_dir=config["paths"]["feature_sets_dir"],
            random_state=int(modeling.get("random_state", 42)),
        )
        subsets = {name: payload["features"] for name, payload in selected.items()}
        feature_selection_scores = {name: payload["cv_score"] for name, payload in selected.items()}

    task_results: Dict[str, Any] = {
        "feature_selection_scores": feature_selection_scores,
        "subsets": {},
    }

    for subset_name, subset_features in subsets.items():
        split = base_split
        X_train = split["X_train"][subset_features]
        X_test = split["X_test"][subset_features]
        models_dir = Path(config["paths"]["models_dir"]) / task / subset_name
        trained = train_models(
            X_train=X_train,
            y_train=split["y_train"],
            X_test=X_test,
            y_test=split["y_test"],
            feature_names=subset_features,
            output_dir=str(models_dir),
            task=task,
        )
        model_metadata = {
            "task": task,
            "subset_name": subset_name,
            "feature_names": subset_features,
            "split_manifest": split["split_manifest"],
            **_dataset_manifest_metadata(config),
        }
        with (models_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(model_metadata, handle, indent=2, default=str)
        metrics = _evaluate_trained_models(
            trained_models=trained,
            X_test=X_test,
            y_test=split["y_test"],
            metadata_test=split["metadata_test"],
            feature_names=subset_features,
            task=task,
            subset_name=subset_name,
            config=config,
        )
        task_results["subsets"][subset_name] = {
            "n_features": len(subset_features),
            "features": subset_features,
            "metrics": metrics,
        }

    return task_results


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)
    set_seeds(int(config.get("modeling", {}).get("random_state", 42)))

    if args.clean:
        clean_generated_outputs()

    if args.download:
        run_download(args.config)

    if args.validate:
        run_validation(args.config)

    if args.featurize:
        featurize_data(
            input_path=str(path_from_config(config, "validated_csv")),
            output_path=str(path_from_config(config, "features_csv")),
            feature_list_path="data/processed/features_list.csv",
            use_structure_features=False,
        )

    all_results: Dict[str, Any] = {
        "system_info": get_system_info(),
        "config_path": args.config,
        "tasks": {},
    }

    for task in _tasks(args.task):
        all_results["tasks"][task] = run_task(
            task=task,
            config=config,
            config_path=args.config,
            feature_selection=args.feature_selection,
        )

    results_path = Path(config["paths"]["results_dir"]) / "corrected_pipeline_results.json"
    save_json(all_results, str(results_path))
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the corrected perovskite ML pipeline.")
    parser.add_argument("--config", default="experiments/query_config.yaml", help="Canonical query config path.")
    parser.add_argument("--task", choices=["regression", "classification", "both"], default="both")
    parser.add_argument("--download", action="store_true", help="Download fresh Materials Project candidates.")
    parser.add_argument("--validate", action="store_true", help="Validate downloaded candidates.")
    parser.add_argument("--featurize", action="store_true", help="Generate features from validated materials.")
    parser.add_argument("--feature-selection", action="store_true", help="Create train-only feature subsets.")
    parser.add_argument("--clean", action="store_true", help="Remove generated outputs before running.")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())

