import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, Set

import mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log DVC outputs to MLflow (new metrics_all.json format).")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment", default="retail-demand")

    # NEW training.py default output:
    # metrics/metrics_all.json
    parser.add_argument("--metrics-all", default="metrics/metrics_all.json")

    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--plots-dir", default="metrics/plots")
    return parser.parse_args()


def log_metrics(prefix: str, payload: Dict[str, Any]) -> None:
    """
    Logs:
      - rows as a param
      - numeric metrics (rmse/mae/r2/...) as metrics
    """
    for key, value in payload.items():
        if key == "rows":
            mlflow.log_param(f"{prefix}_rows", value)
        else:
            # Only log numeric-like values as metrics
            try:
                mlflow.log_metric(f"{prefix}_{key}", float(value))
            except (TypeError, ValueError):
                # Skip non-numeric values safely
                pass


def union_model_names(train_block: Dict[str, Any], val_block: Dict[str, Any]) -> Iterable[str]:
    names: Set[str] = set()
    if isinstance(train_block, dict):
        names |= set(train_block.keys())
    if isinstance(val_block, dict):
        names |= set(val_block.keys())
    return sorted(names)


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    metrics_all_path = Path(args.metrics_all)
    metrics_dir = Path(args.metrics_dir)
    models_dir = Path(args.models_dir)
    plots_dir = Path(args.plots_dir)

    if not metrics_all_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_all_path}. Run training first to generate metrics_all.json."
        )

    payload = json.loads(metrics_all_path.read_text())

    # NEW structure expected:
    # {
    #   "train": {"rf": {...}, "gbr": {...}},
    #   "val": {"rf": {...}, "gbr": {...}},
    #   "best_model": "rf"
    # }
    train_block = payload.get("train", {}) if isinstance(payload, dict) else {}
    val_block = payload.get("val", {}) if isinstance(payload, dict) else {}
    best_model = payload.get("best_model")

    model_names = list(union_model_names(train_block, val_block))
    if not model_names:
        raise ValueError(f"No models found in {metrics_all_path}. Expected keys under 'train'/'val'.")

    for name in model_names:
        train_metrics = train_block.get(name, {}) if isinstance(train_block, dict) else {}
        val_metrics = val_block.get(name, {}) if isinstance(val_block, dict) else {}

        # Artifacts (optional)
        model_path = models_dir / f"{name}_model.pkl"
        train_metrics_path = metrics_dir / f"train_metrics_{name}.json"
        val_metrics_path = metrics_dir / f"val_metrics_{name}.json"
        preds_path = metrics_dir / f"val_predictions_{name}.csv"

        with mlflow.start_run(run_name=f"{name}_model"):
            # Params
            mlflow.log_param("model_name", name)
            if best_model is not None:
                mlflow.log_param("best_model", best_model)
                mlflow.log_param("is_best", int(name == best_model))

            # Metrics
            if isinstance(train_metrics, dict) and train_metrics:
                log_metrics("train", train_metrics)
            if isinstance(val_metrics, dict) and val_metrics:
                log_metrics("val", val_metrics)

            # Always log the global summary too (useful to debug)
            mlflow.log_artifact(str(metrics_all_path))

            # Artifacts per model (if exist)
            if model_path.exists():
                mlflow.log_artifact(str(model_path))
            if train_metrics_path.exists():
                mlflow.log_artifact(str(train_metrics_path))
            if val_metrics_path.exists():
                mlflow.log_artifact(str(val_metrics_path))
            if preds_path.exists():
                mlflow.log_artifact(str(preds_path))

            # Plots dir
            if plots_dir.exists():
                mlflow.log_artifact(str(plots_dir))

    print("Logged metrics and artifacts to MLflow.")


if __name__ == "__main__":
    main()
