import argparse
import json
import math
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate demand forecasting model.")
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--val-csv", default="data/processed/val.csv")
    parser.add_argument("--target", default="Weekly_Sales")

    # Outputs
    parser.add_argument("--metrics-out", default="metrics/val_metrics.json")
    parser.add_argument("--preds-out", default="metrics/val_predictions.csv")
    parser.add_argument("--plots-dir", default="metrics/plots")

    # Optional: training summary produced by the NEW training.py
    # Expected structure:
    # {
    #   "train": {"rf": {...}, "gbr": {...}},
    #   "val": {"rf": {...}, "gbr": {...}},
    #   "best_model": "rf"
    # }
    parser.add_argument("--train-metrics-all", default="metrics/metrics_all.json")

    return parser.parse_args()


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "rows": int(len(y_true)),
    }


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    models_dir = Path(args.models_dir)
    val_path = Path(args.val_csv)

    metrics_path = Path(args.metrics_out)
    preds_path = Path(args.preds_out)
    plots_dir = Path(args.plots_dir)

    train_metrics_all_path = Path(args.train_metrics_all)

    # Ensure output dirs exist
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load validation data
    df = pd.read_csv(val_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {val_path}")

    y_true = df[args.target]
    X = df.drop(columns=[args.target])

    # Decide which models to evaluate:
    # - If models/rf_model.pkl and/or models/gbr_model.pkl exist, evaluate those.
    # - Otherwise, evaluate only args.model.
    candidate_models = {
        "rf": models_dir / "rf_model.pkl",
        "gbr": models_dir / "gbr_model.pkl",
    }
    available = {name: path for name, path in candidate_models.items() if path.exists()}

    if available:
        models_to_eval = available
    else:
        if not model_path.exists():
            raise FileNotFoundError(
                f"No candidate models found in {models_dir} and model path does not exist: {model_path}"
            )
        models_to_eval = {"model": model_path}

    metrics_by_model: Dict[str, Dict[str, float]] = {}
    preds_by_model: Dict[str, pd.Series] = {}

    # Evaluate each model
    for name, path in models_to_eval.items():
        model = joblib.load(path)
        preds = model.predict(X)

        m = compute_metrics(y_true, preds)
        metrics_by_model[name] = m
        preds_by_model[name] = preds

        # Save per-model metrics + predictions
        (metrics_path.parent / f"val_metrics_{name}.json").write_text(
            json.dumps(m, indent=2)
        )

        preds_df = X.copy()
        preds_df["actual"] = y_true
        preds_df["prediction"] = preds
        preds_df.to_csv(preds_path.parent / f"val_predictions_{name}.csv", index=False)

    # Pick best model by validation RMSE
    best_name = min(metrics_by_model.items(), key=lambda item: item[1]["rmse"])[0]
    best_metrics = metrics_by_model[best_name]
    best_preds = preds_by_model[best_name]

    # Save "best" outputs
    metrics_path.write_text(json.dumps(best_metrics, indent=2))

    best_preds_df = X.copy()
    best_preds_df["actual"] = y_true
    best_preds_df["prediction"] = best_preds
    best_preds_df.to_csv(preds_path, index=False)

    # Plot 1: Actual vs Predicted (scatter)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, best_preds, s=10, alpha=0.4)
    ax.set_xlabel("Actual Weekly Sales")
    ax.set_ylabel("Predicted Weekly Sales")
    ax.set_title(f"Actual vs Predicted ({best_name})")
    fig.tight_layout()
    fig.savefig(plots_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # Plot 2: Residual distribution
    residuals = y_true - best_preds
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=50, alpha=0.8)  # no hardcoded color
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals_hist.png", dpi=150)
    plt.close(fig)

    # Plot 3: Train vs Val metrics comparison (if metrics_all.json exists)
    if train_metrics_all_path.exists():
        train_all = json.loads(train_metrics_all_path.read_text())

        # New training.py structure
        train_block = train_all.get("train", {})
        val_block = train_all.get("val", {})

        keys = ["rmse", "mae", "r2"]
        model_names = list(metrics_by_model.keys())

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for idx, metric in enumerate(keys):
            train_vals = [train_block.get(name, {}).get(metric) for name in model_names]

            # For validation bars: prefer current evaluation results, fallback to file if missing
            val_vals = [
                metrics_by_model.get(name, {}).get(metric, val_block.get(name, {}).get(metric))
                for name in model_names
            ]

            x = range(len(model_names))
            ax = axes[idx]
            ax.bar([i - 0.2 for i in x], train_vals, width=0.4, label="Train")
            ax.bar([i + 0.2 for i in x], val_vals, width=0.4, label="Validation")
            ax.set_xticks(list(x), model_names)
            ax.set_title(metric.upper())
            if idx == 0:
                ax.set_ylabel("Metric Value")

        axes[0].legend()
        fig.suptitle("Train vs Validation Metrics by Model")
        fig.tight_layout()
        fig.savefig(plots_dir / "metrics_comparison.png", dpi=150)
        plt.close(fig)
    else:
        # Not an error — just skip comparison plot
        pass

    print(f"Best model (by VAL RMSE): {best_name}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {preds_path}")
    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
