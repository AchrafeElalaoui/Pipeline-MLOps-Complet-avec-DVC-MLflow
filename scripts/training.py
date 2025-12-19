import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train demand forecasting model (train/val).")
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--val-csv", default="data/processed/val.csv")
    parser.add_argument("--target", default="Weekly_Sales")
    parser.add_argument("--model-out", default="models/model.pkl")
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_pipeline(cat_cols: List[str], num_cols: List[str], model) -> Pipeline:
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, cat_cols),
            ("num", num_pipeline, num_cols),
        ]
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "rows": int(len(y_true)),
    }


def split_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in df columns: {list(df.columns)}")
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_csv)
    val_path = Path(args.val_csv)
    model_out = Path(args.model_out)
    metrics_dir = Path(args.metrics_dir)
    models_dir = model_out.parent

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train, y_train = split_X_y(train_df, args.target)
    X_val, y_val = split_X_y(val_df, args.target)

    # Détection simple des colonnes catégorielles / numériques
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    models = {
        "rf": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            min_samples_split=4,
            max_features="sqrt",
            random_state=args.random_state,
            n_jobs=-1,
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=args.random_state,
        ),
    }

    metrics_payload: Dict[str, Dict[str, Dict[str, float]]] = {"train": {}, "val": {}}
    pipelines: Dict[str, Pipeline] = {}

    # Entraîner et évaluer chaque modèle
    for name, model in models.items():
        pipe = build_pipeline(cat_cols, num_cols, model)
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe

        pred_train = pipe.predict(X_train)
        pred_val = pipe.predict(X_val)

        m_train = compute_metrics(y_train, pred_train)
        m_val = compute_metrics(y_val, pred_val)

        metrics_payload["train"][name] = m_train
        metrics_payload["val"][name] = m_val

        # Sauvegarde modèle + métriques par modèle
        joblib.dump(pipe, models_dir / f"{name}_model.pkl")
        (metrics_dir / f"train_metrics_{name}.json").write_text(json.dumps(m_train, indent=2))
        (metrics_dir / f"val_metrics_{name}.json").write_text(json.dumps(m_val, indent=2))

    # Choix du meilleur modèle selon RMSE sur validation
    best_name = min(metrics_payload["val"].items(), key=lambda item: item[1]["rmse"])[0]
    best_pipe = pipelines[best_name]
    best_metrics = metrics_payload["val"][best_name]

    # Sauvegarde meilleur modèle final + résumé global
    joblib.dump(best_pipe, model_out)
    (metrics_dir / "best_model.json").write_text(json.dumps({"best_model": best_name}, indent=2))
    (metrics_dir / "best_val_metrics.json").write_text(json.dumps(best_metrics, indent=2))
    (metrics_dir / "metrics_all.json").write_text(json.dumps(metrics_payload | {"best_model": best_name}, indent=2))

    print(f"Best model (by VAL RMSE): {best_name}")
    print(f"Saved best model to: {model_out}")
    print(f"Saved metrics to: {metrics_dir}")


if __name__ == "__main__":
    main()
