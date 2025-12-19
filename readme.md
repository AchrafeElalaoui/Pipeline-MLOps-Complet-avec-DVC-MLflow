# Pipeline MLOps: DVC + MLflow Integration

This repo now includes DVC pipeline tracking plus MLflow experiment logging for
the retail demand workflow (preprocess -> train -> eval).

## What I changed (and why)

### Commands I ran

0) `git --version` and `dvc --version`
   - Why: quick checks to confirm required tools are installed.

1) `git init`
   - Why: DVC works best with Git versioning; this creates the local Git repo.

2) `dvc init`
   - Why: initializes DVC metadata and enables pipeline tracking.

3) `dvc add data/train.csv data/features.csv data/stores.csv`
   - Why: tracks raw datasets with DVC instead of Git, keeping Git lightweight.

4) `dvc stage add -n preprocess -d scripts/preprocessing.py -d data/train.csv -d data/features.csv -d data/stores.csv -o data/processed python scripts/preprocessing.py --train data/train.csv --features data/features.csv --stores data/stores.csv --output-dir data/processed`
   - Why: defines the preprocessing stage in a reproducible DVC pipeline.

5) `dvc stage add -n train -d scripts/training.py -d data/processed/train.csv -o models/model.pkl -o models/rf_model.pkl -o models/gbr_model.pkl -m metrics/train_metrics.json -m metrics/train_metrics_rf.json -m metrics/train_metrics_gbr.json -m metrics/train_metrics_all.json python scripts/training.py --train-csv data/processed/train.csv --model-out models/model.pkl --metrics-out metrics/train_metrics.json`
   - Why: defines training with all model artifacts and metrics tracked by DVC.

6) `dvc stage add -n eval -d scripts/evaluating.py -d data/processed/val.csv -d models/model.pkl -d models/rf_model.pkl -d models/gbr_model.pkl -d metrics/train_metrics.json -d metrics/train_metrics_all.json -o metrics/val_predictions.csv -o metrics/val_predictions_rf.csv -o metrics/val_predictions_gbr.csv -o metrics/plots -m metrics/val_metrics.json -m metrics/val_metrics_rf.json -m metrics/val_metrics_gbr.json python scripts/evaluating.py --model models/model.pkl --models-dir models --val-csv data/processed/val.csv --metrics-out metrics/val_metrics.json --preds-out metrics/val_predictions.csv --plots-dir metrics/plots --train-metrics metrics/train_metrics.json --train-metrics-all metrics/train_metrics_all.json`
   - Why: defines evaluation with metrics, predictions, and plots tracked by DVC.

Note: I initially tried `dvc stage add ... --cmd` but DVC expects the command as
the positional argument, so I re-ran the correct syntax above.

### Files added or modified

- `.dvc/` and `.dvcignore`
  - Why: created by `dvc init` for DVC configuration and ignore rules.
- `.git/`
  - Why: created by `git init` to enable Git versioning for DVC metadata.
- `dvc.yaml`
  - Why: holds the DVC pipeline stages for preprocess/train/eval.
- `data/train.csv.dvc`, `data/features.csv.dvc`, `data/stores.csv.dvc`
  - Why: metadata files created by `dvc add` to track raw datasets.
- `data/.gitignore`
  - Why: ignores raw data files in Git because they are tracked by DVC.
- `models/.gitignore`, `metrics/.gitignore`
  - Why: ignores generated artifacts/metrics in Git (they are tracked by DVC).
- `.gitignore`
  - Why: ignores MLflow artifacts and Python cache files in Git.
- `scripts/mlflow_log.py`
  - Why: logs DVC outputs (metrics, models, plots) into MLflow runs.

## How to run the DVC pipeline

1) Reproduce the full pipeline:
```
dvc repro
```

2) View tracked metrics:
```
dvc metrics show
```

3) Check plots and predictions:
- `metrics/plots/` contains PNGs.
- `metrics/val_predictions_*.csv` contains predictions.

## MLflow tracking (without changing training/eval scripts)

1) Install MLflow if needed:
```
pip install mlflow
```

2) Log the latest DVC outputs to MLflow:
```
python scripts/mlflow_log.py --tracking-uri file:./mlruns --experiment retail-demand
```

3) Open the MLflow UI:
```
mlflow ui --backend-store-uri file:./mlruns
```

## Why this setup

- DVC handles data and pipeline reproducibility (preprocess/train/eval).
- MLflow tracks experiments and artifacts without changing the training code.
- The pipeline and logging are decoupled, so you can re-run and compare runs
  easily while keeping Git history clean.
