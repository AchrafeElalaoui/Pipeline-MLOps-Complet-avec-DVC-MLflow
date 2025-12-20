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

7) `git add .`
   - Why: stage all new DVC/MLflow files for the first commit.

8) `git commit -m "Integrate DVC pipeline and MLflow logging"`
   - Why: record the initial integration in Git history.

9) `git branch -M main`
   - Why: align the local branch name with GitHub's default.

10) `git remote add origin https://github.com/AchrafeElalaoui/Pipeline-MLOps-Complet-avec-DVC-MLflow.git`
   - Why: connect the local repo to the GitHub remote.

11) `git push -u origin main`
   - Why: publish the initial commit to GitHub.

12) `git add readme.md`
   - Why: stage this README update that documents all actions.

13) `git commit -m "Document integration and publish steps"`
   - Why: record the documentation update in Git history.

14) `git push`
   - Why: publish the README update to GitHub.

15) `git restore --staged readme.md`
   - Why: unstage an accidental README deletion.

16) `git checkout -- readme.md`
   - Why: restore the README content from the last commit.

17) `git log --oneline -- .dvc/config`
   - Why: confirm which commits touched DVC remote config.

18) `git show d80e8c5:.dvc/config`
   - Why: verify the previous baseline content of `.dvc/config`.

19) `@'... '@ | python -`
   - Why: restore Google Drive client ID/secret to `.dvc/config.local` and remove `.env`.

20) `git revert eb05551`
   - Why: return the repo to the state before moving GDrive secrets into `.env`.

Additional read-only checks (directory listings, file previews, `git status`)
were used to verify state but did not change any files.

### Manual edits (no shell command)

- Added FastAPI app in `app/main.py` with `/predict`, `/health`, and `/models`.
- Added Streamlit UI in `streamlit_app.py` to call the FastAPI backend.
- Added Docker support (`Dockerfile`, `Dockerfile.streamlit`, `docker-compose.yml`).
- Added dependency files (`requirements.txt`, `requirements-streamlit.txt`).
- Added multi-model selection support and UI model picker.
- Added configurable request timeouts/retries for the Streamlit client.
- Restored Google Drive client ID/secret entries in `.dvc/config.local` (local-only).
- Removed `.env` that previously stored Google Drive secrets (local-only).
- Added a CML GitHub Actions workflow that runs only on `main`.
- Added metrics/plots publishing to the CML report.
- Pinned DVC/CML versions in the workflow to avoid pip resolution depth errors.
- Added `matplotlib` to `requirements.txt` for evaluation plots.

### Files added or modified

- `.dvc/` and `.dvcignore`
  - Why: created by `dvc init` for DVC configuration and ignore rules.
- `.git/`
  - Why: created by `git init` to enable Git versioning for DVC metadata.
- `dvc.yaml`
  - Why: holds the DVC pipeline stages for preprocess/train/eval.
- `dvc.lock`
  - Why: records the exact pipeline state and file hashes after the last run.
- `data/train.csv.dvc`, `data/features.csv.dvc`, `data/stores.csv.dvc`
  - Why: metadata files created by `dvc add` to track raw datasets.
- `data/.gitignore`
  - Why: ignores raw data files in Git because they are tracked by DVC.
- `models/.gitignore`, `metrics/.gitignore`
  - Why: ignores generated artifacts/metrics in Git (they are tracked by DVC).
- `.gitignore`
  - Why: ignores MLflow artifacts and Python cache files in Git.
- `metrics/best_model.json`
  - Why: snapshot of the currently selected best model name.
- `metrics/best_val_metrics.json`
  - Why: snapshot of validation metrics for the best model.
- `metrics/metrics_all.json`
  - Why: combined train/validation metrics summary across models.
- `scripts/mlflow_log.py`
  - Why: logs DVC outputs (metrics, models, plots) into MLflow runs.
- `app/main.py`
  - Why: FastAPI service that loads trained models and serves predictions.
- `app/__init__.py`
  - Why: marks the FastAPI module for reliable imports.
- `requirements.txt`
  - Why: pins API dependencies for local/Docker runs.
- `Dockerfile`
  - Why: builds the FastAPI container image.
- `streamlit_app.py`
  - Why: Streamlit UI for single and batch predictions via FastAPI.
- `requirements-streamlit.txt`
  - Why: pins Streamlit UI dependencies.
- `Dockerfile.streamlit`
  - Why: builds the Streamlit UI container image.
- `docker-compose.yml`
  - Why: runs FastAPI and Streamlit together with shared model volume.
- `.github/workflows/cml.yaml`
  - Why: runs the DVC pipeline on `main` and posts a CML report with metrics and plots.

### Local-only files (ignored by Git)

- `.dvc/config.local`
  - Why: stores Google Drive credentials locally without committing them.
- `.dvc/gdrive-creds.json`
  - Why: OAuth token cache created by DVC during Google Drive auth.

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

## FastAPI serving (Docker)

The API loads models from `models/` and expects inputs that match the
preprocessed feature columns used for training.

### Build and run with Docker

1) Build the image:
```
docker build -t retail-demand-api .
```

2) Run via Docker Compose:
```
docker compose up --build
```

The API will be available at `http://localhost:8000`.

### Multi-model prediction

You can choose which model to use by setting `model_name` in the request:

```
{"model_name":"rf","records":[...]}
```

List available models:

```
GET /models
```

Environment variables:
- `MODELS_DIR` (default: `models`)
- `DEFAULT_MODEL_NAME` (default: `model`)
- `MODEL_PATH` (optional absolute/relative path override)

### Example request

```
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"model_name\":\"rf\",\"records\":[{\"Store\":1,\"Dept\":1,\"IsHoliday\":0,\"Temperature\":42.31,\"Fuel_Price\":2.572,\"MarkDown1\":0,\"MarkDown2\":0,\"MarkDown3\":0,\"MarkDown4\":0,\"MarkDown5\":0,\"CPI\":211.0963582,\"Unemployment\":8.106,\"Type\":\"A\",\"Size\":151315,\"Year\":2010,\"Month\":2,\"Week\":5}]}"
```

## Streamlit UI (FastAPI backend)

The UI sends requests to the FastAPI `/predict` endpoint and supports single or
batch predictions via CSV upload. You can choose which model to use from the UI.

### Run with Docker Compose

```
docker compose up --build
```

- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

### Run locally (no Docker)

1) Install UI dependencies:
```
pip install -r requirements-streamlit.txt
```

2) Start Streamlit:
```
streamlit run streamlit_app.py
```

3) Set the backend URL if needed:
```
$env:FASTAPI_URL="http://localhost:8000"
```

Optional timeout settings:
```
$env:FASTAPI_TIMEOUT="120"
$env:FASTAPI_RETRIES="1"
```

### Why this setup

- Streamlit remains a thin client; all predictions stay in FastAPI.
- The UI is isolated from the API, so you can deploy them independently.
- Timeouts are configurable to handle large model loads or slow responses.

## CML (GitHub Actions)

The workflow runs **only on `main`** and posts a CML report that includes
metrics and plots from `metrics/plots/`.

### Required GitHub secrets

- `CML_TOKEN` (PAT with `repo` scope for posting comments)
- `DVC_GDRIVE_SERVICE_ACCOUNT_JSON` (service account JSON as a single-line string)

### What it does

1) Pulls data from the DVC Google Drive remote.
2) Runs `dvc repro` to generate metrics and plots.
3) Publishes plots and metrics into a CML report.

If comment posting fails, the report is still written to the GitHub Actions
job summary.

Note: the workflow uses `dvc pull --force` to overwrite any tracked outputs
left in the workspace from previous runs.

### Dependency notes

- The workflow pins `dvc==3.64.0` and `dvc-gdrive==3.0.1` to avoid resolver depth issues.
- CML is installed via the GitHub Action `iterative/setup-cml@v2`.
- `matplotlib` is listed in `requirements.txt` to support evaluation plots.
- The service account JSON is written to `.dvc/gdrive-sa.json`, and the config uses a relative path (`gdrive-sa.json`).
