# Pipeline MLOps complet avec DVC, MLflow, FastAPI et Streamlit

This repository delivers an end-to-end MLOps workflow for retail demand
forecasting. It combines data versioning and reproducible pipelines (DVC),
experiment tracking and model registry (MLflow), a production-grade API
(FastAPI), a user-facing dashboard (Streamlit), and containerization (Docker).

The goal is to provide a professional, reproducible pipeline from raw data to
deployed predictions, with clear traceability of data, models, metrics, and
artifacts.

## Key capabilities

- Data versioning and pipeline reproducibility with DVC.
- Training and evaluation of multiple models (Random Forest, Gradient Boosting).
- Centralized metrics and plots for model comparison.
- MLflow tracking for parameters, metrics, and artifacts.
- MLflow Model Registry registration for all candidate models.
- FastAPI serving for real-time predictions.
- Streamlit UI for single and batch predictions.
- Docker and docker-compose for reproducible deployment.
- CML GitHub Actions workflow for automated reporting in CI.

## Architecture overview

Data flows through a reproducible pipeline and then into serving components:

```
data/*.csv
  -> DVC pipeline (preprocess -> train -> eval -> log_mlflow)
  -> metrics/, models/, plots/
  -> MLflow tracking + Model Registry
  -> FastAPI /predict
  -> Streamlit UI
```

## Repository structure

```
app/                     # FastAPI service
scripts/                 # preprocessing, training, evaluation, MLflow logging
data/                    # raw data tracked by DVC
data/processed/          # processed train/val splits (DVC outputs)
models/                  # trained models (DVC outputs)
metrics/                 # metrics, predictions, plots (DVC outputs)
mlruns/                  # MLflow local tracking store
.dvc/                    # DVC configuration
.github/workflows/       # CML workflow for CI reporting
Dockerfile               # FastAPI container
Dockerfile.streamlit     # Streamlit container
docker-compose.yml       # API + UI orchestration
requirements.txt         # API + pipeline dependencies
requirements-streamlit.txt # Streamlit dependencies
```

## Data and features

- Sources: `data/train.csv`, `data/features.csv`, `data/stores.csv`
- Target: `Weekly_Sales`
- Features used by the model:
  `Store`, `Dept`, `IsHoliday`, `Temperature`, `Fuel_Price`, `MarkDown1-5`,
  `CPI`, `Unemployment`, `Type`, `Size`, `Year`, `Month`, `Week`

The preprocessing step merges data sources, handles missing values, extracts
time-based features, and produces train/validation splits in `data/processed/`.

## DVC pipeline

The pipeline is declared in `dvc.yaml` with four stages:

- `preprocess`: builds `data/processed/` from raw CSVs.
- `train`: trains RF and GBR models, selects the best model, saves artifacts.
- `eval`: evaluates models, saves metrics, predictions, and plots.
- `log_mlflow`: logs the latest DVC outputs to MLflow.

Key commands:

```
dvc repro
dvc status
dvc metrics show
```

## Training and evaluation

Models implemented:

- `RandomForestRegressor` (RF)
- `GradientBoostingRegressor` (GBR)

Training uses an sklearn `Pipeline` with:

- Categorical features: `SimpleImputer` + `OneHotEncoder`
- Numeric features: `SimpleImputer`

Outputs:

- Models:
  - `models/rf_model.pkl`
  - `models/gbr_model.pkl`
  - `models/model.pkl` (best model by validation RMSE)
- Metrics:
  - `metrics/metrics_all.json` (train + val metrics for all models)
  - `metrics/best_val_metrics.json`
  - `metrics/train_metrics_*.json`, `metrics/val_metrics_*.json`
- Predictions:
  - `metrics/val_predictions_*.csv`
- Plots:
  - `metrics/plots/` (comparison and residual plots)

## MLflow tracking and Model Registry

MLflow logging is handled by `scripts/mlflow_log.py`:

- Logs train and validation metrics for each model.
- Logs artifacts (metrics JSON, predictions CSV, plots, and model files).
- Logs MLflow model format and registers each model in the registry.

Default tracking store: `mlruns/` (local file store).

Example:

```
python scripts/mlflow_log.py --tracking-uri file:./mlruns --experiment retail-demand
mlflow ui --backend-store-uri file:./mlruns
```

Registry names use the prefix `retail-demand`, e.g.:

- `retail-demand-rf`
- `retail-demand-gbr`

Model promotion (Staging/Production) is done from the MLflow UI.

## Serving with FastAPI

FastAPI loads models from `models/` and exposes:

- `GET /health` to check available models
- `GET /models` to list models
- `POST /predict` for predictions

Example payload:

```json
{
  "model_name": "rf",
  "records": [
    {
      "Store": 1,
      "Dept": 1,
      "IsHoliday": 0,
      "Temperature": 42.31,
      "Fuel_Price": 2.572,
      "MarkDown1": 0,
      "MarkDown2": 0,
      "MarkDown3": 0,
      "MarkDown4": 0,
      "MarkDown5": 0,
      "CPI": 211.0963582,
      "Unemployment": 8.106,
      "Type": "A",
      "Size": 151315,
      "Year": 2010,
      "Month": 2,
      "Week": 5
    }
  ]
}
```

## Streamlit UI

`streamlit_app.py` provides:

- Single prediction form
- Batch CSV upload
- Model selection from `/models`

The UI communicates with FastAPI through:

- `FASTAPI_URL`
- `FASTAPI_TIMEOUT`
- `FASTAPI_RETRIES`

## Docker and docker-compose

Run everything locally:

```
docker compose up --build
```

Services:

- `api`: FastAPI on `http://localhost:8000`
- `ui`: Streamlit on `http://localhost:8501`

## DVC remote (Google Drive)

The DVC remote is configured in `.dvc/config`. For local credentials, use
`.dvc/config.local` (ignored by Git). In CI, a service account is injected to
enable `dvc pull` and `dvc push`.

## CML workflow (CI)

`.github/workflows/cml.yaml`:

- Pulls data with DVC
- Reproduces the pipeline
- Logs MLflow artifacts
- Publishes metrics and plots to the GitHub summary/PR

Required GitHub secrets:

- `CML_TOKEN` (PAT with repo scope)
- `DVC_GDRIVE_SERVICE_ACCOUNT_JSON` (service account JSON as a single-line string)

The Google Drive folder must be shared with the service account email.

## Quickstart (local)

```
pip install -r requirements.txt
dvc pull
dvc repro
python scripts/mlflow_log.py --tracking-uri file:./mlruns --experiment retail-demand
mlflow ui --backend-store-uri file:./mlruns
```

API only:

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Streamlit only:

```
streamlit run streamlit_app.py
```

## Configuration reference

- `MODELS_DIR`: directory for model files (default `models`)
- `MODEL_PATH`: optional absolute path to a single model file
- `DEFAULT_MODEL_NAME`: default model served by FastAPI
- `FASTAPI_URL`: FastAPI URL used by Streamlit
- `FASTAPI_TIMEOUT`: request timeout (seconds)
- `FASTAPI_RETRIES`: retry count on timeout

## Notes

- Data and large artifacts are tracked by DVC (not Git).
- The MLflow Model Registry requires a registry-capable tracking backend if you
  want to use advanced features beyond local file tracking.
