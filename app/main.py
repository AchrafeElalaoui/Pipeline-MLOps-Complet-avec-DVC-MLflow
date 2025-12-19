import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

FEATURE_COLUMNS = [
    "Store",
    "Dept",
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
    "Type",
    "Size",
    "Year",
    "Month",
    "Week",
]

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODEL_PATH_ENV = os.getenv("MODEL_PATH")
MODEL_PATH = Path(MODEL_PATH_ENV) if MODEL_PATH_ENV else None
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "model")


class DemandRecord(BaseModel):
    Store: int
    Dept: int
    IsHoliday: Union[bool, int] = 0
    Temperature: Optional[float] = None
    Fuel_Price: Optional[float] = None
    MarkDown1: Optional[float] = None
    MarkDown2: Optional[float] = None
    MarkDown3: Optional[float] = None
    MarkDown4: Optional[float] = None
    MarkDown5: Optional[float] = None
    CPI: Optional[float] = None
    Unemployment: Optional[float] = None
    Type: Optional[str] = None
    Size: Optional[float] = None
    Year: int
    Month: int
    Week: int


class PredictionRequest(BaseModel):
    records: List[DemandRecord]
    model_name: Optional[str] = None


class PredictionResponse(BaseModel):
    predictions: List[float]
    model_name: str


app = FastAPI(title="Retail Demand Forecast API")


def list_models() -> Dict[str, Path]:
    models: Dict[str, Path] = {}
    if MODEL_PATH and MODEL_PATH.exists():
        models["model"] = MODEL_PATH

    if not MODELS_DIR.exists():
        return dict(sorted(models.items()))

    default_path = MODELS_DIR / "model.pkl"
    if default_path.exists() and "model" not in models:
        models["model"] = default_path

    for path in MODELS_DIR.glob("*_model.pkl"):
        name = path.stem.replace("_model", "")
        models[name] = path

    return dict(sorted(models.items()))


def resolve_model_path(model_name: Optional[str]) -> Path:
    models = list_models()
    if not model_name and MODEL_PATH and MODEL_PATH.exists():
        return MODEL_PATH

    name = model_name or DEFAULT_MODEL_NAME

    if name in models:
        return models[name]

    if name and name.endswith(".pkl"):
        candidate = MODELS_DIR / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Model '{name}' not found. Available: {', '.join(models.keys())}"
    )


@lru_cache(maxsize=8)
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


def records_to_frame(records: List[DemandRecord]) -> pd.DataFrame:
    payload = [record.dict() for record in records]
    frame = pd.DataFrame(payload)

    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = None

    frame = frame[FEATURE_COLUMNS]
    frame["IsHoliday"] = frame["IsHoliday"].fillna(0).astype(int)
    return frame


@app.get("/health")
def health() -> dict:
    models = list_models()
    return {
        "status": "ok",
        "models_available": list(models.keys()),
        "default_model": DEFAULT_MODEL_NAME,
        "models_dir": str(MODELS_DIR),
    }


@app.get("/models")
def models() -> dict:
    models_map = list_models()
    return {
        "models": list(models_map.keys()),
        "default_model": DEFAULT_MODEL_NAME,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model_path = resolve_model_path(request.model_name)
        model = load_model(str(model_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    frame = records_to_frame(request.records)
    preds = model.predict(frame)
    return PredictionResponse(
        predictions=[float(value) for value in preds],
        model_name=request.model_name or DEFAULT_MODEL_NAME,
    )
