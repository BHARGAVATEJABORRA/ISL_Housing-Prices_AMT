"""FastAPI service for the ISL housing model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from isl_housing.utils import normalise_columns

app = FastAPI(title="ISL Housing Predictor")


class PredictionRequest(BaseModel):
    rows: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    predictions: List[float]
    lower: Optional[List[float]] = None
    upper: Optional[List[float]] = None


def _load_artifacts() -> tuple[Any, Dict[str, Any]]:
    artifact_dir = Path("models")
    model_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.yaml"

    if not model_path.exists():
        raise FileNotFoundError("Model artifacts not found")

    pipeline = joblib.load(model_path)
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = yaml.safe_load(fh) or {}
    return pipeline, metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        pipeline, metadata = _load_artifacts()
    except FileNotFoundError as exc:  # pragma: no cover - environment dependent
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    df = normalise_columns(pd.DataFrame(request.rows))
    preds = pipeline.predict(df)

    if metadata.get("log_target", True):
        preds = np.expm1(preds)

    preds = preds.astype(float)
    lower = upper = None
    conformal_q = metadata.get("conformal_quantile")
    if conformal_q is not None:
        lower = np.clip(preds - conformal_q, a_min=0, a_max=None).tolist()
        upper = (preds + conformal_q).tolist()

    return PredictionResponse(predictions=preds.tolist(), lower=lower, upper=upper)


@app.get("/health")
def health() -> Dict[str, str]:  # pragma: no cover - trivial endpoint
    return {"status": "ok"}
