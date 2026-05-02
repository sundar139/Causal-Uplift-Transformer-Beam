from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import AliasChoices, BaseModel, Field

from causal_uplift.baselines import TwoModelUpliftBaseline
from causal_uplift.config import AppConfig

MODEL: TwoModelUpliftBaseline | None = None
MODEL_LOADED = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global MODEL, MODEL_LOADED
    config = AppConfig.from_env()
    model_path = Path(config.model_dir) / "smoke_uplift_baseline.joblib"
    if model_path.exists():
        MODEL = joblib.load(model_path)
        MODEL_LOADED = True
    else:
        MODEL = None
        MODEL_LOADED = False
    yield


app = FastAPI(title="Causal Uplift Smoke API", version="0.1.0", lifespan=lifespan)


class FeatureRow(BaseModel):
    features: list[float] = Field(default_factory=list)


class PredictRequest(BaseModel):
    rows: list[FeatureRow] = Field(
        default_factory=list,
        validation_alias=AliasChoices("rows", "records"),
    )


class PredictItem(BaseModel):
    treatment_probability: float
    control_probability: float
    uplift: float
    recommend_treatment: bool


class PredictResponse(BaseModel):
    model_loaded: bool
    predictions: list[PredictItem]


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": MODEL_LOADED}


@app.get("/version")
def version() -> dict[str, str]:
    return {"service": "causal-uplift-transformer-beam", "version": "0.1.0"}


@app.post("/predict_uplift", response_model=PredictResponse)
def predict_uplift(payload: PredictRequest) -> PredictResponse:
    if not MODEL_LOADED or MODEL is None:
        placeholders = [
            PredictItem(
                treatment_probability=0.5,
                control_probability=0.5,
                uplift=0.0,
                recommend_treatment=False,
            )
            for _ in payload.rows
        ]
        return PredictResponse(model_loaded=False, predictions=placeholders)

    expected_columns = MODEL.feature_columns
    expected_width = len(expected_columns)

    processed_rows: list[list[float]] = []
    for row in payload.rows:
        values = list(row.features)
        if len(values) < expected_width:
            values.extend([0.0] * (expected_width - len(values)))
        if len(values) > expected_width:
            values = values[:expected_width]
        processed_rows.append(values)

    frame = pd.DataFrame(processed_rows, columns=expected_columns)
    prediction = MODEL.predict_uplift(frame)

    items = [
        PredictItem(
            treatment_probability=float(tp),
            control_probability=float(cp),
            uplift=float(u),
            recommend_treatment=bool(u > 0.0),
        )
        for tp, cp, u in zip(
            prediction.treatment_proba,
            prediction.control_proba,
            prediction.uplift,
            strict=True,
        )
    ]

    return PredictResponse(model_loaded=True, predictions=items)
