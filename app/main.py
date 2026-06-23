import os
import logging
import numpy as np
import pandas as pd
import re
import joblib
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crop_predictor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", os.path.join(PROJECT_ROOT, "artifacts"))
MODEL_NAME = os.environ.get("CROP_MODEL_NAME", "VotingClassifier_Ensemble")
FEATURE_COLUMNS_PATH = os.path.join(ARTIFACT_DIR, "preprocessors", "feature_columns.joblib")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "models", f"{MODEL_NAME}.joblib")
ENCODERS_PATH = os.path.join(ARTIFACT_DIR, "preprocessors", "label_encoders.joblib")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "preprocessors", "standard_scaler.joblib")
CLIMATE_PATH = os.path.join(ARTIFACT_DIR, "preprocessors", "climate_constants.joblib")

MONTHS_SHORT = ["jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_FULL = ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december']
MONTH_TO_IDX = {m: i for i, m in enumerate(MONTH_FULL)}
MONTH_ABBR_MAP = {
    'jan': 'january', 'feb': 'february', 'mar': 'march', 'apr': 'april', 'may': 'may',
    'jun': 'june', 'jul': 'july', 'aug': 'august', 'sep': 'september',
    'oct': 'october', 'nov': 'november', 'dec': 'december'
}

artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        artifacts["model"] = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}")

        encoders = joblib.load(ENCODERS_PATH)
        artifacts["le_crop"] = encoders["le_crop"]
        artifacts["le_season"] = encoders["le_season"]
        artifacts["le_district"] = encoders["le_district"]
        logger.info(f"Loaded encoders from {ENCODERS_PATH}")

        artifacts["scaler"] = joblib.load(SCALER_PATH)
        logger.info(f"Loaded scaler from {SCALER_PATH}")

        artifacts["climate_constants"] = joblib.load(CLIMATE_PATH)
        logger.info(f"Loaded climate constants from {CLIMATE_PATH}")

        artifacts["feature_columns"] = joblib.load(FEATURE_COLUMNS_PATH)
        logger.info(f"Loaded {len(artifacts['feature_columns'])} feature columns")

        yield
    except Exception as exc:
        logger.exception("Failed to load startup artifacts.")
        raise exc


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CropInput(BaseModel):
    district: str
    season: str
    area: float
    transplant_month: str
    growth_period: str
    harvest_period: str
    min_temp: float
    max_temp: float
    min_relative_humidity: float
    max_relative_humidity: float


def _clean_month_string(s):
    s = str(s).strip().lower()
    for abbr, full in MONTH_ABBR_MAP.items():
        s = re.sub(fr'\b{abbr}\b', full, s)
    return s


def _clean_category(s):
    return str(s).strip().lower()


def _encode_span(span):
    vec = [0] * 12
    span = _clean_month_string(span)
    parts = re.split(r'\s+to\s+', span)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        return vec
    if parts[0] in MONTH_TO_IDX and parts[1] in MONTH_TO_IDX:
        start, end = MONTH_TO_IDX[parts[0]], MONTH_TO_IDX[parts[1]]
        if start <= end:
            for i in range(start, end + 1):
                vec[i] = 1
        else:
            for i in range(start, 12):
                vec[i] = 1
            for i in range(0, end + 1):
                vec[i] = 1
    return vec


def data_input_pipeline(user_input, scaler, le_district, le_season, climate_constants, feature_columns):
    features = {}

    district = _clean_category(user_input["district"])
    season = _clean_category(user_input["season"])

    if district not in list(le_district.classes_):
        raise ValueError(f"Unknown district: {district}")
    if season not in list(le_season.classes_):
        raise ValueError(f"Unknown season: {season}")

    features["district_enc"] = int(le_district.transform([district])[0])
    features["season_enc"] = int(le_season.transform([season])[0])

    transplant = _clean_month_string(user_input["transplant_month"])
    if transplant not in MONTH_TO_IDX:
        raise ValueError(f"Unknown transplant month: {user_input['transplant_month']}")
    features["transplant_month"] = MONTH_TO_IDX[transplant]

    growth_vec = _encode_span(user_input["growth_period"])
    for i, m in enumerate(MONTHS_SHORT):
        features[f"growth_{m}"] = int(growth_vec[i])

    harvest_vec = _encode_span(user_input["harvest_period"])
    for i, m in enumerate(MONTHS_SHORT):
        features[f"harvest_{m}"] = int(harvest_vec[i])

    area = float(user_input.get("area", 0.0))
    max_temp = float(user_input.get("max_temp", 0.0))
    min_temp = float(user_input.get("min_temp", 0.0))
    max_rh = float(user_input.get("max_relative_humidity", 0.0))
    min_rh = float(user_input.get("min_relative_humidity", 0.0))

    cc = climate_constants
    features["climate_risk_score"] = (
        ((max_temp - cc["mu_T"]) / cc["sigma_T"]) +
        ((min_temp - cc["mu_T"]) / cc["sigma_T"]) +
        ((max_rh - cc["mu_H"]) / cc["sigma_H"]) +
        ((min_rh - cc["mu_H"]) / cc["sigma_H"])
    )

    features["area_log"] = np.log1p(area)

    processed_df = pd.DataFrame([features])
    processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)

    if "area_log" in processed_df.columns:
        processed_df["area_log"] = (
            processed_df["area_log"] - float(scaler.mean_[0])
        ) / float(scaler.scale_[0])

    return processed_df


def _top_crop_predictions(model, processed_data, le_crop, limit=5):
    model_input = processed_data.to_numpy()

    if not hasattr(model, "predict_proba"):
        pred = int(model.predict(model_input)[0])
        return [{
            "rank": 1,
            "crop": le_crop.inverse_transform([pred])[0],
            "confidence": None,
        }]

    probabilities = model.predict_proba(model_input)[0]
    model_classes = getattr(model, "classes_", np.arange(len(probabilities)))
    top_indexes = np.argsort(probabilities)[::-1][:limit]

    predictions = []
    for rank, probability_index in enumerate(top_indexes, start=1):
        crop_class = int(model_classes[probability_index])
        predictions.append({
            "rank": rank,
            "crop": le_crop.inverse_transform([crop_class])[0],
            "confidence": round(float(probabilities[probability_index]), 6),
        })
    return predictions


@app.post("/predict")
async def predict(input_data: CropInput):
    if not artifacts.get("model"):
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")

    try:
        user_input_dict = input_data.model_dump()
        processed_data = data_input_pipeline(
            user_input_dict,
            artifacts["scaler"],
            artifacts["le_district"],
            artifacts["le_season"],
            artifacts["climate_constants"],
            artifacts["feature_columns"],
        )

        model = artifacts["model"]
        le_crop = artifacts["le_crop"]
        predictions = _top_crop_predictions(model, processed_data, le_crop)
        crop_name = predictions[0]["crop"]

        return {
            "status": "success",
            "prediction": crop_name,
            "predictions": predictions,
            "model": MODEL_NAME,
        }

    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "feature_count": len(artifacts.get("feature_columns", [])),
    }


app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
