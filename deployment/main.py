import os
import logging
import joblib
import numpy as np
import pandas as pd
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -------------------------
# Basic logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crop_predictor")

# -------------------------
# Paths & Globals
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_FILES = {
    "scaler": "scaler_area.joblib",
    "label_encoders": "label_encoders.joblib",
    "model": "hybrid_model.joblib",
    # optional fallback features file:
    "model_features": "model_features.joblib",
    "cols_txt": "cols_name.txt",
}
artifacts = {}

# -------------------------
# FastAPI app using lifespan
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load artifacts at startup using robust absolute paths.
    If essential artifacts are missing, raise to prevent app from starting in an inconsistent state.
    """
    try:
        # 1) Scaler
        scaler_path = os.path.join(BASE_DIR, ARTIFACT_FILES["scaler"])
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing artifact: {scaler_path}")
        artifacts["scaler"] = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")

        # 2) Label encoders
        le_path = os.path.join(BASE_DIR, ARTIFACT_FILES["label_encoders"])
        if not os.path.exists(le_path):
            raise FileNotFoundError(f"Missing artifact: {le_path}")
        label_encoders = joblib.load(le_path)
        artifacts["label_encoders"] = label_encoders
        logger.info(f"Loaded label encoders from {le_path}")

        # Resolve target encoder 'crop' name robustly
        if "crop_name" in label_encoders:
            artifacts["le_crop"] = label_encoders["crop_name"]
        elif "crop" in label_encoders:
            artifacts["le_crop"] = label_encoders["crop"]
        else:
            # leave None if not present
            artifacts["le_crop"] = label_encoders.get("target") or label_encoders.get("le_crop", None)

        # 3) Model
        model_path = os.path.join(BASE_DIR, ARTIFACT_FILES["model"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing artifact: {model_path}")
        model = joblib.load(model_path)
        artifacts["model"] = model
        logger.info(f"Loaded model from {model_path}")

        # 4) Model features: try multiple fallbacks
        # Prefer model.feature_names_in_ (scikit-learn >= 1.0)
        model_features = None
        if hasattr(model, "feature_names_in_"):
            model_features = list(getattr(model, "feature_names_in_"))
            logger.info(f"Extracted {len(model_features)} features from model.feature_names_in_.")
        else:
            # try joblib fallback
            mf_path = os.path.join(BASE_DIR, ARTIFACT_FILES["model_features"])
            if os.path.exists(mf_path):
                model_features = joblib.load(mf_path)
                logger.info(f"Loaded model features from {mf_path}.")
            else:
                # try cols_name.txt fallback
                txt_path = os.path.join(BASE_DIR, ARTIFACT_FILES["cols_txt"])
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                        model_features = lines
                        logger.info(f"Loaded model features from {txt_path}.")
        if not model_features:
            raise ValueError(
                "Could not determine model feature names. "
                "Ensure the model has 'feature_names_in_' or provide model_features/joblib or cols_name.txt."
            )
        artifacts["model_features"] = list(model_features)

        logger.info("All artifacts loaded successfully.")
        yield

    except Exception as exc:
        logger.exception("Failed to load startup artifacts.")
        # Re-raise so the app startup fails loudly (so you don't have a running server missing models)
        raise exc
    finally:
        # any cleanup if necessary
        pass


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# -------------------------
# Pydantic schema
# -------------------------
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


# -------------------------
# Preprocessing helpers (kept from your version, made robust)
# -------------------------
def preprocess_pipeline(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    df = df.copy()

    growth_cols = [c for c in df.columns if c.startswith("growth_")]
    harvest_cols = [c for c in df.columns if c.startswith("harvest_")]

    if growth_cols + harvest_cols:
        df[growth_cols + harvest_cols] = df[growth_cols + harvest_cols].astype("int8")

    cat_onehots = []

    if "season_enc" in df.columns:
        season_onehot = pd.get_dummies(df["season_enc"], prefix="season_enc")
        cat_onehots.append(season_onehot)

    if "district_enc" in df.columns:
        district_onehot = pd.get_dummies(df["district_enc"], prefix="district_enc")
        cat_onehots.append(district_onehot)

    if "transplant_month" in df.columns:
        trans_onehot = pd.get_dummies(df["transplant_month"], prefix="transplant_month")
        cat_onehots.append(trans_onehot)

    cat_df = pd.concat(cat_onehots, axis=1) if cat_onehots else pd.DataFrame(index=df.index)

    base_cols = growth_cols + harvest_cols + ["climate_risk_score", "area_log"]
    existing_base_cols = [c for c in base_cols if c in df.columns]

    combined_df = pd.concat([df[existing_base_cols], cat_df], axis=1)

    final_df = pd.DataFrame(0, index=np.arange(len(combined_df)), columns=target_cols)

    shared_cols = [c for c in final_df.columns if c in combined_df.columns]
    if shared_cols:
        final_df.loc[:, shared_cols] = combined_df.loc[:, shared_cols].values

    return final_df


def data_input_pipeline(user_input, scaler, label_encoders, model_features):
    features = {}

    # Normalize district/season names - keep consistent with how encoders were trained
    district = user_input["district"].strip()
    season = user_input["season"].strip()

    # robust encoder retrieval
    le_district = label_encoders.get("district") or label_encoders.get("district_encoder")
    le_season = label_encoders.get("season") or label_encoders.get("season_encoder")

    if le_district is None:
        raise ValueError("District label encoder not found in label_encoders artifact.")
    if le_season is None:
        raise ValueError("Season label encoder not found in label_encoders artifact.")

    if district not in list(le_district.classes_):
        raise ValueError(f"Unknown district: {district}")
    if season not in list(le_season.classes_):
        raise ValueError(f"Unknown season: {season}")

    features["district_enc"] = int(le_district.transform([district])[0])
    features["season_enc"] = int(le_season.transform([season])[0])

    month_abbr_map = {
        "jan": "january",
        "feb": "february",
        "mar": "march",
        "apr": "april",
        "may": "may",
        "jun": "june",
        "jul": "july",
        "aug": "august",
        "sep": "september",
        "oct": "october",
        "nov": "november",
        "dec": "december",
    }

    month_to_idx = {
        "january": 0,
        "february": 1,
        "march": 2,
        "april": 3,
        "may": 4,
        "june": 5,
        "july": 6,
        "august": 7,
        "september": 8,
        "october": 9,
        "november": 10,
        "december": 11,
    }

    transplant = user_input["transplant_month"].strip().lower()
    transplant = month_abbr_map.get(transplant, transplant)
    features["transplant_month"] = month_to_idx.get(transplant, -1)
    if features["transplant_month"] == -1:
        raise ValueError(f"Unknown transplant month: {user_input['transplant_month']}")

    def encode_span(span):
        vec = [0] * 12
        parts = re.split(r"\s*to\s*", span.strip().lower())
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 2:
            return vec
        parts[0] = month_abbr_map.get(parts[0], parts[0])
        parts[1] = month_abbr_map.get(parts[1], parts[1])
        if parts[0] in month_to_idx and parts[1] in month_to_idx:
            start, end = month_to_idx[parts[0]], month_to_idx[parts[1]]
            if start <= end:
                for i in range(start, end + 1):
                    vec[i] = 1
            else:
                for i in range(start, 12):
                    vec[i] = 1
                for i in range(0, end + 1):
                    vec[i] = 1
        return vec

    growth_vec = encode_span(user_input["growth_period"])
    months_short = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    for i, m in enumerate(months_short):
        features[f"growth_{m}"] = int(growth_vec[i])

    harvest_vec = encode_span(user_input["harvest_period"])
    for i, m in enumerate(months_short):
        features[f"harvest_{m}"] = int(harvest_vec[i])

    features["area"] = float(user_input.get("area", 0.0))

    # Climate normalization constants (kept from your code)
    mu_T, sigma_T = 23.960806606031593, 8.928376579084118
    mu_H, sigma_H = 72.11165629487793, 15.164593967354199

    max_temp = float(user_input.get("max_temp", 0.0))
    min_temp = float(user_input.get("min_temp", 0.0))
    max_rh = float(user_input.get("max_relative_humidity", 0.0))
    min_rh = float(user_input.get("min_relative_humidity", 0.0))

    features["climate_risk_score"] = (
        ((max_temp - mu_T) / sigma_T)
        + ((min_temp - mu_T) / sigma_T)
        + ((max_rh - mu_H) / sigma_H)
        + ((min_rh - mu_H) / sigma_H)
    )

    processed_df = pd.DataFrame([features])

    processed_df["area_log"] = np.log1p(processed_df["area"])
    processed_df.drop(columns=["area"], inplace=True)

    expected_cols = [
        "growth_jan", "growth_feb", "growth_mar", "growth_apr", "growth_may",
        "growth_jun", "growth_jul", "growth_aug", "growth_sep", "growth_oct",
        "growth_nov", "growth_dec", "harvest_jan", "harvest_feb", "harvest_mar",
        "harvest_apr", "harvest_may", "harvest_jun", "harvest_jul",
        "harvest_aug", "harvest_sep", "harvest_oct", "harvest_nov",
        "harvest_dec", "season_enc", "district_enc", "transplant_month",
        "climate_risk_score", "area_log"
    ]

    processed_df = processed_df.reindex(columns=expected_cols, fill_value=0)

    cols_to_scale_in_pipeline = ["area_log"]
    cols_existing = [col for col in cols_to_scale_in_pipeline if col in processed_df.columns]
    if cols_existing:
        processed_df[cols_existing] = artifacts["scaler"].transform(processed_df[cols_existing])

    processed_df = preprocess_pipeline(processed_df, model_features)
    return processed_df


# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict(input_data: CropInput):
    if not artifacts.get("model"):
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")

    try:
        user_input_dict = input_data.model_dump()
        processed_data = data_input_pipeline(
            user_input_dict,
            artifacts["scaler"],
            artifacts["label_encoders"],
            artifacts["model_features"],
        )

        model = artifacts["model"]

        # Prefer predict_proba if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(processed_data)
            # probs shape = (n_samples, n_classes)
            top_indices = probs[0].argsort()[-3:][::-1]
            # Determine the encoded class labels for these indices
            if hasattr(model, "classes_"):
                encoded_labels = list(model.classes_[top_indices])
            else:
                encoded_labels = [int(i) for i in top_indices]

            # Map to human readable crop names if encoder exists
            if artifacts.get("le_crop") is not None:
                top_crops = artifacts["le_crop"].inverse_transform(encoded_labels)
            else:
                top_crops = [str(l) for l in encoded_labels]

            top_probs = probs[0][top_indices]

            results = [{"crop": crop, "probability": float(prob)} for crop, prob in zip(top_crops, top_probs)]
            return {"status": "success", "predictions": results}

        else:
            # Fallback: model does not implement predict_proba
            pred = model.predict(processed_data)
            # If the model returns encoded labels, try to decode
            if artifacts.get("le_crop") is not None:
                try:
                    pred_name = artifacts["le_crop"].inverse_transform(pred)
                except Exception:
                    pred_name = [str(p) for p in pred]
            else:
                pred_name = [str(p) for p in pred]


            return {"status": "success", "predictions": [{"crop": pred_name[0], "probability": 1.0}]}

    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------
# Run locally
# -------------------------
# Serve static files from the same directory (so frontend and backend are same origin)
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
