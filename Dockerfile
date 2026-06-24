FROM python:3.10-slim

RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

COPY artifacts/preprocessors/feature_columns.joblib  artifacts/preprocessors/
COPY artifacts/preprocessors/label_encoders.joblib   artifacts/preprocessors/
COPY artifacts/preprocessors/climate_constants.joblib artifacts/preprocessors/
COPY artifacts/preprocessors/standard_scaler.joblib  artifacts/preprocessors/
COPY artifacts/models/ResidualCatBoost_RF.joblib     artifacts/models/

COPY app/ .

# Create minimal crop_recommendation package with only model_configs.py.
# Empty __init__.py files prevent pulling in training-only deps
RUN mkdir -p /app/crop_recommendation/pipeline
COPY src/crop_recommendation/pipeline/model_configs.py /app/crop_recommendation/pipeline/model_configs.py
RUN echo "" > /app/crop_recommendation/__init__.py && \
    echo "" > /app/crop_recommendation/pipeline/__init__.py

ENV ARTIFACT_DIR=/app/artifacts
ENV CROP_MODEL_NAME=ResidualCatBoost_RF
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
