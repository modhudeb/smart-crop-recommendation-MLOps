# Smart Crop Recommendation MLOps

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/DVC-Reproducible%20Pipeline-945DD6?style=flat&logo=dvc&logoColor=white)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving%20Layer-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-oriented machine learning project for crop recommendation in Bangladesh. The repository combines a reproducible DVC pipeline, MLflow/DAGsHub experiment tracking, exploratory analysis, model artifacts, and a FastAPI web application for serving crop predictions.

<p align="center">
  <img src="images/ui-homepage.png" alt="Smart Crop Recommendation web interface" width="86%">
</p>

## Overview

Smart Crop Recommendation MLOps predicts suitable crops from agronomic, seasonal, regional, and climate-related inputs. The project is structured to support both experimentation and reproducible delivery:

- A staged DVC pipeline for ingestion, preprocessing, feature generation, splitting, training, and evaluation.
- Experiment tracking through MLflow and DAGsHub.
- A FastAPI prediction service with a static browser UI.
- Report-ready metrics, diagnostic figures, and notebook outputs.
- A clean project layout separating production code, experiments, artifacts, data, reports, and notebooks.

## Results Snapshot

The current evaluation artifact reports strong classification performance on the repository's test split:

| Metric | Score |
|---|---:|
| Accuracy | 0.9631 |
| Weighted Precision | 0.9678 |
| Weighted Recall | 0.9631 |
| Weighted F1 Score | 0.9623 |

<p align="center">
  <img src="images/ui-predicted-crops.png" alt="Top crop recommendation results" width="82%">
</p>

## Visual Analysis

The repository includes analysis figures for model behavior, statistical comparison, and crop distribution. These are useful for project reports, model review, and stakeholder communication.

| Crop Distribution | Model Diagnostics |
|---|---|
| <img src="images/Top_15_Crop_Composition_per_Transplant_Month_by_production.png" alt="Top crop composition by transplant month" width="100%"> | <img src="images/ecum-model-diag.png" alt="Model diagnostics" width="100%"> |

| Feature Impact | Statistical Confidence |
|---|---|
| <img src="images/SHAP_Beeswarm_Feature_Impact_Predictions_Alternative.png" alt="SHAP feature impact beeswarm" width="100%"> | <img src="images/Bootstrap_95_Confidence_Intervals.png" alt="Bootstrap confidence intervals" width="100%"> |

## Project Structure

```text
.
├── app/                                  # FastAPI service, static UI, app artifacts
├── artifacts/
│   ├── encoders/                         # Label encoders and reusable encoding artifacts
│   ├── models/                           # Trained model artifacts
│   └── preprocessors/                    # Scalers and preprocessing constants
├── data/
│   ├── external/                         # Original source dataset
│   ├── raw/                              # Ingestion output
│   ├── processed/                        # Cleaned and encoded data
│   ├── features/                         # Feature-engineered data
│   └── splits/                           # Train/test splits
├── experiments/pipeline_v2/              # Experimental second-generation pipeline
├── images/                               # README-ready visual assets
├── notebooks/                            # Exploratory notebooks and notebook metadata
├── reports/
│   ├── figures/                          # Full report figures
│   ├── logs/                             # Pipeline logs
│   └── metrics/                          # Evaluation metrics and confusion matrix
├── src/crop_recommendation/pipeline/     # DVC production pipeline stages
├── dvc.yaml                              # DVC stage definition
├── dvc.lock                              # Locked DVC dependency/output metadata
├── params.yaml                           # Model hyperparameters
└── README.md
```

## Pipeline

The canonical workflow is defined in `dvc.yaml`.

```text
data/external/SPAS-Dataset-BD.csv
  -> data/raw/SPAS-Dataset-BD.csv
  -> data/processed/processed_data.csv
  -> data/features/featured_data.csv
  -> data/splits/train.csv + data/splits/test.csv
  -> artifacts/models/random_forest_model.joblib
  -> reports/metrics/evaluation_metrics.json
  -> reports/metrics/confusion_matrix.png
```

Run the full reproducible pipeline:

```bash
dvc repro
```

Run stages manually:

```bash
python src/crop_recommendation/pipeline/data_ingestion.py
python src/crop_recommendation/pipeline/preprocessing.py
python src/crop_recommendation/pipeline/feature_engineering.py
python src/crop_recommendation/pipeline/splitter.py
python src/crop_recommendation/pipeline/model_training.py
python src/crop_recommendation/pipeline/model_eval.py
```

## Application

The serving layer is implemented with FastAPI and serves both the API and the static web interface from `app/`.

```bash
cd app
python main.py
```

The app starts at:

```text
http://127.0.0.1:8000/
```

### Prediction Request

`POST /predict`

```json
{
  "district": "string",
  "season": "string",
  "area": 10.0,
  "transplant_month": "june",
  "growth_period": "jun to oct",
  "harvest_period": "sep to nov",
  "min_temp": 18.0,
  "max_temp": 32.0,
  "min_relative_humidity": 55.0,
  "max_relative_humidity": 88.0
}
```

The response returns up to the top three crop recommendations when the model supports probability estimates.

## Data and Features

The pipeline uses the SPAS Dataset BD crop recommendation dataset. The preprocessing stage standardizes column names, cleans categorical values, encodes crop season and district information, vectorizes growth and harvest month ranges, converts numeric fields, and prepares a supervised classification target.

Core feature groups include:

- Regional and seasonal attributes.
- Area and production-related variables.
- Temperature and humidity statistics.
- Growth, harvest, and transplant month encodings.
- Label-encoded crop, district, and season fields.

## Experiment Tracking

Model training initializes DAGsHub and MLflow tracking in:

```text
src/crop_recommendation/pipeline/model_training.py
```

When running training locally, ensure MLflow/DAGsHub credentials are configured, or adjust the tracking setup for a local MLflow backend.

## Installation

Create and activate a Python environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the core dependencies:

```bash
pip install -U pandas numpy scikit-learn joblib pyyaml matplotlib seaborn dvc mlflow dagshub fastapi uvicorn pydantic
```

## Notes for GitHub Users

- Large `.joblib` model artifacts are ignored by default and should be managed through DVC, release assets, or external artifact storage.
- The FastAPI app expects deployment artifacts in `app/`, including `hybrid_model.joblib`, `label_encoders.joblib`, and `scaler_area.joblib`.
- `experiments/pipeline_v2/` is intentionally separated from the DVC production pipeline.

## License

This project is distributed under the terms of the [MIT License](LICENSE).

## References

If you use the dataset or methodology in academic work, cite the original dataset and associated publication:

```bibtex
@dataset{spas_dataset_bd,
  title     = {SPAS Dataset BD},
  publisher = {Mendeley Data},
  doi       = {10.17632/tszv6k3vky.1},
  url       = {https://data.mendeley.com/datasets/tszv6k3vky/1}
}

@article{spas_dib_2023,
  title   = {SPAS: A Dataset for Smart Crop Prediction and Recommendation in Bangladesh},
  journal = {Data in Brief},
  year    = {2023},
  doi     = {10.1016/j.dib.2023.109756},
  url     = {https://www.sciencedirect.com/science/article/pii/S2352340923007535}
}
```
