"""Train a single model variant — used by modular DVC foreach stages."""

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import yaml
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_PIPELINE_DIR, "..", "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from crop_recommendation.pipeline.model_configs import (  # noqa: E402
    build_logistic_regression,
    build_lightgbm,
    build_xgboost,
    build_mlp,
    build_catboost,
    build_calibrated_catboost_svc,
    build_calibrated_catboost_rf,
    build_voting_classifier,
)
from crop_recommendation.pipeline.model_trainer import (  # noqa: E402
    train_single_model,
    calculate_metrics,
)


def _project_root():
    return os.path.abspath(os.path.join(_PIPELINE_DIR, "..", "..", ".."))


def _setup_logger(log_dir, model_name):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"TrainModel.{model_name}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join(log_dir, f"train_{model_name}.log"))
        fmt = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(fmt)
        fh.setFormatter(fmt)
        ch.setLevel(logging.INFO)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger


def load_params(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_name, random_state, n_classes, best_rf_params):
    if model_name == "LogisticRegression":
        return build_logistic_regression(random_state)
    if model_name == "RandomForest_Tuned":
        return clone(RandomForestClassifier(**best_rf_params, random_state=random_state, n_jobs=-1))
    if model_name == "LightGBM":
        return build_lightgbm(random_state)
    if model_name == "XGBoost":
        return build_xgboost(random_state, n_classes)
    if model_name == "MLP":
        return build_mlp(random_state)
    if model_name == "CatBoost":
        return build_catboost(random_state)
    if model_name == "ResidualCatBoost_SVC":
        return build_calibrated_catboost_svc(random_state)
    if model_name == "ResidualCatBoost_RF":
        return build_calibrated_catboost_rf(random_state, best_rf_params)
    if model_name == "VotingClassifier_Ensemble":
        return build_voting_classifier(random_state, best_rf_params)
    raise ValueError(f"Unknown model: {model_name}")


def train_model(model_name, arrays_path, rf_params_path, model_path, metrics_path, params_path):
    root = _project_root()
    log_dir = os.path.join(root, "reports", "logs")
    logger = _setup_logger(log_dir, model_name)

    params = load_params(params_path)
    random_state = params.get("model_training", {}).get("random_state", 42)

    arrays = np.load(arrays_path)
    X_orig, y_orig = arrays["X_orig"], arrays["y_orig"]
    X_synth, y_synth = arrays["X_synth"], arrays["y_synth"]
    X_full, y_full = arrays["X_full"], arrays["y_full"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]

    with open(rf_params_path, "r") as f:
        best_rf_params = json.load(f)

    n_classes = len(np.unique(y_orig))
    model = build_model(model_name, random_state, n_classes, best_rf_params)

    if model_name == "VotingClassifier_Ensemble":
        logger.info(f"Training {model_name}...")
        model.fit(X_full, y_full)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics["model"] = model_name
        final_model = model
    else:
        final_model, metrics = train_single_model(
            model_name,
            model,
            X_orig,
            y_orig,
            X_synth,
            y_synth,
            X_full,
            y_full,
            X_test,
            y_test,
            random_state,
            logger,
        )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    joblib.dump(final_model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved metrics to {metrics_path}")
    return metrics


if __name__ == "__main__":
    root = _project_root()
    parser = argparse.ArgumentParser(description="Train one model for the modular DVC pipeline.")
    parser.add_argument("--model", required=True, help="Model name, e.g. LightGBM")
    parser.add_argument(
        "--arrays-path",
        default=os.path.join(root, "data", "training", "training_arrays.npz"),
    )
    parser.add_argument(
        "--rf-params-path",
        default=os.path.join(root, "data", "training", "rf_best_params.json"),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Output model path (default: artifacts/models/<model>.joblib)",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Output metrics path (default: reports/metrics/models/<model>.json)",
    )
    parser.add_argument(
        "--params-path",
        default=os.path.join(root, "params.yaml"),
    )
    args = parser.parse_args()

    model_path = args.model_path or os.path.join(root, "artifacts", "models", f"{args.model}.joblib")
    metrics_path = args.metrics_path or os.path.join(root, "reports", "metrics", "models", f"{args.model}.json")

    metrics = train_model(
        args.model,
        args.arrays_path,
        args.rf_params_path,
        model_path,
        metrics_path,
        args.params_path,
    )
    print(f"{args.model}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
