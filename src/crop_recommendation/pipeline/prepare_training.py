"""Prepare shared training artifacts (TVAE augmentation + RF hyperparameter tuning).

This stage is expensive and shared by every model stage. DVC caches it independently
so changing one model config does not re-run augmentation or RF search.
"""

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_PIPELINE_DIR, "..", "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from crop_recommendation.pipeline.model_trainer import (  # noqa: E402
    run_tvae_augmentation,
    tune_random_forest,
)


def _project_root():
    return os.path.abspath(os.path.join(_PIPELINE_DIR, "..", "..", ".."))


def _setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("PrepareTraining")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join(log_dir, "prepare_training.log"))
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


def run(train_path, test_path, arrays_path, rf_params_path, feature_columns_path, params_path):
    root = _project_root()
    log_dir = os.path.join(root, "reports", "logs")
    model_save_dir = os.path.join(root, "artifacts", "models")
    os.makedirs(os.path.dirname(arrays_path), exist_ok=True)
    os.makedirs(os.path.dirname(feature_columns_path), exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    logger = _setup_logger(log_dir)
    params = load_params(params_path)
    random_state = params.get("model_training", {}).get("random_state", 42)
    target_col = "crop_name_enc"

    logger.info("Loading train/test splits.")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_cols = [c for c in train_df.columns if c != target_col]

    logger.info("Running shared TVAE augmentation.")
    X_orig, y_orig, X_synth, y_synth, feature_cols = run_tvae_augmentation(
        train_df,
        target_col,
        model_save_dir,
        logger,
    )
    X_full = np.concatenate([X_orig, X_synth])
    y_full = np.concatenate([y_orig, y_synth])

    logger.info("Running shared Random Forest hyperparameter search.")
    best_rf_params = tune_random_forest(X_full, y_full, random_state)
    logger.info(f"Best RF params: {best_rf_params}")

    np.savez_compressed(
        arrays_path,
        X_orig=X_orig,
        y_orig=y_orig,
        X_synth=X_synth,
        y_synth=y_synth,
        X_full=X_full,
        y_full=y_full,
        X_test=test_df.drop(columns=[target_col]).values,
        y_test=test_df[target_col].astype(int).values,
    )
    with open(rf_params_path, "w") as f:
        json.dump(best_rf_params, f, indent=2)
    joblib.dump(feature_cols, feature_columns_path)

    logger.info(f"Saved training arrays to {arrays_path}")
    logger.info(f"Saved RF params to {rf_params_path}")
    logger.info(f"Saved feature columns to {feature_columns_path}")


if __name__ == "__main__":
    root = _project_root()
    parser = argparse.ArgumentParser(description="Prepare shared training data for modular model stages.")
    parser.add_argument(
        "--train-path",
        default=os.path.join(root, "data", "splits", "train.csv"),
    )
    parser.add_argument(
        "--test-path",
        default=os.path.join(root, "data", "splits", "test.csv"),
    )
    parser.add_argument(
        "--arrays-path",
        default=os.path.join(root, "data", "training", "training_arrays.npz"),
    )
    parser.add_argument(
        "--rf-params-path",
        default=os.path.join(root, "data", "training", "rf_best_params.json"),
    )
    parser.add_argument(
        "--feature-columns-path",
        default=os.path.join(root, "artifacts", "preprocessors", "feature_columns.joblib"),
    )
    parser.add_argument(
        "--params-path",
        default=os.path.join(root, "params.yaml"),
    )
    args = parser.parse_args()
    run(
        args.train_path,
        args.test_path,
        args.arrays_path,
        args.rf_params_path,
        args.feature_columns_path,
        args.params_path,
    )
    print("Prepare training complete.")
