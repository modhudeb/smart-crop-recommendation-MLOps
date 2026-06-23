"""Model training orchestrator — delegates to model_configs and model_trainer."""
import os
import json
import logging
import joblib
import yaml
import numpy as np
import pandas as pd

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

import sys
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_PIPELINE_DIR, "..", "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from crop_recommendation.pipeline.encoding import CategoricalEncoder
from crop_recommendation.pipeline.feature_engineering import FeatureEngineering
from crop_recommendation.pipeline.model_configs import (
    build_logistic_regression,
    build_lightgbm,
    build_xgboost,
    build_mlp,
    build_catboost,
    build_calibrated_catboost_svc,
    build_calibrated_catboost_rf,
    build_voting_classifier,
)
from crop_recommendation.pipeline.model_trainer import (
    run_tvae_augmentation,
    tune_random_forest,
    train_single_model,
)


def load_params(param_path="params.yaml"):
    with open(param_path, "r") as f:
        return yaml.safe_load(f)


def configure_mlflow(project_root, experiment_name="crop-recommendation"):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    mlflow_dir = os.path.join(project_root, "mlruns")
    artifact_location = f"file:///{mlflow_dir.replace(os.sep, '/')}"

    if not tracking_uri:
        db_path = os.path.join(project_root, "mlflow.db")
        tracking_uri = f"sqlite:///{db_path.replace(os.sep, '/')}"

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
    mlflow.set_experiment(experiment_name)
    return tracking_uri


class ModelTraining:
    def __init__(self, train_path=None, model_save_dir=None,
                 feature_save_dir=None, log_dir=None):
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        self.train_path = train_path or os.path.join(_root, "data", "splits", "train.csv")
        self.model_save_dir = model_save_dir or os.path.join(_root, "artifacts", "models")
        self.feature_save_dir = feature_save_dir or os.path.join(_root, "artifacts", "preprocessors")
        self.log_dir = log_dir or os.path.join(_root, "reports", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.feature_save_dir, exist_ok=True)

        self.logger = logging.getLogger("ModelTraining")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            fh = logging.FileHandler(os.path.join(self.log_dir, "model_training.log"))
            ch.setLevel(logging.INFO)
            fh.setLevel(logging.DEBUG)
            fmt = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s",
                                    datefmt="%Y-%m-%d %H:%M:%S")
            ch.setFormatter(fmt)
            fh.setFormatter(fmt)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        self.models = {}
        self.results = {}
        self.logger.info("ModelTraining pipeline initialized.")

    # ── Data ────────────────────────────────────────────────────────────

    def load_data(self):
        self.logger.info(f"Loading training data from: {self.train_path}")
        return pd.read_csv(self.train_path)

    def prepare_data(self, df):
        target = "crop_name_enc"
        X = df.drop(columns=[target]).values
        y = df[target].astype(int).values
        self.logger.info(f"Features: {X.shape}, Target: {y.shape}")
        return X, y, list(df.drop(columns=[target]).columns)

    # ── Training ────────────────────────────────────────────────────────

    def train_all(self, train_df, X_test, y_test, feature_cols, random_state=42):
        target_col = "crop_name_enc"

        X_orig, y_orig, X_synth, y_synth, feature_cols = run_tvae_augmentation(
            train_df, target_col, self.model_save_dir, self.logger,
        )
        X_full = np.concatenate([X_orig, X_synth])
        y_full = np.concatenate([y_orig, y_synth])

        best_rf_params = tune_random_forest(X_full, y_full, random_state)
        self.logger.info(f"Best RF params: {best_rf_params}")
        n_classes = len(np.unique(y_orig))

        # Standard models
        standard_builders = {
            "LogisticRegression": lambda: build_logistic_regression(random_state),
            "RandomForest_Tuned": lambda: build_logistic_regression(random_state),  # placeholder, replaced below
            "LightGBM": lambda: build_lightgbm(random_state),
            "XGBoost": lambda: build_xgboost(random_state, n_classes),
            "MLP": lambda: build_mlp(random_state),
            "CatBoost": lambda: build_catboost(random_state),
        }

        # Replace RF placeholder with tuned model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.base import clone
        standard_builders["RandomForest_Tuned"] = lambda: clone(
            RandomForestClassifier(**best_rf_params, random_state=random_state, n_jobs=-1)
        )

        for name, builder in standard_builders.items():
            model = builder()
            final_model, metrics = train_single_model(
                name, model, X_orig, y_orig, X_synth, y_synth,
                X_full, y_full, X_test, y_test, random_state, self.logger,
            )
            self.models[name] = final_model
            self.results[name] = metrics

        # Calibrated CatBoost variants
        for name, builder in {
            "ResidualCatBoost_SVC": lambda: build_calibrated_catboost_svc(random_state),
            "ResidualCatBoost_RF": lambda: build_calibrated_catboost_rf(random_state, best_rf_params),
        }.items():
            model = builder()
            final_model, metrics = train_single_model(
                name, model, X_orig, y_orig, X_synth, y_synth,
                X_full, y_full, X_test, y_test, random_state, self.logger,
            )
            self.models[name] = final_model
            self.results[name] = metrics

        # Voting ensemble
        self.logger.info("Training VotingClassifier_Ensemble...")
        voting = build_voting_classifier(random_state, best_rf_params)
        voting.fit(X_full, y_full)
        self.models["VotingClassifier_Ensemble"] = voting
        y_pred_v = voting.predict(X_test)
        y_prob_v = voting.predict_proba(X_test)
        from crop_recommendation.pipeline.model_trainer import calculate_metrics
        m = calculate_metrics(y_test, y_pred_v, y_prob_v)
        m["model"] = "VotingClassifier_Ensemble"
        self.results["VotingClassifier_Ensemble"] = m

        return self.results

    # ── Artifacts ──────────────────────────────────────────────────────

    def save_artifacts(self, encoder, feature_engineer, feature_cols):
        if isinstance(encoder, dict):
            encoders = encoder
        else:
            encoders = {
                "le_crop": encoder.le_crop,
                "le_season": encoder.le_season,
                "le_district": encoder.le_district,
            }
        joblib.dump(encoders, os.path.join(self.feature_save_dir, "label_encoders.joblib"))
        joblib.dump(feature_engineer.scaler, os.path.join(self.feature_save_dir, "standard_scaler.joblib"))
        joblib.dump(feature_engineer.climate_constants, os.path.join(self.feature_save_dir, "climate_constants.joblib"))
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(self.model_save_dir, f"{name}.joblib"))
        joblib.dump(feature_cols, os.path.join(self.feature_save_dir, "feature_columns.joblib"))
        self.logger.info(f"Artifacts saved to {self.model_save_dir} and {self.feature_save_dir}")

    # ── MLflow ─────────────────────────────────────────────────────────

    def _log_to_mlflow(self, project_root, results):
        tracking_uri = configure_mlflow(project_root)

        params = load_params(os.path.join(project_root, "params.yaml"))

        with mlflow.start_run(run_name="training-pipeline"):
            mlflow.log_param("cv_folds", params.get("model_training", {}).get("cv_folds", 5))
            mlflow.log_param("random_state", params.get("model_training", {}).get("random_state", 42))
            mlflow.log_param("test_size", params.get("split_data", {}).get("test_size", 0.30))
            mlflow.log_param("augmentation_enabled", params.get("augmentation", {}).get("enabled", True))
            from crop_recommendation.pipeline.model_trainer import HAS_SDV
            mlflow.log_param("sdv_installed", HAS_SDV)
            # log per-model params
            for section in ["logistic_regression", "random_forest_tuned", "lightgbm",
                            "xgboost", "mlp", "catboost", "calibrated_catboost_svc",
                            "calibrated_catboost_rf", "voting_classifier"]:
                cfg = params.get("model_training", {}).get(section, {})
                if isinstance(cfg, dict):
                    for k, v in cfg.items():
                        mlflow.log_param(f"{section}.{k}", json.dumps(v) if isinstance(v, (list, dict)) else v)

            for name, metrics in results.items():
                for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "kappa", "roc_auc"]:
                    val = metrics.get(key)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        mlflow.log_metric(f"{name}_{key}", round(val, 4))

            for name, model in self.models.items():
                try:
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        name=name,
                        registered_model_name=name,
                    )
                    self.logger.info(f"Registered model: {name}")
                except Exception as e:
                    self.logger.warning(f"MLflow register fail {name}: {e}")
                    # Fallback: log as plain artifact
                    path = os.path.join(self.model_save_dir, f"{name}.joblib")
                    if os.path.exists(path):
                        try:
                            mlflow.log_artifact(path, artifact_path="models")
                        except Exception as e2:
                            self.logger.warning(f"MLflow artifact fail {name}: {e2}")

            csv_path = os.path.join(project_root, "reports", "metrics", "training_results.csv")
            if os.path.exists(csv_path):
                mlflow.log_artifact(csv_path, artifact_path="reports")

            self.logger.info(f"MLflow logged to: {tracking_uri}")

    # ── Main ───────────────────────────────────────────────────────────

    def run(self):
        self.logger.info("Starting model training pipeline.")
        train_df = self.load_data()
        X_train_raw, y_train_raw, feature_cols = self.prepare_data(train_df)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        test_df = pd.read_csv(os.path.join(project_root, "data", "splits", "test.csv"))
        X_test = test_df.drop(columns=["crop_name_enc"]).values
        y_test = test_df["crop_name_enc"].astype(int).values

        results = self.train_all(train_df, X_test, y_test, feature_cols)

        pp_dir = os.path.join(project_root, "artifacts", "preprocessors")
        enc_path = os.path.join(pp_dir, "label_encoders.joblib")
        encoder = joblib.load(enc_path) if os.path.exists(enc_path) else None
        fe = FeatureEngineering.__new__(FeatureEngineering)
        fe.scaler = joblib.load(os.path.join(pp_dir, "standard_scaler.joblib")) \
            if os.path.exists(os.path.join(pp_dir, "standard_scaler.joblib")) else None
        fe.climate_constants = joblib.load(os.path.join(pp_dir, "climate_constants.joblib")) \
            if os.path.exists(os.path.join(pp_dir, "climate_constants.joblib")) else None
        if encoder and fe.scaler:
            self.save_artifacts(encoder, fe, feature_cols)

        results_df = pd.DataFrame(results).T
        results_df.to_csv(os.path.join(project_root, "reports", "metrics", "training_results.csv"))
        self.logger.info("Model training pipeline completed.")

        if HAS_MLFLOW:
            self._log_to_mlflow(project_root, results)

        return results


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    trainer = ModelTraining(
        train_path=os.path.join(project_root, "data", "splits", "train.csv"),
        model_save_dir=os.path.join(project_root, "artifacts", "models"),
        feature_save_dir=os.path.join(project_root, "artifacts", "preprocessors"),
        log_dir=os.path.join(project_root, "reports", "logs"),
    )
    results = trainer.run()
    print("\nModel training complete.")
    for name, metrics in results.items():
        print(f"  {name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
