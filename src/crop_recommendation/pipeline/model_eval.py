import os
import numpy as np
import pandas as pd
import logging
import json
import joblib

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, roc_auc_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon


class ModelEvaluation:
    def __init__(self, test_path=None,
                 model_dir=None,
                 feature_dir=None,
                 metrics_save_path=None,
                 cm_save_path=None,
                 log_dir=None):
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        test_path = test_path or os.path.join(_root, "data", "splits", "test.csv")
        model_dir = model_dir or os.path.join(_root, "artifacts", "models")
        feature_dir = feature_dir or os.path.join(_root, "artifacts", "preprocessors")
        metrics_save_path = metrics_save_path or os.path.join(_root, "reports", "metrics", "evaluation_metrics.json")
        cm_save_path = cm_save_path or os.path.join(_root, "reports", "metrics", "confusion_matrix.png")
        log_dir = log_dir or os.path.join(_root, "reports", "logs")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(cm_save_path), exist_ok=True)

        self.logger = logging.getLogger("ModelEvaluation")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            fh = logging.FileHandler(os.path.join(log_dir, "model_evaluation.log"))
            ch.setLevel(logging.INFO)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        self.test_path = test_path
        self.model_dir = model_dir
        self.feature_dir = feature_dir
        self.metrics_save_path = metrics_save_path
        self.cm_save_path = cm_save_path
        self.logger.info("ModelEvaluation pipeline initialized.")

    def load_test_data(self):
        self.logger.info(f"Loading test data from: {self.test_path}")
        test_df = pd.read_csv(self.test_path)
        target_column = 'crop_name_enc'
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        X_test = X_test.fillna(0)
        self.logger.info(f"Test features shape: {X_test.shape}")
        return X_test, y_test

    def load_model(self, model_name):
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        self.logger.info(f"Loading model from: {model_path}")
        return joblib.load(model_path)

    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_score_weighted': f1_score(y_test, y_pred, average='weighted'),
            'kappa': cohen_kappa_score(y_test, y_pred),
        }
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob, average='weighted', multi_class='ovr')
            except Exception:
                metrics['roc_auc'] = np.nan
        else:
            metrics['roc_auc'] = np.nan

        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report

        self.logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                         f"F1: {metrics['f1_score_weighted']:.4f}, "
                         f"Balanced Acc: {metrics['balanced_accuracy']:.4f}")
        return metrics, y_pred, y_prob

    def save_metrics(self, all_metrics):
        serializable = {}
        for name, m in all_metrics.items():
            serializable[name] = {k: v for k, v in m.items() if k != 'classification_report'}
        with open(self.metrics_save_path, 'w') as f:
            json.dump(serializable, f, indent=4)
        self.logger.info(f"Metrics saved to: {self.metrics_save_path}")

    def save_confusion_matrix(self, y_true, y_pred, model_name='best_model'):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, cmap='Blues', annot=True, fmt='d')
        plt.title(f"Confusion Matrix: {model_name}", fontsize=14)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(self.cm_save_path, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Confusion matrix saved to: {self.cm_save_path}")

    def run(self):
        self.logger.info("Starting model evaluation pipeline.")
        X_test, y_test = self.load_test_data()

        model_names = [
            'LogisticRegression', 'RandomForest_Tuned', 'LightGBM', 'XGBoost',
            'MLP', 'CatBoost', 'ResidualCatBoost_SVC', 'ResidualCatBoost_RF',
            'VotingClassifier_Ensemble'
        ]

        all_metrics = {}
        best_model_name = None
        best_balanced_acc = -1

        for name in model_names:
            try:
                model = self.load_model(name)
                metrics, y_pred, y_prob = self.evaluate_model(model, X_test, y_test, name)
                all_metrics[name] = metrics
                if metrics['balanced_accuracy'] > best_balanced_acc:
                    best_balanced_acc = metrics['balanced_accuracy']
                    best_model_name = name
            except FileNotFoundError:
                self.logger.warning(f"Model file not found: {name}. Skipping.")
            except Exception as e:
                self.logger.error(f"Error evaluating {name}: {e}")

        if best_model_name:
            self.logger.info(f"Best model: {best_model_name} (Balanced Acc: {best_balanced_acc:.4f})")
            best_model = self.load_model(best_model_name)
            y_pred = best_model.predict(X_test)
            self.save_confusion_matrix(y_test, y_pred, best_model_name)

        self.save_metrics(all_metrics)
        self.logger.info("Model evaluation pipeline completed.")

        # ── MLflow logging ──────────────────────────────────────────────
        if HAS_MLFLOW:
            self._log_to_mlflow(all_metrics, best_model_name)

        return all_metrics

    def _log_to_mlflow(self, all_metrics, best_model_name):
        """Log evaluation metrics and artifacts to MLflow."""
        import mlflow

        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if not tracking_uri:
            mlflow_dir = os.path.join(_root, "mlruns")
            tracking_uri = f"file:///{mlflow_dir.replace(os.sep, '/')}"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("crop-recommendation")

        with mlflow.start_run(run_name="evaluation-pipeline"):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_param("num_models_evaluated", len(all_metrics))

            for name, metrics in all_metrics.items():
                for metric_key in ["accuracy", "balanced_accuracy", "precision_weighted",
                                   "recall_weighted", "f1_score_weighted", "kappa", "roc_auc"]:
                    val = metrics.get(metric_key, None)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        mlflow.log_metric(f"{name}_{metric_key}", round(val, 4))

            # Log evaluation metrics JSON
            if os.path.exists(self.metrics_save_path):
                mlflow.log_artifact(self.metrics_save_path, artifact_path="reports")

            # Log confusion matrix PNG
            if os.path.exists(self.cm_save_path):
                mlflow.log_artifact(self.cm_save_path, artifact_path="reports")

            self.logger.info(f"MLflow evaluation logged to: {tracking_uri}")


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    log_dir = os.path.join(project_root, "reports", "logs")
    evaluator = ModelEvaluation(log_dir=log_dir)
    all_metrics = evaluator.run()
    print("\nEvaluation complete.")
