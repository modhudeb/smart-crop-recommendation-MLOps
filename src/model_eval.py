import os
import pandas as pd
import logging
import joblib
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluation:
    """
    Handles loading the trained model and evaluating its performance
    on the test dataset. Saves metrics and a confusion matrix.
    """

    def __init__(self, 
                 test_path: str = "./data/splits/test.csv", 
                 model_path: str = "./models/random_forest_model.joblib", 
                 metrics_save_path: str = "./metrics/evaluation_metrics.json",
                 cm_save_path: str = "./metrics/confusion_matrix.png",
                 log_dir: str = "logs"):
        """
        Initializes the ModelEvaluation pipeline.

        Args:
            test_path (str): Path to the test data CSV.
            model_path (str): Path to the trained model file.
            metrics_save_path (str): Path to save the evaluation metrics (JSON).
            cm_save_path (str): Path to save the confusion matrix plot (PNG).
            log_dir (str): Directory to store log files.
        """
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(cm_save_path), exist_ok=True)

        # Logger setup
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
        self.model_path = model_path
        self.metrics_save_path = metrics_save_path
        self.cm_save_path = cm_save_path
        self.model = None
        self.logger.info("ModelEvaluation pipeline initialized.")

    def load_model(self):
        """Loads the trained model from disk."""
        try:
            self.logger.info(f"Loading model from: {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.logger.info("Model loaded successfully.")
        except FileNotFoundError:
            self.logger.error(f"Model file not found at: {self.model_path}")
            raise
        except Exception as e:
            self.logger.exception(f"Error loading model: {str(e)}")
            raise

    def load_test_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Loads and prepares the test data."""
        try:
            self.logger.info(f"Loading test data from: {self.test_path}")
            test_df = pd.read_csv(self.test_path)
            
            target_column = 'crop_name_enc'
            if target_column not in test_df.columns:
                self.logger.error(f"Target column '{target_column}' not found in test data.")
                raise ValueError(f"Target column '{target_column}' not found.")
                
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            # Ensure consistency with training (fill NaNs)
            X_test = X_test.fillna(0)
            
            self.logger.info(f"Test features shape: {X_test.shape}")
            self.logger.info(f"Test target shape: {y_test.shape}")
            
            return X_test, y_test
        except FileNotFoundError:
            self.logger.error(f"Test data file not found at: {self.test_path}")
            raise
        except Exception as e:
            self.logger.exception(f"Error loading test data: {str(e)}")
            raise

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Generates predictions and calculates evaluation metrics."""
        if self.model is None:
            self.logger.error("Model is not loaded. Run load_model() first.")
            raise ValueError("Model has not been loaded.")
            
        try:
            self.logger.info("Generating predictions on test data...")
            y_pred = self.model.predict(X_test)
            
            self.logger.info("Calculating evaluation metrics.")
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            report = classification_report(y_test, y_pred, output_dict=True)
            
            metrics = {
                'accuracy': accuracy,
                'precision_weighted': precision,
                'recall_weighted': recall,
                'f1_score_weighted': f1,
                'classification_report': report
            }
            
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"Precision (Weighted): {precision:.4f}")
            self.logger.info(f"Recall (Weighted): {recall:.4f}")
            self.logger.info(f"F1-Score (Weighted): {f1:.4f}")
            
            # Save metrics to JSON
            with open(self.metrics_save_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"Metrics saved to: {self.metrics_save_path}")

            # Generate and save confusion matrix
            self.save_confusion_matrix(y_test, y_pred)

        except Exception as e:
            self.logger.exception(f"An error occurred during model evaluation: {str(e)}")
            raise
            
    def save_confusion_matrix(self, y_true, y_pred):
        """Generates, plots, and saves a confusion matrix."""
        self.logger.info("Generating confusion matrix plot.")
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        
        try:
            plt.savefig(self.cm_save_path, bbox_inches='tight')
            self.logger.info(f"Confusion matrix plot saved to: {self.cm_save_path}")
        except Exception as e:
            self.logger.exception(f"Failed to save confusion matrix plot: {str(e)}")

    def run(self):
        """Executes the full model evaluation pipeline."""
        self.logger.info("Starting model evaluation run.")
        self.load_model()
        X_test, y_test = self.load_test_data()
        self.evaluate_model(X_test, y_test)
        self.logger.info("Model evaluation pipeline finished.")


if __name__ == "__main__":
    try:
        evaluator = ModelEvaluation(
            test_path="./data/splits/test.csv",
            model_path="./models/random_forest_model.joblib",
            metrics_save_path="./metrics/evaluation_metrics.json",
            cm_save_path="./metrics/confusion_matrix.png",
            log_dir="./logs"
        )
        evaluator.run()
        print("\nModel evaluation complete. Metrics and confusion matrix saved.")

    except FileNotFoundError:
        logging.error("A required file (test.csv or model.joblib) was not found. Please run previous steps.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the evaluation process: {e}")