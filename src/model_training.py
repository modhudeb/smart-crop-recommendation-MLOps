import os
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
import joblib
import yaml
import mlflow
import dagshub



# MLOps CONFIG
dagshub.init(repo_owner='modhuofficial', repo_name='smart-crop-recommendation-MLOps', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/modhuofficial/smart-crop-recommendation-MLOps.mlflow")




class ModelTraining:
    """
    Handles the training of the model and saving it to disk.
    """

    def __init__(self, 
                 train_path: str = "./data/splits/train.csv", 
                 model_save_path: str = "./models/random_forest_model.joblib", 
                 log_dir: str = "logs",
                 params: dict = None):
        """
        Initializes the ModelTraining pipeline.

        Args:
            train_path (str): Path to the training data CSV.
            model_save_path (str): Path to save the trained model file.
            log_dir (str): Directory to store log files.
        """
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        # Logger setup
        self.logger = logging.getLogger("ModelTraining")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            fh = logging.FileHandler(os.path.join(log_dir, "model_training.log"))
            ch.setLevel(logging.INFO)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        self.train_path = train_path
        self.model_save_path = model_save_path
        self.params = params if params is not None else {}
        
        # Initialize the model (RandomForestClassifier based on your notebook)
        self.model = RandomForestClassifier(**params)
        
        self.logger.info("ModelTraining pipeline initialized.")

    def load_data(self) -> pd.DataFrame:
        """Loads the training data."""
        try:
            self.logger.info(f"Loading training data from: {self.train_path}")
            train_df = pd.read_csv(self.train_path)
            return train_df
        except FileNotFoundError:
            self.logger.error(f"Training data file not found at: {self.train_path}")
            raise
        except Exception as e:
            self.logger.exception(f"Error loading training data: {str(e)}")
            raise

    def prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Separates features (X) and target (y) from the DataFrame."""
        self.logger.info("Preparing data for training.")
        
        target_column = 'crop_name_enc'
        if target_column not in df.columns:
            self.logger.error(f"Target column '{target_column}' not found.")
            raise ValueError(f"Target column '{target_column}' not found.")
            
        X_train = df.drop(columns=[target_column])
        y_train = df[target_column]
        
        # Handle any potential NaNs that might have been missed or introduced
        X_train = X_train.fillna(0)
        
        self.logger.info(f"Features shape: {X_train.shape}")
        self.logger.info(f"Target shape: {y_train.shape}")        
        return X_train, y_train

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fits the model on the training data."""
        try:
            self.logger.info("Starting model training...")
            self.model.fit(X_train, y_train)
            self.logger.info("Model training completed successfully.")
        except Exception as e:
            self.logger.exception(f"An error occurred during model training: {str(e)}")
            raise

    def save_model(self):
        """Saves the trained model to disk using joblib."""
        try:
            joblib.dump(self.model, self.model_save_path)
            self.logger.info(f"Model saved successfully to: {self.model_save_path}")
        except Exception as e:
            self.logger.exception(f"Error saving the model: {str(e)}")
            raise

    def run(self):
        """Executes the full model training pipeline."""
        self.logger.info("Starting model training run.")
        train_df = self.load_data()
        X_train, y_train = self.prepare_data(train_df)
        with mlflow.start_run():
            X_dataset = mlflow.data.from_pandas(df=X_train, name="training_data")
            y_dataset = mlflow.data.from_pandas(df=y_train.to_frame(name="crop_name_enc"), name="training_data")
            mlflow.log_input(dataset=X_dataset, context='training')
            mlflow.log_input(dataset=y_dataset, context='training')

            self.train_model(X_train, y_train)
            self.save_model()

            mlflow.log_params(params=self.params)
            mlflow.sklearn.log_model(
                sk_model=self.model,
                name='models_crop_recommend',
                registered_model_name='RandomForestHyper-crop-recommend'
            )
            # print("=")
            # mlflow.log_artifact(self.train_path, 'data')

            self.logger.info("Model training pipeline finished.")



def load_params(param_path: str = "params.yaml") -> dict:
    """Loads parameters from a YAML file."""
    with open(param_path, 'r') as file:
        params = yaml.safe_load(file)
    return params


if __name__ == "__main__":
    try:
        trainer = ModelTraining(
            train_path="./data/splits/train.csv",
            model_save_path="./models/random_forest_model.joblib",
            log_dir="./logs",
            params=load_params(param_path='params.yaml').get('model_training', {}).get('random_forest', {})
        )
        
        trainer.run()
        print("\nModel training complete and model saved.")

    except FileNotFoundError:
        logging.error("The 'train.csv' file was not found. Please run the data splitting script first.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the training process: {e}")