import os
import pandas as pd
import logging


class DataIngestion:
    def __init__(self, data_path: str, log_dir: str = "logs"):
        """Handles dataset ingestion and logging setup."""
        os.makedirs(log_dir, exist_ok=True)

        # Logger setup
        self.logger = logging.getLogger("Data-Handler")
        self.logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            fh = logging.FileHandler(os.path.join(log_dir, "data_history.log"))
            ch.setLevel(logging.DEBUG)
            fh.setLevel(logging.DEBUG)

            formatter = logging.Formatter(
                "%(asctime)s - [%(levelname)s] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)

            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        self.data_path = data_path
        self.logger.info("DataIngestion initialized successfully.")

    def load_store_data(self, save_path: str) -> pd.DataFrame:
        """Load dataset from source and save it to a local folder for reproducibility."""
        try:
            self.logger.info(f"Loading dataset from: {self.data_path}")
            df = pd.read_csv(self.data_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            self.logger.info(f"Dataset saved successfully to: {save_path}")
            return df

        except FileNotFoundError:
            self.logger.error(f"File not found at: {self.data_path}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("The CSV file is empty.")
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error during load_store_data: {str(e)}")
            raise

    def load_data(self) -> pd.DataFrame:
        """Load dataset directly without saving (for quick reads)."""
        try:
            self.logger.info(f"Reading dataset from: {self.data_path}")
            df = pd.read_csv(self.data_path)
            self.logger.info("Dataset loaded successfully.")
            return df

        except FileNotFoundError:
            self.logger.error(f"File not found at: {self.data_path}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("The CSV file is empty.")
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error while loading data: {str(e)}")
            raise

    def preview_data(self, df: pd.DataFrame, n: int = 5):
        """Preview the top records of the dataset."""
        try:
            self.logger.debug(f"Previewing first {n} records:\n{df.head(n).to_string(index=False)}")
        except Exception as e:
            self.logger.warning(f"Unable to preview data: {str(e)}")


if __name__ == "__main__":
    # Example standalone run
    data_path = "./miscellaneous/SPAS-Dataset-BD.csv"
    save_path = "./data/raw/SPAS-Dataset-BD.csv"

    ingestion = DataIngestion(data_path=data_path, log_dir="./logs")

    df = ingestion.load_store_data(save_path)
    # ingestion.preview_data(df, n=3)

    df_loaded = ingestion.load_data()
    ingestion.logger.info(f"Final DataFrame shape: {df_loaded.shape}")
