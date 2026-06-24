import os
import pandas as pd
import logging


class DataIngestion:
    def __init__(self, data_path, log_dir="reports/logs"):
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger("DataIngestion")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            fh = logging.FileHandler(os.path.join(log_dir, "data_ingestion.log"))
            ch.setLevel(logging.DEBUG)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        self.data_path = data_path
        self.logger.info("DataIngestion initialized.")

    def load_store_data(self, save_path):
        self.logger.info(f"Loading dataset from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        self.logger.info(f"Dataset saved to: {save_path}")
        return df

    def load_data(self):
        self.logger.info(f"Reading dataset from: {self.data_path}")
        return pd.read_csv(self.data_path)

    def run(self, save_path):
        return self.load_store_data(save_path)


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_path = os.path.join(project_root, "data", "external", "SPAS-Dataset-BD.csv")
    save_path = os.path.join(project_root, "data", "raw", "SPAS-Dataset-BD.csv")
    log_dir = os.path.join(project_root, "reports", "logs")
    os.makedirs(log_dir, exist_ok=True)
    ingestion = DataIngestion(data_path=data_path, log_dir=log_dir)
    df = ingestion.run(save_path)
    print(f"Data ingestion complete. Shape: {df.shape}")
