import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging


class DataSplit:
    def __init__(self, df, test_size=0.30, random_state=42,
                 save_path_train="./data/splits/train.csv",
                 save_path_test="./data/splits/test.csv",
                 log_dir="reports/logs"):
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_path_train), exist_ok=True)
        os.makedirs(os.path.dirname(save_path_test), exist_ok=True)

        self.logger = logging.getLogger("DataSplit")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            fh = logging.FileHandler(os.path.join(log_dir, "data_split.log"))
            ch.setLevel(logging.INFO)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        self.df = df.copy()
        self.test_size = test_size
        self.random_state = random_state
        self.save_path_train = save_path_train
        self.save_path_test = save_path_test
        self.logger.info("DataSplit pipeline initialized.")

    def split_data(self):
        self.logger.info("Splitting data into training and testing sets.")
        target_column = 'crop_name_enc'
        if target_column not in self.df.columns:
            self.logger.error(f"Target column '{target_column}' not found.")
            raise ValueError(f"Target column '{target_column}' not found.")

        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        self.logger.info(f"Training set shape: {train_df.shape}")
        self.logger.info(f"Testing set shape: {test_df.shape}")

        return train_df, test_df

    def save_splits(self, train_df, test_df):
        try:
            train_df.to_csv(self.save_path_train, index=False)
            self.logger.info(f"Training data saved to: {self.save_path_train}")
            test_df.to_csv(self.save_path_test, index=False)
            self.logger.info(f"Testing data saved to: {self.save_path_test}")
        except Exception as e:
            self.logger.exception(f"Error saving data splits: {e}")
            raise

    def run(self):
        self.logger.info("Starting data splitting run.")
        train_df, test_df = self.split_data()
        self.save_splits(train_df, test_df)
        self.logger.info("Data splitting completed successfully.")
        return train_df, test_df


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    featured_data_path = os.path.join(project_root, "data", "features", "featured_data.csv")
    log_dir = os.path.join(project_root, "reports", "logs")
    df = pd.read_csv(featured_data_path)
    splitter = DataSplit(df=df, log_dir=log_dir)
    train_data, test_data = splitter.run()
    print(f"\nData splitting complete. Train: {train_data.shape}, Test: {test_data.shape}")
