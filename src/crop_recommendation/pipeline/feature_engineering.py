import os
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler


class FeatureEngineering:
    def __init__(self, df, save_path="./data/features/featured_data.csv", log_dir="reports/logs"):
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.logger = logging.getLogger("FeatureEngineering")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            fh = logging.FileHandler(os.path.join(log_dir, "feature_engineering.log"))
            ch.setLevel(logging.INFO)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        self.df = df.copy()
        self.save_path = save_path
        self.scaler = None
        self.climate_constants = None
        self.logger.info("FeatureEngineering pipeline initialized.")

    def compute_climate_constants(self):
        df_valid = self.df[(self.df["production"] > 0) & (self.df["area"] > 0)].copy()
        mu_T = df_valid[["max_temp", "min_temp"]].stack().mean()
        sigma_T = df_valid[["max_temp", "min_temp"]].stack().std()
        mu_H = df_valid[["max_relative_humidity", "min_relative_humidity"]].stack().mean()
        sigma_H = df_valid[["max_relative_humidity", "min_relative_humidity"]].stack().std()
        self.climate_constants = {"mu_T": mu_T, "sigma_T": sigma_T, "mu_H": mu_H, "sigma_H": sigma_H}
        self.logger.info(
            f"Climate constants: mu_T={mu_T:.4f}, sigma_T={sigma_T:.4f}, " f"mu_H={mu_H:.4f}, sigma_H={sigma_H:.4f}"
        )

    def create_climate_risk_score(self):
        self.logger.info("Creating climate risk score.")
        cc = self.climate_constants
        self.df["climate_risk_score"] = (
            ((self.df["max_temp"] - cc["mu_T"]) / cc["sigma_T"])
            + ((self.df["min_temp"] - cc["mu_T"]) / cc["sigma_T"])
            + ((self.df["max_relative_humidity"] - cc["mu_H"]) / cc["sigma_H"])
            + ((self.df["min_relative_humidity"] - cc["mu_H"]) / cc["sigma_H"])
        )

    def drop_weather_columns(self):
        self.logger.info("Dropping raw weather columns.")
        weather_cols = [
            "avg_temp",
            "avg_humidity",
            "max_temp",
            "min_temp",
            "max_relative_humidity",
            "min_relative_humidity",
        ]
        self.df.drop(columns=[c for c in weather_cols if c in self.df.columns], inplace=True)

    def log_transform_and_scale(self):
        self.logger.info("Log-transforming and scaling numeric features.")
        self.df["area_log"] = np.log1p(self.df["area"])
        self.df["production_log"] = np.log1p(self.df["production"])
        self.scaler = StandardScaler()
        cols_to_scale = ["area_log", "production_log", "ap_ratio"]
        self.df[cols_to_scale] = self.scaler.fit_transform(self.df[cols_to_scale])

    def drop_redundant(self):
        self.logger.info("Dropping redundant columns (area, production, ap_ratio, production_log).")
        self.df.drop(
            columns=["area", "production", "ap_ratio", "production_log"],
            inplace=True,
            errors="ignore",
        )

    def save_featured_data(self):
        try:
            self.df.to_csv(self.save_path, index=False)
            self.logger.info(f"Feature-engineered dataset saved to: {self.save_path}")
        except Exception as e:
            self.logger.exception(f"Error saving featured data: {e}")
            raise

    def run(self):
        self.logger.info("Starting feature engineering run.")
        self.compute_climate_constants()
        self.create_climate_risk_score()
        self.drop_weather_columns()
        self.log_transform_and_scale()
        self.drop_redundant()
        self.df.fillna(0, inplace=True)
        self.save_featured_data()
        self.logger.info(f"Feature engineering completed. Final shape: {self.df.shape}")
        return self.df


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_path = os.path.join(project_root, "data", "processed", "encoded_data.csv")
    save_path = os.path.join(project_root, "data", "features", "featured_data.csv")
    log_dir = os.path.join(project_root, "reports", "logs")
    df = pd.read_csv(data_path)
    fe = FeatureEngineering(df=df, save_path=save_path, log_dir=log_dir)
    featured_df = fe.run()
    pp_dir = os.path.join(project_root, "artifacts", "preprocessors")
    os.makedirs(pp_dir, exist_ok=True)
    joblib.dump(fe.scaler, os.path.join(pp_dir, "standard_scaler.joblib"))
    joblib.dump(fe.climate_constants, os.path.join(pp_dir, "climate_constants.joblib"))
    print(f"\nFeature engineering complete. Shape: {featured_df.shape}")
    print(f"Columns: {list(featured_df.columns)}")
