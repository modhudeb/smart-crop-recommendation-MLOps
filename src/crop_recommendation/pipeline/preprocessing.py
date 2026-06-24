import os
import re
import numpy as np
import pandas as pd
import logging

MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
MONTH_TO_IDX = {m: i for i, m in enumerate(MONTHS)}
MONTH_ABBR_MAP = {
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "may": "may",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}


def _clean_month_string(s):
    s = str(s).strip().lower()
    for abbr, full in MONTH_ABBR_MAP.items():
        s = re.sub(rf"\b{abbr}\b", full, s)
    return s


def _month_range_to_vector(s):
    s = _clean_month_string(s)
    split = re.split(r"\s+to\s+", s)
    vec = [0] * 12
    if len(split) == 2:
        start, end = split
        if start in MONTH_TO_IDX and end in MONTH_TO_IDX:
            i1, i2 = MONTH_TO_IDX[start], MONTH_TO_IDX[end]
            if i1 <= i2:
                for i in range(i1, i2 + 1):
                    vec[i] = 1
            else:
                for i in range(i1, 12):
                    vec[i] = 1
                for i in range(0, i2 + 1):
                    vec[i] = 1
    return vec


class Preprocessing:
    def __init__(self, df, save_path="./data/processed/processed_data.csv", log_dir="reports/logs"):
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.logger = logging.getLogger("Preprocessing")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            fh = logging.FileHandler(os.path.join(log_dir, "preprocessing.log"))
            ch.setLevel(logging.INFO)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        self.df = df.copy()
        self.save_path = save_path
        self.logger.info("Preprocessing pipeline initialized.")

    def clean_columns(self):
        self.logger.info("Cleaning column names and categorical string values.")
        self.df.columns = self.df.columns.str.strip().str.replace(" ", "_").str.lower()
        for col in ["season", "crop_name", "district"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.strip().str.lower()
        if "transplant" in self.df.columns:
            self.df["transplant"] = self.df["transplant"].apply(_clean_month_string)

    def process_growth_harvest(self):
        self.logger.info("Processing growth and harvest months into feature vectors.")
        growth_vectors = self.df["growth"].apply(_month_range_to_vector).to_list()
        harvest_vectors = self.df["harvest"].apply(_month_range_to_vector).to_list()
        growth_df = pd.DataFrame(growth_vectors, columns=[f"growth_{m[:3]}" for m in MONTHS])
        harvest_df = pd.DataFrame(harvest_vectors, columns=[f"harvest_{m[:3]}" for m in MONTHS])
        self.df = pd.concat([self.df.reset_index(drop=True), growth_df, harvest_df], axis=1)
        self.df.drop(columns=["growth", "harvest"], inplace=True)

    def convert_numeric_and_handle_missing(self):
        self.logger.info("Converting data types, calculating AP ratio, handling missing values.")
        self.df["area"] = pd.to_numeric(self.df["area"], errors="coerce")
        self.df["production"] = pd.to_numeric(self.df["production"], errors="coerce")
        self.df["ap_ratio"] = (self.df["area"] / self.df["production"].replace(0, np.nan)).astype(float)
        self.df["area"] = self.df["area"].astype(int)
        self.df["production"] = self.df["production"].astype(int)
        self.df.dropna(subset=["area", "production"], inplace=True)

        weather_cols = [
            "avg_temp",
            "min_temp",
            "max_temp",
            "avg_humidity",
            "min_relative_humidity",
            "max_relative_humidity",
        ]
        self.df[weather_cols] = self.df[weather_cols].apply(pd.to_numeric, errors="coerce")

    def filter_invalid(self):
        self.logger.info("Filtering invalid rows.")
        self.df = self.df[self.df["crop_name"] != "#ref!"].copy()
        self.df = self.df[self.df["season"].notna()].copy()
        self.df = self.df[(self.df["production"] > 0) & (self.df["area"] > 0)].copy()

    def save_processed(self):
        try:
            self.df.to_csv(self.save_path, index=False)
            self.logger.info(f"Processed dataset saved to: {self.save_path}")
        except Exception as e:
            self.logger.exception(f"Error saving processed data: {e}")
            raise

    def run(self):
        self.logger.info("Starting preprocessing run.")
        self.clean_columns()
        self.process_growth_harvest()
        self.convert_numeric_and_handle_missing()
        self.filter_invalid()
        self.save_processed()
        self.logger.info(f"Preprocessing completed. Final shape: {self.df.shape}")
        return self.df


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_path = os.path.join(project_root, "data", "raw", "SPAS-Dataset-BD.csv")
    save_path = os.path.join(project_root, "data", "processed", "processed_data.csv")
    log_dir = os.path.join(project_root, "reports", "logs")
    df = pd.read_csv(data_path)
    preprocessor = Preprocessing(df=df, save_path=save_path, log_dir=log_dir)
    processed_df = preprocessor.run()
    print(f"\nPreprocessing complete. Shape: {processed_df.shape}")
