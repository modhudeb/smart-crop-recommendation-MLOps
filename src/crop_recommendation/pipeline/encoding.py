import os
import re
import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.preprocessing import LabelEncoder

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
IDX_TO_MONTH = {v: k for k, v in MONTH_TO_IDX.items()}
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


def clean_month_string(s):
    s = str(s).strip().lower()
    for abbr, full in MONTH_ABBR_MAP.items():
        s = re.sub(rf"\b{abbr}\b", full, s)
    return s


def month_range_to_vector(s):
    s = clean_month_string(s)
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


class CategoricalEncoder:
    def __init__(self):
        self.le_crop = LabelEncoder()
        self.le_season = LabelEncoder()
        self.le_district = LabelEncoder()
        self.is_fitted = False

    def fit(self, df):
        df = df.copy()
        df = df[df["crop_name"] != "#ref!"]
        df = df[df["season"].notna()]
        self.le_crop.fit(df["crop_name"])
        self.le_season.fit(df["season"])
        self.le_district.fit(df["district"])
        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call .fit() first.")
        df = df.copy()
        df["crop_name_enc"] = self.le_crop.transform(df["crop_name"])
        df["season_enc"] = self.le_season.transform(df["season"])
        df["district_enc"] = self.le_district.transform(df["district"])
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def inverse_transform_crop(self, values):
        if isinstance(values, (int, np.integer)):
            values = [values]
        return self.le_crop.inverse_transform(values)

    def save(self, filepath):
        joblib.dump(
            {
                "le_crop": self.le_crop,
                "le_season": self.le_season,
                "le_district": self.le_district,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)


def encode_data(df, encoder=None, log_dir="reports/logs"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("Encoding")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join(log_dir, "encoding.log"))
        formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)

    df = df.copy()
    df = df[df["crop_name"] != "#ref!"]
    df = df[df["season"].notna()].copy()

    if encoder is None:
        encoder = CategoricalEncoder()
        encoder.fit(df)

    df = encoder.transform(df)
    df["transplant_month"] = df["transplant"].apply(lambda x: MONTH_TO_IDX[clean_month_string(x)])
    df.drop(columns=["crop_name", "season", "district", "transplant"], inplace=True)

    logger.info(f"Encoded data: {df.shape}, Target classes: {df['crop_name_enc'].nunique()}")
    return df, encoder


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_path = os.path.join(project_root, "data", "processed", "processed_data.csv")
    save_path = os.path.join(project_root, "data", "processed", "encoded_data.csv")
    encoder_path = os.path.join(project_root, "artifacts", "preprocessors", "label_encoders.joblib")
    log_dir = os.path.join(project_root, "reports", "logs")
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    df = pd.read_csv(data_path)
    df_enc, encoder = encode_data(df, log_dir=log_dir)
    df_enc.to_csv(save_path, index=False)
    encoder.save(encoder_path)
    print(f"Encoded shape: {df_enc.shape}")
    print(f"Columns: {list(df_enc.columns)}")
    print(f"Saved to: {save_path}")
