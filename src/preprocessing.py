import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

class Preprocessing:
    """
    Preprocessing pipeline for the agricultural dataset.
    Handles:
        - Column formatting and cleaning
        - Month string cleaning and vectorization for growth/harvest periods
        - Numeric conversions and feature creation (e.g., AP ratio)
        - Label encoding for categorical features
        - Saving the processed dataset to disk
    """

    def __init__(self, df: pd.DataFrame, save_path: str = "./data/processed/processed_data.csv", log_dir: str = "logs"):
        """
        Initializes the Preprocessing pipeline.

        Args:
            df (pd.DataFrame): The raw input DataFrame.
            save_path (str): The path to save the processed CSV file.
            log_dir (str): The directory to store log files.
        """
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Logger setup
        self.logger = logging.getLogger("Preprocessing")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            fh = logging.FileHandler(os.path.join(log_dir, "preprocessing.log"))
            ch.setLevel(logging.INFO)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

        self.df = df.copy()
        self.save_path = save_path
        self.logger.info("Preprocessing pipeline initialized.")

        # Month mapping configuration
        self.months = ['january', 'february', 'march', 'april', 'may', 'june',
                       'july', 'august', 'september', 'october', 'november', 'december']
        self.month_to_idx = {m: i for i, m in enumerate(self.months)}
        self.month_abbr_map = {
            'jan': 'january', 'feb': 'february', 'mar': 'march', 'apr': 'april', 'may': 'may',
            'jun': 'june', 'jul': 'july', 'aug': 'august', 'sep': 'september',
            'oct': 'october', 'nov': 'november', 'dec': 'december'
        }

    def clean_columns(self):
        """Cleans column names and categorical string columns."""
        self.logger.info("Cleaning column names and categorical string values.")
        self.df.columns = self.df.columns.str.strip().str.replace(' ', '_').str.lower()
        for col in ['season', 'crop_name', 'district']:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.strip().str.lower()

    def _clean_month_string(self, s):
        """Standardizes month names from abbreviations to full names."""
        s = str(s).strip().lower()
        for abbr, full in self.month_abbr_map.items():
            s = re.sub(fr'\\b{abbr}\\b', full, s)
        return s

    def _month_range_to_vector(self, s):
        """Converts a month range string (e.g., 'jun to oct') to a 12-element binary vector."""
        s = self._clean_month_string(s)
        split = re.split(r'\\s+to\\s+', s)
        vec = [0] * 12
        if len(split) == 2:
            start, end = split
            if start in self.month_to_idx and end in self.month_to_idx:
                i1, i2 = self.month_to_idx[start], self.month_to_idx[end]
                if i1 <= i2:
                    for i in range(i1, i2 + 1): vec[i] = 1
                else:  # Handles year-wraps like 'nov to feb'
                    for i in range(i1, 12): vec[i] = 1
                    for i in range(0, i2 + 1): vec[i] = 1
        return vec

    def process_growth_harvest(self):
        """Processes 'growth' and 'harvest' month ranges into binary vectors."""
        self.logger.info("Processing growth and harvest months into feature vectors.")
        growth_vectors = self.df['growth'].apply(self._month_range_to_vector).to_list()
        harvest_vectors = self.df['harvest'].apply(self._month_range_to_vector).to_list()

        growth_df = pd.DataFrame(growth_vectors, columns=[f'growth_{m[:3]}' for m in self.months])
        harvest_df = pd.DataFrame(harvest_vectors, columns=[f'harvest_{m[:3]}' for m in self.months])

        self.df = pd.concat([self.df.reset_index(drop=True), growth_df, harvest_df], axis=1)
        self.df.drop(columns=['growth', 'harvest'], inplace=True)

    def process_transplant(self):
        """Cleans and encodes the 'transplant' month."""
        self.logger.info("Cleaning and encoding transplant month.")
        self.df['transplant'] = self.df['transplant'].apply(self._clean_month_string)
        self.df['transplant_month'] = self.df['transplant'].apply(
            lambda x: self.month_to_idx.get(x, np.nan)
        )
        self.df.drop(columns=['transplant'], inplace=True)

    def convert_numeric_and_handle_missing(self):
        """Converts columns to numeric types, calculates AP ratio, and handles missing values."""
        self.logger.info("Converting data types, creating AP ratio, and handling missing values.")
        self.df['area'] = pd.to_numeric(self.df['area'], errors='coerce')
        self.df['production'] = pd.to_numeric(self.df['production'], errors='coerce')

        # Drop rows where area or production could not be converted
        self.df.dropna(subset=['area', 'production'], inplace=True)
        self.df['area'] = self.df['area'].astype(int)
        self.df['production'] = self.df['production'].astype(int)
        
        # Calculate AP ratio, handle division by zero
        self.df['ap_ratio'] = (self.df['area'] / self.df['production'].replace(0, np.nan)).astype(float)

        weather_cols = ['avg_temp', 'min_temp', 'max_temp', 'avg_humidity',
                        'min_relative_humidity', 'max_relative_humidity']
        self.df[weather_cols] = self.df[weather_cols].apply(pd.to_numeric, errors='coerce')

    def encode_categories(self):
        """Encodes categorical columns using LabelEncoder and cleans invalid rows."""
        self.logger.info("Encoding categorical columns with LabelEncoder.")
        # Remove invalid or placeholder rows
        self.df = self.df[self.df['crop_name'] != '#ref!'].copy()
        self.df.dropna(subset=['season'], inplace=True)

        # Label encoding
        self.df['crop_name_enc'] = LabelEncoder().fit_transform(self.df['crop_name'])
        self.df['season_enc'] = LabelEncoder().fit_transform(self.df['season'])
        self.df['district_enc'] = LabelEncoder().fit_transform(self.df['district'])

        self.df.drop(columns=['crop_name', 'season', 'district'], inplace=True)

    def save_processed(self):
        """Saves the processed DataFrame to a CSV file."""
        try:
            self.df.to_csv(self.save_path, index=False)
            self.logger.info(f"Processed dataset saved successfully to: {self.save_path}")
        except Exception as e:
            self.logger.exception(f"Error saving processed data: {str(e)}")
            raise

    def run(self):
        """Executes the full preprocessing pipeline in order."""
        self.logger.info("Starting preprocessing run.")
        self.clean_columns()
        self.process_growth_harvest()
        self.process_transplant()
        self.convert_numeric_and_handle_missing()
        self.encode_categories()
        self.save_processed()
        self.logger.info(f"Preprocessing completed. Final DataFrame shape: {self.df.shape}")
        return self.df


if __name__ == "__main__":
    try:
        raw_data_path = "./data/raw/SPAS-Dataset-BD.csv"
        processed_data_path = "./data/processed/processed_data.csv"
        
        df_raw = pd.read_csv(raw_data_path)
        
        preprocessor = Preprocessing(df=df_raw, save_path=processed_data_path, log_dir="./logs")
        processed_df = preprocessor.run()
        
        print("\nPreprocessing complete.")

    except FileNotFoundError:
        logging.error(f"The file was not found at {raw_data_path}. Please ensure the path is correct.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")