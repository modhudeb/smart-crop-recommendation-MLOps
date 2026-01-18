import os
import pandas as pd
import numpy as np
import logging





class FeatureEngineering:
    """
    Performs feature engineering on the preprocessed dataset.
    Handles:
        - Interaction features between weather and soil
        - Polynomial features for key numeric columns
        - Aggregated features by district
        - Saving the feature-engineered dataset
    """

    def __init__(self, df: pd.DataFrame, save_path: str = "./data/featured/featured_data.csv", log_dir: str = "logs"):
        """
        Initializes the FeatureEngineering pipeline.

        Args:
            df (pd.DataFrame): The preprocessed input DataFrame.
            save_path (str): The path to save the feature-engineered CSV file.
            log_dir (str): The directory to store log files.
        """
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Logger setup
        self.logger = logging.getLogger("FeatureEngineering")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            fh = logging.FileHandler(os.path.join(log_dir, "feature_engineering.log"))
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
        self.logger.info("FeatureEngineering pipeline initialized.")

    def create_weather_interaction_features(self):
        """Creates interaction features from weather data."""
        self.logger.info("Creating weather interaction features.")
        self.df['temp_range'] = self.df['max_temp'] - self.df['min_temp']
        self.df['humidity_range'] = self.df['max_relative_humidity'] - self.df['min_relative_humidity']
        # Interaction between temperature and humidity
        self.df['temp_humidity_interaction'] = self.df['avg_temp'] * self.df['avg_humidity']

    def create_polynomial_features(self):
        """Creates polynomial features for 'area' and 'production'."""
        self.logger.info("Creating polynomial features.")
        self.df['area_sq'] = self.df['area']**2
        self.df['production_sq'] = self.df['production']**2

    def create_district_aggregated_features(self):
        """Creates aggregated features grouped by district."""
        self.logger.info("Creating district-level aggregated features.")
        if 'district_enc' in self.df.columns:
            # Average production per district
            avg_prod_by_district = self.df.groupby('district_enc')['production'].transform('mean')
            self.df['avg_prod_by_district'] = avg_prod_by_district

            # Average area per district
            avg_area_by_district = self.df.groupby('district_enc')['area'].transform('mean')
            self.df['avg_area_by_district'] = avg_area_by_district
        else:
            self.logger.warning("'district_enc' not found. Skipping district-based features.")


    def save_featured_data(self):
        """Saves the DataFrame with new features to a CSV file."""
        try:
            self.df.to_csv(self.save_path, index=False)
            self.logger.info(f"Feature-engineered dataset saved to: {self.save_path}")
        except Exception as e:
            self.logger.exception(f"Error saving featured data: {str(e)}")
            raise

    def run(self):
        """Executes the full feature engineering pipeline."""
        self.logger.info("Starting feature engineering run.")
        # self.create_weather_interaction_features()
        # self.create_polynomial_features()
        # self.create_district_aggregated_features()
        
        # Filling potential NaN values created during feature generation
        self.df.fillna(0, inplace=True)
        
        self.save_featured_data()
        self.logger.info(f"Feature engineering completed. Final DataFrame shape: {self.df.shape}")
        return self.df


if __name__ == "__main__":
    try:
        # Example standalone run
        processed_data_path = "./data/processed/processed_data.csv"
        featured_data_path = "./data/featured/featured_data.csv"
        
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data not found at {processed_data_path}. Please run the preprocessing script first.")

        df_processed = pd.read_csv(processed_data_path)
        
        feature_engineer = FeatureEngineering(df=df_processed, save_path=featured_data_path, log_dir="./logs")
        featured_df = feature_engineer.run()
        
        print("\nFeature engineering complete.")

    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An unexpected error occurred during the feature engineering process: {e}")