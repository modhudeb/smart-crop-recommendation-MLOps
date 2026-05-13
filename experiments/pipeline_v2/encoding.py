"""
encoding.py - Encode categorical features and create supervised learning targets
Extracted from: Encoding section of CropRecomV2.ipynb
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path


class CategoricalEncoder:
    """Handles encoding and decoding of categorical features."""
    
    def __init__(self):
        self.le_crop = LabelEncoder()
        self.le_season = LabelEncoder()
        self.le_district = LabelEncoder()
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'CategoricalEncoder':
        """Fit label encoders on data."""
        df = df.copy()
        
        # Remove invalid entries before fitting
        df = df[df['crop_name'] != '#ref!']
        df = df[df['season'].notna()]
        
        self.le_crop.fit(df['crop_name'])
        self.le_season.fit(df['season'])
        self.le_district.fit(df['district'])
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encodings to dataframe."""
        if not self.is_fitted:
            raise ValueError("Encoder not fitted. Call .fit() first.")
        
        df = df.copy()
        df['crop_name_enc'] = self.le_crop.transform(df['crop_name'])
        df['season_enc'] = self.le_season.transform(df['season'])
        df['district_enc'] = self.le_district.transform(df['district'])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform_crop(self, encoded_values):
        """Decode crop names."""
        if isinstance(encoded_values, (int, np.integer)):
            encoded_values = [encoded_values]
        return self.le_crop.inverse_transform(encoded_values)
    
    def inverse_transform_season(self, encoded_values):
        """Decode season names."""
        if isinstance(encoded_values, (int, np.integer)):
            encoded_values = [encoded_values]
        return self.le_season.inverse_transform(encoded_values)
    
    def inverse_transform_district(self, encoded_values):
        """Decode district names."""
        if isinstance(encoded_values, (int, np.integer)):
            encoded_values = [encoded_values]
        return self.le_district.inverse_transform(encoded_values)
    
    def save(self, filepath: str):
        """Save encoders to disk."""
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'CategoricalEncoder':
        """Load encoders from disk."""
        return joblib.load(filepath)


def encode_data(df: pd.DataFrame, encoder: CategoricalEncoder = None) -> tuple:
    """
    Encode categorical features and prepare data for modeling.
    
    Args:
        df: Preprocessed dataframe
        encoder: CategoricalEncoder instance (creates new if None)
        
    Returns:
        Tuple of (encoded_df, encoder)
    """
    df = df.copy()
    
    # Remove invalid data
    df = df[df['crop_name'] != '#ref!']
    df = df[df['season'].notna()].copy()
    
    # Create or use provided encoder
    if encoder is None:
        encoder = CategoricalEncoder()
        encoder.fit(df)
    
    # Apply encoding
    df = encoder.transform(df)
    
    # Create transplant_month as numeric
    from preprocessing import MONTH_TO_IDX, clean_month_string
    df['transplant_month'] = df['transplant'].apply(
        lambda x: MONTH_TO_IDX[clean_month_string(x)]
    )
    
    # Drop original categorical columns
    df.drop(columns=['crop_name', 'season', 'district', 'transplant'], inplace=True)
    
    return df, encoder


def create_supervised_targets(df: pd.DataFrame, target_col: str = 'crop_name_enc') -> tuple:
    """
    Create X (features) and y (target) for supervised learning.
    
    Args:
        df: Encoded dataframe
        target_col: Column to use as target (default: crop recommendation)
        
    Returns:
        Tuple of (X, y) where y is the target and X is features
    """
    # Remove rows with missing values in key columns
    df = df.dropna(subset=[target_col])
    
    # Target variable
    y = df[target_col].values
    
    # Features (all except target and seasonal one-hots we might drop)
    X = df.drop(columns=[target_col])
    
    return X, y


def get_feature_names(df: pd.DataFrame, exclude_target: str = 'crop_name_enc') -> list:
    """Get list of feature names after encoding."""
    cols = list(df.columns)
    if exclude_target in cols:
        cols.remove(exclude_target)
    return cols


if __name__ == "__main__":
    from data_loading import load_raw_data, create_working_copy
    from preprocessing import preprocess_data
    
    # Load and preprocess
    df = load_raw_data()
    df_prep, _ = create_working_copy(df)
    df_prep = preprocess_data(df_prep)
    
    # Encode
    df_enc, encoder = encode_data(df_prep)
    
    print(f"Encoded data shape: {df_enc.shape}")
    print(f"Columns: {list(df_enc.columns)}")
    print(f"\nTarget unique values (crops): {df_enc['crop_name_enc'].nunique()}")
    
    # Create X, y
    X, y = create_supervised_targets(df_enc)
    print(f"\nFeature matrix X shape: {X.shape}")
    print(f"Target y shape: {y.shape}")
    print(f"Target unique values: {len(np.unique(y))}")
