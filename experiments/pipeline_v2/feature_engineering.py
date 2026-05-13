"""
feature_engineering.py - Create derived features for improved model performance
Extracted from: Feature Engineering section of CropRecomV2.ipynb
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from math import log2


def create_entropy_feature(df: pd.DataFrame, col_name: str = 'entropy_score') -> pd.DataFrame:
    """
    Create entropy score based on growth and harvest patterns.
    Higher entropy = more diverse seasonal patterns.
    """
    df = df.copy()
    
    growth_cols = [col for col in df.columns if 'growth_' in col]
    harvest_cols = [col for col in df.columns if 'harvest_' in col]
    
    def calc_entropy(row):
        combined = list(row[growth_cols].values) + list(row[harvest_cols].values)
        combined = np.array(combined) / (np.sum(combined) + 1e-8)
        return entropy(combined, base=2)
    
    df[col_name] = df.apply(calc_entropy, axis=1)
    return df


def create_seasonal_concentration(df: pd.DataFrame, col_name: str = 'seasonal_concentration') -> pd.DataFrame:
    """
    Create seasonal concentration feature.
    High value = concentrated in few months; Low value = spread across many months.
    """
    df = df.copy()
    
    growth_cols = [col for col in df.columns if 'growth_' in col]
    
    def concentration(row):
        growth_pattern = row[growth_cols].values
        n_months = np.sum(growth_pattern > 0)
        if n_months == 0:
            return 0
        return n_months / len(growth_pattern)
    
    df[col_name] = df.apply(concentration, axis=1)
    return df


def create_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived climate features."""
    df = df.copy()
    
    weather_cols = ['avg_temp', 'min_temp', 'max_temp', 'avg_humidity',
                    'min_relative_humidity', 'max_relative_humidity']
    available_cols = [col for col in weather_cols if col in df.columns]
    
    if len(available_cols) >= 2:
        # Temperature range
        temp_cols = [col for col in ['min_temp', 'max_temp'] if col in df.columns]
        if len(temp_cols) == 2:
            df['temp_range'] = df['max_temp'] - df['min_temp']
        
        # Humidity range
        humidity_cols = [col for col in ['min_relative_humidity', 'max_relative_humidity'] 
                        if col in df.columns]
        if len(humidity_cols) == 2:
            df['humidity_range'] = df['max_relative_humidity'] - df['min_relative_humidity']
        
        # Climate extremeness (standardized deviation from mean)
        df['climate_extremeness'] = 0.0
        for col in available_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df['climate_extremeness'] += np.abs((df[col] - mean_val) / std_val)
        df['climate_extremeness'] /= len(available_cols)
    
    return df


def create_production_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Create production efficiency features."""
    df = df.copy()
    
    # Avoid division by zero
    df['production_efficiency'] = np.where(
        df['ap_ratio'] > 0,
        1.0 / df['ap_ratio'],
        0
    )
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between important variables."""
    df = df.copy()
    
    # Temperature-Humidity interaction
    if 'avg_temp' in df.columns and 'avg_humidity' in df.columns:
        df['temp_humidity_interaction'] = df['avg_temp'] * df['avg_humidity']
    
    # Season-Transplant interaction (encoded)
    if 'season_enc' in df.columns and 'transplant_month' in df.columns:
        df['season_transplant_sync'] = (df['season_enc'] == df['transplant_month']).astype(int)
    
    return df


def engineer_features(df: pd.DataFrame, create_all: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    
    Args:
        df: Encoded dataframe
        create_all: If True, create all engineered features
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    if create_all:
        df = create_entropy_feature(df)
        df = create_seasonal_concentration(df)
        df = create_climate_features(df)
        df = create_production_efficiency(df)
        df = create_interaction_features(df)
    
    return df


def get_engineered_feature_names() -> list:
    """Return list of engineered feature names."""
    return [
        'entropy_score',
        'seasonal_concentration',
        'temp_range',
        'humidity_range',
        'climate_extremeness',
        'production_efficiency',
        'temp_humidity_interaction',
        'season_transplant_sync'
    ]


if __name__ == "__main__":
    from data_loading import load_raw_data, create_working_copy
    from preprocessing import preprocess_data
    from encoding import encode_data
    
    # Load, preprocess, encode
    df = load_raw_data()
    df_prep, _ = create_working_copy(df)
    df_prep = preprocess_data(df_prep)
    df_enc, _ = encode_data(df_prep)
    
    # Engineer features
    df_eng = engineer_features(df_enc)
    
    print(f"Shape after feature engineering: {df_eng.shape}")
    print(f"New columns: {list(df_eng.columns[-10:])}")
