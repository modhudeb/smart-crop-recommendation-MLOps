"""
data_loading.py - Load raw data and create working copies
Extracted from: Loading Data section of CropRecomV2.ipynb
"""

import pandas as pd
from pathlib import Path


def load_raw_data(csv_path: str = 'data/raw/SPAS-Dataset-BD.csv') -> pd.DataFrame:
    """
    Load raw CSV data.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with raw data
    """
    if not Path(csv_path).exists():
        # Try alternative paths
        alt_paths = [
            'SPAS-Dataset-BD.csv',
            'miscellaneous/SPAS-Dataset-BD.csv',
            'data/raw/SPAS-Dataset-BD.csv',
        ]
        for alt_path in alt_paths:
            if Path(alt_path).exists():
                csv_path = alt_path
                break
    
    df = pd.read_csv(csv_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def create_working_copy(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create working copies: one for preprocessing, one for EDA.
    Keep raw dataset unchanged.
    
    Args:
        raw_df: Raw dataframe from CSV
        
    Returns:
        Tuple of (df_preprocessing, df_eda)
    """
    df_preprocessing = raw_df.copy(deep=True)
    df_eda = raw_df.copy(deep=True)
    return df_preprocessing, df_eda


if __name__ == "__main__":
    # Test data loading
    df = load_raw_data()
    df_prep, df_eda = create_working_copy(df)
    print(f"\nPreprocessing copy shape: {df_prep.shape}")
    print(f"EDA copy shape: {df_eda.shape}")
