"""
preprocessing.py - Data cleaning, transformation, and feature creation
Extracted from: Preprocessing section of CropRecomV2.ipynb
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import skew


# ============================================================================
# Month Configuration and Utilities
# ============================================================================

MONTHS = ['january', 'february', 'march', 'april', 'may', 'june',
          'july', 'august', 'september', 'october', 'november', 'december']

MONTH_TO_IDX = {m: i for i, m in enumerate(MONTHS)}
IDX_TO_MONTH = {v: k for k, v in MONTH_TO_IDX.items()}

MONTH_ABBR_MAP = {
    'jan': 'january', 'feb': 'february', 'mar': 'march', 'apr': 'april', 'may': 'may',
    'jun': 'june', 'jul': 'july', 'aug': 'august', 'sep': 'september',
    'oct': 'october', 'nov': 'november', 'dec': 'december'
}


def clean_month_string(s: str) -> str:
    """Convert abbreviations and normalize month string."""
    s = str(s).strip().lower()
    for abbr, full in MONTH_ABBR_MAP.items():
        s = re.sub(fr'\b{abbr}\b', full, s)
    return s


def month_range_to_vector(s: str) -> list:
    """
    Convert month range string (e.g., "january to march") to one-hot vector.
    """
    s = clean_month_string(s)
    split = re.split(r'\s+to\s+', s)
    vec = [0] * 12
    if len(split) == 2:
        start, end = split
        if start in MONTH_TO_IDX and end in MONTH_TO_IDX:
            i1, i2 = MONTH_TO_IDX[start], MONTH_TO_IDX[end]
            if i1 <= i2:
                for i in range(i1, i2+1):
                    vec[i] = 1
            else:
                # Wrap-around for ranges like "nov to feb"
                for i in range(i1, 12):
                    vec[i] = 1
                for i in range(0, i2+1):
                    vec[i] = 1
    return vec


def month_to_index(m: str) -> int:
    """Convert month string to numeric index (0-11)."""
    m = clean_month_string(m)
    return MONTH_TO_IDX[m] if m in MONTH_TO_IDX else np.nan


# ============================================================================
# Column Cleaning and Normalization
# ============================================================================

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names: lowercase, strip whitespace, replace spaces with underscores
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    return df


def normalize_text_columns(df: pd.DataFrame, text_cols: list = None) -> pd.DataFrame:
    """
    Normalize text columns: strip whitespace and convert to lowercase
    """
    if text_cols is None:
        text_cols = ['season', 'crop_name', 'district', 'transplant']
    
    df = df.copy()
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()
    return df


# ============================================================================
# Feature Creation from Month Ranges
# ============================================================================

def create_month_one_hot_features(df: pd.DataFrame, 
                                   growth_col: str = 'growth',
                                   harvest_col: str = 'harvest') -> pd.DataFrame:
    """
    Create one-hot encoded month vectors for growth and harvest seasons.
    
    Args:
        df: DataFrame with growth and harvest columns
        growth_col: Name of growth column
        harvest_col: Name of harvest column
        
    Returns:
        DataFrame with new columns and original columns removed
    """
    df = df.copy()
    
    # Create vectors for growth and harvest months
    growth_vectors = df[growth_col].apply(month_range_to_vector).tolist()
    harvest_vectors = df[harvest_col].apply(month_range_to_vector).tolist()
    
    # Create DataFrames
    growth_df = pd.DataFrame(growth_vectors, columns=[f'growth_{m[:3]}' for m in MONTHS])
    harvest_df = pd.DataFrame(harvest_vectors, columns=[f'harvest_{m[:3]}' for m in MONTHS])
    
    # Concatenate and drop originals
    df = pd.concat([df, growth_df, harvest_df], axis=1)
    df.drop(columns=[growth_col, harvest_col], inplace=True)
    
    return df


def normalize_transplant_month(df: pd.DataFrame, transplant_col: str = 'transplant') -> pd.DataFrame:
    """Normalize transplant month strings."""
    df = df.copy()
    df[transplant_col] = df[transplant_col].apply(clean_month_string)
    return df


# ============================================================================
# Numeric Feature Conversion and Engineering
# ============================================================================

def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert area and production to numeric; create area/production ratio;
    convert weather columns to float.
    """
    df = df.copy()
    
    # Area and production
    df['area'] = pd.to_numeric(df['area'], errors='coerce')
    df['production'] = pd.to_numeric(df['production'], errors='coerce')
    df['ap_ratio'] = df['area'] / df['production']
    df['ap_ratio'] = df['ap_ratio'].astype(float)
    df['area'] = df['area'].astype(int)
    df['production'] = df['production'].astype(int)
    
    # Weather columns
    weather_cols = ['avg_temp', 'min_temp', 'max_temp', 'avg_humidity', 
                    'min_relative_humidity', 'max_relative_humidity']
    for col in weather_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def create_area_log_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create log-transformed area feature for data augmentation."""
    df = df.copy()
    df['area_log'] = np.log1p(df['area'])
    return df


def create_climate_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic climate risk score based on weather variables.
    Used for augmentation stratification.
    """
    df = df.copy()
    
    weather_cols = ['avg_temp', 'min_temp', 'max_temp', 'avg_humidity']
    available_cols = [col for col in weather_cols if col in df.columns]
    
    if available_cols:
        # Normalize each column to [0, 1]
        normalized = df[available_cols].copy()
        for col in available_cols:
            min_val = normalized[col].min()
            max_val = normalized[col].max()
            if max_val > min_val:
                normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        
        # Average across weather variables
        df['climate_risk_score'] = normalized.mean(axis=1)
    else:
        df['climate_risk_score'] = 0.5
    
    return df


# ============================================================================
# Main Preprocessing Pipeline
# ============================================================================

def preprocess_data(df: pd.DataFrame, drop_invalid: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Normalize column names and text
    2. Create month one-hot features
    3. Convert numeric columns
    4. Clean invalid data
    
    Args:
        df: Raw dataframe
        drop_invalid: Whether to drop rows with invalid crop names and missing seasons
        
    Returns:
        Preprocessed dataframe
    """
    df = df.copy()
    
    # Step 1: Normalize
    df = normalize_column_names(df)
    df = normalize_text_columns(df)
    
    # Step 2: Month features
    df = create_month_one_hot_features(df)
    df = normalize_transplant_month(df)
    
    # Step 3: Numeric conversion
    df = convert_numeric_columns(df)
    
    # Step 4: Clean invalid data
    if drop_invalid:
        df = df[df['crop_name'] != '#ref!']
        df = df[df['season'].notna()].copy()
    
    return df


if __name__ == "__main__":
    from data_loading import load_raw_data, create_working_copy
    
    df = load_raw_data()
    df_prep, _ = create_working_copy(df)
    df_prep = preprocess_data(df_prep)
    
    print(f"Preprocessed shape: {df_prep.shape}")
    print(f"\nColumns: {list(df_prep.columns)}")
    print(f"\nFirst few rows:")
    print(df_prep.head())
