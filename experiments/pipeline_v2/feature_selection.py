"""
feature_selection.py - Select most important features using statistical methods
Extracted from: Feature selection section of CropRecomV2.ipynb
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler


def mutual_information_selection(X: pd.DataFrame, y: np.ndarray, 
                                  n_features: int = None, top_k: float = 0.8) -> list:
    """
    Select features based on mutual information with target.
    
    Args:
        X: Feature matrix
        y: Target variable
        n_features: If specified, select top n features. Otherwise use top_k ratio
        top_k: Fraction of features to keep (0-1)
        
    Returns:
        List of selected feature names
    """
    mi_scores = mutual_info_classif(X, y, random_state=42)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    if n_features is None:
        n_features = max(1, int(len(X.columns) * top_k))
    
    selected = feature_importance.head(n_features)['feature'].tolist()
    return selected


def f_statistic_selection(X: pd.DataFrame, y: np.ndarray,
                         n_features: int = None, top_k: float = 0.8) -> list:
    """
    Select features based on F-statistic (ANOVA).
    
    Args:
        X: Feature matrix
        y: Target variable
        n_features: If specified, select top n features. Otherwise use top_k ratio
        top_k: Fraction of features to keep (0-1)
        
    Returns:
        List of selected feature names
    """
    f_scores, _ = f_classif(X, y)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'f_score': f_scores
    }).sort_values('f_score', ascending=False)
    
    if n_features is None:
        n_features = max(1, int(len(X.columns) * top_k))
    
    selected = feature_importance.head(n_features)['feature'].tolist()
    return selected


def correlation_based_selection(X: pd.DataFrame, y: np.ndarray,
                               n_features: int = None, top_k: float = 0.8) -> list:
    """
    Select features based on correlation with target (for regression-like targets).
    
    Args:
        X: Feature matrix
        y: Target variable (numeric)
        n_features: If specified, select top n features. Otherwise use top_k ratio
        top_k: Fraction of features to keep (0-1)
        
    Returns:
        List of selected feature names
    """
    # Create temporary dataframe with target
    temp_df = X.copy()
    temp_df['target'] = y
    
    correlations = temp_df.corr()['target'].drop('target').abs().sort_values(ascending=False)
    
    if n_features is None:
        n_features = max(1, int(len(X.columns) * top_k))
    
    selected = correlations.head(n_features).index.tolist()
    return selected


def ensemble_selection(X: pd.DataFrame, y: np.ndarray,
                      n_features: int = None, top_k: float = 0.8,
                      methods: list = None) -> list:
    """
    Ensemble feature selection using multiple methods.
    Select features that appear in multiple methods.
    
    Args:
        X: Feature matrix
        y: Target variable
        n_features: If specified, select top n features. Otherwise use top_k ratio
        top_k: Fraction of features to keep (0-1)
        methods: List of selection methods to use ['mi', 'f', 'corr']
        
    Returns:
        List of selected feature names
    """
    if methods is None:
        methods = ['mi', 'f']  # Default: MI and F-statistic
    
    # Get selections from each method
    selections = {}
    if 'mi' in methods:
        selections['mi'] = set(mutual_information_selection(X, y, n_features, top_k))
    if 'f' in methods:
        selections['f'] = set(f_statistic_selection(X, y, n_features, top_k))
    if 'corr' in methods:
        try:
            selections['corr'] = set(correlation_based_selection(X, y, n_features, top_k))
        except:
            pass  # Skip if target is not numeric
    
    # Find features appearing in majority of methods
    all_selected = {}
    for feature in X.columns:
        count = sum(1 for s in selections.values() if feature in s)
        all_selected[feature] = count
    
    # Sort by frequency
    sorted_features = sorted(all_selected.items(), key=lambda x: -x[1])
    
    if n_features is None:
        n_features = max(1, int(len(X.columns) * top_k))
    
    selected = [f for f, _ in sorted_features[:n_features]]
    return selected


def select_features(X: pd.DataFrame, y: np.ndarray,
                   method: str = 'ensemble', n_features: int = None,
                   top_k: float = 0.8) -> list:
    """
    Select features using specified method.
    
    Args:
        X: Feature matrix
        y: Target variable
        method: 'mi', 'f', 'corr', or 'ensemble'
        n_features: Number of features to select
        top_k: Fraction of features to keep
        
    Returns:
        List of selected feature names
    """
    if method == 'mi':
        return mutual_information_selection(X, y, n_features, top_k)
    elif method == 'f':
        return f_statistic_selection(X, y, n_features, top_k)
    elif method == 'corr':
        return correlation_based_selection(X, y, n_features, top_k)
    elif method == 'ensemble':
        return ensemble_selection(X, y, n_features, top_k)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    from data_loading import load_raw_data, create_working_copy
    from preprocessing import preprocess_data
    from encoding import encode_data, create_supervised_targets
    from feature_engineering import engineer_features
    
    # Build pipeline
    df = load_raw_data()
    df_prep, _ = create_working_copy(df)
    df_prep = preprocess_data(df_prep)
    df_enc, _ = encode_data(df_prep)
    df_eng = engineer_features(df_enc)
    X, y = create_supervised_targets(df_eng)
    
    # Select features
    selected = select_features(X, y, method='ensemble', top_k=0.8)
    
    print(f"Total features: {X.shape[1]}")
    print(f"Selected features: {len(selected)}")
    print(f"Selected: {selected[:10]}...")
