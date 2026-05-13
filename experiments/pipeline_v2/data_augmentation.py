"""
data_augmentation.py - Generate synthetic data using TVAE and CTGAN
Extracted from: Data augmentation section of CropRecomV2.ipynb
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

try:
    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    HAS_SDV = True
except ImportError:
    HAS_SDV = False

try:
    from ctgan import CTGAN
    HAS_CTGAN = True
except ImportError:
    HAS_CTGAN = False


class DataAugmentor:
    """Handle data augmentation using synthetic data generation."""
    
    def __init__(self, method: str = 'tvae'):
        """
        Initialize augmentor.
        
        Args:
            method: 'tvae', 'ctgan', or 'random'
        """
        if method == 'tvae' and not HAS_SDV:
            raise ImportError("SDV not installed. Install with: pip install sdv")
        if method == 'ctgan' and not HAS_CTGAN:
            raise ImportError("CTGAN not installed. Install with: pip install ctgan")
        
        self.method = method
        self.generator = None
        self.scaler = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        """
        Fit the data generator on training data.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            
        Returns:
            self
        """
        # Combine X and y if provided
        if y is not None:
            df = X.copy()
            df['target'] = y
        else:
            df = X.copy()
        
        if self.method == 'tvae':
            self._fit_tvae(df)
        elif self.method == 'ctgan':
            self._fit_ctgan(df)
        elif self.method == 'random':
            self.scaler = StandardScaler()
            self.scaler.fit(df.select_dtypes(include=[np.number]))
        
        self.is_fitted = True
        return self
    
    def _fit_tvae(self, df: pd.DataFrame):
        """Fit TVAE synthesizer."""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        self.generator = TVAESynthesizer(metadata)
        self.generator.fit(df)
    
    def _fit_ctgan(self, df: pd.DataFrame):
        """Fit CTGAN synthesizer."""
        # CTGAN expects numeric data
        discrete_cols = df.select_dtypes(include=['int', 'int64']).columns.tolist()
        self.generator = CTGAN(epochs=300)
        self.generator.fit(df, discrete_columns=discrete_cols)
    
    def generate(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic data.
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            DataFrame with synthetic data
        """
        if not self.is_fitted:
            raise ValueError("Augmentor not fitted. Call .fit() first.")
        
        if self.method == 'tvae':
            synthetic_df = self.generator.sample(n_samples)
        elif self.method == 'ctgan':
            synthetic_df = self.generator.sample(n_samples)
        elif self.method == 'random':
            # Generate random noise and transform
            numeric_cols = self.scaler.get_feature_names_out() if hasattr(
                self.scaler, 'get_feature_names_out') else []
            synthetic_df = pd.DataFrame(
                self.scaler.inverse_transform(np.random.randn(n_samples, len(numeric_cols))),
                columns=numeric_cols
            )
        
        return synthetic_df
    
    def augment_dataset(self, X: pd.DataFrame, y: np.ndarray,
                       augmentation_factor: float = 1.0) -> tuple:
        """
        Augment dataset by adding synthetic samples.
        
        Args:
            X: Feature matrix
            y: Target variable
            augmentation_factor: Ratio of synthetic to original samples (1.0 = 100% augmentation)
            
        Returns:
            Tuple of (X_augmented, y_augmented)
        """
        if augmentation_factor < 0:
            raise ValueError("augmentation_factor must be >= 0")
        
        if augmentation_factor == 0:
            return X.copy(), y.copy()
        
        # Fit augmentor
        self.fit(X, y)
        
        # Generate synthetic samples
        n_synthetic = int(len(X) * augmentation_factor)
        if n_synthetic > 0:
            synthetic_data = self.generate(n_synthetic)
            
            # Ensure same columns
            if 'target' in synthetic_data.columns:
                y_synthetic = synthetic_data['target'].values
                X_synthetic = synthetic_data.drop(columns=['target'])
            else:
                X_synthetic = synthetic_data
                y_synthetic = np.random.choice(y, n_synthetic)
            
            # Combine original and synthetic
            X_augmented = pd.concat([X, X_synthetic], axis=0, ignore_index=True)
            y_augmented = np.concatenate([y, y_synthetic])
        else:
            X_augmented = X.copy()
            y_augmented = y.copy()
        
        return X_augmented, y_augmented


def random_oversampling(X: pd.DataFrame, y: np.ndarray,
                       sampling_strategy: str = 'all') -> tuple:
    """
    Simple random oversampling of minority classes.
    
    Args:
        X: Feature matrix
        y: Target variable
        sampling_strategy: 'all', 'minority', or float (ratio of minority to majority)
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    from sklearn.utils import class_weight
    
    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_count = class_counts.max()
    
    X_resampled = [X.copy()]
    y_resampled = [y.copy()]
    
    for class_label, count in zip(unique_classes, class_counts):
        if count < max_count:
            # Need to oversample this class
            class_indices = np.where(y == class_label)[0]
            n_synthetic = max_count - count
            
            synthetic_indices = np.random.choice(class_indices, size=n_synthetic)
            X_resampled.append(X.iloc[synthetic_indices])
            y_resampled.append(np.full(n_synthetic, class_label))
    
    X_final = pd.concat(X_resampled, ignore_index=True)
    y_final = np.concatenate(y_resampled)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(y_final))
    return X_final.iloc[shuffle_idx], y_final[shuffle_idx]


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
    
    print(f"Original shape: {X.shape}, Target distribution: {np.unique(y, return_counts=True)}")
    
    # Try random oversampling
    X_aug, y_aug = random_oversampling(X, y)
    print(f"Augmented shape: {X_aug.shape}, Target distribution: {np.unique(y_aug, return_counts=True)}")
    
    # Try TVAE if available
    if HAS_SDV:
        augmentor = DataAugmentor(method='tvae')
        X_aug2, y_aug2 = augmentor.augment_dataset(X, y, augmentation_factor=0.5)
        print(f"TVAE augmented shape: {X_aug2.shape}")
