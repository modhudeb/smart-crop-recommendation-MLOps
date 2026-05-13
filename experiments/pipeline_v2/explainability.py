"""
explainability.py - Model explainability using SHAP
Extracted from: Model Explainability section of CropRecomV2.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class SHAPExplainer:
    """Use SHAP for model explainability."""
    
    def __init__(self, model, X_background: pd.DataFrame = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            X_background: Background data for SHAP (if None, use small sample)
        """
        if not HAS_SHAP:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        self.model = model
        
        # Use subset for speed
        if X_background is None:
            self.X_background = X_background.sample(min(100, len(X_background)), random_state=42)
        else:
            self.X_background = X_background
        
        # Create explainer
        self.explainer = self._create_explainer()
        self.shap_values = None
    
    def _create_explainer(self):
        """Create appropriate SHAP explainer based on model type."""
        model_name = type(self.model).__name__
        
        # Use TreeExplainer for tree-based models
        if model_name in ['RandomForestClassifier', 'LGBMClassifier', 'XGBClassifier', 'CatBoostClassifier']:
            return shap.TreeExplainer(self.model)
        
        # Use KernelExplainer for others
        else:
            return shap.KernelExplainer(
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                self.X_background
            )
    
    def calculate_shap_values(self, X: pd.DataFrame, check_additivity: bool = False):
        """
        Calculate SHAP values for samples.
        
        Args:
            X: Features to explain
            check_additivity: Whether to check SHAP additivity
            
        Returns:
            SHAP values
        """
        self.shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)
        return self.shap_values
    
    def plot_summary(self, X: pd.DataFrame, plot_type: str = 'bar', figsize: tuple = (12, 8)):
        """
        Plot SHAP summary plots.
        
        Args:
            X: Features to explain
            plot_type: 'bar', 'dot', or 'violin'
            figsize: Figure size
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=figsize)
        
        # Handle multiclass case
        if isinstance(self.shap_values, list):
            # For multiclass, plot for first class
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        if plot_type == 'bar':
            shap.summary_plot(shap_vals, X, plot_type='bar', show=False)
        elif plot_type == 'dot':
            shap.summary_plot(shap_vals, X, plot_type='dot', show=False)
        elif plot_type == 'violin':
            shap.summary_plot(shap_vals, X, plot_type='violin', show=False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_force(self, X: pd.DataFrame, index: int = 0):
        """
        Plot SHAP force plot for a single instance.
        
        Args:
            X: Features
            index: Row index to explain
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Handle multiclass
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        shap.force_plot(
            self.explainer.expected_value if not isinstance(self.explainer.expected_value, list) 
            else self.explainer.expected_value[0],
            shap_vals[index],
            X.iloc[index]
        )
    
    def plot_waterfall(self, X: pd.DataFrame, index: int = 0, figsize: tuple = (12, 8)):
        """
        Plot SHAP waterfall plot (detailed breakdown).
        
        Args:
            X: Features
            index: Row index to explain
            figsize: Figure size
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=figsize)
        
        # Create Explanation object
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[0]
        
        explanation = shap.Explanation(
            values=shap_vals[index],
            base_values=expected_value,
            data=X.iloc[index].values,
            feature_names=X.columns.tolist()
        )
        
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, X: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """
        Get feature importance from SHAP values.
        
        Args:
            X: Features
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance ranking
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Handle multiclass
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        # Calculate mean absolute SHAP value per feature
        importance = np.abs(shap_vals).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_k)


class FeatureImportanceComparison:
    """Compare feature importance from multiple sources."""
    
    @staticmethod
    def permutation_importance(model, X: pd.DataFrame, y: np.ndarray,
                              n_repeats: int = 10) -> pd.DataFrame:
        """
        Calculate permutation importance.
        
        Args:
            model: Trained model
            X: Features
            y: Labels
            n_repeats: Number of permutation iterations
            
        Returns:
            DataFrame with importance scores
        """
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=42
        )
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': result.importances_mean,
            'std': result.importances_std
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    @staticmethod
    def plot_comparison(shap_importance: pd.DataFrame,
                       perm_importance: pd.DataFrame,
                       figsize: tuple = (14, 6)):
        """
        Plot comparison of SHAP vs Permutation importance.
        
        Args:
            shap_importance: DataFrame from SHAP
            perm_importance: DataFrame from permutation importance
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # SHAP importance
        axes[0].barh(shap_importance['feature'][:10], shap_importance['importance'][:10])
        axes[0].set_xlabel('Mean |SHAP value|')
        axes[0].set_title('SHAP Feature Importance')
        
        # Permutation importance
        axes[1].barh(perm_importance['feature'][:10], perm_importance['importance'][:10])
        axes[1].set_xlabel('Permutation Importance')
        axes[1].set_title('Permutation Feature Importance')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if HAS_SHAP:
        from data_loading import load_raw_data, create_working_copy
        from preprocessing import preprocess_data
        from encoding import encode_data, create_supervised_targets
        from feature_engineering import engineer_features
        from models import ModelTrainer
        from sklearn.model_selection import train_test_split
        
        # Build pipeline
        df = load_raw_data()
        df_prep, _ = create_working_copy(df)
        df_prep = preprocess_data(df_prep)
        df_enc, _ = encode_data(df_prep)
        df_eng = engineer_features(df_enc)
        X, y = create_supervised_targets(df_eng)
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        trainer = ModelTrainer()
        trainer.train_model(X_train, y_train, 'RandomForest', hp_search=False)
        model = trainer.models['RandomForest']
        
        # SHAP explanation
        explainer = SHAPExplainer(model, X_train.sample(100))
        shap_imp = explainer.get_feature_importance(X_test)
        print("SHAP Feature Importance:")
        print(shap_imp)
    else:
        print("SHAP not installed. Install with: pip install shap")
