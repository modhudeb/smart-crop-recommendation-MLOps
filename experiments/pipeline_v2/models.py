"""
models.py - Model training and hyperparameter tuning
Extracted from: Modeling sections of CropRecomV2.ipynb
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and manage multiple models."""
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.scalers = {}
        self.cv_results = {}
    
    def get_logistic_regression(self) -> dict:
        """Get Logistic Regression model and params."""
        return {
            'model': LogisticRegression(random_state=self.random_state, max_iter=1000,
                                       n_jobs=self.n_jobs),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear'],
            },
            'needs_scaler': True,
        }
    
    def get_random_forest(self) -> dict:
        """Get Random Forest model and params."""
        return {
            'model': RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
            'needs_scaler': False,
        }
    
    def get_lightgbm(self) -> dict:
        """Get LightGBM model and params."""
        return {
            'model': LGBMClassifier(random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, 20],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 100],
            },
            'needs_scaler': False,
        }
    
    def get_xgboost(self) -> dict:
        """Get XGBoost model and params."""
        return {
            'model': XGBClassifier(random_state=self.random_state, n_jobs=self.n_jobs, verbosity=0),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 1.0],
            },
            'needs_scaler': False,
        }
    
    def get_mlp(self) -> dict:
        """Get MLP Neural Network and params."""
        return {
            'model': MLPClassifier(random_state=self.random_state, max_iter=500),
            'params': {
                'hidden_layer_sizes': [(100,), (100, 50), (200, 100), (300, 150)],
                'learning_rate': ['constant', 'adaptive'],
                'alpha': [0.0001, 0.001, 0.01],
            },
            'needs_scaler': True,
        }
    
    def get_catboost(self) -> dict:
        """Get CatBoost model and params."""
        return {
            'model': CatBoostClassifier(random_state=self.random_state, verbose=0),
            'params': {
                'iterations': [100, 200, 300],
                'depth': [5, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1],
            },
            'needs_scaler': False,
        }
    
    def get_all_models(self) -> dict:
        """Get all available models."""
        return {
            'LogisticRegression': self.get_logistic_regression(),
            'RandomForest': self.get_random_forest(),
            'LightGBM': self.get_lightgbm(),
            'XGBoost': self.get_xgboost(),
            'MLP': self.get_mlp(),
            'CatBoost': self.get_catboost(),
        }
    
    def train_model(self, X: pd.DataFrame, y: np.ndarray, model_name: str,
                   hp_search: bool = True, n_iter: int = 20,
                   cv_folds: int = 5) -> tuple:
        """
        Train a single model with optional hyperparameter search.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_name: Name of model to train
            hp_search: Whether to perform hyperparameter search
            n_iter: Number of iterations for RandomizedSearchCV
            cv_folds: Number of CV folds
            
        Returns:
            Tuple of (trained_model, cv_scores)
        """
        if model_name not in self.get_all_models():
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.get_all_models()[model_name]
        model = model_config['model']
        needs_scaler = model_config['needs_scaler']
        
        # Scale if needed
        X_train = X.copy()
        if needs_scaler:
            scaler = StandardScaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            self.scalers[model_name] = scaler
        
        # Hyperparameter search
        if hp_search and 'params' in model_config:
            searcher = RandomizedSearchCV(
                model,
                model_config['params'],
                n_iter=n_iter,
                cv=StratifiedKFold(cv_folds, shuffle=True, random_state=self.random_state),
                n_jobs=self.n_jobs,
                scoring='balanced_accuracy',
                random_state=self.random_state
            )
            searcher.fit(X_train, y)
            model = searcher.best_estimator_
            cv_scores = searcher.cv_results_['mean_test_score']
        else:
            # Just train with CV
            model.fit(X_train, y)
            skf = StratifiedKFold(cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(
                model, X_train, y, cv=skf, scoring='balanced_accuracy'
            )
        
        self.models[model_name] = model
        self.cv_results[model_name] = cv_scores
        
        return model, cv_scores
    
    def train_all_models(self, X: pd.DataFrame, y: np.ndarray,
                        models: list = None, hp_search: bool = True,
                        cv_folds: int = 5) -> dict:
        """
        Train multiple models.
        
        Args:
            X: Feature matrix
            y: Target variable
            models: List of model names to train (None = all)
            hp_search: Whether to perform hyperparameter search
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with results
        """
        if models is None:
            models = list(self.get_all_models().keys())
        
        results = {}
        for model_name in models:
            print(f"Training {model_name}...")
            try:
                model, cv_scores = self.train_model(
                    X, y, model_name,
                    hp_search=hp_search,
                    cv_folds=cv_folds
                )
                results[model_name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores
                }
                print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            except Exception as e:
                print(f"  Error: {e}")
        
        return results
    
    def get_best_model(self) -> tuple:
        """Get the best trained model."""
        if not self.models:
            raise ValueError("No models trained yet")
        
        best_name = max(self.cv_results, key=lambda x: self.cv_results[x].mean())
        return best_name, self.models[best_name]
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        X_pred = X.copy()
        if model_name in self.scalers:
            X_pred = pd.DataFrame(
                self.scalers[model_name].transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return self.models[model_name].predict(X_pred)
    
    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        X_pred = X.copy()
        if model_name in self.scalers:
            X_pred = pd.DataFrame(
                self.scalers[model_name].transform(X),
                columns=X.columns,
                index=X.index
            )
        
        model = self.models[model_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_pred)
        else:
            return model.decision_function(X_pred)


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
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(X, y, models=['RandomForest', 'LightGBM'],
                                       hp_search=False, cv_folds=3)
    
    best_name, best_model = trainer.get_best_model()
    print(f"\nBest model: {best_name}")
