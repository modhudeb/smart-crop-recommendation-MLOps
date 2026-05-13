"""
model_evaluation.py - Evaluate and compare model performance
Extracted from: Model Evaluation and Comparison sections of CropRecomV2.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import ttest_rel, wilcoxon
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluate and compare model performance."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = {}
        self.predictions = {}
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         y_prob: np.ndarray = None) -> dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for multiclass ROC)
            
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'kappa': cohen_kappa_score(y_true, y_pred),
        }
        
        # ROC AUC (for binary classification only)
        if len(np.unique(y_true)) == 2 and y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            except:
                pass
        
        return metrics
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: np.ndarray,
                      model_name: str = 'Model') -> dict:
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name for logging
            
        Returns:
            Dictionary with metrics and predictions
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
        
        # Metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_prob)
        
        # Store results
        self.results[model_name] = metrics
        self.predictions[model_name] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
        }
        
        return metrics
    
    def evaluate_multiple_models(self, models: dict, X_test: pd.DataFrame,
                                y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate multiple models.
        
        Args:
            models: Dictionary {model_name: trained_model}
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with results for all models
        """
        for name, model in models.items():
            self.evaluate_model(model, X_test, y_test, name)
        
        return pd.DataFrame(self.results).T
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: np.ndarray,
                            cv_folds: int = 5, metrics: list = None) -> dict:
        """
        Perform k-fold cross-validation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv_folds: Number of folds
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary with CV results
        """
        if metrics is None:
            metrics = ['accuracy', 'f1_weighted', 'balanced_accuracy']
        
        cv_results = {}
        skf = StratifiedKFold(cv_folds, shuffle=True, random_state=self.random_state)
        
        for metric in metrics:
            scores = cross_val_score(model, X, y, cv=skf, scoring=metric, n_jobs=-1)
            cv_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
        
        return cv_results
    
    def statistical_comparison(self, model1_preds: np.ndarray,
                             model2_preds: np.ndarray,
                             y_true: np.ndarray,
                             test_name: str = 'wilcoxon') -> dict:
        """
        Statistical test to compare two models.
        
        Args:
            model1_preds: Predictions from model 1
            model2_preds: Predictions from model 2
            y_true: True labels
            test_name: 'ttest' or 'wilcoxon'
            
        Returns:
            Dictionary with test results
        """
        # Calculate per-sample metrics (correct/incorrect)
        correct1 = (model1_preds == y_true).astype(int)
        correct2 = (model2_preds == y_true).astype(int)
        
        if test_name == 'ttest':
            stat, pval = ttest_rel(correct1, correct2)
        else:  # wilcoxon
            stat, pval = wilcoxon(correct1, correct2)
        
        return {
            'statistic': stat,
            'p_value': pval,
            'significant': pval < 0.05,
            'model1_correct': correct1.sum(),
            'model2_correct': correct2.sum(),
        }
    
    def plot_confusion_matrix(self, model_name: str, figsize: tuple = (12, 10)):
        """Plot confusion matrix for a model."""
        if model_name not in self.predictions:
            raise ValueError(f"No predictions for {model_name}")
        
        preds = self.predictions[model_name]
        cm = confusion_matrix(preds['y_true'], preds['y_pred'])
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure(figsize=figsize)
        disp.plot()
        plt.title(f'Confusion Matrix: {model_name}')
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, metric: str = 'f1', figsize: tuple = (12, 6)):
        """Plot comparison of models on a specific metric."""
        if not self.results:
            raise ValueError("No results to plot")
        
        metrics_df = pd.DataFrame(self.results).T
        
        if metric not in metrics_df.columns:
            raise ValueError(f"Metric {metric} not found")
        
        plt.figure(figsize=figsize)
        metrics_df[metric].sort_values(ascending=False).plot(kind='barh', color='steelblue')
        plt.xlabel(metric.replace('_', ' ').title())
        plt.ylabel('Model')
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
        plt.tight_layout()
        plt.show()
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all evaluated models."""
        return pd.DataFrame(self.results).T


if __name__ == "__main__":
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, 
                                       models=['RandomForest', 'LightGBM'],
                                       hp_search=False, cv_folds=3)
    
    # Evaluate
    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate_multiple_models(
        {name: res['model'] for name, res in results.items()},
        X_test, y_test
    )
    
    print("Evaluation Results:")
    print(eval_results)
