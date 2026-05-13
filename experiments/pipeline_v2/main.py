"""
main.py - Main orchestration pipeline combining all modules
Orchestrates the complete ML pipeline:
1. Data Loading
2. Preprocessing
3. Encoding
4. Feature Engineering
5. Feature Selection
6. Data Augmentation
7. Model Training
8. Model Evaluation
9. Model Explainability
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Import all modules
from setup import get_config, HAS_SDV, HAS_CTGAN
from data_loading import load_raw_data, create_working_copy
from preprocessing import preprocess_data
from encoding import encode_data, CategoricalEncoder, create_supervised_targets
from feature_engineering import engineer_features
from feature_selection import select_features
from data_augmentation import DataAugmentor, random_oversampling
from models import ModelTrainer
from model_evaluation import ModelEvaluator
from explainability import SHAPExplainer, FeatureImportanceComparison


class CropRecommendationPipeline:
    """Complete ML pipeline orchestrator."""
    
    def __init__(self, config: dict = None):
        """Initialize pipeline with configuration."""
        self.config = config or get_config()
        self.data = {}
        self.models = {}
        self.encoder = None
        self.trainer = ModelTrainer(random_state=self.config['random_state'])
        self.evaluator = ModelEvaluator(random_state=self.config['random_state'])
    
    def step_1_load_data(self, csv_path: str = None) -> pd.DataFrame:
        """Step 1: Load raw data."""
        print("\n" + "="*80)
        print("STEP 1: LOADING DATA")
        print("="*80)
        
        df = load_raw_data(csv_path) if csv_path else load_raw_data()
        df_prep, df_eda = create_working_copy(df)
        
        self.data['raw'] = df
        self.data['prep'] = df_prep
        self.data['eda'] = df_eda
        
        print(f"✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        return df_prep
    
    def step_2_preprocess(self) -> pd.DataFrame:
        """Step 2: Preprocess data."""
        print("\n" + "="*80)
        print("STEP 2: PREPROCESSING")
        print("="*80)
        
        df = self.data['prep']
        df_prep = preprocess_data(df)
        
        self.data['preprocessed'] = df_prep
        print(f"✓ Preprocessed to {df_prep.shape[0]} rows, {df_prep.shape[1]} columns")
        return df_prep
    
    def step_3_encode(self) -> tuple:
        """Step 3: Encode categorical features."""
        print("\n" + "="*80)
        print("STEP 3: ENCODING")
        print("="*80)
        
        df = self.data['preprocessed']
        df_enc, encoder = encode_data(df)
        
        self.data['encoded'] = df_enc
        self.encoder = encoder
        
        print(f"✓ Encoded data: {df_enc.shape}")
        print(f"✓ Target classes: {df_enc['crop_name_enc'].nunique()}")
        return df_enc, encoder
    
    def step_4_feature_engineering(self) -> pd.DataFrame:
        """Step 4: Engineer features."""
        print("\n" + "="*80)
        print("STEP 4: FEATURE ENGINEERING")
        print("="*80)
        
        df = self.data['encoded']
        df_eng = engineer_features(df)
        
        self.data['engineered'] = df_eng
        print(f"✓ Created engineered features: {df_eng.shape[1] - df.shape[1]} new features")
        return df_eng
    
    def step_5_feature_selection(self, method: str = 'ensemble', top_k: float = 0.8) -> list:
        """Step 5: Select important features."""
        print("\n" + "="*80)
        print("STEP 5: FEATURE SELECTION")
        print("="*80)
        
        df = self.data['engineered']
        X, y = create_supervised_targets(df)
        
        selected_features = select_features(X, y, method=method, top_k=top_k)
        
        self.data['selected_features'] = selected_features
        print(f"✓ Selected {len(selected_features)} features (from {X.shape[1]})")
        print(f"  Top 10: {selected_features[:10]}")
        
        return selected_features
    
    def step_6_data_augmentation(self, augmentation_factor: float = 1.0,
                                 method: str = 'oversample') -> tuple:
        """Step 6: Augment data."""
        print("\n" + "="*80)
        print("STEP 6: DATA AUGMENTATION")
        print("="*80)
        
        df = self.data['engineered']
        X, y = create_supervised_targets(df)
        selected_features = self.data['selected_features']
        
        # Select features
        X = X[selected_features]
        
        if method == 'oversample':
            X_aug, y_aug = random_oversampling(X, y)
        else:
            augmentor = DataAugmentor(method=method)
            X_aug, y_aug = augmentor.augment_dataset(X, y, augmentation_factor)
        
        self.data['X_augmented'] = X_aug
        self.data['y_augmented'] = y_aug
        
        print(f"✓ Original shape: {X.shape}")
        print(f"✓ Augmented shape: {X_aug.shape}")
        print(f"  Class distribution: {dict(zip(*np.unique(y_aug, return_counts=True)))}")
        
        return X_aug, y_aug
    
    def step_7_train_models(self, models: list = None, test_size: float = 0.2,
                           hp_search: bool = False, cv_folds: int = 5) -> dict:
        """Step 7: Train models."""
        print("\n" + "="*80)
        print("STEP 7: MODEL TRAINING")
        print("="*80)
        
        X = self.data['X_augmented']
        y = self.data['y_augmented']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config['random_state'],
            stratify=y
        )
        
        self.data['X_train'] = X_train
        self.data['X_test'] = X_test
        self.data['y_train'] = y_train
        self.data['y_test'] = y_test
        
        print(f"✓ Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        
        # Train models
        if models is None:
            models = ['RandomForest', 'LightGBM', 'XGBoost']
        
        results = self.trainer.train_all_models(
            X_train, y_train, models=models,
            hp_search=hp_search, cv_folds=cv_folds
        )
        
        self.models['train_results'] = results
        print(f"✓ Trained {len(results)} models")
        
        return results
    
    def step_8_evaluate_models(self) -> pd.DataFrame:
        """Step 8: Evaluate models."""
        print("\n" + "="*80)
        print("STEP 8: MODEL EVALUATION")
        print("="*80)
        
        trained_models = {name: res['model'] for name, res in self.models['train_results'].items()}
        X_test = self.data['X_test']
        y_test = self.data['y_test']
        
        # Evaluate
        eval_results = self.evaluator.evaluate_multiple_models(trained_models, X_test, y_test)
        
        self.models['evaluation'] = eval_results
        print("\n" + eval_results.to_string())
        
        return eval_results
    
    def step_9_model_explainability(self, model_name: str = None) -> dict:
        """Step 9: Generate model explanations."""
        print("\n" + "="*80)
        print("STEP 9: MODEL EXPLAINABILITY")
        print("="*80)
        
        if model_name is None:
            best_model_name, _ = self.trainer.get_best_model()
            model_name = best_model_name
        
        model = self.trainer.models[model_name]
        X_test = self.data['X_test']
        
        print(f"✓ Explaining model: {model_name}")
        
        # Feature importance from permutation
        try:
            from sklearn.inspection import permutation_importance
            perm_imp = FeatureImportanceComparison.permutation_importance(
                model, X_test, self.data['y_test'], n_repeats=5
            )
            print(f"\nTop 5 Important Features (Permutation):")
            print(perm_imp.head(5).to_string())
            
            self.models['permutation_importance'] = perm_imp
        except Exception as e:
            print(f"  Warning: Could not compute permutation importance: {e}")
        
        # SHAP if available
        try:
            explainer = SHAPExplainer(model, self.data['X_train'].sample(100))
            shap_imp = explainer.get_feature_importance(X_test, top_k=10)
            print(f"\nTop 10 Important Features (SHAP):")
            print(shap_imp.to_string())
            
            self.models['shap_importance'] = shap_imp
        except Exception as e:
            print(f"  Warning: SHAP not available: {e}")
        
        return self.models
    
    def run_full_pipeline(self, csv_path: str = None, models: list = None,
                         test_size: float = 0.2, augmentation_factor: float = 1.0,
                         hp_search: bool = False, cv_folds: int = 5) -> dict:
        """
        Run complete pipeline from start to finish.
        
        Args:
            csv_path: Path to CSV data file
            models: List of models to train
            test_size: Test set fraction
            augmentation_factor: Data augmentation ratio
            hp_search: Whether to perform hyperparameter search
            cv_folds: Cross-validation folds
            
        Returns:
            Dictionary with all results
        """
        print("\n" + "#"*80)
        print("# CROP RECOMMENDATION ML PIPELINE v2")
        print("#"*80)
        
        # Execute pipeline steps
        self.step_1_load_data(csv_path)
        self.step_2_preprocess()
        self.step_3_encode()
        self.step_4_feature_engineering()
        self.step_5_feature_selection()
        self.step_6_data_augmentation(augmentation_factor=augmentation_factor)
        self.step_7_train_models(models=models, test_size=test_size,
                               hp_search=hp_search, cv_folds=cv_folds)
        eval_results = self.step_8_evaluate_models()
        self.step_9_model_explainability()
        
        # Summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETED")
        print("="*80)
        best_model_name, best_model = self.trainer.get_best_model()
        print(f"✓ Best model: {best_model_name}")
        print(f"✓ Balanced Accuracy: {eval_results.loc[best_model_name, 'balanced_accuracy']:.4f}")
        
        return {
            'trainer': self.trainer,
            'evaluator': self.evaluator,
            'encoder': self.encoder,
            'data': self.data,
            'models': self.models,
            'evaluation': eval_results,
        }
    
    def save_artifacts(self, output_dir: str = 'artifacts'):
        """Save trained models and encoders."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save encoder
        self.encoder.save(f"{output_dir}/encoder.pkl")
        
        # Save trained models
        for model_name, model in self.trainer.models.items():
            import joblib
            joblib.dump(model, f"{output_dir}/{model_name}.pkl")
        
        print(f"\n✓ Artifacts saved to {output_dir}/")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Crop Recommendation ML Pipeline')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--models', nargs='+', default=['RandomForest', 'LightGBM', 'XGBoost'],
                       help='Models to train')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--augmentation', type=float, default=1.0, help='Augmentation factor')
    parser.add_argument('--hp-search', action='store_true', help='Hyperparameter search')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    parser.add_argument('--output', type=str, default='artifacts', help='Output directory')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = CropRecommendationPipeline()
    results = pipeline.run_full_pipeline(
        csv_path=args.data,
        models=args.models,
        test_size=args.test_size,
        augmentation_factor=args.augmentation,
        hp_search=args.hp_search,
        cv_folds=args.cv_folds
    )
    
    # Save artifacts
    pipeline.save_artifacts(args.output)
    
    return results


if __name__ == "__main__":
    # Run as script
    results = main()
