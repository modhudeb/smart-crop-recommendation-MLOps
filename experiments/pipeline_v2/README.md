# Crop Recommendation ML Pipeline v2

Complete modular machine learning pipeline for crop recommendation system. Extracted from the CropRecomV2.ipynb notebook and organized into reusable Python modules.

## Project Structure

```
pipelines_v2/
├── __init__.py                 # Package initialization
├── setup.py                    # Import all libraries and configuration
├── data_loading.py             # Data loading from CSV
├── preprocessing.py            # Data cleaning and transformation
├── encoding.py                 # Categorical encoding
├── feature_engineering.py      # Feature creation and derivation
├── feature_selection.py        # Feature importance and selection
├── data_augmentation.py        # Synthetic data generation
├── models.py                   # Model training and hyperparameter tuning
├── model_evaluation.py         # Model evaluation and comparison
├── explainability.py           # SHAP explainability
├── main.py                     # Pipeline orchestration
└── README.md                   # This file
```

## Modules Description

### 1. **setup.py** - Configuration & Libraries
- Imports all required libraries
- Configuration management
- Feature flags for optional dependencies (SDV, CTGAN, SHAP)

### 2. **data_loading.py** - Data Input
- `load_raw_data()` - Load CSV file
- `create_working_copy()` - Create separate dataframes for preprocessing and EDA

### 3. **preprocessing.py** - Data Cleaning
- Column name normalization
- Month string parsing and one-hot encoding
- Numeric conversions
- Feature creation (area_log, climate_risk_score)
- `preprocess_data()` - Full preprocessing pipeline

### 4. **encoding.py** - Categorical Encoding
- `CategoricalEncoder` - Fit/transform label encoders
- Encode crop names, seasons, districts
- Save/load encoders
- `create_supervised_targets()` - Create X, y for modeling

### 5. **feature_engineering.py** - Derived Features
- Entropy scores from seasonal patterns
- Seasonal concentration metrics
- Climate extremeness features
- Production efficiency ratios
- Interaction features
- `engineer_features()` - Full feature engineering pipeline

### 6. **feature_selection.py** - Feature Importance
- Mutual Information (MI) selection
- F-statistic (ANOVA) selection
- Correlation-based selection
- Ensemble selection (majority voting across methods)
- `select_features()` - Main selection function

### 7. **data_augmentation.py** - Synthetic Data
- TVAE-based augmentation (requires SDV)
- CTGAN-based augmentation (requires CTGAN)
- Random oversampling for imbalanced classes
- `DataAugmentor` - Main augmentation class
- `random_oversampling()` - Simple oversampling

### 8. **models.py** - Model Training
- 6 pre-configured models:
  - Logistic Regression
  - Random Forest
  - LightGBM
  - XGBoost
  - MLP Neural Network
  - CatBoost
- Hyperparameter search with RandomizedSearchCV
- `ModelTrainer` - Main training class
- Prediction and probability estimation

### 9. **model_evaluation.py** - Model Assessment
- Comprehensive metrics (accuracy, precision, recall, F1, balanced accuracy, Cohen's kappa)
- k-Fold cross-validation
- Statistical significance testing (t-test, Wilcoxon)
- Confusion matrix plotting
- Model comparison visualizations
- `ModelEvaluator` - Main evaluation class

### 10. **explainability.py** - Model Interpretation
- SHAP-based feature importance
- Summary plots (bar, dot, violin)
- Force plots and waterfall plots
- Permutation importance
- Feature importance comparison
- `SHAPExplainer` - Main SHAP class

### 11. **main.py** - Pipeline Orchestration
- `CropRecommendationPipeline` - Complete pipeline orchestrator
- 9-step pipeline execution:
  1. Load Data
  2. Preprocess
  3. Encode
  4. Feature Engineering
  5. Feature Selection
  6. Data Augmentation
  7. Model Training
  8. Model Evaluation
  9. Model Explainability
- CLI interface with argument parsing
- Artifact saving

## Quick Start

### Basic Usage

```python
from pipelines_v2.main import CropRecommendationPipeline

# Create pipeline
pipeline = CropRecommendationPipeline()

# Run complete pipeline
results = pipeline.run_full_pipeline(
    csv_path='data/raw/SPAS-Dataset-BD.csv',
    models=['RandomForest', 'LightGBM', 'XGBoost'],
    test_size=0.2,
    augmentation_factor=1.0,
    hp_search=False,
    cv_folds=5
)

# Save artifacts
pipeline.save_artifacts(output_dir='artifacts')
```

### Command Line Usage

```bash
# Run with defaults
python -m pipelines_v2.main

# Run with custom parameters
python -m pipelines_v2.main \
    --data data/raw/SPAS-Dataset-BD.csv \
    --models RandomForest LightGBM XGBoost \
    --test-size 0.2 \
    --augmentation 1.0 \
    --cv-folds 5 \
    --output artifacts
```

### Step-by-Step Usage

```python
from pipelines_v2 import (
    data_loading, preprocessing, encoding,
    feature_engineering, feature_selection,
    data_augmentation, models, model_evaluation
)

# Load data
df = data_loading.load_raw_data('data/raw/SPAS-Dataset-BD.csv')
df_prep, df_eda = data_loading.create_working_copy(df)

# Preprocess
df_prep = preprocessing.preprocess_data(df_prep)

# Encode
df_enc, encoder = encoding.encode_data(df_prep)

# Engineer features
df_eng = feature_engineering.engineer_features(df_enc)

# Select features
X, y = encoding.create_supervised_targets(df_eng)
selected_features = feature_selection.select_features(X, y, method='ensemble')

# Augment data
X_aug, y_aug = data_augmentation.random_oversampling(X[selected_features], y)

# Train models
trainer = models.ModelTrainer()
results = trainer.train_all_models(X_aug, y_aug, models=['RandomForest'])

# Evaluate
evaluator = model_evaluation.ModelEvaluator()
eval_df = evaluator.evaluate_multiple_models(
    {name: res['model'] for name, res in results.items()},
    X_test, y_test
)
```

## Module Dependencies

### Required
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- seaborn

### Optional
- lightgbm (for LightGBM model)
- xgboost (for XGBoost model)
- catboost (for CatBoost model)
- sdv (for TVAE data augmentation)
- ctgan (for CTGAN data augmentation)
- shap (for model explainability)

Install all optional dependencies:
```bash
pip install lightgbm xgboost catboost sdv ctgan shap
```

## Configuration

Default configuration in `setup.py`:
```python
{
    'random_state': 42,
    'cv_folds': 5,
    'test_size': 0.2,
    'n_jobs': -1,  # Use all available CPU cores
}
```

Modify configuration:
```python
from pipelines_v2.main import CropRecommendationPipeline

custom_config = {
    'random_state': 123,
    'cv_folds': 10,
    'test_size': 0.15,
    'n_jobs': 4,
}

pipeline = CropRecommendationPipeline(config=custom_config)
```

## Features

- ✅ **Modular Design** - Each functional component is separate and reusable
- ✅ **Complete Pipeline** - End-to-end ML workflow from data to model explanation
- ✅ **Multiple Models** - 6 different algorithms with hyperparameter tuning
- ✅ **Data Augmentation** - TVAE, CTGAN, and random oversampling
- ✅ **Feature Engineering** - Domain-specific and derived features
- ✅ **Feature Selection** - MI, F-statistic, correlation, and ensemble methods
- ✅ **Model Evaluation** - Comprehensive metrics and statistical testing
- ✅ **Explainability** - SHAP and permutation importance
- ✅ **CLI Interface** - Easy command-line usage
- ✅ **Artifact Saving** - Save models and encoders for deployment

## Pipeline Steps

### Step 1: Data Loading
- Load CSV data
- Create working copies for preprocessing and EDA

### Step 2: Preprocessing
- Normalize column names and text columns
- Parse month ranges to one-hot vectors
- Convert numeric columns
- Create derived features (ap_ratio, area_log, etc.)
- Handle missing values

### Step 3: Encoding
- Fit label encoders for categorical features
- Encode crop names, seasons, districts
- Create transplant month numeric index

### Step 4: Feature Engineering
- Entropy scores from seasonal patterns
- Seasonal concentration metrics
- Climate-based features
- Production efficiency ratios
- Interaction features between variables

### Step 5: Feature Selection
- Compute feature importance using multiple methods
- Ensemble voting to select top features
- Reduce dimensionality while preserving performance

### Step 6: Data Augmentation
- Generate synthetic samples to balance classes
- Use TVAE or CTGAN for realistic synthetic data
- Or use simple random oversampling

### Step 7: Model Training
- Train multiple models with cross-validation
- Hyperparameter tuning using RandomizedSearchCV
- Scale features if needed (for algorithms like LogReg, MLP)

### Step 8: Model Evaluation
- Compute comprehensive metrics
- Compare model performance
- Statistical significance testing
- Generate confusion matrices

### Step 9: Model Explainability
- SHAP values for feature importance
- Permutation importance
- Visualizations and interpretations

## Output Artifacts

After running the pipeline, artifacts are saved to the specified directory:
- `encoder.pkl` - Label encoder for categorical features
- `RandomForest.pkl`, `LightGBM.pkl`, etc. - Trained models
- `.scalers` - StandardScaler if used (internal to models)

## Performance Metrics

Pipeline evaluates models using:
- **Accuracy** - Overall correctness
- **Precision** - Positive prediction accuracy (weighted)
- **Recall** - True positive rate (weighted)
- **F1-Score** - Harmonic mean of precision and recall
- **Balanced Accuracy** - Average of per-class accuracies
- **Cohen's Kappa** - Inter-rater agreement considering chance
- **ROC-AUC** - Area under ROC curve (binary classification)

## Example Output

```
================================================================================
CROP RECOMMENDATION ML PIPELINE v2
================================================================================

================================================================================
STEP 1: LOADING DATA
================================================================================
✓ Loaded 12345 rows, 21 columns

================================================================================
STEP 2: PREPROCESSING
================================================================================
✓ Preprocessed to 12000 rows, 45 columns

================================================================================
STEP 3: ENCODING
================================================================================
✓ Encoded data: (12000, 45)
✓ Target classes: 72

...

================================================================================
STEP 8: MODEL EVALUATION
================================================================================
                     accuracy  precision    recall       f1  balanced_accuracy     kappa
RandomForest         0.850000   0.851234  0.850000  0.849876       0.845000  0.837321
LightGBM             0.862000   0.863456  0.862000  0.861876       0.858000  0.851234
XGBoost              0.875000   0.876543  0.875000  0.874876       0.871000  0.864567

================================================================================
PIPELINE COMPLETED
================================================================================
✓ Best model: XGBoost
✓ Balanced Accuracy: 0.8710

✓ Artifacts saved to artifacts/
```

## Troubleshooting

### Memory Issues
- Reduce `cv_folds` parameter
- Reduce training data size
- Use `hp_search=False` to skip hyperparameter tuning

### Missing Dependencies
Install optional packages:
```bash
pip install lightgbm xgboost catboost sdv ctgan shap
```

### Slow Training
- Reduce number of models
- Set `hp_search=False`
- Reduce `augmentation_factor`
- Use `n_jobs=-1` in configuration for parallel processing

## Contributing

To add new models:
1. Add method in `ModelTrainer.get_all_models()`
2. Include model configuration and hyperparameters
3. Specify if feature scaling is needed

To add new features:
1. Create function in `feature_engineering.py`
2. Call in `engineer_features()` pipeline
3. Update feature names list

## License

Same as parent project

## References

- Original Notebook: `miscellaneous/CropRecomV2.ipynb`
- Documentation: See individual module docstrings
- Configuration: `setup.py`
