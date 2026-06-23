"""Training utilities: metrics, CV loop, TVAE augmentation, and per-model training."""
import os
import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, roc_auc_score,
)
from tqdm.auto import tqdm

try:
    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    HAS_SDV = True
except ImportError:
    HAS_SDV = False


def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'kappa': cohen_kappa_score(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')
        except Exception:
            metrics['roc_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
    return metrics


def stratified_cv_train(model, model_name, X, y, X_synth=None, y_synth=None,
                        n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold_id, (train_idx, val_idx) in enumerate(
        tqdm(skf.split(X, y), total=n_splits, desc=f"[{model_name}] CV", unit="fold"),
        start=1
    ):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if X_synth is not None and y_synth is not None:
            X_tr = np.concatenate([X_tr, X_synth])
            y_tr = np.concatenate([y_tr, y_synth])

        fold_model = clone(model)
        if hasattr(fold_model, "verbose"):
            fold_model.verbose = 0
        if hasattr(fold_model, "random_state"):
            fold_model.random_state = random_state

        fold_model.fit(X_tr, y_tr)
        y_pred = fold_model.predict(X_val)
        y_prob = fold_model.predict_proba(X_val) if hasattr(fold_model, "predict_proba") else None

        metrics = {
            'model': model_name,
            'fold': fold_id,
            'accuracy': accuracy_score(y_val, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_val, y_pred, average='macro', zero_division=0),
            'kappa': cohen_kappa_score(y_val, y_pred),
        }
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_val, y_prob, multi_class='ovr', average='weighted')
            except ValueError:
                metrics['roc_auc'] = np.nan
        else:
            metrics['roc_auc'] = np.nan

        fold_metrics.append(metrics)
        print(f"[{model_name}] Fold {fold_id}/{n_splits} | "
              f"Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | "
              f"AUC: {metrics['roc_auc']:.4f}")

    return pd.DataFrame(fold_metrics)


def run_tvae_augmentation(df_train, target_col, model_save_dir, logger):
    if not HAS_SDV:
        logger.warning("SDV not installed. Skipping TVAE augmentation.")
        feature_cols = [c for c in df_train.columns if c != target_col]
        X = df_train[feature_cols].values
        y = df_train[target_col].astype(int).values
        return X, y, X, y, feature_cols

    logger.info("Running TVAE data augmentation.")
    cat_cols = [c for c in df_train.columns if c.startswith(('growth_', 'harvest_'))]
    cat_cols += ['season_enc', 'district_enc', 'transplant_month', target_col]

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    metadata.update_columns(column_names=cat_cols, sdtype='categorical')

    gan = TVAESynthesizer(metadata, enforce_rounding=False, epochs=500)
    gan.fit(df_train)
    syn_data = gan.sample(len(df_train))

    feature_cols = [c for c in df_train.columns if c != target_col]
    X_orig = df_train[feature_cols].values
    y_orig = df_train[target_col].astype(int).values
    X_synth = syn_data[feature_cols].values
    y_synth = syn_data[target_col].astype(int).values

    logger.info(f"Original: {len(y_orig)}, Synthetic: {len(y_synth)}, Total: {len(y_orig) + len(y_synth)}")
    joblib.dump(gan, os.path.join(model_save_dir, "tvae_gan.joblib"))
    return X_orig, y_orig, X_synth, y_synth, feature_cols


def tune_random_forest(X_full, y_full, random_state=42):
    """Run RandomizedSearchCV for Random Forest and return best params."""
    rf_param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [10, 30, 40, None],
        'min_samples_split': [1, 2, 5],
        'min_samples_leaf': [1, 2, 5],
    }
    rf_base = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf_base, rf_param_grid, n_iter=10, cv=5,
        verbose=1, random_state=random_state, n_jobs=-1,
        scoring='balanced_accuracy',
    )
    rf_search.fit(X_full, y_full)
    return rf_search.best_params_


def train_single_model(name, model, X_orig, y_orig, X_synth, y_synth,
                       X_full, y_full, X_test, y_test,
                       n_splits, random_state, logger):
    """Train one model: CV + final fit + test evaluation. Returns (model, cv_df, metrics)."""
    logger.info(f"Training {name}...")

    cv_df = stratified_cv_train(
        model, name, X_orig, y_orig,
        X_synth=X_synth, y_synth=y_synth,
        n_splits=n_splits, random_state=random_state,
    )

    final_model = clone(model)
    final_model.fit(X_full, y_full)

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test) if hasattr(final_model, "predict_proba") else None
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    metrics['model'] = name

    logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    return final_model, cv_df, metrics
