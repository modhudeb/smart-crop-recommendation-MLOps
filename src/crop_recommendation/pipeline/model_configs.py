"""Model configurations and custom estimator definitions."""

import numpy as np
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from catboost import CatBoostClassifier, Pool


class CalibratedCatBoost(BaseEstimator, ClassifierMixin):
    """CatBoost base + meta-learner on stacked probabilities."""

    def __init__(
        self,
        iterations=500,
        depth=6,
        learning_rate=0.05,
        val_fraction=0.2,
        meta_model=None,
        meta_weight=0.25,
        verbose=0,
        random_state=42,
        cat_features=None,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.val_fraction = val_fraction
        self.meta_model = meta_model
        self.meta_weight = meta_weight
        self.verbose = verbose
        self.random_state = random_state
        self.cat_features = cat_features

    def fit(self, X, y):
        from sklearn.utils.validation import check_X_y
        from sklearn.utils.multiclass import unique_labels

        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_fraction, random_state=self.random_state)
        tr_idx, val_idx = next(sss.split(X, y))
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            random_seed=self.random_state,
            verbose=self.verbose >= 2,
            task_type="CPU",
            thread_count=-1,
        )
        self.model_.fit(Pool(X_tr, y_tr, cat_features=self.cat_features))

        base_val_prob = self.model_.predict_proba(X_val)
        meta_X_val = np.hstack([base_val_prob, -np.log(base_val_prob + 1e-9)])

        self.meta_model_ = clone(
            self.meta_model if self.meta_model is not None else SVC(kernel="linear", probability=True)
        )
        self.meta_model_.fit(meta_X_val, y_val)
        return self

    def predict_proba(self, X):
        base_prob = self.model_.predict_proba(X)
        meta_X = np.hstack([base_prob, -np.log(base_prob + 1e-9)])
        meta_prob = self.meta_model_.predict_proba(meta_X)
        return (1 - self.meta_weight) * base_prob + self.meta_weight * meta_prob

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ── Model factory functions ──────────────────────────────────────────────


def build_logistic_regression(random_state=42):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000, random_state=random_state)


def build_lightgbm(random_state=42):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.6,
        random_state=random_state,
        verbose=-1,
    )


def build_xgboost(random_state=42, n_classes=None):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )


def build_mlp(random_state=42):
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=400,
        random_state=random_state,
    )


def build_catboost(random_state=42):
    return CatBoostClassifier(
        iterations=200,
        learning_rate=0.04,
        depth=None,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=random_state,
        verbose=0,
    )


def build_calibrated_catboost_svc(random_state=42):
    meta = SVC(kernel="linear", probability=True, random_state=random_state)
    return CalibratedCatBoost(
        iterations=200,
        depth=None,
        learning_rate=0.04,
        val_fraction=0.3,
        meta_model=meta,
        meta_weight=0.2,
        verbose=0,
        random_state=random_state,
    )


def build_calibrated_catboost_rf(random_state=42, rf_params=None):
    from sklearn.ensemble import RandomForestClassifier

    meta = RandomForestClassifier(**(rf_params or {}), random_state=random_state)
    return CalibratedCatBoost(
        iterations=200,
        depth=None,
        learning_rate=0.04,
        val_fraction=0.3,
        meta_model=meta,
        meta_weight=0.2,
        verbose=0,
        random_state=random_state,
    )


def build_voting_classifier(random_state=42, rf_params=None):
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.neural_network import MLPClassifier

    estimators = [
        ("rf", RandomForestClassifier(**(rf_params or {}), random_state=random_state)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400, random_state=random_state)),
        (
            "cat",
            CatBoostClassifier(
                iterations=200,
                learning_rate=0.04,
                depth=6,
                loss_function="MultiClass",
                verbose=0,
                random_seed=random_state,
            ),
        ),
    ]
    return VotingClassifier(estimators=estimators, voting="soft")
