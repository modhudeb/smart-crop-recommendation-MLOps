"""
setup.py - Import all required libraries and set up configuration
Extracted from: Setup section of CropRecomV2.ipynb
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import log2
from scipy.stats import skew, entropy

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from scipy.stats import ttest_rel, wilcoxon
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tqdm.auto import tqdm

# Optional: Data augmentation library
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

# Configure visualization
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)


def get_config():
    """Return configuration dictionary"""
    return {
        'random_state': 42,
        'cv_folds': 5,
        'test_size': 0.2,
        'n_jobs': -1,
    }


if __name__ == "__main__":
    print("Setup module loaded successfully")
    print(f"SDV available: {HAS_SDV}")
    print(f"CTGAN available: {HAS_CTGAN}")
