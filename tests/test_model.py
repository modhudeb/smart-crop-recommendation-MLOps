import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "models", "ResidualCatBoost_RF.joblib")
MODEL_EXISTS = os.path.exists(MODEL_PATH)


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model artifacts not available")
class TestModelLoading:
    def test_model_loads_successfully(self):
        import joblib
        model = joblib.load(MODEL_PATH)
        assert model is not None

    def test_model_has_predict_method(self):
        import joblib
        model = joblib.load(MODEL_PATH)
        assert hasattr(model, "predict")

    def test_model_has_predict_proba(self):
        import joblib
        model = joblib.load(MODEL_PATH)
        assert hasattr(model, "predict_proba")

    def test_model_predict_returns_valid_shape(self):
        import joblib
        import numpy as np
        model = joblib.load(MODEL_PATH)
        sample = np.zeros((1, 29))
        pred = model.predict(sample)
        assert pred.shape == (1,)

    def test_model_predict_proba_returns_probabilities(self):
        import joblib
        import numpy as np
        model = joblib.load(MODEL_PATH)
        sample = np.zeros((1, 29))
        proba = model.predict_proba(sample)
        assert proba.shape[0] == 1
        assert proba.shape[1] > 1
        assert abs(proba.sum() - 1.0) < 1e-5
