import os
import sys
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_EXISTS = os.path.exists(os.path.join(ARTIFACT_DIR, "models", "ResidualCatBoost_RF.joblib"))


@pytest.fixture
def client():
    os.environ["ARTIFACT_DIR"] = ARTIFACT_DIR
    os.environ["CROP_MODEL_NAME"] = "ResidualCatBoost_RF"
    from app.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model artifacts not available")
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "model" in data
        assert "feature_count" in data


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model artifacts not available")
class TestPredictEndpoint:
    def test_predict_returns_prediction(self, client):
        payload = {
            "district": "dhaka",
            "season": "kharif 1",
            "area": 10.5,
            "transplant_month": "April",
            "growth_period": "May to June",
            "harvest_period": "September to October",
            "min_temp": 20.0,
            "max_temp": 35.0,
            "min_relative_humidity": 30.0,
            "max_relative_humidity": 70.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "prediction" in data
        assert "predictions" in data
        assert len(data["predictions"]) > 0

    def test_predict_missing_field_returns_422(self, client):
        payload = {
            "district": "dhaka",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_district_returns_400(self, client):
        payload = {
            "district": "unknown_place",
            "season": "kharif 1",
            "area": 10.5,
            "transplant_month": "April",
            "growth_period": "May to June",
            "harvest_period": "September to October",
            "min_temp": 20.0,
            "max_temp": 35.0,
            "min_relative_humidity": 30.0,
            "max_relative_humidity": 70.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
