import os
import sys
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture
def sample_input():
    return {
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


@pytest.fixture
def mock_artifacts():
    scaler = MagicMock()
    scaler.mean_ = np.array([2.5])
    scaler.scale_ = np.array([1.2])

    le_district = MagicMock()
    le_district.classes_ = ["dhaka", "chattogram", "khulna", "rajshahi"]
    le_district.transform.side_effect = lambda x: [["dhaka", "chattogram", "khulna", "rajshahi"].index(v) for v in x]

    le_season = MagicMock()
    le_season.classes_ = ["kharif 1", "kharif 2", "rabi"]

    le_crop = MagicMock()
    le_crop.inverse_transform.side_effect = lambda x: [f"crop_{v}" for v in x[0]]

    climate_constants = {"mu_T": 28.0, "sigma_T": 5.0, "mu_H": 60.0, "sigma_H": 15.0}

    feature_columns = [
        "growth_jan", "growth_feb", "growth_mar", "growth_apr",
        "growth_may", "growth_jun", "growth_jul", "growth_aug",
        "growth_sep", "growth_oct", "growth_nov", "growth_dec",
        "harvest_jan", "harvest_feb", "harvest_mar", "harvest_apr",
        "harvest_may", "harvest_jun", "harvest_jul", "harvest_aug",
        "harvest_sep", "harvest_oct", "harvest_nov", "harvest_dec",
        "season_enc", "district_enc", "transplant_month",
        "climate_risk_score", "area_log",
    ]

    return {
        "scaler": scaler,
        "le_district": le_district,
        "le_season": le_season,
        "le_crop": le_crop,
        "climate_constants": climate_constants,
        "feature_columns": feature_columns,
    }
