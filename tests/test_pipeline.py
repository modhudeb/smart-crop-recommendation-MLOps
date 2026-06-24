import pytest
from app.main import (
    _clean_month_string,
    _clean_category,
    _encode_span,
    data_input_pipeline,
)


class TestCleanMonthString:
    def test_expands_abbreviations(self):
        assert _clean_month_string("jan") == "january"
        assert _clean_month_string("feb") == "february"
        assert _clean_month_string("dec") == "december"

    def test_handles_mixed_case(self):
        assert _clean_month_string("JAN") == "january"
        assert _clean_month_string("Feb") == "february"

    def test_passes_through_full_names(self):
        assert _clean_month_string("january") == "january"
        assert _clean_month_string("march") == "march"

    def test_replaces_multiple_abbreviations(self):
        result = _clean_month_string("jan to dec")
        assert result == "january to december"


class TestCleanCategory:
    def test_strips_and_lowers(self):
        assert _clean_category("  North  ") == "north"
        assert _clean_category("SUMMER") == "summer"


class TestEncodeSpan:
    def test_single_month(self):
        result = _encode_span("May to June")
        assert result[4] == 1  # may
        assert result[5] == 1  # june
        assert sum(result) == 2

    def test_wraps_year(self):
        result = _encode_span("Nov to Feb")
        assert result[10] == 1  # nov
        assert result[11] == 1  # dec
        assert result[0] == 1  # jan
        assert result[1] == 1  # feb
        assert sum(result) == 4

    def test_returns_zeros_for_invalid(self):
        result = _encode_span("invalid")
        assert sum(result) == 0

    def test_full_year(self):
        result = _encode_span("January to December")
        assert sum(result) == 12


class TestDataInputPipeline:
    def test_output_has_expected_columns(self, sample_input, mock_artifacts):
        result = data_input_pipeline(
            sample_input,
            mock_artifacts["scaler"],
            mock_artifacts["le_district"],
            mock_artifacts["le_season"],
            mock_artifacts["climate_constants"],
            mock_artifacts["feature_columns"],
        )
        assert len(result.columns) == len(mock_artifacts["feature_columns"])

    def test_output_is_single_row(self, sample_input, mock_artifacts):
        result = data_input_pipeline(
            sample_input,
            mock_artifacts["scaler"],
            mock_artifacts["le_district"],
            mock_artifacts["le_season"],
            mock_artifacts["climate_constants"],
            mock_artifacts["feature_columns"],
        )
        assert len(result) == 1

    def test_invalid_district_raises(self, sample_input, mock_artifacts):
        sample_input["district"] = "unknown_place"
        with pytest.raises(ValueError, match="Unknown district"):
            data_input_pipeline(
                sample_input,
                mock_artifacts["scaler"],
                mock_artifacts["le_district"],
                mock_artifacts["le_season"],
                mock_artifacts["climate_constants"],
                mock_artifacts["feature_columns"],
            )

    def test_invalid_season_raises(self, sample_input, mock_artifacts):
        sample_input["season"] = "unknown_season"
        with pytest.raises(ValueError, match="Unknown season"):
            data_input_pipeline(
                sample_input,
                mock_artifacts["scaler"],
                mock_artifacts["le_district"],
                mock_artifacts["le_season"],
                mock_artifacts["climate_constants"],
                mock_artifacts["feature_columns"],
            )
