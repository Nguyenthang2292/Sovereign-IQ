import pandas as pd
import numpy as np
import pytest

from modules import xgboost_prediction_labeling as labeling


def test_apply_directional_labels_assigns_expected_classes(monkeypatch):
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 120.0, 104.0, 90.0],
            "ATR_RATIO_14_50": [1.0] * 5,
        }
    )

    result = labeling.apply_directional_labels(df.copy())

    assert result.loc[0, "TargetLabel"] == "UP"
    assert result.loc[2, "TargetLabel"] == "DOWN"
    assert result.loc[1, "TargetLabel"] == "NEUTRAL"
    assert result.loc[0, "DynamicThreshold"] == 0.05
    assert result.loc[0, "Target"] == labeling.LABEL_TO_ID["UP"]


def test_apply_directional_labels_empty_dataframe(monkeypatch):
    """Test apply_directional_labels with empty DataFrame."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [],
            "ATR_RATIO_14_50": [],
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_apply_directional_labels_single_row(monkeypatch):
    """Test apply_directional_labels with single row."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [100.0],
            "ATR_RATIO_14_50": [1.0],
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert len(result) == 1
    assert "TargetLabel" in result.columns
    assert "Target" in result.columns


def test_apply_directional_labels_missing_atr_ratio(monkeypatch):
    """Test apply_directional_labels with missing ATR_RATIO column."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0],
        }
    )

    # Should handle missing ATR_RATIO gracefully
    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert "TargetLabel" in result.columns


def test_apply_directional_labels_with_nan(monkeypatch):
    """Test apply_directional_labels with NaN values."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [100.0, np.nan, 102.0],
            "ATR_RATIO_14_50": [1.0, 1.0, 1.0],
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert "TargetLabel" in result.columns
