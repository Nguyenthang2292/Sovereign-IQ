"""
Unit tests for the get_prediction_window utility function in the XGBoost module.

This test suite verifies:
- Correct mapping from known input timeframes to their associated prediction windows.
- Default handling for unknown, invalid, or empty inputs.
- Case-insensitivity of inputs to get_prediction_window.
- Exhaustive coverage for all known configured timeframes.
- Graceful handling or appropriate error raising for None input.
"""

from modules.xgboost.utils import get_prediction_window


def test_get_prediction_window_known_timeframes():
    assert get_prediction_window("1h") == "24h"
    assert get_prediction_window("4h") == "48h"
    assert get_prediction_window("30m") == "12h"
    assert get_prediction_window("1d") == "7d"


def test_get_prediction_window_defaults_for_unknown():
    assert get_prediction_window("13m") == "next sessions"
    assert get_prediction_window("invalid") == "next sessions"
    assert get_prediction_window("") == "next sessions"


def test_get_prediction_window_case_insensitive():
    """Test get_prediction_window is case insensitive."""
    assert get_prediction_window("1H") == "24h"
    assert get_prediction_window("4H") == "48h"
    assert get_prediction_window("1d") == "7d"
    assert get_prediction_window("1D") == "7d"


def test_get_prediction_window_all_known_timeframes():
    """Test get_prediction_window for all known timeframes."""
    from config import PREDICTION_WINDOWS

    for timeframe, expected in PREDICTION_WINDOWS.items():
        assert get_prediction_window(timeframe) == expected
        # Test uppercase
        assert get_prediction_window(timeframe.upper()) == expected


def test_get_prediction_window_none_input():
    """Test get_prediction_window with None input."""
    # Should handle None gracefully or raise appropriate error
    try:
        result = get_prediction_window(None)
        assert isinstance(result, str)
    except (TypeError, AttributeError):
        # Acceptable to raise error for None
        pass
