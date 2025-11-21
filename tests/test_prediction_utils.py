from modules.xgboost_prediction_utils import get_prediction_window


def test_get_prediction_window_known_timeframes():
    assert get_prediction_window("1h") == "24h"
    assert get_prediction_window("4h") == "48h"


def test_get_prediction_window_defaults_for_unknown():
    assert get_prediction_window("13m") == "next sessions"
