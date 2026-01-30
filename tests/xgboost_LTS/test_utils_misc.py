import numpy as np
import pytest

from modules.xgboost_LTS.utils.utils import get_prediction_window


def test_get_prediction_window_known_timeframe():
    assert get_prediction_window("1h") == "24h"


def test_get_prediction_window_unknown_timeframe():
    assert get_prediction_window("10m") == "next sessions"
