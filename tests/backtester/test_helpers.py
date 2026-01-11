
from contextlib import contextmanager
from unittest.mock import patch

"""
Helper utilities for backtester tests.
"""



@contextmanager
def mock_all_signal_calculators(
    osc_signal=1,
    osc_confidence=0.7,
    spc_signal=1,
    spc_confidence=0.6,
    xgb_signal=1,
    xgb_confidence=0.8,
    hmm_signal=1,
    hmm_confidence=0.65,
    rf_signal=1,
    rf_confidence=0.75,
):
    """
    Context manager to mock all signal calculators.

    This prevents API calls during testing by mocking all signal calculator
    functions that would normally fetch data from exchanges.

    Args:
        osc_signal, osc_confidence: Range Oscillator signal and confidence
        spc_signal, spc_confidence: SPC signal and confidence
        xgb_signal, xgb_confidence: XGBoost signal and confidence
        hmm_signal, hmm_confidence: HMM signal and confidence
        rf_signal, rf_confidence: Random Forest signal and confidence

    Usage:
        with mock_all_signal_calculators():
            # Your test code here - no API calls will be made
            result = backtester.backtest(...)
    """
    with (
        patch("core.signal_calculators.get_range_oscillator_signal", return_value=(osc_signal, osc_confidence)),
        patch("core.signal_calculators.get_spc_signal", return_value=(spc_signal, spc_confidence)),
        patch("core.signal_calculators.get_xgboost_signal", return_value=(xgb_signal, xgb_confidence)),
        patch("core.signal_calculators.get_hmm_signal", return_value=(hmm_signal, hmm_confidence)),
        patch("core.signal_calculators.get_random_forest_signal", return_value=(rf_signal, rf_confidence)),
    ):
        yield


@contextmanager
def mock_no_signals():
    """Context manager to mock all signal calculators to return no signals."""
    with mock_all_signal_calculators(
        osc_signal=0,
        osc_confidence=0.0,
        spc_signal=0,
        spc_confidence=0.0,
        xgb_signal=0,
        xgb_confidence=0.0,
        hmm_signal=0,
        hmm_confidence=0.0,
        rf_signal=0,
        rf_confidence=0.0,
    ):
        yield
