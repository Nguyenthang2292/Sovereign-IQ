
from pathlib import Path
from unittest.mock import MagicMock
import sys
import warnings

"""
Shared fixtures for backtester tests.
"""


# Mock optuna BEFORE any other imports
# This must happen before core.signal_calculators is imported
if "optuna" not in sys.modules:
    sys.modules["optuna"] = MagicMock()

# Add project root to path (same as tests/conftest.py)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_ohlcv_data():
    """Generate mock OHLCV data for testing."""

    def _generate_data(periods=200, base_price=100.0, volatility=0.5):
        dates = pd.date_range("2023-01-01", periods=periods, freq="h")
        # Create realistic price movement with random walk
        returns = np.random.randn(periods) * volatility
        prices = base_price + np.cumsum(returns)

        # Ensure prices are positive
        prices = np.maximum(prices, base_price * 0.1)

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.uniform(1000, 10000, periods),
            },
            index=dates,
        )

        return df

    return _generate_data


@pytest.fixture
def mock_data_fetcher(mock_ohlcv_data):
    """Create a fully mocked DataFetcher that doesn't call real APIs."""

    def fake_fetch(symbol, **kwargs):
        """Mock fetch function that returns generated data."""
        limit = kwargs.get("limit", 200)
        df = mock_ohlcv_data(periods=limit)
        return df, "binance"

    # Create a complete mock DataFetcher
    fetcher = SimpleNamespace()
    fetcher.fetch_ohlcv_with_fallback_exchange = fake_fetch
    fetcher.fetch_binance_account_balance = Mock(return_value=None)
    fetcher.market_prices = {}
    fetcher._ohlcv_dataframe_cache = {}

    return fetcher


@pytest.fixture
def mock_signal_calculators():
    """Mock all signal calculator functions to avoid API calls.

    Use this fixture in tests to prevent API calls to exchanges.
    """
    # Patch at the usage site in HybridSignalCalculator
    with (
        patch("modules.position_sizing.core.hybrid_signal_calculator.get_range_oscillator_signal") as mock_osc,
        patch("modules.position_sizing.core.hybrid_signal_calculator.get_spc_signal") as mock_spc,
        patch("modules.position_sizing.core.hybrid_signal_calculator.get_xgboost_signal") as mock_xgb,
        patch("modules.position_sizing.core.hybrid_signal_calculator.get_hmm_signal") as mock_hmm,
        patch("modules.position_sizing.core.hybrid_signal_calculator.get_random_forest_signal") as mock_rf,
    ):
        # Set default return values
        mock_osc.return_value = (1, 0.7)  # LONG signal with 70% confidence
        mock_spc.return_value = (1, 0.6)  # LONG signal with 60% confidence
        mock_xgb.return_value = (1, 0.8)  # LONG signal with 80% confidence
        mock_hmm.return_value = (1, 0.65)  # LONG signal with 65% confidence
        mock_rf.return_value = (1, 0.75)  # LONG signal with 75% confidence

        yield {
            "range_oscillator": mock_osc,
            "spc": mock_spc,
            "xgboost": mock_xgb,
            "hmm": mock_hmm,
            "random_forest": mock_rf,
        }


@pytest.fixture(autouse=True, scope="function")
def auto_mock_signal_calculators():
    """Automatically mock signal calculators for all tests to prevent API calls.

    This is set to autouse=True to ensure no API calls are made by default.
    Individual tests can override this if needed.
    """
    # Import core.signal_calculators first (optuna is already mocked)
    # Then patch the functions
    try:
        import core.signal_calculators  # noqa: F401

        # Patch at definition site
        with (
            patch("core.signal_calculators.get_range_oscillator_signal", return_value=(1, 0.7)),
            patch("core.signal_calculators.get_spc_signal", return_value=(1, 0.6)),
            patch("core.signal_calculators.get_xgboost_signal", return_value=(1, 0.8)),
            patch("core.signal_calculators.get_hmm_signal", return_value=(1, 0.65)),
            patch("core.signal_calculators.get_random_forest_signal", return_value=(1, 0.75)),
        ):
            yield
    except (ImportError, AttributeError):
        # If import fails, skip patching (tests will need to handle this)
        yield


@pytest.fixture
def mock_hybrid_signal_calculator():
    """Mock HybridSignalCalculator to return predictable signals."""
    with patch("modules.position_sizing.core.hybrid_signal_calculator.HybridSignalCalculator") as mock_class:
        mock_instance = MagicMock()
        # Return alternating signals for testing
        signals = [1, 0, 1, -1, 0, 1] * 100  # Repeat pattern
        confidences = [0.8, 0.0, 0.7, 0.6, 0.0, 0.75] * 100

        def calculate_signal(df, symbol, timeframe, period_index, signal_type, **kwargs):
            idx = period_index % len(signals)
            return (signals[idx], confidences[idx])

        mock_instance.calculate_hybrid_signal = MagicMock(side_effect=calculate_signal)
        mock_instance.get_cache_stats = MagicMock(
            return_value={
                "signal_cache_size": 0,
                "signal_cache_max_size": 200,
                "data_cache_size": 0,
                "data_cache_max_size": 10,
            }
        )
        mock_instance.enabled_indicators = ["range_oscillator", "spc", "xgboost", "hmm", "random_forest"]
        mock_instance.use_confidence_weighting = True
        mock_instance.min_indicators_agreement = 3

        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_regime_detector():
    """Mock RegimeDetector to return predictable regimes."""
    with patch("modules.position_sizing.core.regime_detector.RegimeDetector") as mock_class:
        mock_instance = MagicMock()
        # Return alternating regimes
        regimes = ["BULLISH", "NEUTRAL", "BEARISH", "NEUTRAL"] * 25

        def detect_regime(symbol, timeframe, limit):
            # Simple hash-based regime selection for consistency
            hash_val = hash(f"{symbol}{timeframe}") % len(regimes)
            return regimes[hash_val]

        mock_instance.detect_regime = MagicMock(side_effect=detect_regime)
        mock_class.return_value = mock_instance
        yield mock_instance
