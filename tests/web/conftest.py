"""
Shared fixtures for web module tests.
"""

import sys
from pathlib import Path

# Add project root to sys.path for imports
# This ensures web module and its dependencies can be imported
project_root = Path(__file__).parent.parent.parent
project_root_str = str(project_root)

import json
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Import app - add project root to path so absolute imports like 'from web.utils' work
# We only need project_root, not web_dir, since app.py now uses absolute imports
# CRITICAL: Remove tests/web from sys.path if it exists, as it conflicts with web package
# (tests/web/__init__.py would make Python think tests/web is the 'web' package)
tests_web_path = str(project_root / "tests" / "web")
if tests_web_path in sys.path:
    sys.path.remove(tests_web_path)

# Remove project_root from path if it exists (to avoid duplicates and ensure it's at position 0)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)

# Insert at position 0 to ensure it's checked FIRST
sys.path.insert(0, project_root_str)

# Now import app - app.py uses 'from web.api import ...' which needs project_root in path
from web.app import app


@pytest.fixture
def client():
    """Create FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    np.random.seed(42)

    base_price = 50000
    prices = []
    for i in range(100):
        change = np.random.randn() * 100
        base_price = max(base_price + change, 1000)
        prices.append(base_price)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * (1 + abs(np.random.randn() * 0.01)) for p in prices],
            "low": [p * (1 - abs(np.random.randn() * 0.01)) for p in prices],
            "close": [p * (1 + np.random.randn() * 0.005) for p in prices],
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )

    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    return df


@pytest.fixture
def empty_ohlcv_df():
    """Create empty OHLCV DataFrame for testing error cases."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def mock_exchange_manager():
    """Create mock ExchangeManager."""
    return Mock()


@pytest.fixture
def mock_data_fetcher(sample_ohlcv_df):
    """Create mock DataFetcher that returns sample data."""
    mock_fetcher = Mock()
    mock_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (sample_ohlcv_df, "binance")
    return mock_fetcher


@pytest.fixture
def mock_data_fetcher_empty(empty_ohlcv_df):
    """Create mock DataFetcher that returns empty data."""
    mock_fetcher = Mock()
    mock_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (empty_ohlcv_df, "binance")
    return mock_fetcher


@pytest.fixture
def mock_data_fetcher_none():
    """Create mock DataFetcher that returns None."""
    mock_fetcher = Mock()
    mock_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (None, "binance")
    return mock_fetcher


@pytest.fixture
def mock_chart_generator():
    """Create mock ChartGenerator."""
    mock_gen = Mock()
    mock_gen.create_chart.return_value = "/fake/path/to/chart.png"
    return mock_gen


@pytest.fixture
def mock_gemini_analyzer():
    """Create mock GeminiChartAnalyzer."""
    mock_analyzer = Mock()
    mock_analyzer.analyze_chart.return_value = (
        "This is a sample analysis. The chart shows a bullish trend with "
        "strong support at $50,000. Long signal recommended."
    )
    return mock_analyzer


@pytest.fixture
def mock_gemini_analyzer_short():
    """Create mock GeminiChartAnalyzer that returns short signal."""
    mock_analyzer = Mock()
    mock_analyzer.analyze_chart.return_value = (
        "This is a sample analysis. The chart shows a bearish trend with "
        "resistance at $55,000. Short signal recommended."
    )
    return mock_analyzer


@pytest.fixture
def mock_multi_timeframe_coordinator():
    """Create mock MultiTimeframeCoordinator."""
    mock_coordinator = Mock()
    mock_coordinator.analyze_deep.return_value = {
        "timeframes": {
            "1h": {
                "analysis": "1h analysis text",
                "signal": "LONG",
                "confidence": 0.7,
                "chart_path": "/fake/path/to/chart_1h.png",
            },
            "4h": {
                "analysis": "4h analysis text",
                "signal": "LONG",
                "confidence": 0.8,
                "chart_path": "/fake/path/to/chart_4h.png",
            },
        },
        "aggregated": {"signal": "LONG", "confidence": 0.75, "weights_used": {"1h": 0.3, "4h": 0.7}},
    }
    return mock_coordinator


@pytest.fixture
def mock_batch_scanner():
    """Create mock MarketBatchScanner."""
    mock_scanner = Mock()
    mock_scanner.scan_market.return_value = {
        "summary": {"total_scanned": 10, "long_count": 3, "short_count": 2, "none_count": 5},
        "long_symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        "short_symbols": ["ADA/USDT", "SOL/USDT"],
        "long_symbols_with_confidence": [
            {"symbol": "BTC/USDT", "confidence": 0.8},
            {"symbol": "ETH/USDT", "confidence": 0.7},
            {"symbol": "BNB/USDT", "confidence": 0.6},
        ],
        "short_symbols_with_confidence": [
            {"symbol": "ADA/USDT", "confidence": 0.7},
            {"symbol": "SOL/USDT", "confidence": 0.6},
        ],
        "all_results": {
            "BTC/USDT": {"signal": "LONG", "confidence": 0.8},
            "ETH/USDT": {"signal": "LONG", "confidence": 0.7},
        },
        "results_file": "/fake/path/to/results.json",
    }
    return mock_scanner


@pytest.fixture
def temp_charts_dir(tmp_path):
    """Create temporary directory for charts."""
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    return charts_dir


@pytest.fixture
def temp_results_dir(tmp_path):
    """Create temporary directory for results."""
    results_dir = tmp_path / "results" / "batch_scan"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _setup_common_mocks(sample_ohlcv_df, tmp_path, mock_data_fetcher_class, mock_log_mgr, mock_task_mgr):
    """
    Helper function to setup common mocks for DataFetcher, LogManager, and TaskManager.

    Args:
        sample_ohlcv_df: Sample OHLCV DataFrame to return from DataFetcher
        tmp_path: Temporary path for log file
        mock_data_fetcher_class: Mock class for DataFetcher
        mock_log_mgr: Mock function for get_log_manager
        mock_task_mgr: Mock function for get_task_manager

    Returns:
        tuple: (mock_data_fetcher, mock_log_manager, mock_task_manager)
    """
    # Setup mock_data_fetcher
    mock_data_fetcher = Mock()
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (sample_ohlcv_df, "binance")
    mock_data_fetcher_class.return_value = mock_data_fetcher

    # Setup mock_log_manager
    mock_log_manager = Mock()
    mock_log_manager.create_log_file.return_value = tmp_path / "test.log"
    mock_log_mgr.return_value = mock_log_manager

    # Setup mock_task_manager
    mock_task_manager = Mock()
    mock_task_mgr.return_value = mock_task_manager

    return mock_data_fetcher, mock_log_manager, mock_task_manager


@pytest.fixture
def sample_batch_results_json(temp_results_dir):
    """Create sample batch results JSON file."""
    results_file = temp_results_dir / "test_results.json"
    results_data = {
        "summary": {"total_scanned": 10, "long_count": 3, "short_count": 2, "none_count": 5},
        "long_symbols": ["BTC/USDT", "ETH/USDT"],
        "short_symbols": ["ADA/USDT"],
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f)
    return results_file


@pytest.fixture
def sample_batch_results_json_no_summary(temp_results_dir):
    """Create sample batch results JSON file without summary."""
    results_file = temp_results_dir / "test_no_summary.json"
    results_data = {"long_symbols": ["BTC/USDT"], "timestamp": datetime.now().isoformat()}
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f)
    return results_file


@pytest.fixture
def chart_analyzer_mocks(sample_ohlcv_df, tmp_path):
    """
    Fixture that patches common chart analyzer dependencies and yields mock objects.

    This fixture applies patches for:
    - ExchangeManager
    - DataFetcher (with default return: sample_ohlcv_df, 'binance')
    - ChartGenerator (with default chart_path)
    - GeminiChartAnalyzer (with default analysis text)
    - get_charts_dir (returns tmp_path)
    - get_log_manager (with default log file)
    - get_task_manager (with default mock task manager)

    Yields a dict with all mock objects for easy access and customization.
    """

    chart_path = str(tmp_path / "chart.png")

    with (
        patch("web.api.chart_analyzer.ExchangeManager") as mock_exchange,
        patch("web.api.chart_analyzer.DataFetcher") as mock_data_fetcher_class,
        patch("web.api.chart_analyzer.ChartGenerator") as mock_chart_gen_class,
        patch("web.api.chart_analyzer.GeminiChartAnalyzer") as mock_gemini_class,
        patch("web.api.chart_analyzer.get_charts_dir", return_value=tmp_path) as mock_get_charts_dir,
        patch("web.api.chart_analyzer.get_log_manager") as mock_log_mgr,
        patch("web.api.chart_analyzer.get_task_manager") as mock_task_mgr,
    ):
        # Setup common mocks using helper
        mock_data_fetcher, mock_log_manager, mock_task_manager = _setup_common_mocks(
            sample_ohlcv_df, tmp_path, mock_data_fetcher_class, mock_log_mgr, mock_task_mgr
        )

        # Setup chart analyzer specific mocks
        mock_chart_gen = Mock()
        mock_chart_gen.create_chart.return_value = chart_path
        mock_chart_gen_class.return_value = mock_chart_gen

        mock_gemini = Mock()
        mock_gemini.analyze_chart.return_value = "Analysis text"
        mock_gemini_class.return_value = mock_gemini

        # Yield all mocks in a dict for easy access
        yield {
            "exchange": mock_exchange,
            "data_fetcher_class": mock_data_fetcher_class,
            "data_fetcher": mock_data_fetcher,
            "chart_gen_class": mock_chart_gen_class,
            "chart_gen": mock_chart_gen,
            "gemini_class": mock_gemini_class,
            "gemini": mock_gemini,
            "get_charts_dir": mock_get_charts_dir,
            "log_manager": mock_log_manager,
            "log_mgr": mock_log_mgr,
            "task_manager": mock_task_manager,
            "task_mgr": mock_task_mgr,
            "chart_path": chart_path,
        }


@pytest.fixture
def multi_timeframe_mocks(sample_ohlcv_df, tmp_path):
    """
    Fixture that patches multi-timeframe analysis dependencies and yields mock objects.

    This fixture applies patches for:
    - ExchangeManager
    - DataFetcher (with default return: sample_ohlcv_df, 'binance')
    - MultiTimeframeCoordinator (with default analyze_deep return)
    - get_charts_dir (returns tmp_path)
    - get_log_manager (with default log file)
    - get_task_manager (with default mock task manager)

    Yields a dict with all mock objects for easy access and customization.
    """

    with (
        patch("web.api.chart_analyzer.ExchangeManager") as mock_exchange,
        patch("web.api.chart_analyzer.DataFetcher") as mock_data_fetcher_class,
        patch("web.api.chart_analyzer.MultiTimeframeCoordinator") as mock_mtf_class,
        patch("web.api.chart_analyzer.get_charts_dir", return_value=tmp_path) as mock_get_charts_dir,
        patch("web.api.chart_analyzer.get_log_manager") as mock_log_mgr,
        patch("web.api.chart_analyzer.get_task_manager") as mock_task_mgr,
    ):
        # Setup common mocks using helper
        mock_data_fetcher, mock_log_manager, mock_task_manager = _setup_common_mocks(
            sample_ohlcv_df, tmp_path, mock_data_fetcher_class, mock_log_mgr, mock_task_mgr
        )

        # Setup multi-timeframe specific mocks
        mock_mtf = Mock()
        mock_mtf.analyze_deep.return_value = {
            "timeframes": {
                "1h": {
                    "analysis": "1h analysis",
                    "signal": "LONG",
                    "confidence": 0.7,
                    "chart_path": str(tmp_path / "chart_1h.png"),
                },
                "4h": {
                    "analysis": "4h analysis",
                    "signal": "LONG",
                    "confidence": 0.8,
                    "chart_path": str(tmp_path / "chart_4h.png"),
                },
            },
            "aggregated": {"signal": "LONG", "confidence": 0.75, "weights_used": {"1h": 0.3, "4h": 0.7}},
        }
        mock_mtf_class.return_value = mock_mtf

        # Yield all mocks in a dict for easy access
        yield {
            "exchange": mock_exchange,
            "data_fetcher_class": mock_data_fetcher_class,
            "data_fetcher": mock_data_fetcher,
            "mtf_class": mock_mtf_class,
            "mtf": mock_mtf,
            "get_charts_dir": mock_get_charts_dir,
            "log_manager": mock_log_manager,
            "log_mgr": mock_log_mgr,
            "task_manager": mock_task_manager,
            "task_mgr": mock_task_mgr,
        }
