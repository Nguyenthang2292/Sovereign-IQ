"""
Tests for batch scan service with new architectural changes.

Tests cover:
- Batch scan execution with valid configuration
- Graceful handling of empty results
- Configuration validation
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from modules.gemini_chart_analyzer.core.exceptions import ScanConfigurationError
from modules.gemini_chart_analyzer.core.scanner_types import BatchScanResult
from modules.gemini_chart_analyzer.services.batch_scan_service import (
    BatchScanConfig,
    run_batch_scan,
)


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for test configurations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def valid_batch_config():
    """Create a valid batch scan configuration for testing."""
    return BatchScanConfig(
        timeframe="1h",
        timeframes=["1h", "4h"],
        max_symbols=100,
        limit=700,
        cooldown=2.5,
        enable_pre_filter=True,
        pre_filter_mode="voting",
        pre_filter_percentage=10.0,
        fast_mode=True,
        spc_config={
            "preset": None,
            "volatility_adjustment": False,
            "use_correlation_weights": False,
            "time_decay_factor": None,
            "interpolation_mode": None,
            "min_flip_duration": None,
            "flip_confidence_threshold": None,
            "enable_mtf": False,
            "mtf_timeframes": None,
            "mtf_require_alignment": None,
        },
        rf_model_path=None,
    )


@pytest.fixture
def mock_analyzer():
    """Create a mock analyzer for testing."""
    analyzer = MagicMock()
    analyzer.analyze_chart.return_value = json.dumps(
        {"signal": "LONG", "confidence": "high", "reasoning": "Chart shows strong upward trend"}
    )
    return analyzer


@pytest.fixture
def mock_data_fetcher():
    """Create a mock data fetcher for testing."""
    fetcher = MagicMock()
    fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (pd.DataFrame(), "binance")
    return fetcher


def test_run_batch_scan_with_valid_config(valid_batch_config):
    """Test batch scan execution with valid configuration."""
    mock_result = BatchScanResult(
        long_symbols=["BTC/USDT"],
        short_symbols=[],
        none_symbols=[],
        all_results={"BTC/USDT": {"signal": "LONG", "confidence": 0.8}},
        summary={"total_symbols": 1, "long_count": 1},
        results_file="test_results.json",
    )

    with (
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.MarketBatchScanner") as mock_scanner_class,
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.generate_html_report") as mock_html_report,
    ):
        mock_scanner = MagicMock()
        mock_scanner.scan_market.return_value = mock_result
        mock_scanner_class.return_value = mock_scanner
        mock_html_report.return_value = "test_report.html"

        # Run the batch scan
        results = run_batch_scan(valid_batch_config)

        # Verify results structure
        assert results is not None
        assert hasattr(results, "long_symbols") or "long_symbols" in results

        # Verify scanner was called with correct configuration
        mock_scanner.scan_market.assert_called_once()
        call_kwargs = mock_scanner.scan_market.call_args[1]
        assert call_kwargs["timeframe"] == "1h"
        assert call_kwargs["timeframes"] == ["1h", "4h"]
        assert call_kwargs["enable_pre_filter"] is True
        assert call_kwargs["pre_filter_percentage"] == 10.0


def test_run_batch_scan_handles_empty_results(valid_batch_config):
    """Test batch scan gracefully handles no signals from analyzer."""
    mock_result = BatchScanResult(
        long_symbols=[],
        short_symbols=[],
        none_symbols=["BTC/USDT"],
        all_results={"BTC/USDT": {"signal": "NONE", "confidence": 0.0}},
        summary={"total_symbols": 1, "long_count": 0, "short_count": 0},
        results_file="test_results.json",
    )

    with (
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.MarketBatchScanner") as mock_scanner_class,
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.generate_html_report"),
    ):
        mock_scanner = MagicMock()
        mock_scanner.scan_market.return_value = mock_result
        mock_scanner_class.return_value = mock_scanner

        # Run the batch scan
        results = run_batch_scan(valid_batch_config)

        # Verify results are handled gracefully
        assert results is not None
        assert hasattr(results, "long_symbols") or "long_symbols" in results


def test_run_batch_scan_configuration_validation(valid_batch_config):
    """Test batch scan validates configuration parameters."""
    mock_result = MagicMock()
    mock_result.long_symbols = []
    mock_result.short_symbols = []
    mock_result.none_symbols = []
    mock_result.all_results = {}
    mock_result.summary = {}
    mock_result.results_file = "test_results.json"

    with patch("modules.gemini_chart_analyzer.services.batch_scan_service.MarketBatchScanner") as mock_scanner_class:
        mock_scanner = MagicMock()
        # Simulate validation error
        mock_scanner.scan_market.side_effect = ScanConfigurationError("Invalid timeframe")
        mock_scanner_class.return_value = mock_scanner

        # Test with invalid timeframe
        invalid_config = BatchScanConfig(
            timeframe="invalid",  # Invalid timeframe
            timeframes=None,
            max_symbols=100,
            limit=700,
            cooldown=2.5,
            enable_pre_filter=False,
            pre_filter_mode=None,
            fast_mode=True,
            spc_config=None,
            rf_model_path=None,
        )

        with pytest.raises(ScanConfigurationError):
            run_batch_scan(invalid_config)


def test_run_batch_scan_with_pre_filter(valid_batch_config):
    """Test batch scan with pre-filter enabled."""
    valid_batch_config.enable_pre_filter = True
    valid_batch_config.pre_filter_mode = "voting"
    valid_batch_config.pre_filter_percentage = 10.0

    mock_result = BatchScanResult(
        long_symbols=["BTC/USDT"],
        short_symbols=[],
        none_symbols=[],
        all_results={"BTC/USDT": {"signal": "LONG", "confidence": 0.8}},
        summary={"total_symbols": 1, "long_count": 1},
        results_file="test_results.json",
    )

    with (
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.MarketBatchScanner") as mock_scanner_class,
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.generate_html_report"),
    ):
        mock_scanner = MagicMock()
        mock_scanner.scan_market.return_value = mock_result
        mock_scanner_class.return_value = mock_scanner

        # Run the batch scan
        results = run_batch_scan(valid_batch_config)

        # Verify pre-filter was configured
        call_kwargs = mock_scanner.scan_market.call_args[1]
        assert call_kwargs["enable_pre_filter"] is True
        assert call_kwargs["pre_filter_percentage"] == 10.0
        assert results is not None


def test_run_batch_scan_symbol_limiting(valid_batch_config):
    """Test batch scan respects max_symbols limit."""
    valid_batch_config.max_symbols = 2  # Limit to 2 symbols

    mock_result = BatchScanResult(
        long_symbols=["BTC/USDT", "ETH/USDT"],
        short_symbols=[],
        none_symbols=[],
        all_results={"BTC/USDT": {"signal": "LONG"}, "ETH/USDT": {"signal": "LONG"}},
        summary={"total_symbols": 2, "long_count": 2},
        results_file="test_results.json",
    )

    with (
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.MarketBatchScanner") as mock_scanner_class,
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.generate_html_report"),
    ):
        mock_scanner = MagicMock()
        mock_scanner.scan_market.return_value = mock_result
        mock_scanner_class.return_value = mock_scanner

        # Run the batch scan
        results = run_batch_scan(valid_batch_config)

        # Verify max_symbols was passed
        call_kwargs = mock_scanner.scan_market.call_args[1]
        assert call_kwargs["max_symbols"] == 2
        assert results is not None


def test_run_batch_scan_timeframe_multiple(valid_batch_config):
    """Test batch scan with multiple timeframes."""
    valid_batch_config.timeframes = ["1h", "4h", "1d"]

    mock_result = BatchScanResult(
        long_symbols=["BTC/USDT"],
        short_symbols=[],
        none_symbols=[],
        all_results={"BTC/USDT": {"signal": "LONG", "confidence": 0.8}},
        summary={"total_symbols": 1, "long_count": 1},
        results_file="test_results.json",
    )

    with (
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.MarketBatchScanner") as mock_scanner_class,
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.generate_html_report"),
    ):
        mock_scanner = MagicMock()
        mock_scanner.scan_market.return_value = mock_result
        mock_scanner_class.return_value = mock_scanner

        # Run the batch scan
        results = run_batch_scan(valid_batch_config)

        # Verify multiple timeframes were passed
        call_kwargs = mock_scanner.scan_market.call_args[1]
        assert call_kwargs["timeframes"] == ["1h", "4h", "1d"]
        assert results is not None


def test_batch_config_creation_from_dict():
    """Test BatchScanConfig creation from dictionary."""
    config_dict = {
        "timeframe": "4h",
        "timeframes": ["4h", "1d"],
        "max_symbols": 50,
        "limit": 500,
        "cooldown": 3.0,
        "enable_pre_filter": True,
        "pre_filter_mode": "hybrid",
        "pre_filter_percentage": 15.0,
        "fast_mode": False,
        "spc_config": {
            "preset": "medium_risk",
            "volatility_adjustment": True,
            "use_correlation_weights": True,
            "time_decay_factor": 0.5,
            "interpolation_mode": "linear",
            "min_flip_duration": 5,
            "flip_confidence_threshold": 0.7,
            "enable_mtf": True,
            "mtf_timeframes": ["1h", "4h"],
            "mtf_require_alignment": False,
        },
        "rf_model_path": "models/random_forest_model.pkl",
    }

    config = BatchScanConfig(**config_dict)

    assert config.timeframe == "4h"
    assert config.timeframes == ["4h", "1d"]
    assert config.max_symbols == 50
    assert config.limit == 500
    assert config.cooldown == 3.0
    assert config.enable_pre_filter is True
    assert config.pre_filter_mode == "hybrid"
    assert config.pre_filter_percentage == 15.0
    assert config.fast_mode is False
    assert config.spc_config is not None
    assert config.spc_config["preset"] == "medium_risk"
    assert config.rf_model_path == "models/random_forest_model.pkl"


def test_run_batch_scan_model_fallback(valid_batch_config):
    """Test batch scan handles errors gracefully."""
    mock_result = MagicMock()
    mock_result.long_symbols = []
    mock_result.short_symbols = []
    mock_result.none_symbols = []
    mock_result.all_results = {}
    mock_result.summary = {}
    mock_result.results_file = "test_results.json"

    with patch("modules.gemini_chart_analyzer.services.batch_scan_service.MarketBatchScanner") as mock_scanner_class:
        mock_scanner = MagicMock()
        # Simulate error that gets handled
        from modules.gemini_chart_analyzer.core.exceptions import GeminiAnalysisError

        mock_scanner.scan_market.side_effect = GeminiAnalysisError("Model not found")
        mock_scanner_class.return_value = mock_scanner

        # Run the batch scan - should raise the error
        with pytest.raises(GeminiAnalysisError):
            run_batch_scan(valid_batch_config)


def test_run_batch_scan_with_rf_model(valid_batch_config):
    """Test batch scan with random forest model enabled."""
    valid_batch_config.rf_model_path = "models/random_forest_model.pkl"

    mock_result = BatchScanResult(
        long_symbols=["BTC/USDT"],
        short_symbols=[],
        none_symbols=[],
        all_results={"BTC/USDT": {"signal": "LONG", "confidence": 0.8}},
        summary={"total_symbols": 1, "long_count": 1},
        results_file="test_results.json",
    )

    with (
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.MarketBatchScanner") as mock_scanner_class,
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.generate_html_report"),
    ):
        mock_scanner = MagicMock()
        mock_scanner.scan_market.return_value = mock_result
        mock_scanner_class.return_value = mock_scanner

        # Run the batch scan
        results = run_batch_scan(valid_batch_config)

        # Verify RF model path was passed to scanner
        assert mock_scanner_class.call_args[1]["rf_model_path"] == "models/random_forest_model.pkl"
        assert results is not None
