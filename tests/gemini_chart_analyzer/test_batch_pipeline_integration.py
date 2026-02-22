import json
from unittest.mock import MagicMock, patch

import pytest
from modules.gemini_chart_analyzer.core.scanner_types import BatchScanResult, SignalResult, SymbolScanResult
from modules.gemini_chart_analyzer.services.batch_scan_service import BatchScanConfig, run_batch_scan


@pytest.fixture
def dummy_symbols():
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]


@patch("modules.gemini_chart_analyzer.services.batch_scan_service.MarketBatchScanner")
def test_batch_pipeline_integration(mock_scanner_class, dummy_symbols):
    """Test the batch pipeline End-to-End with mocked external APIs."""

    # Mock scanner instance
    mock_scanner_instance = MagicMock()
    mock_scanner_class.return_value = mock_scanner_instance

    # Prepare mocked scan result
    mock_results = {}
    for i, sym in enumerate(dummy_symbols):
        signal = "LONG" if i % 2 == 0 else "SHORT"
        mock_results[sym] = SymbolScanResult(
            timeframes={"1h": SignalResult(signal=signal, confidence=0.85)},
            aggregated=SignalResult(signal=signal, confidence=0.85),
        )

    mock_batch_scan_result = BatchScanResult(
        all_results=mock_results,
        summary={"LONG": 3, "SHORT": 2, "NONE": 0},
        total_batches=1,
        batches_processed=1,
    )
    mock_batch_scan_result.long_symbols = [s for s, r in mock_results.items() if r.aggregated.signal == "LONG"]
    mock_scanner_instance.scan_market.return_value = mock_batch_scan_result

    # Run the pipeline
    config_dict = {
        "timeframe": "1h",
        "limit": 500,
        "cooldown": 1.0,
        "pre_filter": {"use_pre_filter": False},
    }
    config = BatchScanConfig.model_validate(config_dict)

    # Actually run_batch_scan doesn't need data_fetcher argument
    result = run_batch_scan(config)

    # Convert BatchScanResult or dict to verify
    assert result is not None
    assert len(result.all_results) == 5
    assert "BTC/USDT" in result.all_results
    assert result.all_results["BTC/USDT"].aggregated.signal == "LONG"
    assert result.all_results["ETH/USDT"].aggregated.signal == "SHORT"
    assert result.summary["LONG"] == 3
    assert result.summary["SHORT"] == 2

    # Verify calls
    mock_scanner_class.assert_called_once()
    mock_scanner_instance.scan_market.assert_called_once()
