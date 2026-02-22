"""Tests for batch_scan_service pre-filter orchestration."""

from unittest.mock import Mock, patch

from modules.gemini_chart_analyzer.core.scanner_types import BatchScanResult
from modules.gemini_chart_analyzer.services.batch_scan_service import BatchScanConfig, PreFilterConfig, run_batch_scan


def _empty_result() -> BatchScanResult:
    """Create an empty BatchScanResult to avoid report generation side effects."""
    return BatchScanResult(
        long_symbols=[],
        short_symbols=[],
        none_symbols=[],
        summary={"total_scanned": 0},
        results_file="",
    )


def test_run_batch_scan_without_prefilter_skips_worker() -> None:
    """Pre-filter worker is not invoked when pre_filter.enabled is False."""
    with (
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.MarketBatchScanner") as mock_scanner_class,
        patch("modules.gemini_chart_analyzer.core.prefilter.workflow.run_prefilter_worker") as mock_prefilter,
    ):
        mock_scanner = Mock()
        mock_scanner.scan_market.return_value = _empty_result()
        mock_scanner_class.return_value = mock_scanner

        config = BatchScanConfig(timeframe="1h", max_symbols=5, pre_filter=PreFilterConfig(enabled=False))
        result = run_batch_scan(config)

        assert isinstance(result, BatchScanResult)
        mock_prefilter.assert_not_called()
        scan_call = mock_scanner.scan_market.call_args
        scan_config = scan_call.args[0]
        assert scan_config.initial_symbols is None


def test_run_batch_scan_with_prefilter_sets_initial_symbols() -> None:
    """Pre-filter worker output is forwarded to ScanConfig.initial_symbols."""
    selected_symbols = ["BTC/USDT", "ETH/USDT"]

    with (
        patch("modules.gemini_chart_analyzer.services.batch_scan_service.MarketBatchScanner") as mock_scanner_class,
        patch("modules.gemini_chart_analyzer.cli.batch_scanner.utils.init_components") as mock_init_components,
        patch("modules.gemini_chart_analyzer.core.prefilter.workflow.run_prefilter_worker") as mock_prefilter,
    ):
        mock_scanner = Mock()
        mock_scanner.get_all_symbols.return_value = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        mock_scanner.scan_market.return_value = _empty_result()
        mock_scanner_class.return_value = mock_scanner

        mock_init_components.return_value = (Mock(), Mock())
        mock_prefilter.return_value = selected_symbols

        config = BatchScanConfig(
            timeframe="1h",
            max_symbols=10,
            pre_filter=PreFilterConfig(enabled=True, percentage=50.0, mode="voting"),
        )
        result = run_batch_scan(config)

        assert isinstance(result, BatchScanResult)
        mock_prefilter.assert_called_once()
        scan_call = mock_scanner.scan_market.call_args
        scan_config = scan_call.args[0]
        assert scan_config.initial_symbols == selected_symbols
