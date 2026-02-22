import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.gemini_chart_analyzer.services.chart_analysis_service import SingleAnalysisConfig, run_chart_analysis


@pytest.fixture
def dummy_ohlcv_data():
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100 + i for i in range(100)],
            "high": [110 + i for i in range(100)],
            "low": [90 + i for i in range(100)],
            "close": [105 + i for i in range(100)],
            "volume": [1000 + i * 10 for i in range(100)],
        },
        index=dates,
    )
    return df


@patch("modules.gemini_chart_analyzer.services.chart_analysis_service.GeminiChartAnalyzer")
@patch("modules.gemini_chart_analyzer.services.chart_analysis_service.ChartGenerator")
def test_single_pipeline_integration(mock_chart_gen_class, mock_gemini_analyzer_class, dummy_ohlcv_data):
    """Test the single analyzer pipeline End-to-End with mocked external APIs."""
    # Setup mocks
    mock_df = dummy_ohlcv_data
    mock_data_fetcher = MagicMock()
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "MockExchange")

    mock_generator_instance = MagicMock()
    mock_generator_instance.create_chart.return_value = "dummy_chart.png"
    mock_chart_gen_class.return_value = mock_generator_instance

    mock_analyzer_instance = MagicMock()
    mock_analyzer_instance.analyze_chart.return_value = "MOCK GEMINI ANALYSIS TEXT WITH LONG SIGNAL."
    mock_gemini_analyzer_class.return_value = mock_analyzer_instance

    # We mock generate_html_report to avoid creating actual files, or we let it run and clean up
    # Actually, let's mock it for speed, or let it run in a temp dir if possible.
    # The run_chart_analysis function uses HTMLReportGenerator.
    with patch(
        "modules.gemini_chart_analyzer.services.chart_analysis_service.generate_html_report"
    ) as mock_html_gen_func:
        mock_html_gen_func.return_value = "dummy_report.html"

        # Run pipeline
        config = SingleAnalysisConfig(symbol="BTC/USDT", timeframe="1h")
        result = run_chart_analysis(config, mock_data_fetcher)

        # Assertions
        assert result is not None
        assert result["symbol"] == "BTC/USDT"
        assert result["timeframe"] == "1h"
        assert "MOCK GEMINI ANALYSIS TEXT" in result["analysis"]
        assert result["chart_path"] == "dummy_chart.png"
        assert result["html_report_path"] == "dummy_report.html"

        # Verify calls
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.assert_called_once_with(
            symbol="BTC/USDT", timeframe="1h", limit=500, check_freshness=False
        )
        mock_generator_instance.create_chart.assert_called_once()
        mock_analyzer_instance.analyze_chart.assert_called_once()
        mock_html_gen_func.assert_called_once()
