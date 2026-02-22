"""Tests for __init__ file imports."""

import pytest


def test_module_imports():
    """Test that main module can be imported without errors."""
    try:
        from modules.gemini_chart_analyzer import (
            GeminiChartAnalyzer,
            MarketBatchScanner,
            SignalAggregator,
        )

        assert MarketBatchScanner is not None
        assert GeminiChartAnalyzer is not None
        assert SignalAggregator is not None
    except ImportError as e:
        pytest.fail(f"Module import failed: {e}")
