"""Tests for ChartGenerator."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from modules.gemini_chart_analyzer.core.generators.chart_generator import (
    ChartGenerator,
)


class TestChartGenerator:
    """Test suite for ChartGenerator."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")
        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": 50000 + pd.Series(range(100)) * 10,
                "high": 50100 + pd.Series(range(100)) * 10,
                "low": 49900 + pd.Series(range(100)) * 10,
                "close": 50050 + pd.Series(range(100)) * 10,
                "volume": 1000 + pd.Series(range(100)),
            }
        )

    def test_add_indicators_ma(self, sample_ohlcv_data):
        """Test adding Moving Average indicators."""
        generator = ChartGenerator()
        indicators = {"MA": {"periods": [20, 50], "type": "SMA"}}

        df = generator._add_indicators(sample_ohlcv_data, indicators)

        assert "MA_20" in df.columns
        assert "MA_50" in df.columns

    def test_add_indicators_rsi(self, sample_ohlcv_data):
        """Test adding RSI indicator."""
        generator = ChartGenerator()
        indicators = {"RSI": {"period": 14}}

        df = generator._add_indicators(sample_ohlcv_data, indicators)

        assert "RSI_14" in df.columns

    def test_add_indicators_macd(self, sample_ohlcv_data):
        """Test adding MACD indicator."""
        generator = ChartGenerator()
        indicators = {"MACD": {"fast": 12, "slow": 26, "signal": 9}}

        df = generator._add_indicators(sample_ohlcv_data, indicators)

        assert "MACD" in df.columns
        assert "MACD_signal" in df.columns
        assert "MACD_hist" in df.columns

    def test_add_indicators_bb(self, sample_ohlcv_data):
        """Test adding Bollinger Bands indicator."""
        generator = ChartGenerator()
        indicators = {"BB": {"period": 20, "std": 2}}

        df = generator._add_indicators(sample_ohlcv_data, indicators)

        assert "BB_upper_20" in df.columns
        assert "BB_middle_20" in df.columns
        assert "BB_lower_20" in df.columns

    def test_add_indicators_multiple(self, sample_ohlcv_data):
        """Test adding multiple indicators at once."""
        generator = ChartGenerator()
        indicators = {
            "MA": {"periods": [20], "type": "SMA"},
            "RSI": {"period": 14},
            "MACD": {"fast": 12, "slow": 26, "signal": 9},
        }

        df = generator._add_indicators(sample_ohlcv_data, indicators)

        assert "MA_20" in df.columns
        assert "RSI_14" in df.columns
        assert "MACD" in df.columns

    def test_add_indicators_none(self, sample_ohlcv_data):
        """Test that None indicators don't add any columns."""
        generator = ChartGenerator()

        df = generator._add_indicators(sample_ohlcv_data, None)

        # DataFrame should remain unchanged
        assert list(df.columns) == list(sample_ohlcv_data.columns)

    def test_chart_output_path_generation(self, sample_ohlcv_data):
        """Test that chart output path is generated correctly."""
        generator = ChartGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_chart.png")

            result_path = generator.create_chart(
                sample_ohlcv_data,
                "BTC/USDT",
                "1h",
                output_path=output_path,
            )

            assert os.path.exists(result_path)
            assert result_path == output_path

    def test_chart_output_path_auto_generated(self, sample_ohlcv_data):
        """Test that chart output path is auto-generated if not provided."""
        generator = ChartGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "modules.gemini_chart_analyzer.core.generators.chart_generator.get_charts_dir",
                return_value=Path(temp_dir),
            ):
                result_path = generator.create_chart(
                    sample_ohlcv_data,
                    "BTC/USDT",
                    "1h",
                    output_path=None,
                )

            assert os.path.exists(result_path)
            assert "BTC_USDT" in result_path
            assert "1h" in result_path
            assert result_path.endswith(".png")

    def test_chart_symbol_sanitization(self, sample_ohlcv_data):
        """Test that symbol is sanitized in output path."""
        generator = ChartGenerator()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "modules.gemini_chart_analyzer.core.generators.chart_generator.get_charts_dir",
                return_value=Path(temp_dir),
            ):
                result_path = generator.create_chart(
                    sample_ohlcv_data,
                    "BTC/USDT:PERP",
                    "1h",
                    output_path=None,
                )

            assert os.path.exists(result_path)
            assert ":" not in os.path.basename(result_path)
            assert "BTC_USDT" in result_path

    def test_create_chart_with_indicators(self, sample_ohlcv_data):
        """Test creating chart with various indicators."""
        generator = ChartGenerator()
        indicators = {
            "MA": {"periods": [20, 50], "type": "SMA"},
            "RSI": {"period": 14},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_chart.png")
            result_path = generator.create_chart(
                sample_ohlcv_data,
                "ETH/USDT",
                "4h",
                indicators=indicators,
                output_path=output_path,
            )

            assert os.path.exists(result_path)
