"""Tests for GeminiBatchChartAnalyzer."""

import json
from unittest.mock import patch

from modules.gemini_chart_analyzer.core.analyzers.gemini_batch_chart_analyzer import (
    GeminiBatchChartAnalyzer,
)


def _signal_field(signal_obj, field_name: str):
    """Read signal field from dataclass/object or dict result."""
    if isinstance(signal_obj, dict):
        return signal_obj[field_name]
    return getattr(signal_obj, field_name)


class TestGeminiBatchChartAnalyzer:
    """Test suite for GeminiBatchChartAnalyzer."""

    def test_extract_json_from_text_fenced_block(self):
        """Test JSON extraction from fenced code block."""
        analyzer = GeminiBatchChartAnalyzer(api_key="test_key")
        response_text = """
        ```json
        {"BTC/USDT": {"signal": "LONG", "confidence": 0.85}}
        ```
        """
        result = analyzer._extract_json_from_text(response_text)

        assert result is not None
        assert "BTC/USDT" in result

    def test_extract_json_from_text_bare_json(self):
        """Test JSON extraction from bare JSON (no code block)."""
        analyzer = GeminiBatchChartAnalyzer(api_key="test_key")
        response_text = '{"ETH/USDT": {"signal": "SHORT", "confidence": 0.70}}'
        result = analyzer._extract_json_from_text(response_text)

        assert result is not None
        assert "ETH/USDT" in result

    def test_extract_json_from_text_malformed(self):
        """Test JSON extraction from malformed text."""
        analyzer = GeminiBatchChartAnalyzer(api_key="test_key")
        response_text = "Here is some text with JSON-like content: {incomplete"

        result = analyzer._extract_json_from_text(response_text)

        assert result is None

    def test_extract_json_from_text_nested_braces(self):
        """Test JSON extraction from text with nested braces."""
        analyzer = GeminiBatchChartAnalyzer(api_key="test_key")
        response_text = """
        Some text before
        ```json
        {
            "BTC/USDT": {"signal": "LONG", "confidence": 0.85},
            "ETH/USDT": {"signal": {"nested": "value"}, "confidence": 0.70}
        }
        ```
        Some text after
        """
        result = analyzer._extract_json_from_text(response_text)

        assert result is not None
        assert "BTC/USDT" in result

    def test_parse_json_response_valid(self):
        """Test parsing valid JSON response."""
        analyzer = GeminiBatchChartAnalyzer(api_key="test_key")
        expected_symbols = ["BTC/USDT", "ETH/USDT"]
        json_str = json.dumps(
            {
                "BTC/USDT": {"signal": "LONG", "confidence": 0.85},
                "ETH/USDT": {"signal": "SHORT", "confidence": 0.70},
            }
        )

        result = analyzer._parse_json_response(json_str, expected_symbols)

        assert len(result) == 2
        assert "BTC/USDT" in result
        assert "ETH/USDT" in result
        assert _signal_field(result["BTC/USDT"], "signal") == "LONG"
        assert _signal_field(result["ETH/USDT"], "signal") == "SHORT"

    def test_parse_json_response_invalid_signal(self):
        """Test parsing JSON with invalid signal values."""
        analyzer = GeminiBatchChartAnalyzer(api_key="test_key")
        expected_symbols = ["BTC/USDT"]
        json_str = json.dumps(
            {
                "BTC/USDT": {"signal": "INVALID", "confidence": 0.85},
            }
        )

        result = analyzer._parse_json_response(json_str, expected_symbols)

        assert len(result) == 1
        # Invalid signal should default to NONE
        assert _signal_field(result["BTC/USDT"], "signal") == "NONE"

    def test_parse_json_response_missing_symbols(self):
        """Test parsing JSON with missing expected symbols."""
        analyzer = GeminiBatchChartAnalyzer(api_key="test_key")
        expected_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        json_str = json.dumps(
            {
                "BTC/USDT": {"signal": "LONG", "confidence": 0.85},
                "ETH/USDT": {"signal": "SHORT", "confidence": 0.70},
            }
        )

        result = analyzer._parse_json_response(json_str, expected_symbols)

        assert len(result) == 3
        # Missing symbol should default to NONE
        assert _signal_field(result["BNB/USDT"], "signal") == "NONE"
        assert _signal_field(result["BNB/USDT"], "confidence") == 0.0

    def test_parse_json_response_partial_symbols(self):
        """Test parsing JSON with partial symbol list."""
        analyzer = GeminiBatchChartAnalyzer(api_key="test_key")
        expected_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"]
        json_str = json.dumps(
            {
                "BTC/USDT": {"signal": "LONG", "confidence": 0.85},
                "ETH/USDT": {"signal": "SHORT", "confidence": 0.70},
            }
        )

        result = analyzer._parse_json_response(json_str, expected_symbols)

        assert len(result) == 4
        assert _signal_field(result["ADA/USDT"], "signal") == "NONE"

    def test_is_actionable_signal(self):
        """Test identification of actionable signals."""
        analyzer = GeminiBatchChartAnalyzer(api_key="test_key")

        long_signal = {"signal": "LONG", "confidence": 0.85}
        short_signal = {"signal": "SHORT", "confidence": 0.70}
        none_signal = {"signal": "NONE", "confidence": 0.50}

        assert analyzer._is_actionable_signal(long_signal) is True
        assert analyzer._is_actionable_signal(short_signal) is True
        assert analyzer._is_actionable_signal(none_signal) is False

    def test_clamp_confidence(self):
        """Test confidence clamping to [0.0, 1.0]."""
        analyzer = GeminiBatchChartAnalyzer(api_key="test_key")

        assert analyzer._clamp_confidence(1.5) == 1.0
        assert analyzer._clamp_confidence(0.85) == 0.85
        assert analyzer._clamp_confidence(0.5) == 0.5
        assert analyzer._clamp_confidence(0.0) == 0.0
        assert analyzer._clamp_confidence(-0.5) == 0.0

    @patch("modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer.PIL.Image.open")
    @patch.object(GeminiBatchChartAnalyzer, "_analyze_with_custom_prompt")
    def test_analyze_batch_chart_success(self, mock_analyze, mock_open):
        """Test successful batch chart analysis."""
        mock_analyze.return_value = json.dumps(
            {
                "BTC/USDT": {"signal": "LONG", "confidence": 0.85},
            }
        )
        analyzer = GeminiBatchChartAnalyzer(api_key="test_key", cooldown_seconds=0)

        result = analyzer.analyze_batch_chart(
            "test.png",
            batch_id=1,
            total_batches=10,
            symbols=["BTC/USDT"],
        )

        assert len(result) == 1
        assert "BTC/USDT" in result
        assert _signal_field(result["BTC/USDT"], "signal") == "LONG"
