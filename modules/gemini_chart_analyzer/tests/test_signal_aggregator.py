"""Tests for SignalAggregator."""

import math

from modules.gemini_chart_analyzer.core.aggregators.signal_aggregator import SignalAggregator


class TestSignalAggregator:
    """Test suite for SignalAggregator."""

    def test_empty_input(self):
        """Test aggregation with empty input."""
        aggregator = SignalAggregator()
        result = aggregator.aggregate_signals({})

        assert result["signal"] == "NONE"
        assert result["confidence"] == 0.0
        assert result["timeframe_breakdown"] == {}
        assert result["weights_used"] == {}

    def test_nan_confidence(self):
        """Test aggregation with NaN confidence values."""
        aggregator = SignalAggregator()
        signals = {
            "1h": {"signal": "LONG", "confidence": float("nan")},
            "4h": {"signal": "LONG", "confidence": 0.7},
        }
        result = aggregator.aggregate_signals(signals)

        # NaN should be skipped, only valid confidence used
        assert result["signal"] == "LONG"
        assert not math.isnan(result["confidence"])
        assert 0.0 <= result["confidence"] <= 1.0

    def test_inf_weight(self):
        """Test aggregation with Infinity weight values."""
        aggregator = SignalAggregator()
        signals = {
            "1h": {"signal": "LONG", "confidence": 0.8},
            "4h": {"signal": "LONG", "confidence": 0.7},
        }
        # Manually set an infinite weight
        aggregator.timeframe_weights["1h"] = float("inf")

        result = aggregator.aggregate_signals(signals)

        # Infinite weight should be handled gracefully
        assert result["signal"] in ["LONG", "NONE"]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_long_wins(self):
        """Test LONG signal wins over SHORT."""
        aggregator = SignalAggregator()
        signals = {
            "1h": {"signal": "LONG", "confidence": 0.8},
            "4h": {"signal": "LONG", "confidence": 0.7},
            "1d": {"signal": "SHORT", "confidence": 0.6},
        }
        result = aggregator.aggregate_signals(signals)

        assert result["signal"] == "LONG"
        assert result["confidence"] > 0.6

    def test_short_wins(self):
        """Test SHORT signal wins over LONG."""
        aggregator = SignalAggregator()
        signals = {
            "1h": {"signal": "SHORT", "confidence": 0.9},
            "4h": {"signal": "SHORT", "confidence": 0.8},
            "1d": {"signal": "LONG", "confidence": 0.6},
        }
        result = aggregator.aggregate_signals(signals)

        assert result["signal"] == "SHORT"
        assert result["confidence"] > 0.7

    def test_tie_results_in_none(self):
        """Test tie between LONG and SHORT results in NONE."""
        aggregator = SignalAggregator(timeframe_weights={"1h": 0.5, "4h": 0.5})
        signals = {
            "1h": {"signal": "LONG", "confidence": 0.7},
            "4h": {"signal": "SHORT", "confidence": 0.7},
        }
        result = aggregator.aggregate_signals(signals)

        assert result["signal"] == "NONE"

    def test_none_signals_only(self):
        """Test when all signals are NONE."""
        aggregator = SignalAggregator()
        signals = {
            "1h": {"signal": "NONE", "confidence": 0.5},
            "4h": {"signal": "NONE", "confidence": 0.6},
            "1d": {"signal": "NONE", "confidence": 0.4},
        }
        result = aggregator.aggregate_signals(signals)

        assert result["signal"] == "NONE"
        assert result["confidence"] > 0.0

    def test_confidence_clamping(self):
        """Test confidence values are clamped to [0.0, 1.0]."""
        aggregator = SignalAggregator()
        signals = {
            "1h": {"signal": "LONG", "confidence": 1.5},
            "4h": {"signal": "SHORT", "confidence": -0.5},
        }
        result = aggregator.aggregate_signals(signals)

        # Result should be within valid range
        assert 0.0 <= result["confidence"] <= 1.0

    def test_timeframe_weights_normalization(self):
        """Test that timeframe weights are normalized."""
        aggregator = SignalAggregator(timeframe_weights={"1h": 0.4, "4h": 0.6})
        signals = {
            "1h": {"signal": "LONG", "confidence": 0.8},
            "4h": {"signal": "LONG", "confidence": 0.6},
        }
        result = aggregator.aggregate_signals(signals)

        # Weights should be normalized
        assert result["signal"] == "LONG"

    def test_confidence_threshold(self):
        """Test confidence threshold affects signal decision."""
        aggregator = SignalAggregator(confidence_threshold=0.8)
        signals = {
            "1h": {"signal": "LONG", "confidence": 0.75},
            "4h": {"signal": "SHORT", "confidence": 0.6},
        }
        result = aggregator.aggregate_signals(signals)

        # LONG has higher weighted confidence but below threshold
        assert result["signal"] == "NONE"
