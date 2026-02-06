"""Shared pytest fixtures for auto_trade tests."""

import pytest
from modules.auto_trade.core.atc_scanner import SignalResult
from modules.auto_trade.core.gemini_integration import GeminiSignal


@pytest.fixture
def sample_signal_result():
    """Factory fixture for creating SignalResult objects."""

    def _make_signal(
        symbol: str = "BTC/USDT", score: float = 0.9, signal_type: str = "LONG", xgboost_conf: float = 0.8, **kwargs
    ):
        """Create a SignalResult with default values."""
        details = kwargs.get("details", {"xgboost_conf": xgboost_conf})
        details.update(kwargs.get("extra_details", {}))

        strengths = kwargs.get("strengths", {"5m": 0.8, "15m": 0.7, "1h": 0.9})
        strengths.update(kwargs.get("extra_strengths", {}))

        return SignalResult(symbol, score, signal_type, details, strengths)

    return _make_signal


@pytest.fixture
def sample_gemini_signal():
    """Factory fixture for creating GeminiSignal objects."""

    def _make_signal(symbol: str = "BTC/USDT", **kwargs):
        """Create a GeminiSignal with default values."""
        return GeminiSignal(
            trend=kwargs.get("trend", "UP"),
            signal=kwargs.get("signal", "LONG"),
            confidence=kwargs.get("confidence", 0.9),
            entry=kwargs.get("entry", 50000),
            stop_loss=kwargs.get("stop_loss", 49000),
            take_profit=kwargs.get("take_profit", 52000),
            reasoning=kwargs.get("reasoning", ""),
        )

    return _make_signal
