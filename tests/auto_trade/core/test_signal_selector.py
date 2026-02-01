"""
Tests for SignalSelector.
"""

import pytest

from modules.auto_trade.core.atc_scanner import SignalResult
from modules.auto_trade.core.gemini_integration import GeminiSignal
from modules.auto_trade.core.signal_selector import SignalSelector


class TestSignalSelector:
    @pytest.fixture
    def selector(self):
        return SignalSelector(config={"weight_xgboost": 0.4, "weight_gemini": 0.6, "min_confidence_threshold": 0.5})

    def test_select_best_signal_simple(self, selector):
        """Test selection with a single valid signal."""
        xb_signals = [SignalResult("BTC/USDT", 0.9, "LONG", {"xgboost_conf": "0.8"})]
        gemini_signals = {
            "BTC/USDT": GeminiSignal(
                trend="UP", signal="LONG", confidence=0.9, entry=50000, stop_loss=49000, take_profit=52000
            )
        }

        final = selector.select_best_signal(xb_signals, gemini_signals)

        assert final is not None
        assert final.symbol == "BTC/USDT"
        assert final.signal_type == "LONG"
        # 0.8 * 0.4 + 0.9 * 0.6 = 0.32 + 0.54 = 0.86
        assert final.confidence == pytest.approx(0.86)
        assert final.entry_price == 50000

    def test_select_conflict_resolution(self, selector):
        """Test that conflicting signals are rejected."""
        xb_signals = [SignalResult("BTC/USDT", 0.9, "LONG", {"xgboost_conf": "0.8"})]
        gemini_signals = {
            "BTC/USDT": GeminiSignal(
                trend="DOWN", signal="SHORT", confidence=0.9, entry=50000, stop_loss=51000, take_profit=48000
            )
        }

        final = selector.select_best_signal(xb_signals, gemini_signals)
        assert final is None

    def test_select_ranking(self, selector):
        """Test that the best signal is selected."""
        xb_signals = [
            SignalResult("BTC/USDT", 0.8, "LONG", {"xgboost_conf": "0.7"}),
            SignalResult("ETH/USDT", 0.9, "LONG", {"xgboost_conf": "0.8"}),
        ]
        gemini_signals = {
            "BTC/USDT": GeminiSignal("UP", "LONG", 0.7, 50000, 49000, 52000),  # 0.7*0.4 + 0.7*0.6 = 0.70
            "ETH/USDT": GeminiSignal("UP", "LONG", 0.9, 3000, 2900, 3200),  # 0.8*0.4 + 0.9*0.6 = 0.86
        }

        final = selector.select_best_signal(xb_signals, gemini_signals)

        assert final is not None
        assert final.symbol == "ETH/USDT"
        assert final.confidence == pytest.approx(0.86)

    def test_select_without_gemini(self, selector):
        """Test selection when Gemini data is missing."""
        xb_signals = [SignalResult("BTC/USDT", 0.9, "LONG", {"xgboost_conf": "0.8"})]
        gemini_signals = {}

        final = selector.select_best_signal(xb_signals, gemini_signals)

        assert final is not None
        assert final.symbol == "BTC/USDT"
        # Should fallback to XGBoost confidence
        assert final.confidence == 0.8
