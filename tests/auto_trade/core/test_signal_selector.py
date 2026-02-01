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
        """Test selection when Gemini data is missing (should return None due to missing prices)."""
        xb_signals = [SignalResult("BTC/USDT", 0.9, "LONG", {"xgboost_conf": "0.8"})]
        gemini_signals = {}

        # New strict validation requires price levels, which are missing here
        final = selector.select_best_signal(xb_signals, gemini_signals)

        assert final is None

    def test_reject_zero_prices(self, selector):
        """Test that signals with zero prices are rejected."""
        xb_signals = [SignalResult("BTC/USDT", 0.9, "LONG", {})]
        gemini_signals = {
            "BTC/USDT": GeminiSignal(trend="UP", signal="LONG", confidence=0.9, entry=0, stop_loss=0, take_profit=0)
        }

        final = selector.select_best_signal(xb_signals, gemini_signals)
        assert final is None

    def test_reject_invalid_long_prices(self, selector):
        """Test that invalid LONG price structure is rejected."""
        xb_signals = [SignalResult("BTC/USDT", 0.9, "LONG", {})]
        gemini_signals = {
            "BTC/USDT": GeminiSignal(
                trend="UP",
                signal="LONG",
                confidence=0.9,
                entry=50000,
                stop_loss=51000,
                take_profit=52000,
                # Invalid: SL > Entry
            )
        }

        final = selector.select_best_signal(xb_signals, gemini_signals)
        assert final is None

    def test_empty_signals(self, selector):
        """Test with empty signal inputs."""
        final = selector.select_best_signal([], {})
        assert final is None

    def test_all_below_threshold(self, selector):
        """Test that signals below confidence threshold are rejected."""
        # Config has threshold 0.5. Let's make a signal with confidence 0.4
        xb_signals = [SignalResult("BTC/USDT", 0.9, "LONG", {"xgboost_conf": "0.3"})]  # 0.3*0.4 = 0.12
        gemini_signals = {
            "BTC/USDT": GeminiSignal(
                trend="UP", signal="LONG", confidence=0.4, entry=500, stop_loss=490, take_profit=520
            )  # 0.4*0.6 = 0.24. Total = 0.36 < 0.5
        }

        final = selector.select_best_signal(xb_signals, gemini_signals)
        assert final is None

    def test_custom_weights(self):
        """Test configuration with custom weights."""
        # 100% XGBoost weight
        selector = SignalSelector(config={"weight_xgboost": 1.0, "weight_gemini": 0.0, "min_confidence_threshold": 0.5})

        xb_signals = [SignalResult("BTC/USDT", 0.9, "LONG", {"xgboost_conf": "0.8"})]
        gemini_signals = {
            "BTC/USDT": GeminiSignal(
                trend="UP", signal="LONG", confidence=0.2, entry=500, stop_loss=490, take_profit=520
            )
        }

        # Should rely purely on XGBoost score (0.8) ignoring Gemini score (0.2)
        final = selector.select_best_signal(xb_signals, gemini_signals)
        assert final is not None
        assert final.confidence == 0.8  # Normalized: (0.8*1.0 + 0.2*0.0) / 1.0 = 0.8

    def test_confidence_normalization_cap(self, selector):
        """Test that confidence is capped at 1.0 even if inputs are higher."""
        # Case 1: XGBoost > 1.0 (without Gemini)
        xb_signals = [SignalResult("BTC/USDT", 1.5, "LONG", {"xgboost_conf": "1.5"})]
        gemini_signals = {}
        # NOTE: This returns None now because missing Gemini means missing price levels (strict validation)
        # So we must verify the behavior is consistent (None) OR provide Gemini

        # Let's test providing Gemini but with high scores
        gemini_signals_high = {"BTC/USDT": GeminiSignal("UP", "LONG", 1.5, 500, 490, 520)}

        final = selector.select_best_signal(xb_signals, gemini_signals_high)
        assert final.confidence == 1.0

    def test_calculate_risk_reward(self, selector):
        """Test R/R calculation logic indirectly via log output check or internal method call."""
        # Using internal method directly for unit testing
        from modules.auto_trade.core.signal_selector import FinalSignal

        sig_long = FinalSignal("BTC", "LONG", 50000, 49000, 52000)  # R:1000, R:2000 -> 2.0
        rr_long = selector._calculate_risk_reward_ratio(sig_long)
        assert rr_long == 2.0

        sig_short = FinalSignal("BTC", "SHORT", 50000, 51000, 48000)  # R:1000, R:2000 -> 2.0
        rr_short = selector._calculate_risk_reward_ratio(sig_short)
        assert rr_short == 2.0

    def test_leverage_validation(self):
        """Test that invalid leverage raises ValueError during FinalSignal init."""
        from modules.auto_trade.core.signal_selector import FinalSignal

        # Valid
        FinalSignal("BTC", "LONG", 100, 90, 110, leverage=5)

        # Invalid
        with pytest.raises(ValueError, match="Invalid leverage"):
            FinalSignal("BTC", "LONG", 100, 90, 110, leverage=15)

        with pytest.raises(ValueError, match="Invalid leverage"):
            FinalSignal("BTC", "LONG", 100, 90, 110, leverage=0)
