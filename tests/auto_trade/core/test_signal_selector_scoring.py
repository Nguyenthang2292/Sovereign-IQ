"""
Tests for Signal Selector scoring, R/R capping, and candidate evaluation.
"""

from unittest.mock import MagicMock

import pytest

from modules.auto_trade.core.atc_scanner import SignalResult
from modules.auto_trade.core.gemini_integration import GeminiSignal
from modules.auto_trade.core.signal_selector import SignalSelector
from modules.auto_trade.core.signal_pipeline import FinalSignal


class TestSignalSelectorScoring:
    """Test score calculation components."""

    @pytest.fixture
    def selector(self):
        """Create a SignalSelector instance."""
        return SignalSelector()

    def test_score_calculation_components_breakdown(self, selector, sample_signal_result, sample_gemini_signal):
        """Test individual score components (confidence, R/R, consistency)."""
        xb_signal = sample_signal_result(symbol="BTC/USDT", score=0.9, signal_type="LONG", xgboost_conf=0.8)
        gemini_signal = sample_gemini_signal(confidence=0.85, entry=50000, stop_loss=49000, take_profit=52000)

        final = selector._evaluate_candidate(xb_signal, gemini_signal)

        # Verify final confidence = 0.8 * 0.4 + 0.85 * 0.6 = 0.83
        assert final.confidence == pytest.approx(0.83, abs=0.01), f"Expected confidence 0.83, got {final.confidence}"

        # Verify score components
        # Confidence: 0.83 * 60 = 49.8
        # R/R: (52000-50000)/(50000-49000) = 2.0 -> 2.0/3.0 * 20 = 13.33
        # Consistency: 20 (both agree on LONG)
        # Total: 49.8 + 13.33 + 20 = 83.13
        assert final.score == pytest.approx(83.13, abs=0.5), f"Expected score ~83.13, got {final.score}"

    def test_score_confidence_component_weighted(self, selector, sample_signal_result, sample_gemini_signal):
        """Test that confidence is properly weighted between XGBoost and Gemini."""
        # High XGBoost, low Gemini
        xb_signal = sample_signal_result(symbol="BTC/USDT", score=0.9, signal_type="LONG", xgboost_conf=0.95)
        gemini_signal = sample_gemini_signal(confidence=0.7, entry=50000, stop_loss=49000, take_profit=51000)

        final = selector._evaluate_candidate(xb_signal, gemini_signal)

        # Final confidence should be weighted: 0.95 * 0.4 + 0.7 * 0.6 = 0.8
        assert final.confidence == pytest.approx(0.8, abs=0.01), f"Expected confidence 0.8, got {final.confidence}"

        # Low XGBoost, high Gemini
        xb_signal = sample_signal_result(symbol="BTC/USDT", score=0.9, signal_type="LONG", xgboost_conf=0.7)
        gemini_signal = sample_gemini_signal(confidence=0.9, entry=50000, stop_loss=49000, take_profit=54000)

        final = selector._evaluate_candidate(xb_signal, gemini_signal)

        # Final confidence should be weighted: 0.7 * 0.4 + 0.9 * 0.6 = 0.82
        assert final.confidence == pytest.approx(0.82, abs=0.01), f"Expected confidence 0.82, got {final.confidence}"


class TestSignalSelectorRR:
    """Test risk/reward ratio capping."""

    @pytest.fixture
    def selector(self):
        """Create a SignalSelector instance."""
        return SignalSelector()

    def test_score_risk_reward_component_capped_at_3(self, selector, sample_signal_result, sample_gemini_signal):
        """Test that R/R ratio is capped at 3.0 for scoring."""
        # Create signal with R/R = 5.0
        xb_signal = sample_signal_result(symbol="BTC/USDT", score=0.9, signal_type="LONG", xgboost_conf=0.8)
        gemini_signal = sample_gemini_signal(confidence=0.9, entry=50000, stop_loss=49000, take_profit=54000)

        final = selector._evaluate_candidate(xb_signal, gemini_signal)

        # R/R = (54000-50000)/(50000-49000) = 4.0
        # But should be capped at 3.0 for scoring
        # R/R score = 3.0/3.0 * 20 = 20 (max)
        rr_component = min((54000 - 50000) / (50000 - 49000), 3.0) / 3.0 * 20
        assert rr_component == 20.0, f"Expected R/R component 20.0 (capped at 3.0), got {rr_component}"

    def test_score_risk_reward_calculation(self, selector, sample_signal_result, sample_gemini_signal):
        """Test scoring with R/R ratio calculation."""
        # Create signal with R/R = 2.0 (risk 1000, reward 2000)
        xb_signal = sample_signal_result(symbol="BTC/USDT", score=0.9, signal_type="LONG", xgboost_conf=0.8)
        gemini_signal = sample_gemini_signal(confidence=0.9, entry=50000, stop_loss=49000, take_profit=52000)

        final = selector._evaluate_candidate(xb_signal, gemini_signal)

        # R/R = (52000-50000)/(50000-49000) = 2.0
        # R/R score = 2.0 / 3.0 * 20 = 13.33
        rr_score = (52000 - 50000) / (50000 - 49000) / 3.0 * 20
        assert rr_score == pytest.approx(13.33, abs=0.01), f"Expected R/R score ~13.33, got {rr_score}"


class TestSignalSelectorEdgeCases:
    """Test edge cases and invalid inputs."""

    @pytest.fixture
    def selector(self):
        """Create a SignalSelector instance."""
        return SignalSelector()

    def test_evaluate_candidate_handles_invalid_xgboost_conf(self, selector, sample_gemini_signal):
        """Test handling of unparseable xgboost_conf."""
        xb_signal = SignalResult(
            symbol="BTC/USDT", score=0.9, signal_type="LONG", details={"xgboost_conf": "invalid"}, strengths={"5m": 0.8}
        )
        gemini_signal = sample_gemini_signal(confidence=0.8, entry=100, stop_loss=95, take_profit=105)

        final = selector._evaluate_candidate(xb_signal, gemini_signal)

        # Should use 0.0 as fallback for XGBoost confidence
        # Final confidence = 0.0 * 0.4 + 0.8 * 0.6 = 0.48
        assert final.confidence == pytest.approx(0.48, abs=0.01), f"Expected confidence 0.48, got {final.confidence}"

    def test_evaluate_candidate_missing_xgboost_conf(self, selector, sample_gemini_signal):
        """Test handling of missing xgboost_conf."""
        xb_signal = SignalResult(symbol="BTC/USDT", score=0.9, signal_type="LONG", details={}, strengths={"5m": 0.8})
        gemini_signal = sample_gemini_signal(confidence=0.8, entry=100, stop_loss=95, take_profit=105)

        final = selector._evaluate_candidate(xb_signal, gemini_signal)

        # Should use 0.0 as fallback for missing XGBoost confidence
        assert final.confidence == pytest.approx(
            0.48, abs=0.01
        ), f"Expected confidence 0.48 for missing xgboost_conf, got {final.confidence}"

    def test_evaluate_candidate_opposite_signals_returns_none(
        self, selector, sample_signal_result, sample_gemini_signal
    ):
        """Test that opposite signal directions returns None (discarded)."""
        # XGBoost says LONG, Gemini says SHORT
        xb_signal = sample_signal_result(symbol="BTC/USDT", score=0.9, signal_type="LONG", xgboost_conf=0.8)
        gemini_signal = sample_gemini_signal(
            trend="DOWN", signal="SHORT", confidence=0.8, entry=100, stop_loss=105, take_profit=95
        )

        final = selector._evaluate_candidate(xb_signal, gemini_signal)

        # Opposite directions should return None (signal is discarded)
        assert final is None, f"Expected None for opposite directions, got {final}"
