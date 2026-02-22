import pytest
from unittest.mock import MagicMock
from enum import Enum

from modules.auto_trade.core.signal_pipeline import SignalPipeline
from modules.auto_trade.core.atc_scanner import SignalResult
from modules.auto_trade.core.signal_selector import FinalSignal


class MockSymbolManager:
    def refresh_symbols(self):
        pass

    def get_symbols(self):
        return ["BTCUSDT", "ETHUSDT"]


class MockATCScanner:
    def scan_symbols(self, symbols):
        return [
            SignalResult(symbol="BTCUSDT", signal_type="LONG", score=0.9, strengths={"trend": "strong"}, details={}),
            SignalResult(symbol="ETHUSDT", signal_type="SHORT", score=0.8, strengths={"trend": "weak"}, details={}),
        ]


class MockATCScannerException:
    def scan_symbols(self, symbols):
        raise ValueError("Scanner failed")


class MockXGBoostFilter:
    def filter_signals(self, signals):
        # Keep only BTCUSDT for test
        return [s for s in signals if s.symbol == "BTCUSDT"]


class MockXGBoostFilterEmpty:
    def filter_signals(self, signals):
        return []


class MockGeminiIntegration:
    def is_available(self):
        return True

    async def analyze_candidates_batch_async(self, signals, max_concurrency):
        class MockGeminiSignal:
            def __init__(self):
                self.action = "LONG"
                self.confidence = 0.8
                self.reasoning = "Good"

        return {"BTCUSDT": MockGeminiSignal()}


class MockSignalSelector:
    def select_best_signal(self, xgboost_signals, gemini_results):
        if not xgboost_signals:
            return None
        return FinalSignal(
            symbol="BTCUSDT",
            signal_type="LONG",
            confidence=0.9,
            entry_price=50000,
            take_profit=51000,
            stop_loss=49000,
        )


def test_pipeline_full_run():
    pipeline = SignalPipeline(
        symbol_manager=MockSymbolManager(),
        atc_scanner=MockATCScanner(),
        xgboost_filter=MockXGBoostFilter(),
        gemini_integration=MockGeminiIntegration(),
        signal_selector=MockSignalSelector(),
    )

    result = pipeline.run_pipeline()
    assert result is not None
    assert result.symbol == "BTCUSDT"


def test_pipeline_skips_gemini_when_no_xgboost_candidates():
    pipeline = SignalPipeline(
        symbol_manager=MockSymbolManager(),
        atc_scanner=MockATCScanner(),
        xgboost_filter=MockXGBoostFilterEmpty(),  # Returns empty
        gemini_integration=MockGeminiIntegration(),
        signal_selector=MockSignalSelector(),
    )

    result = pipeline.run_pipeline()
    assert result is None


def test_pipeline_handles_scanner_exception():
    pipeline = SignalPipeline(
        symbol_manager=MockSymbolManager(),
        atc_scanner=MockATCScannerException(),  # Raises exception
        xgboost_filter=MockXGBoostFilter(),
        gemini_integration=MockGeminiIntegration(),
        signal_selector=MockSignalSelector(),
    )

    result = pipeline.run_pipeline()
    assert result is None
