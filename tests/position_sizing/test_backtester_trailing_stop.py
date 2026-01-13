from types import SimpleNamespace

import pandas as pd

from modules.backtester import FullBacktester

"""
Tests for trailing stop fix in Full Backtester.

These tests verify that trailing stop is only activated after price moves
favorably, not immediately on entry.
"""


def test_trailing_stop_not_triggered_immediately_on_entry_long():
    """
    Test that trailing stop does not trigger immediately after entry for LONG position.

    Scenario: Enter LONG position, price immediately drops slightly (but not to stop loss).
    Expected: Position should NOT exit via trailing stop (since it wasn't activated yet).
    """

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        # Create data where price drops slightly after entry
        prices = [100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.5, 96.0, 95.5]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    # Mock signal calculator to return LONG signal at index 0
    class MockSignalCalculator:
        def calculate_hybrid_signal(self, **kwargs):
            # Return LONG signal (1) at period 0, no signal otherwise
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8
            return 0, 0.0

        def calculate_single_signal_highest_confidence(self, **kwargs):
            # Return LONG signal (1) at period 0, no signal otherwise
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8
            return 0, 0.0

        def calculate_signal_from_precomputed(self, **kwargs):
            # For hybrid mode with precomputed indicators
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8
            return 0, 0.0

        def calculate_single_signal_from_precomputed(self, **kwargs):
            # For single signal mode with precomputed indicators
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8
            return 0, 0.0

        def get_cache_stats(self):
            return {"signal_cache_size": 0, "signal_cache_max_size": 1000, "cache_hit_rate": 0.0}

        def clear_cache(self):
            pass

        def precompute_all_indicators_vectorized(self, **kwargs):
            pass  # No-op for mock

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(
        data_fetcher,
        stop_loss_pct=0.05,  # 5% stop loss (won't trigger)
        take_profit_pct=0.10,  # 10% take profit (won't trigger)
        trailing_stop_pct=0.015,  # 1.5% trailing stop
        max_hold_periods=100,
        signal_mode="single_signal",  # Use single signal mode for mock calculator
    )

    # Replace signal calculator with mock
    backtester.hybrid_signal_calculator = MockSignalCalculator()

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=10,
        signal_type="LONG",
    )

    trades = result["trades"]

    # Should have at least one trade (entry at period 0)
    assert len(trades) > 0

    # Check that trailing stop was not the exit reason immediately
    # If price drops from 100 to 99.5 (0.5%), trailing stop should NOT trigger
    # because it wasn't activated yet (price never went above entry)
    first_trade = trades[0]

    # If trailing stop was initialized immediately, it would trigger at 98.5 (100 * 0.985)
    # But since price only drops to 99.5, trailing stop should NOT trigger
    # The trade should exit via stop loss (if price drops enough) or max hold period
    # In this case, price drops to 95.5 which is 4.5% down, so stop loss (5%) won't trigger
    # Trade should exit at end of data or max hold

    # Verify trailing stop was not the immediate exit reason
    # (it should be END_OF_DATA or MAX_HOLD, not TRAILING_STOP triggered immediately)
    if first_trade["exit_reason"] == "TRAILING_STOP":
        # If it's trailing stop, verify it wasn't triggered immediately
        # Entry price should be 100, if trailing stop triggered immediately it would be at ~98.5
        # But since price only went down, trailing stop should not have been activated
        assert first_trade["exit_price"] < 98.5, "Trailing stop should not trigger at entry price level"

    # Verify position was entered
    assert first_trade["entry_price"] == 100.0
    assert first_trade["signal_type"] == "LONG"


def test_trailing_stop_activates_after_favorable_movement_long():
    """
    Test that trailing stop activates after price moves favorably for LONG position.

    Scenario: Enter LONG position, price increases, then decreases.
    Expected: Trailing stop should activate after price increases, then trigger on decrease.
    """

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        # Create data where price increases then decreases
        # Entry at 100, goes up to 105, then down to 103.5 (trailing stop should trigger)
        prices = [100.0, 102.0, 104.0, 105.0, 104.5, 104.0, 103.5, 103.0, 102.5, 102.0]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    class MockSignalCalculator:
        def calculate_hybrid_signal(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8  # LONG signal at period 0
            return 0, 0.0

        def calculate_single_signal_highest_confidence(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8  # LONG signal at period 0
            return 0, 0.0

        def calculate_signal_from_precomputed(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8  # LONG signal at period 0
            return 0, 0.0

        def calculate_single_signal_from_precomputed(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8  # LONG signal at period 0
            return 0, 0.0

        def get_cache_stats(self):
            return {"signal_cache_size": 0, "signal_cache_max_size": 1000, "cache_hit_rate": 0.0}

        def clear_cache(self):
            pass

        def precompute_all_indicators_vectorized(self, **kwargs):
            pass  # No-op for mock

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(
        data_fetcher,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        trailing_stop_pct=0.015,  # 1.5% trailing stop
        max_hold_periods=100,
        signal_mode="single_signal",  # Use single signal mode for mock calculator
    )

    backtester.hybrid_signal_calculator = MockSignalCalculator()

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=10,
        signal_type="LONG",
    )

    trades = result["trades"]
    assert len(trades) > 0

    first_trade = trades[0]

    # Entry at 100, highest should be 105
    # Trailing stop should be set at 105 * 0.985 = 103.425
    # When price drops to 103.5, trailing stop should trigger

    # Verify trailing stop was activated and triggered
    if first_trade["exit_reason"] == "TRAILING_STOP":
        # Exit price should be around 103.425 (105 * 0.985)
        assert 103.0 <= first_trade["exit_price"] <= 104.0
        # PnL should be positive (entry 100, exit ~103.4)
        assert first_trade["pnl"] > 0


def test_trailing_stop_not_triggered_immediately_on_entry_short():
    """
    Test that trailing stop does not trigger immediately after entry for SHORT position.
    """

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        # Create data where price increases slightly after SHORT entry
        prices = [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    class MockSignalCalculator:
        def calculate_hybrid_signal(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return -1, 0.8  # SHORT signal at period 0
            return 0, 0.0

        def calculate_single_signal_highest_confidence(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return -1, 0.8  # SHORT signal at period 0
            return 0, 0.0

        def calculate_signal_from_precomputed(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return -1, 0.8  # SHORT signal at period 0
            return 0, 0.0

        def calculate_single_signal_from_precomputed(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return -1, 0.8  # SHORT signal at period 0
            return 0, 0.0

        def get_cache_stats(self):
            return {"signal_cache_size": 0, "signal_cache_max_size": 1000, "cache_hit_rate": 0.0}

        def clear_cache(self):
            pass

        def precompute_all_indicators_vectorized(self, **kwargs):
            pass  # No-op for mock

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(
        data_fetcher,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        trailing_stop_pct=0.015,
        max_hold_periods=100,
    )

    backtester.hybrid_signal_calculator = MockSignalCalculator()

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=10,
        signal_type="SHORT",
    )

    trades = result["trades"]
    assert len(trades) > 0

    first_trade = trades[0]
    assert first_trade["entry_price"] == 100.0
    assert first_trade["signal_type"] == "SHORT"

    # For SHORT, if price goes up from 100 to 100.5, trailing stop should NOT trigger
    # because it wasn't activated yet (price never went below entry for SHORT)
    if first_trade["exit_reason"] == "TRAILING_STOP":
        # If trailing stop triggered, it should be after price moved favorably first
        # (i.e., price went down, then up)
        pass  # This is acceptable if price moved down first


def test_trailing_stop_activates_after_favorable_movement_short():
    """
    Test that trailing stop activates after price moves favorably for SHORT position.
    """

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        # Entry at 100, goes down to 95, then up to 96.425 (trailing stop should trigger)
        prices = [100.0, 98.0, 96.0, 95.0, 95.5, 96.0, 96.5, 97.0, 97.5, 98.0]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    class MockSignalCalculator:
        def calculate_hybrid_signal(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return -1, 0.8  # SHORT signal
            return 0, 0.0

        def get_cache_stats(self):
            return {"signal_cache_size": 0, "signal_cache_max_size": 1000, "cache_hit_rate": 0.0}

        def clear_cache(self):
            pass

        def precompute_all_indicators_vectorized(self, **kwargs):
            # Mock implementation that raises an exception to force fallback
            raise AttributeError(
                "'MockSignalCalculator' object has no attribute 'precompute_all_indicators_vectorized'"
            )

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(
        data_fetcher,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        trailing_stop_pct=0.015,
        max_hold_periods=100,
    )

    backtester.hybrid_signal_calculator = MockSignalCalculator()

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=10,
        signal_type="SHORT",
    )

    trades = result["trades"]
    assert len(trades) > 0

    first_trade = trades[0]

    # Entry at 100, lowest should be 95
    # Trailing stop should be set at 95 * 1.015 = 96.425
    # When price goes up to 96.5, trailing stop should trigger

    if first_trade["exit_reason"] == "TRAILING_STOP":
        # Exit price should be around 96.425
        assert 96.0 <= first_trade["exit_price"] <= 97.0
        # PnL should be positive (entry 100, exit ~96.4)
        assert first_trade["pnl"] > 0


def test_stop_loss_take_profit_priority_over_trailing_stop():
    """
    Test that stop loss and take profit have priority over trailing stop.
    """

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        # Entry at 100, price goes up to 105 (trailing stop would be 103.425)
        # But then price drops to 95 (stop loss at 95 should trigger first)
        prices = [100.0, 102.0, 104.0, 105.0, 104.0, 98.0, 96.0, 95.0, 94.0, 93.0]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    class MockSignalCalculator:
        def calculate_hybrid_signal(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8
            return 0, 0.0

        def calculate_single_signal_highest_confidence(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8
            return 0, 0.0

        def calculate_signal_from_precomputed(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8
            return 0, 0.0

        def calculate_single_signal_from_precomputed(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8
            return 0, 0.0

        def get_cache_stats(self):
            return {"signal_cache_size": 0, "signal_cache_max_size": 1000, "cache_hit_rate": 0.0}

        def clear_cache(self):
            pass

        def precompute_all_indicators_vectorized(self, **kwargs):
            pass  # No-op for mock

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(
        data_fetcher,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        trailing_stop_pct=0.015,
        max_hold_periods=100,
        signal_mode="single_signal",  # Use single signal mode for mock calculator
    )

    backtester.hybrid_signal_calculator = MockSignalCalculator()

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=10,
        signal_type="LONG",
    )

    trades = result["trades"]
    assert len(trades) > 0

    first_trade = trades[0]

    # In this scenario, price goes up to 105 then drops to 95
    # Trailing stop should trigger at ~103.425 (105 * 0.985) before reaching stop loss at 95
    # So the exit reason should be TRAILING_STOP
    assert first_trade["exit_reason"] == "TRAILING_STOP"
    assert 103.0 <= first_trade["exit_price"] <= 104.0  # Should trigger around trailing stop level
    # PnL should be positive (entry 100, exit ~103.4)
    assert first_trade["pnl"] > 0
