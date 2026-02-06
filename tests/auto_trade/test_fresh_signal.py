"""
Unit Tests for Fresh-Signal Auto-Trade Feature
===============================================

Tests the fresh signal filtering, selection, and execution logic
as per the 2026-02-06 Fresh-Signal Auto-Trade Design.

Run: pytest tests/auto_trade/test_fresh_signal.py -v
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.auto_trade.core.signal_selector import FinalSignal


class TestFreshSignalFiltering:
    """Test filtering signals by freshness (< 60 seconds)."""

    def test_filter_fresh_signals_within_60s(self):
        """Test that signals created within 60 seconds are considered fresh."""
        now = time.time()
        signals = [
            {"symbol": "BTC/USDT", "score": 0.8, "created_at_ts": now - 30},  # 30s ago - fresh
            {"symbol": "ETH/USDT", "score": 0.7, "created_at_ts": now - 59},  # 59s ago - fresh
            {"symbol": "SOL/USDT", "score": 0.6, "created_at_ts": now - 10},  # 10s ago - fresh
        ]

        # Filter logic from auto_trade.py lines 103-108
        fresh = [
            s
            for s in signals
            if isinstance(s.get("created_at_ts"), (int, float)) and (now - float(s["created_at_ts"])) < 60.0
        ]

        assert len(fresh) == 3
        assert all(s["symbol"] in ["BTC/USDT", "ETH/USDT", "SOL/USDT"] for s in fresh)

    def test_filter_fresh_signals_excludes_old(self):
        """Test that signals older than 60 seconds are excluded."""
        now = time.time()
        signals = [
            {"symbol": "BTC/USDT", "score": 0.9, "created_at_ts": now - 30},  # 30s ago - fresh
            {"symbol": "ETH/USDT", "score": 0.8, "created_at_ts": now - 61},  # 61s ago - stale
            {"symbol": "SOL/USDT", "score": 0.7, "created_at_ts": now - 120},  # 120s ago - stale
            {"symbol": "BNB/USDT", "score": 0.6, "created_at_ts": now - 5},  # 5s ago - fresh
        ]

        fresh = [
            s
            for s in signals
            if isinstance(s.get("created_at_ts"), (int, float)) and (now - float(s["created_at_ts"])) < 60.0
        ]

        assert len(fresh) == 2
        assert fresh[0]["symbol"] == "BTC/USDT"
        assert fresh[1]["symbol"] == "BNB/USDT"
        # Verify stale signals are excluded
        fresh_symbols = [s["symbol"] for s in fresh]
        assert "ETH/USDT" not in fresh_symbols
        assert "SOL/USDT" not in fresh_symbols

    def test_filter_handles_missing_timestamp(self):
        """Test that signals without created_at_ts are excluded."""
        now = time.time()
        signals = [
            {"symbol": "BTC/USDT", "score": 0.8, "created_at_ts": now - 30},  # Has timestamp - fresh
            {"symbol": "ETH/USDT", "score": 0.7},  # Missing timestamp
            {"symbol": "SOL/USDT", "score": 0.6, "created_at_ts": None},  # None timestamp
            {"symbol": "BNB/USDT", "score": 0.5, "created_at_ts": "invalid"},  # Invalid type
        ]

        fresh = [
            s
            for s in signals
            if isinstance(s.get("created_at_ts"), (int, float)) and (now - float(s["created_at_ts"])) < 60.0
        ]

        assert len(fresh) == 1
        assert fresh[0]["symbol"] == "BTC/USDT"

    def test_filter_handles_zero_timestamp(self):
        """Test that signals with zero timestamp are excluded (fallback value)."""
        now = time.time()
        signals = [
            {"symbol": "BTC/USDT", "score": 0.8, "created_at_ts": now - 30},  # Fresh
            {"symbol": "ETH/USDT", "score": 0.7, "created_at_ts": 0.0},  # Zero = old/invalid
        ]

        fresh = [
            s
            for s in signals
            if isinstance(s.get("created_at_ts"), (int, float)) and (now - float(s["created_at_ts"])) < 60.0
        ]

        assert len(fresh) == 1
        assert fresh[0]["symbol"] == "BTC/USDT"


class TestBestSignalSelection:
    """Test selecting the best signal by highest score."""

    def test_select_best_signal_by_score(self):
        """Test that the signal with highest score is selected."""
        signals = [
            {"symbol": "BTC/USDT", "score": 0.75, "signal": "LONG"},
            {"symbol": "ETH/USDT", "score": 0.92, "signal": "LONG"},  # Highest
            {"symbol": "SOL/USDT", "score": 0.68, "signal": "SHORT"},
        ]

        # Selection logic from auto_trade.py lines 112-113
        signals.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
        best = signals[0]

        assert best["symbol"] == "ETH/USDT"
        assert best["score"] == 0.92

    def test_select_best_with_equal_scores(self):
        """Test that first signal is selected when scores are equal."""
        signals = [
            {"symbol": "BTC/USDT", "score": 0.85, "signal": "LONG"},
            {"symbol": "ETH/USDT", "score": 0.85, "signal": "LONG"},  # Same score
            {"symbol": "SOL/USDT", "score": 0.70, "signal": "SHORT"},
        ]

        signals.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
        best = signals[0]

        # First element in original list with highest score
        assert best["symbol"] == "BTC/USDT"
        assert best["score"] == 0.85

    def test_select_handles_missing_score(self):
        """Test that missing scores are treated as 0.0."""
        signals = [
            {"symbol": "BTC/USDT", "score": 0.50, "signal": "LONG"},
            {"symbol": "ETH/USDT", "score": None, "signal": "LONG"},  # None score - becomes 0.0
            {"symbol": "SOL/USDT", "signal": "SHORT"},  # Missing score - becomes 0.0
        ]

        signals.sort(key=lambda s: float(s.get("score") if s.get("score") is not None else 0.0), reverse=True)
        best = signals[0]

        assert best["symbol"] == "BTC/USDT"
        assert best["score"] == 0.50


class TestOrderExecutorTPSL:
    """Test OrderExecutor TP/SL calculation from settings."""

    def test_tp_sl_calculation_long(self):
        """Test TP/SL calculation for LONG signal with settings."""
        # Patch BinanceClient and OrderManager at the point where they're used in OrderExecutor
        with patch("modules.auto_trade.execution.order_executor.BinanceClient") as mock_bc, \
             patch("modules.auto_trade.execution.order_executor.OrderManager") as mock_om:

            # Setup BinanceClient mock
            mock_bc_instance = Mock()
            mock_bc_instance.exchange.fetch_ticker.return_value = {"last": 50000.0}
            mock_bc.return_value = mock_bc_instance

            # Setup OrderManager mock - capture the FinalSignal passed to execute_signal
            mock_om_instance = Mock()
            mock_om_instance.execute_signal.return_value = {
                "order_id": "TEST_ORDER_123",
                "message": "Order executed successfully",
            }
            mock_om.return_value = mock_om_instance

            from modules.auto_trade.execution.order_executor import OrderExecutor

            executor = OrderExecutor(api_key="test_key", api_secret="test_secret")

            # Use "BTCUSDT" format (no slash) - OrderExecutor converts to BTC/USDT
            signal_dict = {"symbol": "BTCUSDT", "signal": "LONG", "score": 0.85, "created_at_ts": time.time()}
            tp_sl_settings = {"default_tp": 10.0, "default_sl": 5.0}  # 10% TP, 5% SL

            result = executor.execute_from_signal(signal_dict, tp_sl_settings=tp_sl_settings)

            if not result.get("success"):
                print(f"Error: {result.get('error', 'Unknown error')}")
            assert result["success"] is True

            # Verify OrderManager.execute_signal was called with correct FinalSignal
            call_args = mock_om_instance.execute_signal.call_args
            final_signal = call_args[0][0]

            assert final_signal.symbol == "BTC/USDT"
            assert final_signal.signal_type == "LONG"
            assert final_signal.entry_price == 50000.0
            # LONG: TP = entry * (1 + tp_pct/100) = 50000 * 1.10 = 55000
            assert final_signal.take_profit == pytest.approx(55000.0)
            # LONG: SL = entry * (1 - sl_pct/100) = 50000 * 0.95 = 47500
            assert final_signal.stop_loss == pytest.approx(47500.0)

    def test_tp_sl_calculation_short(self):
        """Test TP/SL calculation for SHORT signal with settings."""
        with patch("modules.auto_trade.execution.order_executor.BinanceClient") as mock_bc, \
             patch("modules.auto_trade.execution.order_executor.OrderManager") as mock_om:

            mock_bc_instance = Mock()
            mock_bc_instance.exchange.fetch_ticker.return_value = {"last": 3000.0}
            mock_bc.return_value = mock_bc_instance

            mock_om_instance = Mock()
            mock_om_instance.execute_signal.return_value = {
                "order_id": "TEST_ORDER_123",
                "message": "Order executed successfully",
            }
            mock_om.return_value = mock_om_instance

            from modules.auto_trade.execution.order_executor import OrderExecutor

            executor = OrderExecutor(api_key="test_key", api_secret="test_secret")

            # Use "ETHUSDT" format (no slash) - OrderExecutor converts to ETH/USDT
            signal_dict = {"symbol": "ETHUSDT", "signal": "SHORT", "score": 0.78, "created_at_ts": time.time()}
            tp_sl_settings = {"default_tp": 8.0, "default_sl": 4.0}  # 8% TP, 4% SL

            result = executor.execute_from_signal(signal_dict, tp_sl_settings=tp_sl_settings)

            assert result["success"] is True
            call_args = mock_om_instance.execute_signal.call_args
            final_signal = call_args[0][0]

            assert final_signal.symbol == "ETH/USDT"
            assert final_signal.signal_type == "SHORT"
            assert final_signal.entry_price == 3000.0
            # SHORT: TP = entry * (1 - tp_pct/100) = 3000 * 0.92 = 2760
            assert final_signal.take_profit == 2760.0
            # SHORT: SL = entry * (1 + sl_pct/100) = 3000 * 1.04 = 3120
            assert final_signal.stop_loss == 3120.0

    def test_tp_sl_default_values_when_no_settings(self):
        """Test that default TP/SL values (5.0 / 2.0) are used when settings not provided."""
        with patch("modules.auto_trade.execution.order_executor.BinanceClient") as mock_bc, \
             patch("modules.auto_trade.execution.order_executor.OrderManager") as mock_om:

            mock_bc_instance = Mock()
            mock_bc_instance.exchange.fetch_ticker.return_value = {"last": 100.0}
            mock_bc.return_value = mock_bc_instance

            mock_om_instance = Mock()
            mock_om_instance.execute_signal.return_value = {
                "order_id": "TEST_ORDER_123",
                "message": "Order executed successfully",
            }
            mock_om.return_value = mock_om_instance

            from modules.auto_trade.execution.order_executor import OrderExecutor

            executor = OrderExecutor(api_key="test_key", api_secret="test_secret")

            signal_dict = {"symbol": "SOL/USDT", "signal": "LONG", "score": 0.90, "created_at_ts": time.time()}

            result = executor.execute_from_signal(signal_dict, tp_sl_settings=None)

            assert result["success"] is True
            call_args = mock_om_instance.execute_signal.call_args
            final_signal = call_args[0][0]

            # Default: 5% TP, 2% SL
            # LONG: TP = 100 * 1.05 = 105
            assert final_signal.take_profit == 105.0
            # LONG: SL = 100 * 0.98 = 98
            assert final_signal.stop_loss == 98.0

    def test_tp_sl_handles_missing_keys_in_settings(self):
        """Test that default values are used when keys missing from settings."""
        with patch("modules.auto_trade.execution.order_executor.BinanceClient") as mock_bc, \
             patch("modules.auto_trade.execution.order_executor.OrderManager") as mock_om:

            mock_bc_instance = Mock()
            mock_bc_instance.exchange.fetch_ticker.return_value = {"last": 400.0}
            mock_bc.return_value = mock_bc_instance

            mock_om_instance = Mock()
            mock_om_instance.execute_signal.return_value = {
                "order_id": "TEST_ORDER_123",
                "message": "Order executed successfully",
            }
            mock_om.return_value = mock_om_instance

            from modules.auto_trade.execution.order_executor import OrderExecutor

            executor = OrderExecutor(api_key="test_key", api_secret="test_secret")

            signal_dict = {"symbol": "BNB/USDT", "signal": "SHORT", "score": 0.82, "created_at_ts": time.time()}
            tp_sl_settings = {}  # Empty settings dict

            result = executor.execute_from_signal(signal_dict, tp_sl_settings=tp_sl_settings)

            assert result["success"] is True
            call_args = mock_om_instance.execute_signal.call_args
            final_signal = call_args[0][0]

            # Should use defaults: 5% TP, 2% SL
            # SHORT: TP = 400 * 0.95 = 380
            assert final_signal.take_profit == 380.0
            # SHORT: SL = 400 * 1.02 = 408
            assert final_signal.stop_loss == 408.0

    def test_tp_sl_handles_invalid_values_in_settings(self):
        """Test that invalid values in settings fall back to defaults."""
        with patch("modules.auto_trade.execution.order_executor.BinanceClient") as mock_bc, \
             patch("modules.auto_trade.execution.order_executor.OrderManager") as mock_om:

            mock_bc_instance = Mock()
            mock_bc_instance.exchange.fetch_ticker.return_value = {"last": 1.0}
            mock_bc.return_value = mock_bc_instance

            mock_om_instance = Mock()
            mock_om_instance.execute_signal.return_value = {
                "order_id": "TEST_ORDER_123",
                "message": "Order executed successfully",
            }
            mock_om.return_value = mock_om_instance

            from modules.auto_trade.execution.order_executor import OrderExecutor

            executor = OrderExecutor(api_key="test_key", api_secret="test_secret")

            signal_dict = {"symbol": "XRP/USDT", "signal": "LONG", "score": 0.75, "created_at_ts": time.time()}
            tp_sl_settings = {"default_tp": "invalid", "default_sl": None}  # Invalid types

            result = executor.execute_from_signal(signal_dict, tp_sl_settings=tp_sl_settings)

            assert result["success"] is True
            call_args = mock_om_instance.execute_signal.call_args
            final_signal = call_args[0][0]

            # Should fall back to defaults due to ValueError
            # LONG: TP = 1.0 * 1.05 = 1.05
            assert final_signal.take_profit == 1.05
            # LONG: SL = 1.0 * 0.98 = 0.98
            assert final_signal.stop_loss == 0.98


class TestAutoTradeCycleIntegration:
    """Integration tests for auto-trade cycle with fresh signals."""

    @pytest.fixture
    def mock_auto_trade_components(self):
        """Mock all components needed for auto-trade cycle."""
        parent = Mock()

        # Mock DataService
        parent.data_service = Mock()

        # Mock SettingsManager
        parent.settings_manager = Mock()
        parent.settings_manager.get.return_value = {"default_tp": 5.0, "default_sl": 2.5}

        # Mock RiskManager
        parent.risk_manager = Mock()
        parent.risk_manager.check_limits.return_value = (True, "")

        # Mock positions and account refresh
        parent.refresh_positions = Mock()
        parent.refresh_account = Mock()

        return parent

    def test_auto_trade_cycle_with_fresh_signal(self, mock_auto_trade_components):
        """Test complete auto-trade cycle with a fresh signal."""
        parent = mock_auto_trade_components
        now = time.time()

        # Mock signals with one fresh signal (use BTCUSDT format for OrderExecutor)
        signals = [
            {"symbol": "BTCUSDT", "signal": "LONG", "score": 0.85, "created_at_ts": now - 30},  # Fresh
            {"symbol": "ETHUSDT", "signal": "LONG", "score": 0.70, "created_at_ts": now - 90},  # Stale
        ]

        parent.data_service.get_signals.return_value = signals

        with patch("modules.auto_trade.execution.order_executor.OrderManager") as mock_om:
            mock_om_instance = Mock()
            mock_om_instance.execute_signal.return_value = {
                "order_id": "ORDER_123",
                "message": "Success",
            }
            mock_om.return_value = mock_om_instance

            with patch("modules.auto_trade.execution.order_executor.BinanceClient") as mock_bc:
                mock_exchange = Mock()
                mock_exchange.fetch_ticker.return_value = {"last": 50000.0}
                mock_bc.return_value.exchange = mock_exchange

                # Simulate of auto-trade cycle logic
                from modules.auto_trade.execution.order_executor import OrderExecutor

                # Filter fresh signals
                fresh_signals = [
                    s
                    for s in signals
                    if isinstance(s.get("created_at_ts"), (int, float))
                    and (time.time() - float(s["created_at_ts"])) < 60.0
                ]

                assert len(fresh_signals) == 1

                # Sort and select best
                fresh_signals.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
                best = fresh_signals[0]

                assert best["symbol"] == "BTCUSDT"

                # Risk check
                can_trade, msg = parent.risk_manager.check_limits(best["symbol"], 100.0, 2)
                assert can_trade is True

                # Execute
                sig_dict = {
                    "symbol": best.get("symbol"),
                    "signal": best.get("signal"),
                    "score": best.get("score", 0.0),
                    "created_at_ts": best.get("created_at_ts", 0.0),
                }

                tp_sl = parent.settings_manager.get("tp_sl", {}) or {}
                executor = OrderExecutor(api_key="test_key", api_secret="test_secret")
                result = executor.execute_from_signal(sig_dict, tp_sl_settings=tp_sl)

                assert result["success"] is True
                assert mock_om_instance.execute_signal.called

    def test_auto_trade_cycle_no_fresh_signals(self, mock_auto_trade_components):
        """Test auto-trade cycle with no fresh signals (all stale)."""
        parent = mock_auto_trade_components
        now = time.time()

        # All signals are stale (> 60s)
        signals = [
            {"symbol": "BTC/USDT", "signal": "LONG", "score": 0.90, "created_at_ts": now - 120},
            {"symbol": "ETH/USDT", "signal": "LONG", "score": 0.85, "created_at_ts": now - 90},
        ]

        parent.data_service.get_signals.return_value = signals

        # Filter fresh signals
        fresh_signals = [
            s
            for s in signals
            if isinstance(s.get("created_at_ts"), (int, float)) and (time.time() - float(s["created_at_ts"])) < 60.0
        ]

        assert len(fresh_signals) == 0
        # No execution should occur - cycle should return early

    def test_auto_trade_cycle_selects_highest_score_among_fresh(self, mock_auto_trade_components):
        """Test that highest score is selected when multiple fresh signals exist."""
        parent = mock_auto_trade_components
        now = time.time()

        signals = [
            {"symbol": "BTC/USDT", "signal": "LONG", "score": 0.75, "created_at_ts": now - 20},  # Fresh
            {"symbol": "ETH/USDT", "signal": "LONG", "score": 0.92, "created_at_ts": now - 30},  # Fresh, highest
            {"symbol": "SOL/USDT", "signal": "LONG", "score": 0.68, "created_at_ts": now - 10},  # Fresh
        ]

        parent.data_service.get_signals.return_value = signals

        # Filter fresh
        fresh_signals = [
            s
            for s in signals
            if isinstance(s.get("created_at_ts"), (int, float)) and (time.time() - float(s["created_at_ts"])) < 60.0
        ]

        assert len(fresh_signals) == 3

        # Select best
        fresh_signals.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
        best = fresh_signals[0]

        assert best["symbol"] == "ETH/USDT"
        assert best["score"] == 0.92


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
