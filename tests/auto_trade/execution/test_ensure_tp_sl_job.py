"""
Regression Tests: EnsureTPSLJob + hybrid-execution guard
=========================================================

1. ensure_tp_sl_job.py - Uses BinanceClient.modify_take_profit() and
   modify_stop_loss() which follow the idempotent cancel-then-create pattern,
   using correct order types (TAKE_PROFIT_MARKET, STOP_MARKET) and
   preventing duplicate conditional orders.

2. auto_trade.py - _trailing_stop_cycle and _negative_breakeven_cycle must
   be no-ops while the WebSocket service is running after fix 2026-02-21.

Import strategy
---------------
``ensure_tp_sl_job`` -> ``binance/__init__.py`` -> long circular chain.
We stub ``modules.auto_trade.execution.binance`` (the package) and
``modules.auto_trade.execution.binance.order_management`` before importing
the unit under test, exposing only what the job actually uses.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Dict
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal sys.modules stubs - break the binance/__init__ import chain
# ---------------------------------------------------------------------------


def _make_pkg_stub(name: str) -> ModuleType:
    stub = ModuleType(name)
    stub.__path__ = []  # mark as package
    stub.__package__ = name
    stub.__spec__ = None
    return stub


def _install_binance_stubs() -> None:
    """Install lightweight stubs for the binance sub-package."""
    pkg_name = "modules.auto_trade.execution.binance"
    om_name = "modules.auto_trade.execution.binance.order_management"

    if pkg_name not in sys.modules:
        pkg_stub = _make_pkg_stub(pkg_name)
        # Add symbols exported by __init__.py that other modules reference
        pkg_stub.BinanceClient = MagicMock(name="BinanceClient")
        sys.modules[pkg_name] = pkg_stub

    if om_name not in sys.modules:
        om_stub = ModuleType(om_name)
        om_stub.__spec__ = None

        def _ccxt_futures_symbol(exchange, symbol: str) -> str:
            return symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol

        def _fetch_all_open_orders(exchange, symbol: str) -> list:
            """Stub: delegates to exchange.fetch_open_orders (mocked in tests)."""
            return exchange.fetch_open_orders(symbol)

        om_stub._ccxt_futures_symbol = _ccxt_futures_symbol
        om_stub._fetch_all_open_orders = _fetch_all_open_orders
        om_stub.OrderManagement = MagicMock(name="OrderManagement")
        sys.modules[om_name] = om_stub


_install_binance_stubs()

# Now the job can be imported without triggering the circular chain
from modules.auto_trade.execution.ensure_tp_sl_job import EnsureTPSLJob  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order(**kw) -> Dict[str, Any]:
    defaults = dict(symbol="BTCUSDT", side="LONG", entry_price=50_000.0, amount=0.01, order_id="AT_123")
    defaults.update(kw)
    return defaults


def _make_settings_manager(tp_pct: float = 5.0, sl_pct: float = 2.5):
    sm = MagicMock()
    sm.get.return_value = {"default_tp": tp_pct, "default_sl": sl_pct}
    return sm


def _make_binance_client(open_orders=None):
    client = MagicMock()
    client.exchange.fetch_open_orders.return_value = open_orders or []
    # modify_take_profit and modify_stop_loss return result dicts with 'id'
    client.modify_take_profit.return_value = {"id": "TP_ORDER_1"}
    client.modify_stop_loss.return_value = {"id": "SL_ORDER_1"}
    return client


# ---------------------------------------------------------------------------
# Fix 1: ensure_tp_sl_job uses modify_take_profit / modify_stop_loss
# ---------------------------------------------------------------------------


class TestEnsureTPSLJobIdempotent:
    def _run_job(self, order=None, tp_pct=5.0, sl_pct=2.5, open_orders=None):
        if order is None:
            order = _make_order()
        binance_client = _make_binance_client(open_orders=open_orders)
        with patch("modules.auto_trade.execution.ensure_tp_sl_job.get_open_positions", return_value=[order]):
            job = EnsureTPSLJob(settings_manager=_make_settings_manager(tp_pct, sl_pct), binance_client=binance_client)
            result = job.run()
        return result, binance_client

    def test_tp_placed_via_modify_take_profit(self):
        """TP should be placed using modify_take_profit (idempotent, correct type)."""
        result, client = self._run_job()
        assert result["tp_added"] == 1
        client.modify_take_profit.assert_called_once()
        call_kwargs = client.modify_take_profit.call_args
        # Verify TP price is above entry for LONG
        tp_price = call_kwargs.kwargs.get("take_profit_price") or call_kwargs[1].get("take_profit_price")
        if tp_price is None and len(call_kwargs.args) >= 3:
            tp_price = call_kwargs.args[2]
        assert tp_price is not None
        assert tp_price > 50_000.0

    def test_sl_placed_via_modify_stop_loss(self):
        """SL should be placed using modify_stop_loss (idempotent, correct type)."""
        result, client = self._run_job()
        assert result["sl_added"] == 1
        client.modify_stop_loss.assert_called_once()
        call_kwargs = client.modify_stop_loss.call_args
        sl_price = call_kwargs.kwargs.get("stop_loss_price") or call_kwargs[1].get("stop_loss_price")
        if sl_price is None and len(call_kwargs.args) >= 3:
            sl_price = call_kwargs.args[2]
        assert sl_price is not None
        assert sl_price < 50_000.0

    def test_both_orders_placed_when_both_missing(self):
        result, client = self._run_job()
        assert result["tp_added"] == 1
        assert result["sl_added"] == 1
        client.modify_take_profit.assert_called_once()
        client.modify_stop_loss.assert_called_once()

    def test_no_raw_create_order_called(self):
        """Must NOT use raw exchange.create_order — that was the duplicate bug."""
        _, client = self._run_job()
        client.exchange.create_order.assert_not_called()

    def test_no_order_placed_when_both_exist(self):
        open_orders = [
            {"info": {"type": "TAKE_PROFIT_MARKET"}, "type": "TAKE_PROFIT_MARKET", "stopPrice": 52500},
            {"info": {"type": "STOP_MARKET"}, "type": "STOP_MARKET", "stopPrice": 48750},
        ]
        result, client = self._run_job(open_orders=open_orders)
        assert result["tp_added"] == 0
        assert result["sl_added"] == 0
        client.modify_take_profit.assert_not_called()
        client.modify_stop_loss.assert_not_called()

    def test_short_sl_above_entry(self):
        result, client = self._run_job(order=_make_order(side="SHORT", entry_price=50_000.0))
        assert result["sl_added"] == 1
        call_kwargs = client.modify_stop_loss.call_args
        sl_price = call_kwargs.kwargs.get("stop_loss_price") or call_kwargs[1].get("stop_loss_price")
        if sl_price is None and len(call_kwargs.args) >= 3:
            sl_price = call_kwargs.args[2]
        assert sl_price > 50_000.0

    def test_short_tp_below_entry(self):
        result, client = self._run_job(order=_make_order(side="SHORT", entry_price=50_000.0))
        assert result["tp_added"] == 1
        call_kwargs = client.modify_take_profit.call_args
        tp_price = call_kwargs.kwargs.get("take_profit_price") or call_kwargs[1].get("take_profit_price")
        if tp_price is None and len(call_kwargs.args) >= 3:
            tp_price = call_kwargs.args[2]
        assert tp_price < 50_000.0

    def test_fail_closed_no_orders_on_detection_failure(self):
        """If fetch_open_orders fails, do NOT create orders (fail-closed)."""
        order = _make_order()
        binance_client = _make_binance_client()
        binance_client.exchange.fetch_open_orders.side_effect = Exception("API error")
        with patch("modules.auto_trade.execution.ensure_tp_sl_job.get_open_positions", return_value=[order]):
            job = EnsureTPSLJob(settings_manager=_make_settings_manager(), binance_client=binance_client)
            result = job.run()
        assert result["tp_added"] == 0
        assert result["sl_added"] == 0
        binance_client.modify_take_profit.assert_not_called()
        binance_client.modify_stop_loss.assert_not_called()

    def test_max_conditional_orders_guard(self):
        """If too many conditional orders exist, refuse to create more."""
        # Simulate 10+ existing SL orders (the duplicate flood scenario)
        many_orders = [
            {"info": {"type": "STOP_MARKET"}, "type": "STOP_MARKET", "stopPrice": 48000 + i} for i in range(12)
        ]
        result, client = self._run_job(open_orders=many_orders)
        # Should detect the flood and refuse to create more
        assert result["tp_added"] == 0
        assert result["sl_added"] == 0
        client.modify_take_profit.assert_not_called()
        client.modify_stop_loss.assert_not_called()

    def test_idempotent_across_runs(self):
        """Running the job twice with existing orders should not create duplicates."""
        # First run: no existing orders → creates TP + SL
        result1, _ = self._run_job()
        assert result1["tp_added"] == 1
        assert result1["sl_added"] == 1

        # Second run: simulating existing orders are now on exchange
        existing_orders = [
            {"info": {"type": "TAKE_PROFIT_MARKET"}, "type": "TAKE_PROFIT_MARKET", "stopPrice": 52500},
            {"info": {"type": "STOP_MARKET"}, "type": "STOP_MARKET", "stopPrice": 48750},
        ]
        result2, client2 = self._run_job(open_orders=existing_orders)
        assert result2["tp_added"] == 0
        assert result2["sl_added"] == 0
        client2.modify_take_profit.assert_not_called()
        client2.modify_stop_loss.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 2: hybrid execution guard
# ---------------------------------------------------------------------------


class TestHybridGuard:
    def _make_parent(self, ws_running: bool):
        parent = MagicMock()
        parent.ws_data_service.is_running = ws_running
        return parent

    def test_trailing_stop_skips_when_ws_running(self):
        from modules.auto_trade.gui.main_window.auto_trade import AutoTradeManager

        manager = AutoTradeManager(parent=self._make_parent(ws_running=True))
        with patch(
            "modules.auto_trade.execution.trailing_stop_job.create_trailing_stop_job",
            side_effect=AssertionError("must not be called"),
        ):
            manager._trailing_stop_cycle()

    def test_trailing_stop_runs_when_ws_not_running(self):
        from modules.auto_trade.gui.main_window.auto_trade import AutoTradeManager

        parent = self._make_parent(ws_running=False)
        parent.data_service.api_key = ""
        parent.data_service.api_secret = ""
        manager = AutoTradeManager(parent=parent)
        manager._get_binance_client = MagicMock(return_value=None)
        mock_job = MagicMock()
        mock_job.run.return_value = {"orders_checked": 0, "orders_updated": 0, "updates": [], "errors": []}
        with patch("modules.auto_trade.execution.trailing_stop_job.create_trailing_stop_job", return_value=mock_job):
            manager._trailing_stop_cycle()
        mock_job.run.assert_called_once()

    def test_negative_be_skips_when_ws_running(self):
        from modules.auto_trade.gui.main_window.auto_trade import AutoTradeManager

        manager = AutoTradeManager(parent=self._make_parent(ws_running=True))
        with patch(
            "modules.auto_trade.execution.negative_breakeven_job.create_negative_breakeven_job",
            side_effect=AssertionError("must not be called"),
        ):
            manager._negative_breakeven_cycle()

    def test_negative_be_runs_when_ws_not_running(self):
        from modules.auto_trade.gui.main_window.auto_trade import AutoTradeManager

        parent = self._make_parent(ws_running=False)
        parent.data_service.api_key = ""
        parent.data_service.api_secret = ""
        manager = AutoTradeManager(parent=parent)
        manager._get_binance_client = MagicMock(return_value=None)
        mock_job = MagicMock()
        mock_job.run.return_value = {"orders_checked": 0, "orders_updated": 0, "updates": [], "errors": []}
        with patch(
            "modules.auto_trade.execution.negative_breakeven_job.create_negative_breakeven_job", return_value=mock_job
        ):
            manager._negative_breakeven_cycle()
        mock_job.run.assert_called_once()

    def test_no_ws_service_does_not_crash(self):
        from modules.auto_trade.gui.main_window.auto_trade import AutoTradeManager

        parent = MagicMock()
        parent.ws_data_service = None
        parent.data_service.api_key = ""
        parent.data_service.api_secret = ""
        manager = AutoTradeManager(parent=parent)
        manager._get_binance_client = MagicMock(return_value=None)
        mock_job = MagicMock()
        mock_job.run.return_value = {"orders_checked": 0, "orders_updated": 0, "updates": [], "errors": []}
        with patch("modules.auto_trade.execution.trailing_stop_job.create_trailing_stop_job", return_value=mock_job):
            manager._trailing_stop_cycle()
        mock_job.run.assert_called_once()
