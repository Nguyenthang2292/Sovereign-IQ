"""
Tests for Binance OrderManagement (TP/SL, -2021 validation).

- Validates that SL is not placed when it would trigger immediately (Binance -2021).
- LONG: SL must be below mark price; SHORT: SL must be above mark price.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import MagicMock, patch

import pytest

from modules.auto_trade.execution.binance.order_management import (
    OrderManagement,
    _get_mark_price_from_exchange,
)

# -----------------------------------------------------------------------------
# _get_mark_price_from_exchange
# -----------------------------------------------------------------------------


def test_get_mark_price_from_exchange_uses_mark_price():
    exchange = MagicMock()
    exchange.fetch_ticker.return_value = {
        "info": {"markPrice": "0.00668"},
        "last": 0.00670,
    }
    assert _get_mark_price_from_exchange(exchange, "SKL/USDT") == 0.00668
    exchange.fetch_ticker.assert_called_once_with("SKL/USDT")


def test_get_mark_price_from_exchange_fallback_to_last():
    exchange = MagicMock()
    exchange.fetch_ticker.return_value = {"info": {}, "last": 0.00670}
    assert _get_mark_price_from_exchange(exchange, "SKL/USDT") == 0.00670


def test_get_mark_price_from_exchange_returns_none_on_error():
    exchange = MagicMock()
    exchange.fetch_ticker.side_effect = Exception("network error")
    assert _get_mark_price_from_exchange(exchange, "SKL/USDT") is None


# -----------------------------------------------------------------------------
# modify_stop_loss: skip when order would trigger immediately (-2021)
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_exchange():
    ex = MagicMock()
    ex.fetch_open_orders.return_value = []
    return ex


@pytest.fixture
def mock_position_long():
    return {
        "symbol": "SKL/USDT:USDT",
        "contracts": 3121.0,
        "side": "long",
        "info": {"positionAmt": "3121"},
    }


@pytest.fixture
def mock_position_short():
    return {
        "symbol": "SKL/USDT:USDT",
        "contracts": 100.0,
        "side": "short",
        "info": {"positionAmt": "-100"},
    }


def test_modify_stop_loss_skips_when_long_and_sl_above_mark(mock_exchange, mock_position_long):
    """LONG position: SL at 0.01 with mark 0.00668 would trigger immediately → skip."""
    with patch(
        "modules.auto_trade.execution.binance.position_management.PositionManagement"
    ) as pm_cls:
        pm_cls.return_value.get_position.return_value = mock_position_long
        mock_exchange.fetch_ticker.return_value = {
            "info": {"markPrice": "0.00668"},
            "last": 0.00668,
        }

        om = OrderManagement(mock_exchange, dry_run=False)
        result = om.modify_stop_loss("SKL/USDT", None, stop_loss_price=0.01)

        assert result is None
        mock_exchange.create_order.assert_not_called()


def test_modify_stop_loss_skips_when_short_and_sl_below_mark(mock_exchange, mock_position_short):
    """SHORT position: SL at 0.005 with mark 0.00668 would trigger immediately → skip."""
    with patch(
        "modules.auto_trade.execution.binance.position_management.PositionManagement"
    ) as pm_cls:
        pm_cls.return_value.get_position.return_value = mock_position_short
        mock_exchange.fetch_ticker.return_value = {
            "info": {"markPrice": "0.00668"},
            "last": 0.00668,
        }

        om = OrderManagement(mock_exchange, dry_run=False)
        result = om.modify_stop_loss("SKL/USDT", None, stop_loss_price=0.005)

        assert result is None
        mock_exchange.create_order.assert_not_called()


def test_modify_stop_loss_places_when_long_and_sl_below_mark(mock_exchange, mock_position_long):
    """LONG: SL at 0.006 below mark 0.00668 is valid → place order."""
    with patch(
        "modules.auto_trade.execution.binance.position_management.PositionManagement"
    ) as pm_cls:
        pm_cls.return_value.get_position.return_value = mock_position_long
        mock_exchange.fetch_ticker.return_value = {
            "info": {"markPrice": "0.00668"},
            "last": 0.00668,
        }
        mock_exchange.create_order.return_value = {"id": "123", "symbol": "SKL/USDT"}

        om = OrderManagement(mock_exchange, dry_run=False)
        result = om.modify_stop_loss("SKL/USDT", None, stop_loss_price=0.006)

        assert result is not None
        assert result.get("id") == "123"
        mock_exchange.create_order.assert_called_once()
        call_kw = mock_exchange.create_order.call_args[1]
        assert call_kw["params"]["stopPrice"] == 0.006
        assert call_kw["side"] == "sell"


def test_modify_stop_loss_places_when_short_and_sl_above_mark(mock_exchange, mock_position_short):
    """SHORT: SL at 0.008 above mark 0.00668 is valid → place order."""
    with patch(
        "modules.auto_trade.execution.binance.position_management.PositionManagement"
    ) as pm_cls:
        pm_cls.return_value.get_position.return_value = mock_position_short
        mock_exchange.fetch_ticker.return_value = {
            "info": {"markPrice": "0.00668"},
            "last": 0.00668,
        }
        mock_exchange.create_order.return_value = {"id": "456", "symbol": "SKL/USDT"}

        om = OrderManagement(mock_exchange, dry_run=False)
        result = om.modify_stop_loss("SKL/USDT", None, stop_loss_price=0.008)

        assert result is not None
        assert result.get("id") == "456"
        mock_exchange.create_order.assert_called_once()
        call_kw = mock_exchange.create_order.call_args[1]
        assert call_kw["params"]["stopPrice"] == 0.008
        assert call_kw["side"] == "buy"


def test_modify_stop_loss_places_when_mark_unavailable_long(mock_exchange, mock_position_long):
    """When mark price cannot be fetched, do not block placement (legacy behavior)."""
    with patch(
        "modules.auto_trade.execution.binance.position_management.PositionManagement"
    ) as pm_cls:
        pm_cls.return_value.get_position.return_value = mock_position_long
        mock_exchange.fetch_ticker.side_effect = Exception("ticker error")

        om = OrderManagement(mock_exchange, dry_run=False)
        mock_exchange.create_order.return_value = {"id": "789"}
        result = om.modify_stop_loss("SKL/USDT", None, stop_loss_price=0.006)

        # No validation when mark is missing → order is attempted
        mock_exchange.create_order.assert_called_once()
        assert result is not None
