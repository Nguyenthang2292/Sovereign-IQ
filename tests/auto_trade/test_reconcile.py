"""
Unit Tests for Reconcile Module
================================

Tests the reconcile_orders_with_binance function to ensure all fixes work correctly.

Run: pytest tests/auto_trade/test_reconcile.py -v
"""

import sys
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.auto_trade.database import get_order_by_client_id, initialize_database, session_scope
from modules.auto_trade.database.reconcile import _normalize_symbol, reconcile_orders_with_binance


def cleanup_test_order(client_order_id: str):
    """Helper function to clean up test orders."""
    with session_scope() as session:
        existing = get_order_by_client_id(session, client_order_id)
        if existing:
            session.delete(existing)
            session.commit()


class TestSymbolNormalization:
    """Test _normalize_symbol function."""

    def test_normalize_already_ccxt_format(self):
        """Test symbols already in CCXT format."""
        assert _normalize_symbol("BTC/USDT") == "BTC/USDT"
        assert _normalize_symbol("ETH/USDT") == "ETH/USDT"

    def test_normalize_binance_format(self):
        """Test converting Binance format to CCXT."""
        assert _normalize_symbol("BTCUSDT") == "BTC/USDT"
        assert _normalize_symbol("ETHUSDT") == "ETH/USDT"
        assert _normalize_symbol("SOLUSDT") == "SOL/USDT"

    def test_normalize_futures_suffix(self):
        """Test removing futures suffix."""
        assert _normalize_symbol("BTCUSDT_PERP") == "BTC/USDT"
        assert _normalize_symbol("ETHUSDT-PERP") == "ETH/USDT"

    def test_normalize_empty_string(self):
        """Test empty string handling."""
        assert _normalize_symbol("") == ""
        assert _normalize_symbol(None) == ""
        assert _normalize_symbol("   ") == ""

    def test_normalize_base_currency_only(self):
        """Test base currency only (append /USDT)."""
        assert _normalize_symbol("BTC") == "BTC/USDT"
        assert _normalize_symbol("ETH") == "ETH/USDT"


class TestReconcileFunction:
    """Test reconcile_orders_with_binance function."""

    @pytest.fixture
    def mock_exchange(self):
        """Create mock CCXT exchange."""
        exchange = Mock()
        exchange.close = Mock()
        # Default empty open orders for stale order checking
        exchange.fetch_open_orders = Mock(return_value=[])
        return exchange

    @pytest.fixture
    def mock_binance_order_filled(self):
        """Mock filled order from Binance."""
        return {
            "id": "123456789",
            "clientOrderId": "AT_1707043200_BTCUSDT_abc123",
            "symbol": "BTC/USDT",
            "type": "MARKET",
            "side": "BUY",
            "status": "FILLED",
            "price": 50000.0,
            "average": 50000.0,
            "amount": 0.01,
            "filled": 0.01,
            "leverage": 2,
            "timestamp": 1707043200000,
            "lastTradeTimestamp": 1707043300000,
            "stopPrice": None,
            "info": {"realizedPnl": "15.50", "stopPrice": None, "takeProfit": None},
        }

    @pytest.fixture
    def mock_binance_order_cancelled(self):
        """Mock cancelled order from Binance."""
        return {
            "id": "987654321",
            "clientOrderId": "AT_1707043300_ETHUSDT_def456",
            "symbol": "ETH/USDT",
            "type": "LIMIT",
            "side": "SELL",
            "status": "CANCELED",
            "price": 3000.0,
            "average": 0,
            "amount": 0.1,
            "filled": 0,
            "leverage": 3,
            "timestamp": 1707043300000,
            "lastTradeTimestamp": None,
            "stopPrice": 2950.0,
            "info": {"stopPrice": "2950.0"},
        }

    def test_authentication_error(self, test_db):
        """Test handling of authentication errors."""
        with patch("modules.auto_trade.database.reconcile.ccxt.binance") as mock_binance:
            import ccxt

            mock_binance.side_effect = ccxt.AuthenticationError("Invalid API key")

            result = reconcile_orders_with_binance("invalid_key", "invalid_secret")

            assert result["inserted"] == 0
            assert result["skipped"] == 0
            assert len(result["errors"]) == 1
            assert "Authentication failed" in result["errors"][0]

    def test_network_error(self, test_db):
        """Test handling of network errors."""
        with patch("modules.auto_trade.database.reconcile.ccxt.binance") as mock_binance:
            import ccxt

            mock_binance.side_effect = ccxt.NetworkError("Connection timeout")

            result = reconcile_orders_with_binance("key", "secret")

            assert result["inserted"] == 0
            assert len(result["errors"]) == 1
            assert "Network error" in result["errors"][0]

    def test_exchange_error(self, test_db):
        """Test handling of exchange errors."""
        with patch("modules.auto_trade.database.reconcile.ccxt.binance") as mock_binance:
            import ccxt

            mock_binance.side_effect = ccxt.ExchangeError("Invalid symbol")

            result = reconcile_orders_with_binance("key", "secret")

            assert result["inserted"] == 0
            assert len(result["errors"]) == 1
            assert "Exchange error" in result["errors"][0]

    def test_status_mapping_filled_to_closed(self, test_db, mock_exchange, mock_binance_order_filled):
        """Test FILLED status is mapped to CLOSED."""
        mock_exchange.fetch_closed_orders = Mock(return_value=[mock_binance_order_filled])

        client_order_id = mock_binance_order_filled["clientOrderId"]
        cleanup_test_order(client_order_id)

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["BTC/USDT"])

            assert result["inserted"] == 1
            assert result["errors"] == []

            # Verify order in DB has CLOSED status
            with session_scope() as session:
                order = get_order_by_client_id(session, client_order_id)
                assert order is not None
                assert order.status == "CLOSED"
                assert order.pnl == 15.50

        cleanup_test_order(client_order_id)

    def test_status_mapping_canceled_to_cancelled(self, test_db, mock_exchange, mock_binance_order_cancelled):
        """Test CANCELED status is mapped to CANCELLED."""
        mock_exchange.fetch_closed_orders = Mock(return_value=[mock_binance_order_cancelled])

        client_order_id = mock_binance_order_cancelled["clientOrderId"]
        cleanup_test_order(client_order_id)

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["ETH/USDT"])

            assert result["inserted"] == 1

            # Verify order in DB has CANCELLED status
            with session_scope() as session:
                order = get_order_by_client_id(session, client_order_id)
                assert order is not None
                assert order.status == "CANCELLED"
                assert order.stop_loss == 2950.0

        cleanup_test_order(client_order_id)

    def test_timestamp_extraction(self, test_db, mock_exchange, mock_binance_order_filled):
        """Test timestamp extraction for created_at and closed_at."""
        mock_exchange.fetch_closed_orders = Mock(return_value=[mock_binance_order_filled])

        # Clean up if order already exists
        with session_scope() as session:
            existing = get_order_by_client_id(session, "AT_1707043200_BTCUSDT_abc123")
            if existing:
                session.delete(existing)
                session.commit()

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["BTC/USDT"])

            assert result["inserted"] == 1, f"Expected 1 inserted, got {result['inserted']}. Errors: {result['errors']}"

            with session_scope() as session:
                order = get_order_by_client_id(session, "AT_1707043200_BTCUSDT_abc123")
                assert order is not None
                assert order.created_at == datetime.fromtimestamp(1707043200)
                assert order.closed_at == datetime.fromtimestamp(1707043300)

                # Clean up after test
                session.delete(order)
                session.commit()

    def test_stop_loss_take_profit_extraction(self, test_db, mock_exchange):
        """Test SL/TP extraction from Binance order."""
        order_with_sl_tp = {
            "id": "111222333",
            "clientOrderId": "AT_1707043400_SOLUSDT_xyz789",
            "symbol": "SOL/USDT",
            "type": "LIMIT",
            "side": "BUY",
            "status": "FILLED",
            "price": 100.0,
            "average": 100.0,
            "amount": 1.0,
            "filled": 1.0,
            "leverage": 5,
            "timestamp": 1707043400000,
            "lastTradeTimestamp": 1707043500000,
            "stopPrice": 95.0,
            "info": {"stopPrice": "95.0", "takeProfit": "110.0"},
        }

        client_order_id = order_with_sl_tp["clientOrderId"]
        cleanup_test_order(client_order_id)

        mock_exchange.fetch_closed_orders = Mock(return_value=[order_with_sl_tp])

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["SOL/USDT"])

            assert result["inserted"] == 1

            with session_scope() as session:
                order = get_order_by_client_id(session, client_order_id)
                assert order is not None
                assert order.stop_loss == 95.0
                assert order.take_profit == 110.0

        cleanup_test_order(client_order_id)

    def test_skip_non_programmatic_orders(self, test_db, mock_exchange):
        """Test that non-AT_ orders are skipped."""
        manual_order = {
            "id": "999888777",
            "clientOrderId": "MANUAL_ORDER_123",
            "symbol": "BTC/USDT",
            "type": "MARKET",
            "side": "BUY",
            "status": "FILLED",
            "price": 50000.0,
            "average": 50000.0,
            "amount": 0.01,
            "filled": 0.01,
            "leverage": 2,
            "timestamp": 1707043600000,
        }

        mock_exchange.fetch_closed_orders = Mock(return_value=[manual_order])

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["BTC/USDT"])

            assert result["inserted"] == 0
            assert result["skipped"] == 0

    def test_skip_duplicate_orders(self, test_db, mock_exchange, mock_binance_order_filled):
        """Test that duplicate orders are skipped."""
        mock_exchange.fetch_closed_orders = Mock(return_value=[mock_binance_order_filled])

        # Clean up if order already exists
        with session_scope() as session:
            existing = get_order_by_client_id(session, "AT_1707043200_BTCUSDT_abc123")
            if existing:
                session.delete(existing)
                session.commit()

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            # First call - should insert
            result1 = reconcile_orders_with_binance("key", "secret", symbols=["BTC/USDT"])
            assert result1["inserted"] == 1
            assert result1["skipped"] == 0

            # Second call - should skip
            result2 = reconcile_orders_with_binance("key", "secret", symbols=["BTC/USDT"])
            assert result2["inserted"] == 0
            assert result2["skipped"] == 1

            # Clean up after test
            with session_scope() as session:
                order = get_order_by_client_id(session, "AT_1707043200_BTCUSDT_abc123")
                if order:
                    session.delete(order)
                    session.commit()

    def test_unsupported_order_type(self, test_db, mock_exchange):
        """Test that unsupported order types are rejected."""
        unsupported_order = {
            "id": "444555666",
            "clientOrderId": "AT_1707043700_BTCUSDT_unsup",
            "symbol": "BTC/USDT",
            "type": "UNSUPPORTED_TYPE",
            "side": "BUY",
            "status": "FILLED",
            "price": 50000.0,
            "average": 50000.0,
            "amount": 0.01,
            "filled": 0.01,
            "leverage": 2,
            "timestamp": 1707043700000,
        }

        mock_exchange.fetch_closed_orders = Mock(return_value=[unsupported_order])

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["BTC/USDT"])

            assert result["inserted"] == 0
            assert len(result["errors"]) == 1
            assert "unsupported order type" in result["errors"][0].lower()

    def test_pagination(self, test_db, mock_exchange):
        """Test pagination for large order sets."""
        # Create 150 orders (more than limit of 100)
        orders_batch1 = [
            {
                "id": f"order_{i}",
                "clientOrderId": f"AT_170704{i:04d}_BTCUSDT_test{i}",
                "symbol": "BTC/USDT",
                "type": "MARKET",
                "side": "BUY",
                "status": "FILLED",
                "price": 50000.0,
                "average": 50000.0,
                "amount": 0.01,
                "filled": 0.01,
                "leverage": 2,
                "timestamp": 1707040000000 + i * 1000,
            }
            for i in range(100)
        ]

        orders_batch2 = [
            {
                "id": f"order_{i}",
                "clientOrderId": f"AT_170704{i:04d}_BTCUSDT_test{i}",
                "symbol": "BTC/USDT",
                "type": "MARKET",
                "side": "BUY",
                "status": "FILLED",
                "price": 50000.0,
                "average": 50000.0,
                "amount": 0.01,
                "filled": 0.01,
                "leverage": 2,
                "timestamp": 1707040000000 + i * 1000,
            }
            for i in range(100, 150)
        ]

        # Clean up existing test orders
        for i in range(150):
            cleanup_test_order(f"AT_170704{i:04d}_BTCUSDT_test{i}")

        # Mock fetch_closed_orders to return batches
        mock_exchange.fetch_closed_orders = Mock(side_effect=[orders_batch1, orders_batch2])

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["BTC/USDT"])

            assert result["inserted"] == 150
            assert mock_exchange.fetch_closed_orders.call_count == 2

        # Clean up after test
        for i in range(150):
            cleanup_test_order(f"AT_170704{i:04d}_BTCUSDT_test{i}")

    def test_exchange_cleanup(self, test_db, mock_exchange, mock_binance_order_filled):
        """Test that exchange connection is properly closed."""
        mock_exchange.fetch_closed_orders = Mock(return_value=[mock_binance_order_filled])

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["BTC/USDT"])

            # Verify exchange.close() was called
            mock_exchange.close.assert_called_once()

    def test_invalid_price_or_amount(self, test_db, mock_exchange):
        """Test that orders with invalid price/amount are rejected."""
        invalid_order = {
            "id": "777888999",
            "clientOrderId": "AT_1707043800_BTCUSDT_invalid",
            "symbol": "BTC/USDT",
            "type": "MARKET",
            "side": "BUY",
            "status": "FILLED",
            "price": 0,
            "average": 0,
            "amount": 0,
            "filled": 0,
            "leverage": 2,
            "timestamp": 1707043800000,
        }

        mock_exchange.fetch_closed_orders = Mock(return_value=[invalid_order])

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["BTC/USDT"])

            assert result["inserted"] == 0
            assert len(result["errors"]) == 1
            assert "invalid amount/price" in result["errors"][0]

    def test_order_type_field_present(self, test_db, mock_exchange, mock_binance_order_filled):
        """Test that order_type field is correctly set in database."""
        mock_exchange.fetch_closed_orders = Mock(return_value=[mock_binance_order_filled])

        # Clean up if order already exists
        with session_scope() as session:
            existing = get_order_by_client_id(session, "AT_1707043200_BTCUSDT_abc123")
            if existing:
                session.delete(existing)
                session.commit()

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["BTC/USDT"])

            assert result["inserted"] == 1

            with session_scope() as session:
                order = get_order_by_client_id(session, "AT_1707043200_BTCUSDT_abc123")
                assert order is not None
                assert order.order_type == "MARKET"

                # Clean up after test
                session.delete(order)
                session.commit()


class TestCloseStaleOrders:
    """Test closing stale OPEN orders not on Binance anymore."""

    @pytest.fixture
    def mock_exchange(self):
        """Create mock CCXT exchange."""
        exchange = Mock()
        exchange.close = Mock()
        return exchange

    def test_close_stale_open_order(self, test_db, mock_exchange):
        """Test that stale OPEN orders are closed when not on Binance anymore."""
        from modules.auto_trade.database import create_order, get_order_by_client_id, session_scope, get_open_positions

        # Clean up any existing open orders first
        with session_scope() as session:
            open_orders = get_open_positions(session)
            for order in open_orders:
                session.delete(order)
            session.commit()

        # Create an OPEN order in DB with unique symbol and UUID to avoid conflicts
        unique_id = str(uuid.uuid4())[:8]
        order_id = f"ORDER_STALE_{unique_id}"
        client_order_id = f"AT_1707043200_XYZUSDT_{unique_id}"
        with session_scope() as session:
            order_data = {
                "order_id": order_id,
                "client_order_id": client_order_id,
                "symbol": "XYZUSDT",
                "side": "LONG",
                "entry_price": 50000.0,
                "amount": 0.01,
                "leverage": 2,
                "status": "OPEN",
                "order_source": "PROGRAMMATIC",
                "execution_mode": "AUTO",
                "created_at": datetime.now(),
            }
            create_order(session, order_data)

        # Mock: Binance returns no open orders (order is gone) and fetch_order returns FILLED
        mock_exchange.fetch_open_orders = Mock(return_value=[])  # No open orders on Binance
        mock_exchange.fetch_closed_orders = Mock(return_value=[])  # No closed orders in range
        mock_exchange.fetch_order = Mock(
            return_value={
                "id": order_id,
                "clientOrderId": client_order_id,
                "symbol": "XYZ/USDT",
                "type": "MARKET",
                "side": "BUY",
                "status": "FILLED",
                "price": 50000.0,
                "average": 50000.0,
                "amount": 0.01,
                "filled": 0.01,
                "timestamp": 1707043200000,
                "lastTradeTimestamp": 1707043300000,
                "info": {"realizedPnl": "25.50"},
            }
        )

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["XYZ/USDT"])

            # Should have closed 1 stale order
            assert result["closed_stale"] == 1, f"Expected closed_stale=1, got {result['closed_stale']}"

            # Verify order is now CLOSED in DB
            with session_scope() as session:
                order = get_order_by_client_id(session, client_order_id)
                assert order is not None
                assert order.status == "CLOSED"
                assert order.closed_at is not None
                assert order.pnl == 25.50

        # Cleanup
        cleanup_test_order(client_order_id)

    def test_close_stale_cancelled_order(self, test_db, mock_exchange):
        """Test that stale OPEN orders are marked CANCELLED when cancelled on Binance."""
        from modules.auto_trade.database import create_order, get_order_by_client_id, session_scope

        # Create an OPEN order in DB
        client_order_id = "AT_1707043200_ETHUSDT_stale002"
        with session_scope() as session:
            order_data = {
                "order_id": "ORDER_STALE_002",
                "client_order_id": client_order_id,
                "symbol": "ETHUSDT",
                "side": "SHORT",
                "entry_price": 3000.0,
                "amount": 0.1,
                "leverage": 3,
                "status": "OPEN",
                "order_source": "PROGRAMMATIC",
                "execution_mode": "AUTO",
                "created_at": datetime.now(),
            }
            create_order(session, order_data)

        # Mock: Order was cancelled on Binance
        mock_exchange.fetch_open_orders = Mock(return_value=[])
        mock_exchange.fetch_closed_orders = Mock(return_value=[])
        mock_exchange.fetch_order = Mock(
            return_value={
                "id": "ORDER_STALE_002",
                "clientOrderId": client_order_id,
                "symbol": "ETH/USDT",
                "type": "LIMIT",
                "side": "SELL",
                "status": "CANCELED",
                "price": 3000.0,
                "average": 0,
                "amount": 0.1,
                "filled": 0,
                "timestamp": 1707043200000,
                "lastTradeTimestamp": None,
                "info": {},
            }
        )

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["ETH/USDT"])

            assert result["closed_stale"] == 1

            # Verify order is now CANCELLED in DB
            with session_scope() as session:
                order = get_order_by_client_id(session, client_order_id)
                assert order is not None
                assert order.status == "CANCELLED"

        # Cleanup
        cleanup_test_order(client_order_id)

    def test_no_close_when_order_still_open(self, test_db, mock_exchange):
        """Test that orders are NOT closed when still open on Binance."""
        from modules.auto_trade.database import create_order, get_order_by_client_id, session_scope

        # Create an OPEN order in DB
        client_order_id = "AT_1707043200_BTCUSDT_active001"
        with session_scope() as session:
            order_data = {
                "order_id": "ORDER_ACTIVE_001",
                "client_order_id": client_order_id,
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 50000.0,
                "amount": 0.01,
                "leverage": 2,
                "status": "OPEN",
                "order_source": "PROGRAMMATIC",
                "execution_mode": "AUTO",
                "created_at": datetime.now(),
            }
            create_order(session, order_data)

        # Mock: Order is still open on Binance
        mock_exchange.fetch_open_orders = Mock(
            return_value=[
                {
                    "id": "ORDER_ACTIVE_001",
                    "clientOrderId": client_order_id,
                    "symbol": "BTC/USDT",
                    "type": "LIMIT",
                    "side": "BUY",
                    "status": "OPEN",
                    "price": 50000.0,
                    "amount": 0.01,
                }
            ]
        )
        mock_exchange.fetch_closed_orders = Mock(return_value=[])

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["BTC/USDT"])

            # Should not close any stale orders
            assert result["closed_stale"] == 0

            # Verify order is still OPEN in DB
            with session_scope() as session:
                order = get_order_by_client_id(session, client_order_id)
                assert order is not None
                assert order.status == "OPEN"

        # Cleanup
        cleanup_test_order(client_order_id)

    def test_close_stale_api_failure(self, test_db, mock_exchange):
        """Test that stale orders are closed even when fetch_order fails."""
        from modules.auto_trade.database import create_order, get_order_by_client_id, session_scope

        # Create an OPEN order in DB
        client_order_id = "AT_1707043200_BTCUSDT_stale003"
        with session_scope() as session:
            order_data = {
                "order_id": "ORDER_STALE_003",
                "client_order_id": client_order_id,
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 50000.0,
                "amount": 0.01,
                "leverage": 2,
                "status": "OPEN",
                "order_source": "PROGRAMMATIC",
                "execution_mode": "AUTO",
                "created_at": datetime.now(),
            }
            create_order(session, order_data)

        # Mock: fetch_order fails but we should still close the stale order
        mock_exchange.fetch_open_orders = Mock(return_value=[])
        mock_exchange.fetch_closed_orders = Mock(return_value=[])
        mock_exchange.fetch_order = Mock(side_effect=Exception("Order not found"))

        with patch("modules.auto_trade.database.reconcile.ccxt.binance", return_value=mock_exchange):
            result = reconcile_orders_with_binance("key", "secret", symbols=["BTC/USDT"])

            assert result["closed_stale"] == 1

            # Verify order is closed (even though API failed)
            with session_scope() as session:
                order = get_order_by_client_id(session, client_order_id)
                assert order is not None
                assert order.status == "CLOSED"
                assert order.closed_at is None  # No timestamp since API failed

        # Cleanup
        cleanup_test_order(client_order_id)


class TestWebSocketSync:
    """Test WebSocket path updates DB when orders close."""

    def test_websocket_sync_closed_order(self):
        """Test that WebSocket order close updates DB."""
        from datetime import datetime
        from unittest.mock import MagicMock, patch

        from modules.auto_trade.database import create_order, get_order_by_client_id, session_scope
        from modules.auto_trade.monitoring.account_monitor import OrderSnapshot

        # Create an OPEN order in DB
        client_order_id = "AT_1707043200_BTCUSDT_ws001"
        with session_scope() as session:
            order_data = {
                "order_id": "ORDER_WS_001",
                "client_order_id": client_order_id,
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 50000.0,
                "amount": 0.01,
                "leverage": 2,
                "status": "OPEN",
                "order_source": "PROGRAMMATIC",
                "execution_mode": "AUTO",
                "created_at": datetime.now(),
            }
            create_order(session, order_data)

        # Create a mock OrderSnapshot (closed)
        now = datetime.now()
        closed_snapshot = OrderSnapshot(
            order_id="ORDER_WS_001",
            client_order_id=client_order_id,
            symbol="BTC/USDT",
            side="buy",
            type="market",
            status="closed",  # WebSocket status
            price=50000.0,
            amount=0.01,
            filled=0.01,
            remaining=0.0,
            timestamp=now,
            last_update_timestamp=now,
        )

        # Import the handler
        from modules.auto_trade.gui.utils.websocket_data_service import WebSocketDataService

        # Create service and call handler
        service = WebSocketDataService(mode="DEMO")

        with patch("modules.auto_trade.database.session_scope") as mock_session_scope:
            mock_session = MagicMock()
            mock_session_scope.return_value.__enter__ = Mock(return_value=mock_session)
            mock_session_scope.return_value.__exit__ = Mock(return_value=False)

            with patch("modules.auto_trade.database.update_order_status_by_client_id") as mock_update:
                mock_update.return_value = True

                # Call the handler
                service._handle_order_update(closed_snapshot)

                # Verify DB was called to update
                mock_update.assert_called_once()
                call_args = mock_update.call_args
                assert call_args[1]["client_order_id"] == client_order_id
                assert call_args[1]["status"] == "CLOSED"

        # Cleanup
        cleanup_test_order(client_order_id)

    def test_websocket_sync_cancelled_order(self):
        """Test that WebSocket order cancel updates DB."""
        from datetime import datetime
        from unittest.mock import MagicMock, patch

        from modules.auto_trade.database import create_order, session_scope
        from modules.auto_trade.monitoring.account_monitor import OrderSnapshot

        # Create an OPEN order in DB
        client_order_id = "AT_1707043200_ETHUSDT_ws002"
        with session_scope() as session:
            order_data = {
                "order_id": "ORDER_WS_002",
                "client_order_id": client_order_id,
                "symbol": "ETHUSDT",
                "side": "SHORT",
                "entry_price": 3000.0,
                "amount": 0.1,
                "leverage": 3,
                "status": "OPEN",
                "order_source": "PROGRAMMATIC",
                "execution_mode": "AUTO",
                "created_at": datetime.now(),
            }
            create_order(session, order_data)

        # Create a mock OrderSnapshot (cancelled)
        now = datetime.now()
        cancelled_snapshot = OrderSnapshot(
            order_id="ORDER_WS_002",
            client_order_id=client_order_id,
            symbol="ETH/USDT",
            side="sell",
            type="limit",
            status="canceled",  # WebSocket status
            price=3000.0,
            amount=0.1,
            filled=0.0,
            remaining=0.1,
            timestamp=now,
            last_update_timestamp=now,
        )

        from modules.auto_trade.gui.utils.websocket_data_service import WebSocketDataService

        service = WebSocketDataService(mode="DEMO")

        with patch("modules.auto_trade.database.session_scope") as mock_session_scope:
            mock_session = MagicMock()
            mock_session_scope.return_value.__enter__ = Mock(return_value=mock_session)
            mock_session_scope.return_value.__exit__ = Mock(return_value=False)

            with patch("modules.auto_trade.database.update_order_status_by_client_id") as mock_update:
                mock_update.return_value = True

                service._handle_order_update(cancelled_snapshot)

                # Verify DB was called with CANCELLED
                mock_update.assert_called_once()
                call_args = mock_update.call_args
                assert call_args[1]["status"] == "CANCELLED"

        # Cleanup
        cleanup_test_order(client_order_id)

    def test_websocket_no_sync_for_open_order(self):
        """Test that WebSocket does NOT sync when order is still open."""
        from datetime import datetime
        from unittest.mock import MagicMock, patch

        from modules.auto_trade.database import create_order, session_scope
        from modules.auto_trade.monitoring.account_monitor import OrderSnapshot

        # Create an OPEN order in DB
        client_order_id = "AT_1707043200_BTCUSDT_ws003"
        with session_scope() as session:
            order_data = {
                "order_id": "ORDER_WS_003",
                "client_order_id": client_order_id,
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 50000.0,
                "amount": 0.01,
                "leverage": 2,
                "status": "OPEN",
                "order_source": "PROGRAMMATIC",
                "execution_mode": "AUTO",
                "created_at": datetime.now(),
            }
            create_order(session, order_data)

        # Create a mock OrderSnapshot (still open)
        now = datetime.now()
        open_snapshot = OrderSnapshot(
            order_id="ORDER_WS_003",
            client_order_id=client_order_id,
            symbol="BTC/USDT",
            side="buy",
            type="limit",
            status="open",  # Still open
            price=50000.0,
            amount=0.01,
            filled=0.0,
            remaining=0.01,
            timestamp=now,
            last_update_timestamp=now,
        )

        from modules.auto_trade.gui.utils.websocket_data_service import WebSocketDataService

        service = WebSocketDataService(mode="DEMO")

        with patch("modules.auto_trade.database.update_order_status_by_client_id") as mock_update:
            service._handle_order_update(open_snapshot)

            # Should NOT call DB update for open orders
            mock_update.assert_not_called()

        # Cleanup
        cleanup_test_order(client_order_id)

    def test_websocket_no_sync_for_manual_order(self):
        """Test that WebSocket does NOT sync non-AT_ orders."""
        from datetime import datetime
        from unittest.mock import MagicMock, patch

        from modules.auto_trade.monitoring.account_monitor import OrderSnapshot

        # Create a manual (non-AT_) order snapshot
        now = datetime.now()
        manual_snapshot = OrderSnapshot(
            order_id="MANUAL_ORDER_001",
            client_order_id="MANUAL_ORDER_001",  # Not AT_ prefixed
            symbol="BTC/USDT",
            side="buy",
            type="market",
            status="closed",
            price=50000.0,
            amount=0.01,
            filled=0.01,
            remaining=0.0,
            timestamp=now,
            last_update_timestamp=now,
        )

        from modules.auto_trade.gui.utils.websocket_data_service import WebSocketDataService

        service = WebSocketDataService(mode="DEMO")

        with patch("modules.auto_trade.database.update_order_status_by_client_id") as mock_update:
            service._handle_order_update(manual_snapshot)

            # Should NOT call DB update for manual orders
            mock_update.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
