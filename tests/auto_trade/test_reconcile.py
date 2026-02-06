"""
Unit Tests for Reconcile Module
================================

Tests the reconcile_orders_with_binance function to ensure all fixes work correctly.

Run: pytest tests/auto_trade/test_reconcile.py -v
"""

import sys
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
    def test_db(self, tmp_path):
        """Create temporary test database."""
        db_path = tmp_path / "test_reconcile.db"
        initialize_database(str(db_path))
        yield str(db_path)

    @pytest.fixture
    def mock_exchange(self):
        """Create mock CCXT exchange."""
        exchange = Mock()
        exchange.close = Mock()
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
