"""
Tests for DynamoDB Order Repository.

Created: 2026-02-20
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from modules.auto_trade.database.repository.dynamodb.orders import DynamoDBOrderRepository


class TestDynamoDBOrderRepository:
    def test_create_order(self, setup_dynamodb_table):
        repo = DynamoDBOrderRepository()

        now = datetime.now(timezone.utc)
        order_data = {
            "order_id": "test_order_1",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_price": 50000.0,
            "amount": 0.01,
            "status": "PENDING",
            "created_at": now,
        }

        order = repo.create_order(order_data)

        assert order["order_id"] == "test_order_1"
        assert order["symbol"] == "BTCUSDT"
        assert order["side"] == "LONG"
        assert order["entry_price"] == 50000.0
        assert "pk" in order
        assert order["pk"] == "ORDER#test_order_1"

    def test_get_open_positions(self, setup_dynamodb_table):
        repo = DynamoDBOrderRepository()

        # Create an OPEN order
        repo.create_order({"order_id": "open_1", "symbol": "BTCUSDT", "status": "OPEN", "order_source": "PROGRAMMATIC"})

        # Create a CLOSED order
        repo.create_order(
            {"order_id": "closed_1", "symbol": "BTCUSDT", "status": "CLOSED", "order_source": "PROGRAMMATIC"}
        )

        # Create OPEN for different symbol
        repo.create_order({"order_id": "open_2", "symbol": "ETHUSDT", "status": "OPEN", "order_source": "PROGRAMMATIC"})

        # Test without symbol
        open_orders = repo.get_open_positions()
        assert len(open_orders) == 2

        # Test with symbol
        btc_open = repo.get_open_positions(symbol="BTCUSDT")
        assert len(btc_open) == 1
        assert btc_open[0]["order_id"] == "open_1"

    def test_update_order_status(self, setup_dynamodb_table):
        repo = DynamoDBOrderRepository()

        repo.create_order({"order_id": "test_update", "symbol": "BTCUSDT", "status": "PENDING"})

        # Update status
        result = repo.update_order_status("test_update", "CLOSED", pnl=15.5)
        assert result is True

        # Verify
        order = repo.get_order_by_id("test_update")
        assert order["status"] == "CLOSED"
        assert order["pnl"] == 15.5
        assert "gsi1sk" in order
        assert order["gsi1sk"].startswith("ORDER#CLOSED#")
