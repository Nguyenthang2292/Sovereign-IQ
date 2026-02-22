"""
Tests for DynamoDB Transactions.

Created: 2026-02-20
"""

import pytest
from datetime import datetime, timezone

from modules.auto_trade.database.repository.dynamodb.transactions import transact_create_order_with_signal
from modules.auto_trade.database.repository.dynamodb.orders import DynamoDBOrderRepository
from modules.auto_trade.database.repository.dynamodb.signals import DynamoDBSignalRepository


class TestDynamoDBTransactions:
    def test_transact_create_order_with_signal(self, setup_dynamodb_table):
        sig_repo = DynamoDBSignalRepository()
        ord_repo = DynamoDBOrderRepository()

        now = datetime.now(timezone.utc)

        # 1. Create signal
        sig_repo.save_signal({"correlation_id": "corr_tx", "symbol": "BTCUSDT", "created_at": now})

        # 2. Perform atomic transaction
        order_data = {"order_id": "ord_tx", "symbol": "BTCUSDT", "created_at": now, "status": "OPEN"}

        result = transact_create_order_with_signal(order_data, "corr_tx", now.isoformat())
        assert result is True

        # 3. Verify Order created
        order = ord_repo.get_order_by_id("ord_tx")
        assert order is not None
        assert order["status"] == "OPEN"

        # 4. Verify Signal updated
        signals = sig_repo.get_recent_signals(limit=1)
        assert signals[0]["executed"] is True
        assert signals[0]["execution_order_id"] == "ord_tx"
