"""
Tests for DynamoDB Signal Repository.

Created: 2026-02-20
"""

import pytest
from datetime import datetime, timezone, timedelta

from modules.auto_trade.database.repository.dynamodb.signals import DynamoDBSignalRepository


class TestDynamoDBSignalRepository:
    def test_save_signal(self, setup_dynamodb_table):
        repo = DynamoDBSignalRepository()

        signal = repo.save_signal(
            {"correlation_id": "sig_1", "symbol": "BTCUSDT", "signal_type": "LONG", "confidence": 0.95}
        )

        assert signal["correlation_id"] == "sig_1"
        assert signal["confidence"] == 0.95
        assert signal["pk"] == "SIGNAL#sig_1"

    def test_mark_signal_executed(self, setup_dynamodb_table):
        repo = DynamoDBSignalRepository()

        repo.save_signal({"correlation_id": "sig_exec", "symbol": "BTCUSDT"})

        result = repo.mark_signal_executed("sig_exec", "order_123")
        assert result is True

        # Verify
        signals = repo.get_recent_signals(limit=1)
        assert signals[0]["executed"] is True
        assert signals[0]["execution_order_id"] == "order_123"

    def test_get_signal_performance_stats(self, setup_dynamodb_table):
        repo = DynamoDBSignalRepository()

        now = datetime.now(timezone.utc)

        # Save WIN
        repo.save_signal({"correlation_id": "s1", "symbol": "BTCUSDT", "created_at": now})
        repo.update_signal_outcome("s1", "WIN", 10.0)

        # Save LOSS
        repo.save_signal({"correlation_id": "s2", "symbol": "BTCUSDT", "created_at": now})
        repo.update_signal_outcome("s2", "LOSS", -5.0)

        # Save old record (should be excluded)
        old_date = now - timedelta(days=40)
        repo.save_signal({"correlation_id": "s3", "symbol": "BTCUSDT", "created_at": old_date})
        repo.update_signal_outcome("s3", "WIN", 20.0)

        stats = repo.get_signal_performance_stats(days=30)

        assert stats["total_signals"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["win_rate"] == 50.0
        assert stats["total_pnl"] == 5.0
