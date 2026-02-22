"""
Integration Tests for DynamoDB
==============================

End-to-end tests for the complete DynamoDB workflow.

Usage:
    pytest tests/auto_trade/database/test_integration_dynamodb.py -v

Created: 2026-02-20
"""

# MUST set environment variables BEFORE importing any modules that use them
import os

os.environ["DB_BACKEND"] = "dynamodb"
os.environ["AWS_ACCESS_KEY_ID"] = "test"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["DYNAMODB_TABLE_NAME"] = "TestAutoTrade"

import sys

# Add conftest path for fixtures
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dynamodb"))

import pytest
from datetime import datetime, timezone
from decimal import Decimal


class TestDynamoDBIntegration:
    """Integration tests for full DynamoDB workflow."""

    def test_full_signal_to_order_workflow(self, setup_dynamodb_table):
        """Test complete workflow: save_signal -> create_order -> mark_signal_executed."""
        from modules.auto_trade.database.repository.factory import (
            get_signal_repository,
            get_order_repository,
        )

        sig_repo = get_signal_repository()
        ord_repo = get_order_repository()

        now = datetime.now(timezone.utc)

        signal = sig_repo.save_signal(
            {
                "correlation_id": "sig_integration_1",
                "symbol": "BTCUSDT",
                "signal_type": "LONG",
                "confidence": 0.85,
                "created_at": now,
                "executed": False,
            }
        )
        assert signal["correlation_id"] == "sig_integration_1"
        assert signal.get("executed") is False or signal.get("executed") is None

        order = ord_repo.create_order(
            {
                "order_id": "order_integration_1",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 50000.0,
                "amount": 0.01,
                "status": "OPEN",
                "created_at": now,
            }
        )
        assert order["order_id"] == "order_integration_1"
        assert order["status"] == "OPEN"

        success = sig_repo.mark_signal_executed("sig_integration_1", "order_integration_1")
        assert success is True

        updated_signal = sig_repo.get_recent_signals(limit=1)[0]
        assert updated_signal["executed"] is True
        assert updated_signal["execution_order_id"] == "order_integration_1"

    def test_order_lifecycle(self, setup_dynamodb_table):
        """Test order lifecycle: create -> update -> close."""
        from modules.auto_trade.database.repository.factory import get_order_repository

        ord_repo = get_order_repository()
        now = datetime.now(timezone.utc)

        order = ord_repo.create_order(
            {
                "order_id": "order_lifecycle_1",
                "symbol": "ETHUSDT",
                "side": "SHORT",
                "entry_price": 3000.0,
                "amount": 0.1,
                "status": "PENDING",
                "created_at": now,
            }
        )
        assert order["status"] == "PENDING"

        opened = ord_repo.update_order_status("order_lifecycle_1", "OPEN")
        assert opened is True

        open_orders = ord_repo.get_open_positions(symbol="ETHUSDT")
        assert len(open_orders) == 1

        closed = ord_repo.update_order_status("order_lifecycle_1", "CLOSED", pnl=25.5)
        assert closed is True

        final_order = ord_repo.get_order_by_id("order_lifecycle_1")
        assert final_order["status"] == "CLOSED"
        assert final_order["pnl"] == 25.5

    def test_signal_outcome_after_close(self, setup_dynamodb_table):
        """Test update_signal_outcome after order is closed."""
        from modules.auto_trade.database.repository.factory import (
            get_order_repository,
            get_signal_repository,
        )

        sig_repo = get_signal_repository()
        ord_repo = get_order_repository()
        now = datetime.now(timezone.utc)

        sig_repo.save_signal(
            {
                "correlation_id": "sig_outcome_1",
                "symbol": "BTCUSDT",
                "signal_type": "LONG",
                "created_at": now,
                "executed": True,
                "execution_order_id": "order_outcome_1",
            }
        )

        ord_repo.create_order(
            {
                "order_id": "order_outcome_1",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "status": "OPEN",
                "created_at": now,
            }
        )

        assert ord_repo.update_order_status("order_outcome_1", "CLOSED", pnl=42.0)
        assert sig_repo.update_signal_outcome("sig_outcome_1", "WIN", outcome_pnl=42.0)

        signal = sig_repo.get_recent_signals(limit=1)[0]
        assert signal["outcome"] == "WIN"
        assert signal["outcome_pnl"] == 42.0

    def test_signal_performance_stats(self, setup_dynamodb_table):
        """Test signal performance statistics calculation."""
        from modules.auto_trade.database.repository.factory import get_signal_repository

        sig_repo = get_signal_repository()
        now = datetime.now(timezone.utc)

        signals = [
            {
                "correlation_id": f"sig_perf_{i}",
                "symbol": "BTCUSDT",
                "signal_type": "LONG",
                "confidence": 0.8,
                "created_at": now,
                "executed": True,
                "outcome": "WIN" if i % 2 == 0 else "LOSS",
                "outcome_pnl": 10.0 if i % 2 == 0 else -5.0,
            }
            for i in range(5)
        ]

        for sig in signals:
            sig_repo.save_signal(sig)

        stats = sig_repo.get_signal_performance_stats(days=30)

        assert "total_signals" in stats
        assert "wins" in stats
        assert "losses" in stats
        assert stats["total_signals"] == 5

    def test_atomic_transaction(self, setup_dynamodb_table):
        """Test atomic create_order_with_signal transaction."""
        from modules.auto_trade.database.repository.factory import (
            get_signal_repository,
            get_order_repository,
        )
        from modules.auto_trade.database.repository.dynamodb.transactions import (
            transact_create_order_with_signal,
        )

        sig_repo = get_signal_repository()
        ord_repo = get_order_repository()
        now = datetime.now(timezone.utc)

        sig_repo.save_signal(
            {
                "correlation_id": "sig_atomic_1",
                "symbol": "BTCUSDT",
                "created_at": now,
            }
        )

        order_data = {
            "order_id": "order_atomic_1",
            "symbol": "BTCUSDT",
            "status": "OPEN",
            "created_at": now,
        }

        result = transact_create_order_with_signal(order_data, "sig_atomic_1", now.isoformat())
        assert result is True

        order = ord_repo.get_order_by_id("order_atomic_1")
        assert order is not None
        assert order["status"] == "OPEN"

        signal = sig_repo.get_recent_signals(limit=1)[0]
        assert signal["executed"] is True
        assert signal["execution_order_id"] == "order_atomic_1"

    def test_system_state_operations(self, setup_dynamodb_table):
        """Test system state get/set operations."""
        from modules.auto_trade.database.repository.factory import (
            get_system_state_repository,
        )

        state_repo = get_system_state_repository()

        state_repo.set_system_state("system.trading_enabled", True, "boolean")

        value = state_repo.get_system_state("system.trading_enabled")
        assert value is True

        state_repo.set_system_state("risk.max_position_size", 10000, "integer")
        value = state_repo.get_system_state("risk.max_position_size")
        assert value == 10000

    def test_audit_log_with_ttl(self, setup_dynamodb_table):
        """Test audit log creation with TTL."""
        from modules.auto_trade.database.repository.factory import (
            get_audit_log_repository,
        )

        audit_repo = get_audit_log_repository()
        now = datetime.now(timezone.utc)

        log = audit_repo.create_audit_log(
            {
                "correlation_id": "corr_audit_1",
                "event_type": "ORDER_CREATED",
                "event_category": "TRADING",
                "severity": "INFO",
                "event_summary": "Test order created",
                "created_at": now,
            }
        )

        assert log["event_type"] == "ORDER_CREATED"
        assert "expire_at" in log
        assert log["expire_at"] > 0

        recent = audit_repo.get_recent_audit_logs(limit=10)
        assert len(recent) >= 1

    def test_martingale_chain_lifecycle(self, setup_dynamodb_table):
        """Test find_or_create_martingale_chain and update_martingale_chain."""
        from modules.auto_trade.database.repository.factory import get_martingale_repository

        martingale_repo = get_martingale_repository()

        chain = martingale_repo.find_or_create_martingale_chain(
            symbol="BTCUSDT",
            initial_order_id="order_chain_1",
            loss=-100.0,
        )

        assert chain["symbol"] == "BTCUSDT"
        assert chain["status"] == "ACTIVE"

        assert martingale_repo.update_martingale_chain(
            chain["chain_id"],
            {
                "current_step": 1,
                "total_loss": -150.0,
                "latest_order_id": "order_chain_2",
            },
        )

        active_chain = martingale_repo.get_martingale_state("BTCUSDT")
        assert active_chain is not None
        assert active_chain["current_step"] == 1
        assert active_chain["latest_order_id"] == "order_chain_2"


class TestRepositoryContextIntegration:
    """Integration tests for RepositoryContext with DynamoDB."""

    def test_context_full_workflow(self, setup_dynamodb_table):
        """Test complete workflow using RepositoryContext."""
        from modules.auto_trade.database.repository.context import RepositoryContext

        ctx = RepositoryContext.from_env()

        now = datetime.now(timezone.utc)

        signal = ctx.signals.save_signal(
            {
                "correlation_id": "ctx_test_sig",
                "symbol": "BTCUSDT",
                "created_at": now,
            }
        )

        order = ctx.orders.create_order(
            {
                "order_id": "ctx_test_order",
                "symbol": "BTCUSDT",
                "status": "OPEN",
                "created_at": now,
            }
        )

        ctx.signals.mark_signal_executed("ctx_test_sig", "ctx_test_order")

        updated_order = ctx.orders.get_order_by_id("ctx_test_order")
        assert updated_order["status"] == "OPEN"
