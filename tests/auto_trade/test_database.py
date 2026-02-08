"""
Unit Tests for Auto Trading System - Database Module
=====================================================

Tests database operations, models, and queries.

Run: pytest tests/auto_trade/test_database.py -v
"""

import sys
from pathlib import Path

import pytest
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import modules.auto_trade.database as db_module
from modules.auto_trade.database import (
    create_order,
    get_all_programmatic_orders,
    get_db_manager,
    get_open_positions,
    get_overall_stats,
    initialize_database,
    is_programmatic_order,
    mark_be_moved,
    mark_signal_executed,
    save_signal,
    session_scope,
    update_order_status,
)
from modules.auto_trade.database.models import Order, Signal


@pytest.mark.usefixtures("test_db")
class TestDatabaseInitialization:
    """Test database initialization."""

    def test_database_creation(self, test_db):
        """Test database file is created."""
        assert Path(test_db).exists(), f"Expected database file at {test_db}"

    def test_tables_created(self, test_db):
        """Test all tables are created."""
        with session_scope() as session:
            session.execute(text("SELECT * FROM orders LIMIT 1"))
            session.execute(text("SELECT * FROM signals LIMIT 1"))
            session.execute(text("SELECT * FROM martingale_chain LIMIT 1"))
            session.execute(text("SELECT * FROM system_state LIMIT 1"))
            session.execute(text("SELECT * FROM audit_log LIMIT 1"))


class TestOrderOperations:
    """Test order CRUD operations."""

    def test_create_order(self, test_db, sample_order_data):
        """Test creating an order."""
        with session_scope() as session:
            # sample_order_data is now a factory from conftest
            order_data = sample_order_data()
            order = create_order(session, order_data)

            assert order is not None, f"Expected order to be created, got None"
            assert (
                getattr(order, "order_id") == "TEST_ORDER_001"
            ), f"Expected order_id 'TEST_ORDER_001', got {getattr(order, 'order_id')}"
            assert getattr(order, "symbol") == "BTCUSDT", f"Expected symbol 'BTCUSDT', got {getattr(order, 'symbol')}"
            assert (
                getattr(order, "order_source") == "PROGRAMMATIC"
            ), f"Expected order_source 'PROGRAMMATIC', got {getattr(order, 'order_source')}"

    def test_get_open_positions(self, test_db, sample_order_data):
        """Test getting open positions."""
        with session_scope() as session:
            # Create order (must be OPEN)
            create_order(session, sample_order_data(status="OPEN"))

            # Get open positions
            positions = get_open_positions(session)

            assert len(positions) == 1, f"Expected 1 position, got {len(positions)}"
            assert (
                getattr(positions[0], "order_id") == "TEST_ORDER_001"
            ), f"Expected order_id 'TEST_ORDER_001', got {getattr(positions[0], 'order_id')}"

    def test_update_order_status(self, test_db, sample_order_data):
        """Test updating order status."""
        with session_scope() as session:
            create_order(session, sample_order_data())
            ok = update_order_status(session, "TEST_ORDER_001", "CLOSED", pnl=125.50)
            assert ok is True, "Expected update_order_status to return True"

            order = session.query(Order).filter(Order.order_id == "TEST_ORDER_001").first()
            assert order is not None, "Expected to find order with order_id 'TEST_ORDER_001'"
            assert getattr(order, "status") == "CLOSED", f"Expected status 'CLOSED', got {getattr(order, 'status')}"
            assert getattr(order, "pnl") == pytest.approx(
                125.50, abs=0.01
            ), f"Expected pnl 125.50, got {getattr(order, 'pnl')}"
            assert getattr(order, "closed_at") is not None, "Expected closed_at to be set"

    def test_mark_be_moved(self, test_db, sample_order_data):
        """Test marking break-even moved."""
        with session_scope() as session:
            create_order(session, sample_order_data())
            ok = mark_be_moved(session, "TEST_ORDER_001", new_stop_loss=50000.0)
            assert ok is True, "Expected mark_be_moved to return True"

            order = session.query(Order).filter(Order.order_id == "TEST_ORDER_001").first()
            assert order is not None, "Expected to find order with order_id 'TEST_ORDER_001'"
            assert getattr(order, "be_moved") is True, "Expected be_moved to be True"
            assert getattr(order, "stop_loss") == pytest.approx(
                50000.0, abs=0.01
            ), f"Expected stop_loss 50000.0, got {getattr(order, 'stop_loss')}"

    def test_programmatic_order_filtering(self, test_db, sample_order_data):
        """Test that only programmatic orders are returned."""
        with session_scope() as session:
            # Create programmatic order
            create_order(session, sample_order_data())

            # Create manual order
            manual_data = sample_order_data(
                order_id="MANUAL_001",
                client_order_id="MANUAL_CLIENT_001",
                order_source="MANUAL"
            )
            create_order(session, manual_data)

            # Get all programmatic orders
            prog_orders = get_all_programmatic_orders(session)

            assert len(prog_orders) == 1, f"Expected 1 programmatic order, got {len(prog_orders)}"
            assert (
                getattr(prog_orders[0], "order_source") == "PROGRAMMATIC"
            ), f"Expected order_source 'PROGRAMMATIC', got {getattr(prog_orders[0], 'order_source')}"

    def test_is_programmatic_order(self, test_db, sample_order_data):
        """Test checking if order is programmatic."""
        with session_scope() as session:
            # Create programmatic order
            create_order(session, sample_order_data())

            # Check
            is_prog = is_programmatic_order(session, "TEST_ORDER_001")

            assert is_prog is True, f"Expected is_programmatic_order to return True, got {is_prog}"


class TestSignalOperations:
    """Test signal operations."""

    def test_save_signal(self, test_db):
        """Test saving a signal."""
        with session_scope() as session:
            signal = save_signal(
                session,
                correlation_id="SIGNAL_001",
                symbol="BTCUSDT",
                signal_type="LONG",
                confidence=0.85,
                atc_score=0.75,
                xgboost_score=0.90,
            )

            assert signal is not None, "Expected signal to be saved, got None"
            assert (
                getattr(signal, "correlation_id") == "SIGNAL_001"
            ), f"Expected correlation_id 'SIGNAL_001', got {getattr(signal, 'correlation_id')}"
            assert getattr(signal, "confidence") == pytest.approx(
                0.85, abs=0.01
            ), f"Expected confidence 0.85, got {getattr(signal, 'confidence')}"

    def test_mark_signal_executed(self, test_db, sample_order_data):
        """Test marking signal as executed (execution_order_id references orders.id)."""
        with session_scope() as session:
            save_signal(session, correlation_id="SIGNAL_001", symbol="BTCUSDT", signal_type="LONG", confidence=0.85)
            order = create_order(session, sample_order_data())

            ok = mark_signal_executed(session, "SIGNAL_001", str(order.id))
            assert ok is True, f"Expected mark_signal_executed to return True, got {ok}"

            signal = session.query(Signal).filter(Signal.correlation_id == "SIGNAL_001").first()
            assert signal is not None, "Expected to find signal with correlation_id 'SIGNAL_001'"
            assert getattr(signal, "executed") is True, "Expected signal.executed to be True"
            assert getattr(signal, "execution_order_id") == str(
                order.id
            ), f"Expected execution_order_id {str(order.id)}, got {getattr(signal, 'execution_order_id')}"


class TestStatistics:
    """Test statistics calculations."""

    def test_overall_stats_empty(self, test_db):
        """Test stats with no orders."""
        with session_scope() as session:
            stats = get_overall_stats(session)

            assert stats["total_trades"] == 0, f"Expected total_trades 0, got {stats['total_trades']}"
            assert stats["win_rate"] == pytest.approx(0.0, abs=0.01), f"Expected win_rate 0.0, got {stats['win_rate']}"

    def test_overall_stats_with_data(self, test_db, sample_order_data):
        """Test stats with orders."""
        with session_scope() as session:
            # Create winning order
            win_data = sample_order_data(status="CLOSED", pnl=125.50)
            create_order(session, win_data)

            # Create losing order
            loss_data = sample_order_data(
                order_id="TEST_ORDER_002",
                client_order_id="AT_12346_ETHUSDT_def456",
                status="CLOSED",
                pnl=-50.25
            )
            create_order(session, loss_data)

            # Get stats
            stats = get_overall_stats(session)

            assert stats["total_trades"] == 2, f"Expected total_trades 2, got {stats['total_trades']}"
            assert stats["winning_trades"] == 1, f"Expected winning_trades 1, got {stats['winning_trades']}"
            assert stats["losing_trades"] == 1, f"Expected losing_trades 1, got {stats['losing_trades']}"
            assert stats["win_rate"] == pytest.approx(
                50.0, abs=0.01
            ), f"Expected win_rate 50.0, got {stats['win_rate']}"
            assert stats["total_pnl"] == pytest.approx(
                75.25, abs=0.01
            ), f"Expected total_pnl 75.25, got {stats['total_pnl']}"


class TestDataValidation:
    """Test data validation."""

    def test_required_fields(self, test_db):
        """Test that required fields are enforced."""
        with session_scope() as session:
            with pytest.raises(ValueError):
                create_order(session, {"symbol": "BTCUSDT"})

    def test_side_validation(self, test_db, sample_order_data):
        """Test that invalid side is rejected."""
        with session_scope() as session:
            invalid_data = sample_order_data(side="INVALID")

            with pytest.raises(Exception):
                create_order(session, invalid_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
