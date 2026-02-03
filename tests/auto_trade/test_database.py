"""
Unit Tests for Auto Trading System - Database Module
=====================================================

Tests database operations, models, and queries.

Run: pytest tests/auto_trade/test_database.py -v
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.auto_trade.database import (
    create_order,
    get_all_programmatic_orders,
    get_last_closed_order,
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


@pytest.fixture
def test_db(tmp_path):
    """Create temporary test database."""
    db_path = tmp_path / "test_auto_trade.db"
    initialize_database(str(db_path))
    yield str(db_path)


@pytest.fixture
def sample_order_data():
    """Sample order data for testing."""
    return {
        "order_id": "TEST_ORDER_001",
        "client_order_id": "AT_12345_BTCUSDT_abc123",
        "symbol": "BTCUSDT",
        "side": "LONG",
        "entry_price": 50000.0,
        "amount": 0.01,
        "leverage": 2,
        "stop_loss": 45000.0,
        "take_profit": 52500.0,
        "status": "OPEN",
        "order_source": "PROGRAMMATIC",
        "execution_mode": "AUTO",
    }


class TestDatabaseInitialization:
    """Test database initialization."""

    def test_database_creation(self, test_db):
        """Test database file is created."""
        assert Path(test_db).exists()

    def test_tables_created(self, test_db):
        """Test all tables are created."""
        with session_scope() as session:
            # Should not raise errors
            session.execute("SELECT * FROM orders LIMIT 1")
            session.execute("SELECT * FROM signals LIMIT 1")
            session.execute("SELECT * FROM martingale_chain LIMIT 1")
            session.execute("SELECT * FROM system_state LIMIT 1")
            session.execute("SELECT * FROM audit_log LIMIT 1")


class TestOrderOperations:
    """Test order CRUD operations."""

    def test_create_order(self, test_db, sample_order_data):
        """Test creating an order."""
        with session_scope() as session:
            order = create_order(session, sample_order_data)

            assert order is not None
            assert order.order_id == "TEST_ORDER_001"
            assert order.symbol == "BTCUSDT"
            assert order.order_source == "PROGRAMMATIC"

    def test_get_open_positions(self, test_db, sample_order_data):
        """Test getting open positions."""
        with session_scope() as session:
            # Create order
            create_order(session, sample_order_data)

            # Get open positions
            positions = get_open_positions(session)

            assert len(positions) == 1
            assert positions[0].order_id == "TEST_ORDER_001"

    def test_update_order_status(self, test_db, sample_order_data):
        """Test updating order status."""
        with session_scope() as session:
            # Create order
            create_order(session, sample_order_data)

            # Update status
            order = update_order_status(session, "TEST_ORDER_001", "CLOSED", pnl=125.50)

            assert order.status == "CLOSED"
            assert order.pnl == 125.50
            assert order.closed_at is not None

    def test_mark_be_moved(self, test_db, sample_order_data):
        """Test marking break-even moved."""
        with session_scope() as session:
            # Create order
            create_order(session, sample_order_data)

            # Mark BE moved
            order = mark_be_moved(session, "TEST_ORDER_001", new_stop_loss=50000.0)

            assert order.be_moved is True
            assert order.stop_loss == 50000.0

    def test_programmatic_order_filtering(self, test_db, sample_order_data):
        """Test that only programmatic orders are returned."""
        with session_scope() as session:
            # Create programmatic order
            create_order(session, sample_order_data)

            # Create manual order
            manual_data = sample_order_data.copy()
            manual_data["order_id"] = "MANUAL_001"
            manual_data["client_order_id"] = "MANUAL_CLIENT_001"
            manual_data["order_source"] = "MANUAL"
            create_order(session, manual_data)

            # Get all programmatic orders
            prog_orders = get_all_programmatic_orders(session)

            # Should only return 1 (programmatic)
            assert len(prog_orders) == 1
            assert prog_orders[0].order_source == "PROGRAMMATIC"

    def test_is_programmatic_order(self, test_db, sample_order_data):
        """Test checking if order is programmatic."""
        with session_scope() as session:
            # Create programmatic order
            create_order(session, sample_order_data)

            # Check
            is_prog = is_programmatic_order(session, "TEST_ORDER_001")

            assert is_prog is True


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

            assert signal is not None
            assert signal.correlation_id == "SIGNAL_001"
            assert signal.confidence == 0.85

    def test_mark_signal_executed(self, test_db, sample_order_data):
        """Test marking signal as executed."""
        with session_scope() as session:
            # Create signal
            save_signal(session, correlation_id="SIGNAL_001", symbol="BTCUSDT", signal_type="LONG", confidence=0.85)

            # Create order
            create_order(session, sample_order_data)

            # Mark signal executed
            signal = mark_signal_executed(session, "SIGNAL_001", "TEST_ORDER_001")

            assert signal.executed is True
            assert signal.order_id == "TEST_ORDER_001"


class TestStatistics:
    """Test statistics calculations."""

    def test_overall_stats_empty(self, test_db):
        """Test stats with no orders."""
        with session_scope() as session:
            stats = get_overall_stats(session)

            assert stats["total_orders"] == 0
            assert stats["win_rate"] == 0.0

    def test_overall_stats_with_data(self, test_db, sample_order_data):
        """Test stats with orders."""
        with session_scope() as session:
            # Create winning order
            win_data = sample_order_data.copy()
            win_data["status"] = "CLOSED"
            win_data["pnl"] = 125.50
            create_order(session, win_data)

            # Create losing order
            loss_data = sample_order_data.copy()
            loss_data["order_id"] = "TEST_ORDER_002"
            loss_data["client_order_id"] = "AT_12346_ETHUSDT_def456"
            loss_data["status"] = "CLOSED"
            loss_data["pnl"] = -50.25
            create_order(session, loss_data)

            # Get stats
            stats = get_overall_stats(session)

            assert stats["total_orders"] == 2
            assert stats["closed_orders"] == 2
            assert stats["winning_orders"] == 1
            assert stats["losing_orders"] == 1
            assert stats["win_rate"] == 50.0
            assert stats["total_pnl"] == 75.25


class TestDataValidation:
    """Test data validation."""

    def test_required_fields(self, test_db):
        """Test that required fields are enforced."""
        with session_scope() as session:
            with pytest.raises(Exception):
                # Missing required fields should fail
                create_order(
                    session,
                    {
                        "symbol": "BTCUSDT"
                        # Missing many required fields
                    },
                )

    def test_side_validation(self, test_db, sample_order_data):
        """Test that invalid side is rejected."""
        with session_scope() as session:
            invalid_data = sample_order_data.copy()
            invalid_data["side"] = "INVALID"

            with pytest.raises(Exception):
                create_order(session, invalid_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
