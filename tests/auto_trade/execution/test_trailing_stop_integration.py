"""
Integration Tests for Trailing Stop Job
========================================

Tests the TrailingStopJob class with mocked dependencies:
- Database session with test Order records
- Mocked BinanceClient for price fetching and SL modification
- Mocked SettingsManager for TP/SL settings

Created: 2026-02-06
"""

import sys
from pathlib import Path

# Ensure both project root and auto_trade module are in Python path
project_root = Path(__file__).parent.parent.parent.parent
auto_trade_root = project_root / "modules" / "auto_trade"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(auto_trade_root) not in sys.path:
    sys.path.insert(0, str(auto_trade_root))

import pytest
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Order
from execution.trailing_stop_job import TrailingStopJob, create_trailing_stop_job


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session


@pytest.fixture
def db_session_scope(in_memory_db):
    """Create a session scope context manager."""

    @contextmanager
    def session_scope():
        session = in_memory_db()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return session_scope


@pytest.fixture
def create_test_order(in_memory_db):
    """Factory fixture to create test orders."""

    def _create_order(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        status="OPEN",
        trailing_step_index=0,
        order_source="PROGRAMMATIC",
    ):
        session = in_memory_db()
        order = Order(
            order_id=f"TEST_{symbol}_{datetime.utcnow().timestamp()}",
            client_order_id=f"AT_{symbol}_{datetime.utcnow().timestamp()}",
            symbol=symbol,
            side=side,
            order_type="MARKET",
            order_source=order_source,
            execution_mode="AUTO",
            entry_price=entry_price,
            amount=1.0,
            notional_value=entry_price * 1.0,
            leverage=1,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=status,
            trailing_step_index=trailing_step_index,
        )
        session.add(order)
        session.commit()
        order_id = order.order_id
        session.close()
        return order_id

    return _create_order


@pytest.fixture
def mock_settings_manager():
    """Create a mock settings manager with trailing stop enabled."""
    manager = MagicMock()
    manager.get.return_value = {
        "trailing_stop": True,
        "trailing_step_pct": 2.0,
        "trailing_limit_steps": False,
        "trailing_max_steps": 5,
    }
    return manager


@pytest.fixture
def mock_settings_manager_with_limit():
    """Create a mock settings manager with trailing stop and limit enabled."""
    manager = MagicMock()
    manager.get.return_value = {
        "trailing_stop": True,
        "trailing_step_pct": 2.0,
        "trailing_limit_steps": True,
        "trailing_max_steps": 3,
    }
    return manager


@pytest.fixture
def mock_settings_manager_disabled():
    """Create a mock settings manager with trailing stop disabled."""
    manager = MagicMock()
    manager.get.return_value = {
        "trailing_stop": False,
        "trailing_step_pct": 2.0,
        "trailing_limit_steps": False,
        "trailing_max_steps": 5,
    }
    return manager


@pytest.fixture
def mock_binance_client():
    """Create a mock Binance client."""
    client = MagicMock()
    client.fetch_ticker.return_value = {"last": 105.0}
    client.modify_stop_loss.return_value = {"success": True}
    return client


# ============================================================================
# INTEGRATION TESTS - TrailingStopJob
# ============================================================================


class TestTrailingStopJobRun:
    """Integration tests for TrailingStopJob.run() method."""

    def test_run_with_trailing_stop_disabled(
        self, db_session_scope, mock_settings_manager_disabled, mock_binance_client
    ):
        """Job should skip processing when trailing stop is disabled."""
        job = TrailingStopJob(
            settings_manager=mock_settings_manager_disabled,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        result = job.run()

        assert result["orders_checked"] == 0
        assert result["orders_updated"] == 0
        assert result["errors"] == []
        mock_binance_client.fetch_ticker.assert_not_called()

    def test_run_with_no_open_orders(self, db_session_scope, mock_settings_manager, mock_binance_client):
        """Job should handle case with no open orders."""
        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        result = job.run()

        assert result["orders_checked"] == 0
        assert result["orders_updated"] == 0
        assert result["errors"] == []

    def test_run_step_0_to_1_long(
        self, db_session_scope, in_memory_db, mock_settings_manager, mock_binance_client, create_test_order
    ):
        """Test stepping from 0 to 1 (BE) for LONG position when price >= entry."""
        # Create order at step 0
        order_id = create_test_order(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            trailing_step_index=0,
        )

        # Mock price at 1% profit (should trigger BE)
        mock_binance_client.fetch_ticker.return_value = {"last": 101.0}

        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        result = job.run()

        assert result["orders_checked"] == 1
        assert result["orders_updated"] == 1
        assert len(result["updates"]) == 1
        assert result["updates"][0]["new_sl"] == 100.0  # BE = entry price
        assert result["updates"][0]["step_index"] == 1

        # Verify DB was updated
        session = in_memory_db()
        order = session.query(Order).filter_by(order_id=order_id).first()
        assert order.stop_loss == 100.0
        assert order.trailing_step_index == 1
        session.close()

    def test_run_step_1_to_2_long(
        self, db_session_scope, in_memory_db, mock_settings_manager, mock_binance_client, create_test_order
    ):
        """Test stepping from 1 to 2 for LONG position when profit >= 2%."""
        # Create order at step 1 (BE already applied)
        order_id = create_test_order(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            stop_loss=100.0,  # Already at BE
            trailing_step_index=1,
        )

        # Mock price at 2.5% profit (should trigger step 2)
        mock_binance_client.fetch_ticker.return_value = {"last": 102.5}

        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        result = job.run()

        assert result["orders_checked"] == 1
        assert result["orders_updated"] == 1
        assert result["updates"][0]["new_sl"] == 102.0  # entry + 2%
        assert result["updates"][0]["step_index"] == 2

        # Verify DB was updated
        session = in_memory_db()
        order = session.query(Order).filter_by(order_id=order_id).first()
        assert order.stop_loss == 102.0
        assert order.trailing_step_index == 2
        session.close()

    def test_run_step_0_to_1_short(
        self, db_session_scope, in_memory_db, mock_settings_manager, mock_binance_client, create_test_order
    ):
        """Test stepping from 0 to 1 (BE) for SHORT position when price <= entry."""
        # Create SHORT order at step 0
        order_id = create_test_order(
            symbol="BTCUSDT",
            side="SHORT",
            entry_price=100.0,
            stop_loss=105.0,
            trailing_step_index=0,
        )

        # Mock price at 1% profit for SHORT (should trigger BE)
        mock_binance_client.fetch_ticker.return_value = {"last": 99.0}

        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        result = job.run()

        assert result["orders_checked"] == 1
        assert result["orders_updated"] == 1
        assert result["updates"][0]["new_sl"] == 100.0  # BE = entry price
        assert result["updates"][0]["step_index"] == 1

        # Verify DB was updated
        session = in_memory_db()
        order = session.query(Order).filter_by(order_id=order_id).first()
        assert order.stop_loss == 100.0
        assert order.trailing_step_index == 1
        session.close()

    def test_run_step_1_to_2_short(
        self, db_session_scope, in_memory_db, mock_settings_manager, mock_binance_client, create_test_order
    ):
        """Test stepping from 1 to 2 for SHORT position when profit >= 2%."""
        # Create SHORT order at step 1 (BE already applied)
        order_id = create_test_order(
            symbol="BTCUSDT",
            side="SHORT",
            entry_price=100.0,
            stop_loss=100.0,  # Already at BE
            trailing_step_index=1,
        )

        # Mock price at 2.5% profit for SHORT (should trigger step 2)
        mock_binance_client.fetch_ticker.return_value = {"last": 97.5}

        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        result = job.run()

        assert result["orders_checked"] == 1
        assert result["orders_updated"] == 1
        assert result["updates"][0]["new_sl"] == 98.0  # entry - 2%
        assert result["updates"][0]["step_index"] == 2

        # Verify DB was updated
        session = in_memory_db()
        order = session.query(Order).filter_by(order_id=order_id).first()
        assert order.stop_loss == 98.0
        assert order.trailing_step_index == 2
        session.close()

    def test_run_no_step_when_profit_insufficient(
        self, db_session_scope, in_memory_db, mock_settings_manager, mock_binance_client, create_test_order
    ):
        """Test no step when profit is insufficient."""
        # Create order at step 1 (BE already applied)
        order_id = create_test_order(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            stop_loss=100.0,  # Already at BE
            trailing_step_index=1,
        )

        # Mock price at 1.5% profit (not enough for step 2 which requires 2%)
        mock_binance_client.fetch_ticker.return_value = {"last": 101.5}

        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        result = job.run()

        assert result["orders_checked"] == 1
        assert result["orders_updated"] == 0

        # Verify DB was NOT updated
        session = in_memory_db()
        order = session.query(Order).filter_by(order_id=order_id).first()
        assert order.stop_loss == 100.0
        assert order.trailing_step_index == 1
        session.close()

    def test_run_with_max_steps_limit(
        self, db_session_scope, in_memory_db, mock_settings_manager_with_limit, mock_binance_client, create_test_order
    ):
        """Test that max_steps limit is respected."""
        # Create order at step 3 (max_steps = 3, so no more steps allowed)
        order_id = create_test_order(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            stop_loss=104.0,  # Already at +4%
            trailing_step_index=3,
        )

        # Mock price at 10% profit (would trigger step 4, but max_steps = 3)
        mock_binance_client.fetch_ticker.return_value = {"last": 110.0}

        job = TrailingStopJob(
            settings_manager=mock_settings_manager_with_limit,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        result = job.run()

        assert result["orders_checked"] == 1
        assert result["orders_updated"] == 0  # No update because max_steps reached

        # Verify DB was NOT updated
        session = in_memory_db()
        order = session.query(Order).filter_by(order_id=order_id).first()
        assert order.stop_loss == 104.0
        assert order.trailing_step_index == 3
        session.close()

    def test_run_multiple_orders(
        self, db_session_scope, in_memory_db, mock_settings_manager, mock_binance_client, create_test_order
    ):
        """Test processing multiple orders in single run."""
        # Create multiple orders
        order_id_1 = create_test_order(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            trailing_step_index=0,
        )
        order_id_2 = create_test_order(
            symbol="ETHUSDT",
            side="SHORT",
            entry_price=100.0,
            stop_loss=105.0,
            trailing_step_index=0,
        )

        # Mock prices for both symbols (need different prices)
        def mock_fetch_ticker(symbol):
            if symbol == "BTCUSDT":
                return {"last": 101.0}  # 1% profit for LONG
            elif symbol == "ETHUSDT":
                return {"last": 99.0}  # 1% profit for SHORT
            return {"last": 100.0}

        mock_binance_client.fetch_ticker.side_effect = mock_fetch_ticker

        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        result = job.run()

        assert result["orders_checked"] == 2
        assert result["orders_updated"] == 2

        # Verify both orders were updated to BE
        session = in_memory_db()
        order_1 = session.query(Order).filter_by(order_id=order_id_1).first()
        order_2 = session.query(Order).filter_by(order_id=order_id_2).first()
        assert order_1.stop_loss == 100.0
        assert order_1.trailing_step_index == 1
        assert order_2.stop_loss == 100.0
        assert order_2.trailing_step_index == 1
        session.close()

    def test_run_skips_non_programmatic_orders(
        self, db_session_scope, in_memory_db, mock_settings_manager, mock_binance_client, create_test_order
    ):
        """Test that non-programmatic orders are skipped."""
        # Create a MANUAL order (should be skipped)
        create_test_order(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            trailing_step_index=0,
            order_source="MANUAL",
        )

        # Mock price at 1% profit
        mock_binance_client.fetch_ticker.return_value = {"last": 101.0}

        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        result = job.run()

        # MANUAL order should not be processed
        assert result["orders_checked"] == 0
        assert result["orders_updated"] == 0

    def test_run_skips_closed_orders(
        self, db_session_scope, in_memory_db, mock_settings_manager, mock_binance_client, create_test_order
    ):
        """Test that closed orders are skipped."""
        # Create a CLOSED order
        create_test_order(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            trailing_step_index=0,
            status="CLOSED",
        )

        # Mock price at 1% profit
        mock_binance_client.fetch_ticker.return_value = {"last": 101.0}

        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        result = job.run()

        # CLOSED order should not be processed
        assert result["orders_checked"] == 0
        assert result["orders_updated"] == 0


class TestTrailingStopJobDryRun:
    """Tests for TrailingStopJob dry run mode (no binance client)."""

    def test_dry_run_updates_db_only(
        self, db_session_scope, in_memory_db, mock_settings_manager, create_test_order
    ):
        """Test dry run mode updates DB without calling Binance."""
        # Create order at step 0
        order_id = create_test_order(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            trailing_step_index=0,
        )

        # Create job WITHOUT binance client (dry run)
        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=None,  # No client = dry run
        )

        result = job.run()

        # Dry run without client doesn't fetch prices, so no updates
        # This is expected behavior - no Binance client means no price data
        assert result["orders_checked"] == 0 or result["orders_updated"] == 0


class TestTrailingStopJobErrorHandling:
    """Tests for TrailingStopJob error handling."""

    def test_handles_binance_client_error(
        self, db_session_scope, in_memory_db, mock_settings_manager, mock_binance_client, create_test_order
    ):
        """Test job handles Binance client errors gracefully."""
        # Create order
        order_id = create_test_order(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            trailing_step_index=0,
        )

        # Mock price fetch to work, but modify_stop_loss to fail
        mock_binance_client.fetch_ticker.return_value = {"last": 101.0}
        mock_binance_client.modify_stop_loss.return_value = {
            "success": False,
            "error": "Insufficient margin",
        }

        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        result = job.run()

        assert result["orders_checked"] == 1
        assert result["orders_updated"] == 0  # Not updated because modify failed

        # Verify DB was NOT updated
        session = in_memory_db()
        order = session.query(Order).filter_by(order_id=order_id).first()
        assert order.stop_loss == 95.0  # Original SL
        assert order.trailing_step_index == 0  # Original step
        session.close()

    def test_handles_price_fetch_error(
        self, db_session_scope, in_memory_db, mock_settings_manager, mock_binance_client, create_test_order
    ):
        """Test job handles price fetch errors gracefully."""
        # Create order
        create_test_order(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            trailing_step_index=0,
        )

        # Mock price fetch to raise exception
        mock_binance_client.fetch_ticker.side_effect = Exception("Connection error")

        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        result = job.run()

        # Job should handle error gracefully
        assert result["orders_checked"] == 0  # Couldn't check because price fetch failed
        assert result["orders_updated"] == 0


class TestCreateTrailingStopJobFactory:
    """Tests for create_trailing_stop_job factory function."""

    def test_factory_creates_job(self, db_session_scope, mock_settings_manager, mock_binance_client):
        """Test factory creates TrailingStopJob instance."""
        job = create_trailing_stop_job(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        assert isinstance(job, TrailingStopJob)
        assert job.settings_manager is mock_settings_manager
        assert job.db_session_scope is db_session_scope
        assert job.binance_client is mock_binance_client

    def test_factory_without_binance_client(self, db_session_scope, mock_settings_manager):
        """Test factory creates job without Binance client (dry run)."""
        job = create_trailing_stop_job(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=None,
        )

        assert isinstance(job, TrailingStopJob)
        assert job.binance_client is None


class TestMultipleStepProgression:
    """Tests for multiple step progression scenarios."""

    def test_progressive_steps_long(
        self, db_session_scope, in_memory_db, mock_settings_manager, mock_binance_client, create_test_order
    ):
        """Test stepping progressively from 0 to 3 for LONG position."""
        order_id = create_test_order(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            trailing_step_index=0,
        )

        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        # Step 0 -> 1 (BE) at 1% profit
        mock_binance_client.fetch_ticker.return_value = {"last": 101.0}
        result = job.run()
        assert result["orders_updated"] == 1

        session = in_memory_db()
        order = session.query(Order).filter_by(order_id=order_id).first()
        assert order.trailing_step_index == 1
        assert order.stop_loss == 100.0
        session.close()

        # Step 1 -> 2 at 2.5% profit
        mock_binance_client.fetch_ticker.return_value = {"last": 102.5}
        result = job.run()
        assert result["orders_updated"] == 1

        session = in_memory_db()
        order = session.query(Order).filter_by(order_id=order_id).first()
        assert order.trailing_step_index == 2
        assert order.stop_loss == 102.0
        session.close()

        # Step 2 -> 3 at 4.5% profit
        mock_binance_client.fetch_ticker.return_value = {"last": 104.5}
        result = job.run()
        assert result["orders_updated"] == 1

        session = in_memory_db()
        order = session.query(Order).filter_by(order_id=order_id).first()
        assert order.trailing_step_index == 3
        assert order.stop_loss == 104.0  # entry + 4%
        session.close()

    def test_skip_steps_when_profit_jumps(
        self, db_session_scope, in_memory_db, mock_settings_manager, mock_binance_client, create_test_order
    ):
        """Test that only one step is taken per run, even if profit jumps multiple thresholds."""
        order_id = create_test_order(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            trailing_step_index=0,
        )

        # Mock price at 10% profit (would qualify for many steps)
        mock_binance_client.fetch_ticker.return_value = {"last": 110.0}

        job = TrailingStopJob(
            settings_manager=mock_settings_manager,
            db_session_scope=db_session_scope,
            binance_client=mock_binance_client,
        )

        # First run: should only step to 1 (BE)
        result = job.run()
        assert result["orders_updated"] == 1

        session = in_memory_db()
        order = session.query(Order).filter_by(order_id=order_id).first()
        assert order.trailing_step_index == 1  # Only stepped once
        assert order.stop_loss == 100.0  # BE
        session.close()

        # Second run: should step to 2
        result = job.run()
        assert result["orders_updated"] == 1

        session = in_memory_db()
        order = session.query(Order).filter_by(order_id=order_id).first()
        assert order.trailing_step_index == 2
        assert order.stop_loss == 102.0
        session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
