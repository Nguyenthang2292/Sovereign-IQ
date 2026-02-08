"""
Tests for Database Statistics and Edge Cases
=============================================

Tests get_overall_stats and other statistics functions with various scenarios.

Run: pytest tests/auto_trade/database/test_queries_stats.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import modules.auto_trade.database as db_module
from modules.auto_trade.database import (
    create_order,
    get_db_manager,
    get_overall_stats,
    initialize_database,
    session_scope,
)


@pytest.fixture
def test_db(tmp_path):
    """Create temporary test database and set global manager to use it."""
    db_path = tmp_path / "test_stats.db"
    db_module._db_manager_instance = None
    initialize_database(str(db_path))
    get_db_manager(str(db_path))
    yield str(db_path)


@pytest.fixture
def sample_order_data():
    """Sample order data for testing."""
    return {
        "order_id": "TEST_ORDER_001",
        "client_order_id": "AT_001_BTCUSDT_abc123",
        "symbol": "BTCUSDT",
        "side": "LONG",
        "entry_price": 50000.0,
        "amount": 0.01,
        "leverage": 2,
        "stop_loss": 45000.0,
        "take_profit": 52500.0,
        "status": "CLOSED",
        "order_source": "PROGRAMMATIC",
        "execution_mode": "AUTO",
    }


class TestGetOverallStatsEmptyDB:
    """Test get_overall_stats with empty database."""

    def test_empty_database_returns_zero_stats(self, test_db):
        """Test that empty database returns all zeros."""
        with session_scope() as session:
            stats = get_overall_stats(session)

            assert stats["total_trades"] == 0, f"Expected total_trades 0, got {stats['total_trades']}"
            assert stats["winning_trades"] == 0, f"Expected winning_trades 0, got {stats['winning_trades']}"
            assert stats["losing_trades"] == 0, f"Expected losing_trades 0, got {stats['losing_trades']}"
            assert stats["win_rate"] == pytest.approx(0.0, abs=0.01), f"Expected win_rate 0.0, got {stats['win_rate']}"
            assert stats["total_pnl"] == pytest.approx(
                0.0, abs=0.01
            ), f"Expected total_pnl 0.0, got {stats['total_pnl']}"
            assert stats["avg_pnl"] == pytest.approx(0.0, abs=0.01), f"Expected avg_pnl 0.0, got {stats['avg_pnl']}"
            assert stats["total_fees"] == pytest.approx(
                0.0, abs=0.01
            ), f"Expected total_fees 0.0, got {stats['total_fees']}"
            assert stats["best_trade"] == pytest.approx(
                0.0, abs=0.01
            ), f"Expected best_trade 0.0, got {stats['best_trade']}"
            assert stats["worst_trade"] == pytest.approx(
                0.0, abs=0.01
            ), f"Expected worst_trade 0.0, got {stats['worst_trade']}"


class TestGetOverallStatsSingleOrder:
    """Test get_overall_stats with single order scenarios."""

    def test_single_winning_order(self, test_db, sample_order_data):
        """Test stats with one winning order."""
        with session_scope() as session:
            data = sample_order_data.copy()
            data["pnl"] = 150.0
            data["commission"] = 2.5
            create_order(session, data)

        with session_scope() as session:
            stats = get_overall_stats(session)

            assert stats["total_trades"] == 1, f"Expected total_trades 1, got {stats['total_trades']}"
            assert stats["winning_trades"] == 1, f"Expected winning_trades 1, got {stats['winning_trades']}"
            assert stats["losing_trades"] == 0, f"Expected losing_trades 0, got {stats['losing_trades']}"
            assert stats["win_rate"] == pytest.approx(
                100.0, abs=0.01
            ), f"Expected win_rate 100.0, got {stats['win_rate']}"
            assert stats["total_pnl"] == pytest.approx(
                150.0, abs=0.01
            ), f"Expected total_pnl 150.0, got {stats['total_pnl']}"
            assert stats["avg_pnl"] == pytest.approx(150.0, abs=0.01), f"Expected avg_pnl 150.0, got {stats['avg_pnl']}"
            assert stats["total_fees"] == pytest.approx(
                2.5, abs=0.01
            ), f"Expected total_fees 2.5, got {stats['total_fees']}"
            assert stats["best_trade"] == pytest.approx(
                150.0, abs=0.01
            ), f"Expected best_trade 150.0, got {stats['best_trade']}"
            assert stats["worst_trade"] == pytest.approx(
                150.0, abs=0.01
            ), f"Expected worst_trade 150.0, got {stats['worst_trade']}"

    def test_single_losing_order(self, test_db, sample_order_data):
        """Test stats with one losing order."""
        with session_scope() as session:
            data = sample_order_data.copy()
            data["pnl"] = -75.0
            data["commission"] = 2.5
            create_order(session, data)

        with session_scope() as session:
            stats = get_overall_stats(session)

            assert stats["total_trades"] == 1, f"Expected total_trades 1, got {stats['total_trades']}"
            assert stats["winning_trades"] == 0, f"Expected winning_trades 0, got {stats['winning_trades']}"
            assert stats["losing_trades"] == 1, f"Expected losing_trades 1, got {stats['losing_trades']}"
            assert stats["win_rate"] == pytest.approx(0.0, abs=0.01), f"Expected win_rate 0.0, got {stats['win_rate']}"
            assert stats["total_pnl"] == pytest.approx(
                -75.0, abs=0.01
            ), f"Expected total_pnl -75.0, got {stats['total_pnl']}"
            assert stats["avg_pnl"] == pytest.approx(-75.0, abs=0.01), f"Expected avg_pnl -75.0, got {stats['avg_pnl']}"
            assert stats["best_trade"] == pytest.approx(
                -75.0, abs=0.01
            ), f"Expected best_trade -75.0, got {stats['best_trade']}"
            assert stats["worst_trade"] == pytest.approx(
                -75.0, abs=0.01
            ), f"Expected worst_trade -75.0, got {stats['worst_trade']}"

    def test_single_breakeven_order(self, test_db, sample_order_data):
        """Test stats with one breakeven order (pnl=0)."""
        with session_scope() as session:
            data = sample_order_data.copy()
            data["pnl"] = 0.0
            create_order(session, data)

        with session_scope() as session:
            stats = get_overall_stats(session)

            assert stats["total_trades"] == 1, f"Expected total_trades 1, got {stats['total_trades']}"
            # Note: pnl=0 is not > 0, so it's not counted as winning
            assert (
                stats["winning_trades"] == 0
            ), f"Expected winning_trades 0 (pnl=0 not > 0), got {stats['winning_trades']}"
            assert (
                stats["losing_trades"] == 0
            ), f"Expected losing_trades 0 (pnl=0 not < 0), got {stats['losing_trades']}"
            assert stats["win_rate"] == pytest.approx(0.0, abs=0.01), f"Expected win_rate 0.0, got {stats['win_rate']}"
            assert stats["total_pnl"] == pytest.approx(
                0.0, abs=0.01
            ), f"Expected total_pnl 0.0, got {stats['total_pnl']}"


class TestGetOverallStatsMultipleOrders:
    """Test get_overall_stats with multiple orders."""

    def test_multiple_win_loss_orders(self, test_db, sample_order_data):
        """Test stats with multiple winning and losing orders."""
        with session_scope() as session:
            # Create 3 winning orders
            for i in range(3):
                data = sample_order_data.copy()
                data["order_id"] = f"WIN_{i}"
                data["client_order_id"] = f"AT_WIN_{i}_BTCUSDT"
                data["pnl"] = 100.0 + (i * 50)  # 100, 150, 200
                data["commission"] = 2.0
                create_order(session, data)

            # Create 2 losing orders
            for i in range(2):
                data = sample_order_data.copy()
                data["order_id"] = f"LOSS_{i}"
                data["client_order_id"] = f"AT_LOSS_{i}_BTCUSDT"
                data["pnl"] = -50.0 - (i * 25)  # -50, -75
                data["commission"] = 2.0
                create_order(session, data)

        with session_scope() as session:
            stats = get_overall_stats(session)

            # Total: 5 trades (3 wins, 2 losses)
            assert stats["total_trades"] == 5, f"Expected total_trades 5, got {stats['total_trades']}"
            assert stats["winning_trades"] == 3, f"Expected winning_trades 3, got {stats['winning_trades']}"
            assert stats["losing_trades"] == 2, f"Expected losing_trades 2, got {stats['losing_trades']}"

            # Win rate: 3/5 = 60%
            assert stats["win_rate"] == pytest.approx(
                60.0, abs=0.01
            ), f"Expected win_rate 60.0, got {stats['win_rate']}"

            # Total PnL: 100+150+200 - 50-75 = 325
            assert stats["total_pnl"] == pytest.approx(
                325.0, abs=0.01
            ), f"Expected total_pnl 325.0, got {stats['total_pnl']}"

            # Avg PnL: 325/5 = 65
            assert stats["avg_pnl"] == pytest.approx(65.0, abs=0.01), f"Expected avg_pnl 65.0, got {stats['avg_pnl']}"

            # Total fees: 5 * 2 = 10
            assert stats["total_fees"] == pytest.approx(
                10.0, abs=0.01
            ), f"Expected total_fees 10.0, got {stats['total_fees']}"

            # Best trade: 200
            assert stats["best_trade"] == pytest.approx(
                200.0, abs=0.01
            ), f"Expected best_trade 200.0, got {stats['best_trade']}"

            # Worst trade: -75
            assert stats["worst_trade"] == pytest.approx(
                -75.0, abs=0.01
            ), f"Expected worst_trade -75.0, got {stats['worst_trade']}"

    def test_all_winning_orders(self, test_db, sample_order_data):
        """Test stats with all winning orders (100% win rate)."""
        with session_scope() as session:
            for i in range(5):
                data = sample_order_data.copy()
                data["order_id"] = f"WIN_{i}"
                data["client_order_id"] = f"AT_WIN_{i}_BTCUSDT"
                data["pnl"] = 50.0
                create_order(session, data)

        with session_scope() as session:
            stats = get_overall_stats(session)

            assert stats["total_trades"] == 5, f"Expected total_trades 5, got {stats['total_trades']}"
            assert stats["winning_trades"] == 5, f"Expected winning_trades 5, got {stats['winning_trades']}"
            assert stats["losing_trades"] == 0, f"Expected losing_trades 0, got {stats['losing_trades']}"
            assert stats["win_rate"] == pytest.approx(
                100.0, abs=0.01
            ), f"Expected win_rate 100.0, got {stats['win_rate']}"

    def test_all_losing_orders(self, test_db, sample_order_data):
        """Test stats with all losing orders (0% win rate)."""
        with session_scope() as session:
            for i in range(5):
                data = sample_order_data.copy()
                data["order_id"] = f"LOSS_{i}"
                data["client_order_id"] = f"AT_LOSS_{i}_BTCUSDT"
                data["pnl"] = -50.0
                create_order(session, data)

        with session_scope() as session:
            stats = get_overall_stats(session)

            assert stats["total_trades"] == 5, f"Expected total_trades 5, got {stats['total_trades']}"
            assert stats["winning_trades"] == 0, f"Expected winning_trades 0, got {stats['winning_trades']}"
            assert stats["losing_trades"] == 5, f"Expected losing_trades 5, got {stats['losing_trades']}"
            assert stats["win_rate"] == pytest.approx(0.0, abs=0.01), f"Expected win_rate 0.0, got {stats['win_rate']}"


class TestGetOverallStatsFilters:
    """Test that get_overall_stats properly filters orders."""

    def test_only_programmatic_orders_counted(self, test_db, sample_order_data):
        """Test that only PROGRAMMATIC orders are included in stats."""
        with session_scope() as session:
            # Create programmatic order
            data = sample_order_data.copy()
            data["pnl"] = 100.0
            create_order(session, data)

            # Create manual order
            manual_data = sample_order_data.copy()
            manual_data["order_id"] = "MANUAL_001"
            manual_data["client_order_id"] = "MANUAL_001_CLIENT"
            manual_data["order_source"] = "MANUAL"
            manual_data["pnl"] = 200.0
            create_order(session, manual_data)

        with session_scope() as session:
            stats = get_overall_stats(session)

            # Only programmatic order should be counted
            assert (
                stats["total_trades"] == 1
            ), f"Expected total_trades 1 (programmatic only), got {stats['total_trades']}"
            assert stats["total_pnl"] == pytest.approx(
                100.0, abs=0.01
            ), f"Expected total_pnl 100.0 (programmatic), got {stats['total_pnl']}"

    def test_only_closed_orders_counted(self, test_db, sample_order_data):
        """Test that only CLOSED orders are included in stats."""
        with session_scope() as session:
            # Create closed order
            data = sample_order_data.copy()
            data["order_id"] = "CLOSED_001"
            data["client_order_id"] = "AT_CLOSED_001"
            data["status"] = "CLOSED"
            data["pnl"] = 100.0
            create_order(session, data)

            # Create open order
            open_data = sample_order_data.copy()
            open_data["order_id"] = "OPEN_001"
            open_data["client_order_id"] = "AT_OPEN_001"
            open_data["status"] = "OPEN"
            create_order(session, open_data)

        with session_scope() as session:
            stats = get_overall_stats(session)

            # Only closed order should be counted
            assert stats["total_trades"] == 1, f"Expected total_trades 1 (closed only), got {stats['total_trades']}"
            assert stats["total_pnl"] == pytest.approx(
                100.0, abs=0.01
            ), f"Expected total_pnl 100.0 (closed), got {stats['total_pnl']}"


class TestGetOverallStatsEdgeCases:
    """Test edge cases for get_overall_stats."""

    def test_very_large_pnl_values(self, test_db, sample_order_data):
        """Test stats with very large PnL values."""
        with session_scope() as session:
            data = sample_order_data.copy()
            data["pnl"] = 999999.99
            data["commission"] = 999.99
            create_order(session, data)

        with session_scope() as session:
            stats = get_overall_stats(session)

            assert stats["total_pnl"] == pytest.approx(
                999999.99, abs=0.01
            ), f"Expected total_pnl 999999.99, got {stats['total_pnl']}"
            assert stats["best_trade"] == pytest.approx(
                999999.99, abs=0.01
            ), f"Expected best_trade 999999.99, got {stats['best_trade']}"

    def test_very_small_pnl_values(self, test_db, sample_order_data):
        """Test stats with very small PnL values."""
        with session_scope() as session:
            data = sample_order_data.copy()
            data["pnl"] = 0.01
            data["commission"] = 0.001
            create_order(session, data)

        with session_scope() as session:
            stats = get_overall_stats(session)

            assert stats["total_pnl"] == pytest.approx(
                0.01, abs=0.001
            ), f"Expected total_pnl 0.01, got {stats['total_pnl']}"

    def test_multiple_symbols(self, test_db, sample_order_data):
        """Test stats aggregated across multiple symbols."""
        with session_scope() as session:
            # Create orders for different symbols
            for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                data = sample_order_data.copy()
                data["order_id"] = f"{symbol}_001"
                data["client_order_id"] = f"AT_{symbol}_001"
                data["symbol"] = symbol
                data["pnl"] = 100.0
                create_order(session, data)

        with session_scope() as session:
            stats = get_overall_stats(session)

            # All 3 orders should be counted regardless of symbol
            assert stats["total_trades"] == 3, f"Expected total_trades 3 (all symbols), got {stats['total_trades']}"
            assert stats["total_pnl"] == pytest.approx(
                300.0, abs=0.01
            ), f"Expected total_pnl 300.0, got {stats['total_pnl']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
