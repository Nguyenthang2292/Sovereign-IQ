"""
Performance benchmarks for auto_trade module.

Benchmarks: get_overall_stats, get_orders_cursor, reconcile (mocked), backup.
Run: pytest tests/auto_trade/integration/test_performance_benchmarks.py -v -s
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.auto_trade.database.config import DEFAULT_SCHEMA_PATH
from modules.auto_trade.database.migrations import MigrationManager
from modules.auto_trade.database.models import Order
from modules.auto_trade.database.queries import get_orders_cursor, get_overall_stats
from modules.auto_trade.database.utils import DatabaseManager

# Benchmark thresholds (seconds)
STATS_10K_MAX_SEC = 5.0
CURSOR_FIRST_PAGE_MAX_SEC = 1.0
BACKUP_10K_MAX_SEC = 10.0
RECONCILE_MOCKED_MAX_SEC = 2.0

NUM_ORDERS_BENCHMARK = 10_001


def _make_order_rows(n: int):
    """Generate n order dicts for bulk insert."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "order_id": f"BENCH_ORDER_{i:08d}",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 50000.0,
                "amount": 0.01,
                "status": "CLOSED",
                "order_source": "PROGRAMMATIC",
                "pnl": 1.0 if i % 2 == 0 else -0.5,
            }
        )
    return rows


@pytest.fixture
def bench_db(tmp_path):
    """Create DB with 10,001 orders for benchmarks."""
    db_path = str(tmp_path / "bench_10k.db")
    manager = MigrationManager(db_path, DEFAULT_SCHEMA_PATH)
    manager.initialize_database()
    manager.auto_migrate()

    db_manager = DatabaseManager(db_path, echo=False)
    with db_manager.session_scope() as session:
        session.bulk_insert_mappings(Order, _make_order_rows(NUM_ORDERS_BENCHMARK))
    return db_path


class TestStatsPerformance:
    """Benchmark get_overall_stats."""

    def test_get_overall_stats_with_10k_orders(self, bench_db):
        """get_overall_stats with 10k+ orders completes within threshold."""
        db_manager = DatabaseManager(bench_db, echo=False)
        with db_manager.session_scope() as session:
            t0 = time.perf_counter()
            stats = get_overall_stats(session)
            elapsed = time.perf_counter() - t0

        assert stats["total_trades"] == NUM_ORDERS_BENCHMARK
        assert elapsed < STATS_10K_MAX_SEC, (
            f"get_overall_stats took {elapsed:.2f}s (max {STATS_10K_MAX_SEC}s)"
        )


class TestCursorPerformance:
    """Benchmark get_orders_cursor pagination."""

    def test_cursor_first_page_with_10k_orders(self, bench_db):
        """First page via get_orders_cursor completes within threshold."""
        db_manager = DatabaseManager(bench_db, echo=False)
        with db_manager.session_scope() as session:
            t0 = time.perf_counter()
            page = get_orders_cursor(session, last_id=None, limit=50)
            elapsed = time.perf_counter() - t0

        assert len(page) == 50
        assert elapsed < CURSOR_FIRST_PAGE_MAX_SEC, (
            f"get_orders_cursor first page took {elapsed:.2f}s (max {CURSOR_FIRST_PAGE_MAX_SEC}s)"
        )


class TestBackupPerformance:
    """Benchmark backup creation."""

    def test_backup_with_10k_orders(self, bench_db, tmp_path):
        """Backup of 10k-order DB completes within threshold."""
        from modules.auto_trade.database.backup import BackupManager

        backup_dir = str(tmp_path / "backups")
        manager = BackupManager(bench_db, backup_dir=backup_dir, max_backups=5, compress=True)

        t0 = time.perf_counter()
        backup_path = manager.create_backup()
        elapsed = time.perf_counter() - t0

        assert backup_path is not None
        assert elapsed < BACKUP_10K_MAX_SEC, (
            f"create_backup took {elapsed:.2f}s (max {BACKUP_10K_MAX_SEC}s)"
        )


class TestReconcilePerformance:
    """Benchmark reconcile with mocked exchange."""

    def test_reconcile_with_mocked_exchange_completes_quickly(self):
        """Reconcile with mocked empty response completes within threshold."""
        try:
            import ccxt
        except ImportError:
            pytest.skip("ccxt not installed")

        from modules.auto_trade.database import reconcile_orders_with_binance

        mock_exchange = MagicMock()
        mock_exchange.fetch_closed_orders.return_value = []
        mock_exchange.fetch_open_orders.return_value = []

        with patch("modules.auto_trade.database.reconcile.ccxt") as mock_ccxt:
            mock_ccxt.binance.return_value = mock_exchange

            t0 = time.perf_counter()
            result = reconcile_orders_with_binance(
                api_key="x",
                api_secret="y",
                symbols=["BTC/USDT"],
                since_hours=24,
                enable_profiling=True,
            )
            elapsed = time.perf_counter() - t0

        assert elapsed < RECONCILE_MOCKED_MAX_SEC, (
            f"reconcile took {elapsed:.2f}s (max {RECONCILE_MOCKED_MAX_SEC}s)"
        )
        assert "inserted" in result
        assert "timing" in result
