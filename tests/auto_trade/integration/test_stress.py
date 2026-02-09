"""
Stress tests for auto_trade module.

Tests behavior under high load: many orders, concurrent reads, concurrent reconcile.
Run: pytest tests/auto_trade/integration/test_stress.py -v
"""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.auto_trade.database.config import DEFAULT_SCHEMA_PATH
from modules.auto_trade.database.migrations import MigrationManager  # type: ignore[attr-defined]
from modules.auto_trade.database.models import Order
from modules.auto_trade.database.queries import get_orders_cursor, get_overall_stats
from modules.auto_trade.database.utils import DatabaseManager

STRESS_ORDERS = 5_000  # Increase to 50_000 for full stress runs
NUM_CONCURRENT_READERS = 5
NUM_CONCURRENT_RECONCILE = 3


def _make_order_rows(n: int):
    """Generate n order dicts for bulk insert."""
    return [
        {
            "order_id": f"STRESS_{i:08d}",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_price": 50000.0,
            "amount": 0.01,
            "status": "CLOSED",
            "order_source": "PROGRAMMATIC",
            "pnl": 1.0 if i % 2 == 0 else -0.5,
        }
        for i in range(n)
    ]


@pytest.fixture
def stress_db(tmp_path):
    """Create DB with many orders for stress testing."""
    db_path = str(tmp_path / "stress.db")
    manager = MigrationManager(db_path, DEFAULT_SCHEMA_PATH)  # type: ignore[attr-defined]
    manager.initialize_database()
    manager.auto_migrate()

    db_manager = DatabaseManager(db_path, echo=False)
    with db_manager.session_scope() as session:
        session.bulk_insert_mappings(Order, _make_order_rows(STRESS_ORDERS))  # type: ignore[arg-type]
    return db_path


class TestHighVolumeOrders:
    """Stress: Large dataset operations."""

    def test_stats_with_high_volume_orders_completes(self, stress_db):
        """get_overall_stats with high volume orders completes (no hard timeout)."""
        db_manager = DatabaseManager(stress_db, echo=False)
        with db_manager.session_scope() as session:
            stats = get_overall_stats(session)

        assert stats["total_trades"] == STRESS_ORDERS
        assert "win_rate" in stats

    def test_cursor_pagination_with_high_volume_orders(self, stress_db):
        """Cursor pagination through high volume orders returns correct pages."""
        db_manager = DatabaseManager(stress_db, echo=False)
        seen_order_ids = set()
        last_id = None
        page_count = 0

        with db_manager.session_scope() as session:
            while True:
                page = get_orders_cursor(session, last_id=last_id, limit=100)
                if not page:
                    break
                page_count += 1
                for order in page:
                    oid = getattr(order, "order_id", None) or getattr(order, "id", None)
                    assert oid not in seen_order_ids, "Duplicate order in pagination"
                    seen_order_ids.add(oid)
                raw_id = getattr(page[-1], "id", None)
                last_id = int(raw_id) if raw_id is not None else None
                if len(page) < 100:
                    break

        assert page_count >= STRESS_ORDERS // 100
        assert len(seen_order_ids) == STRESS_ORDERS


class TestConcurrentReads:
    """Stress: Concurrent read operations."""

    def test_concurrent_stats_reads(self, stress_db):
        """Multiple threads reading stats concurrently do not corrupt DB."""
        errors = []
        results = []

        def reader():
            try:
                db_manager = DatabaseManager(stress_db, echo=False)
                with db_manager.session_scope() as session:
                    stats = get_overall_stats(session)
                    results.append(stats["total_trades"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(NUM_CONCURRENT_READERS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent reads failed: {errors}"
        assert all(r == STRESS_ORDERS for r in results)


class TestReconcileStress:
    """Stress: Reconcile under concurrent load."""

    def test_concurrent_reconcile_calls_serialize_safely(self):
        """Multiple reconcile calls serialize via lock; all complete without error."""
        try:
            import ccxt
        except ImportError:
            pytest.skip("ccxt not installed")

        from modules.auto_trade.database import reconcile_orders_with_binance

        mock_exchange = MagicMock()
        mock_exchange.fetch_closed_orders.return_value = []
        mock_exchange.fetch_open_orders.return_value = []

        results = []
        errors = []

        def run_reconcile():
            try:
                with patch("modules.auto_trade.database.reconcile.ccxt") as mock_ccxt:
                    mock_ccxt.binance.return_value = mock_exchange
                    r = reconcile_orders_with_binance(
                        api_key="x",
                        api_secret="y",
                        symbols=["BTC/USDT"],
                        since_hours=1,
                    )
                    results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_reconcile) for _ in range(NUM_CONCURRENT_RECONCILE)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent reconcile failed: {errors}"
        assert len(results) == NUM_CONCURRENT_RECONCILE
