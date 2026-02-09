"""
Performance tests with 10,000+ orders.

Verifies get_overall_stats and cursor pagination complete in reasonable time
with large datasets. Run: pytest tests/auto_trade/test_performance_10k_orders.py -v
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.auto_trade.database.config import DEFAULT_SCHEMA_PATH
from modules.auto_trade.database.migrations import MigrationManager  # type: ignore[attr-defined]
from modules.auto_trade.database.models import Order
from modules.auto_trade.database.queries import get_orders_cursor, get_overall_stats
from modules.auto_trade.database.utils import DatabaseManager

NUM_ORDERS = 10_001
STATS_MAX_SECONDS = 5.0
CURSOR_FIRST_PAGE_MAX_SECONDS = 1.0


def _make_order_rows(n: int):
    """Generate n order dicts for bulk insert (PROGRAMMATIC, CLOSED, with pnl)."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "order_id": f"PERF_ORDER_{i:08d}",
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
def perf_db(tmp_path):
    """Create DB with schema + migrations and 10,001 PROGRAMMATIC CLOSED orders."""
    db_path = str(tmp_path / "perf_10k.db")
    manager = MigrationManager(db_path, DEFAULT_SCHEMA_PATH)  # type: ignore[attr-defined]
    manager.initialize_database()
    manager.auto_migrate()

    db_manager = DatabaseManager(db_path, echo=False)
    with db_manager.session_scope() as session:
        session.bulk_insert_mappings(Order, _make_order_rows(NUM_ORDERS))  # type: ignore[arg-type]
    return db_path


def test_performance_get_overall_stats_with_10k_orders(perf_db):
    """get_overall_stats with 10,000+ orders completes within STATS_MAX_SECONDS."""
    db_manager = DatabaseManager(perf_db, echo=False)
    with db_manager.session_scope() as session:
        t0 = time.perf_counter()
        stats = get_overall_stats(session)
        elapsed = time.perf_counter() - t0
    assert stats["total_trades"] == NUM_ORDERS
    assert elapsed < STATS_MAX_SECONDS, (
        f"get_overall_stats took {elapsed:.2f}s (max {STATS_MAX_SECONDS}s)"
    )


def test_performance_cursor_first_page_with_10k_orders(perf_db):
    """First page via get_orders_cursor with 10,000+ orders completes within CURSOR_FIRST_PAGE_MAX_SECONDS."""
    db_manager = DatabaseManager(perf_db, echo=False)
    with db_manager.session_scope() as session:
        t0 = time.perf_counter()
        page = get_orders_cursor(session, last_id=None, limit=50)
        elapsed = time.perf_counter() - t0
    assert len(page) == 50
    assert elapsed < CURSOR_FIRST_PAGE_MAX_SECONDS, (
        f"get_orders_cursor first page took {elapsed:.2f}s (max {CURSOR_FIRST_PAGE_MAX_SECONDS}s)"
    )
