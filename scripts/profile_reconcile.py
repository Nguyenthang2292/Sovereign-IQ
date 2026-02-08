"""
Profiling script for reconcile and stats queries.

Usage:
    python scripts/profile_reconcile.py

Environment:
    AUTO_TRADE_PROFILE_RECONCILE=true  # Enable profiling in reconcile
"""

import cProfile
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.auto_trade.database.config import DEFAULT_SCHEMA_PATH
from modules.auto_trade.database.migrations import MigrationManager
from modules.auto_trade.database.queries import get_orders_cursor, get_overall_stats
from modules.auto_trade.database.utils import DatabaseManager


def profile_stats_queries(db_path: str, num_iterations: int = 10):
    """Profile get_overall_stats with multiple iterations."""
    print("\n" + "=" * 60)
    print("PROFILING: get_overall_stats")
    print("=" * 60)

    db_manager = DatabaseManager(db_path, echo=False)

    times = []
    for i in range(num_iterations):
        with db_manager.session_scope() as session:
            t0 = time.perf_counter()
            stats = get_overall_stats(session)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"  Iterations: {num_iterations}")
    print(f"  Average: {avg_time:.4f}s")
    print(f"  Min: {min_time:.4f}s")
    print(f"  Max: {max_time:.4f}s")
    print(f"  Total: {sum(times):.4f}s")

    return stats


def profile_cursor_pagination(db_path: str, num_pages: int = 20, page_size: int = 50):
    """Profile cursor-based pagination."""
    print("\n" + "=" * 60)
    print("PROFILING: Cursor Pagination")
    print("=" * 60)

    db_manager = DatabaseManager(db_path, echo=False)

    times = []
    total_rows = 0
    last_id = None

    for i in range(num_pages):
        with db_manager.session_scope() as session:
            t0 = time.perf_counter()
            page = get_orders_cursor(session, last_id=last_id, limit=page_size)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            total_rows += len(page)
            if page:
                last_id = page[-1].id

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"  Pages fetched: {num_pages}")
    print(f"  Page size: {page_size}")
    print(f"  Total rows: {total_rows}")
    print(f"  Average per page: {avg_time:.4f}s")
    print(f"  Min: {min_time:.4f}s")
    print(f"  Max: {max_time:.4f}s")
    print(f"  Total time: {sum(times):.4f}s")


def profile_with_cprofile(db_path: str):
    """Run cProfile on stats queries."""
    print("\n" + "=" * 60)
    print("CPROFILE: Detailed function profiling")
    print("=" * 60)

    profiler = cProfile.Profile()
    profiler.enable()

    db_manager = DatabaseManager(db_path, echo=False)

    # Profile stats queries
    with db_manager.session_scope() as session:
        for _ in range(5):
            get_overall_stats(session)
            get_orders_cursor(session, last_id=None, limit=50)

    profiler.disable()

    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())


def main():
    """Main profiling function."""
    print("=" * 60)
    print("AUTO_TRADE PERFORMANCE PROFILING")
    print("=" * 60)

    # Use existing test database or create one
    db_path = "data/auto_trade.db"

    # Check if DB exists, if not use a test one
    if not Path(db_path).exists():
        print(f"\nDatabase not found at {db_path}, creating test database...")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "profile_test.db"
            manager = MigrationManager(str(db_path), DEFAULT_SCHEMA_PATH)
            manager.initialize_database()
            manager.auto_migrate()

            # Add test data
            db_manager = DatabaseManager(str(db_path), echo=False)
            from modules.auto_trade.database.models import Order

            test_orders = []
            for i in range(1000):
                test_orders.append(
                    {
                        "order_id": f"PROF_{i:06d}",
                        "symbol": "BTCUSDT",
                        "side": "LONG",
                        "entry_price": 50000.0,
                        "amount": 0.01,
                        "status": "CLOSED" if i % 2 == 0 else "OPEN",
                        "order_source": "PROGRAMMATIC",
                        "pnl": 1.0 if i % 2 == 0 else None,
                    }
                )

            with db_manager.session_scope() as session:
                session.bulk_insert_mappings(Order, test_orders)

            print(f"Created test database with {len(test_orders)} orders")

            # Run profiling
            profile_stats_queries(str(db_path))
            profile_cursor_pagination(str(db_path))
            profile_with_cprofile(str(db_path))
    else:
        print(f"\nUsing existing database: {db_path}")

        # Get basic stats first
        db_manager = DatabaseManager(db_path, echo=False)
        with db_manager.session_scope() as session:
            stats = get_overall_stats(session)
            print(f"\nDatabase contains:")
            print(f"  Total trades: {stats.get('total_trades', 0)}")
            print(f"  Open positions: {stats.get('open_positions', 0)}")
            print(f"  Closed positions: {stats.get('closed_positions', 0)}")

        # Run profiling
        profile_stats_queries(db_path)
        profile_cursor_pagination(db_path)
        profile_with_cprofile(db_path)

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
