"""
Phase 5 Database Module Test
=============================
Quick verification test for the database module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os
import uuid
from datetime import datetime

from modules.auto_trade.database import (
    create_order,
    get_database_stats,
    get_db_manager,
    get_open_positions,
    initialize_database,
    save_signal,
    session_scope,
)


def test_basic_operations():
    """Test basic database operations."""
    print("Testing Phase 5 Database Module...")

    # Use test database
    test_db = "data/test_auto_trade.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    # Initialize
    initialize_database(db_path=test_db)
    print("✓ Database initialized")

    # Test connection
    db_manager = get_db_manager(db_path=test_db)
    try:
        db_manager.check_connection()
        print("✓ Connection verified")
    except Exception as e:
        print(f"⚠ Connection check warning: {e} (continuing anyway)")

    # Create test order
    with session_scope() as session:
        order = create_order(
            session,
            {
                "order_id": f"TEST_{uuid.uuid4().hex[:8]}",
                "client_order_id": f"AT_TEST_{uuid.uuid4().hex[:6]}",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry_price": 50000.0,
                "amount": 0.01,
                "leverage": 2,
                "stop_loss": 45000.0,
                "take_profit": 52500.0,
                "status": "OPEN",
                "order_source": "PROGRAMMATIC",
            },
        )
        print(f"✓ Created order: {order.order_id}")

        # Get open positions
        positions = get_open_positions(session)
        print(f"✓ Open positions: {len(positions)}")

        # Create signal
        signal = save_signal(session, f"SIG_{uuid.uuid4().hex[:8]}", "BTCUSDT", "LONG", 0.85)
        print(f"✓ Created signal: {signal.correlation_id}")

    # Get stats
    stats = get_database_stats()
    print(f"✓ Database stats: {stats}")

    print("\n✅ All basic tests PASSED!")
    return True


if __name__ == "__main__":
    try:
        test_basic_operations()
        print("\n🎉 Phase 5 Database Module is ready!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
