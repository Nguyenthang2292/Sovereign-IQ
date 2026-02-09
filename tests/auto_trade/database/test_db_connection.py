"""
Database connection test script
"""

from modules.auto_trade.database import get_db_manager


def test_db_connection():
    """Test database connection."""
    db_manager = get_db_manager()

    if db_manager.check_connection():
        print("✅ Database hoạt động bình thường")

        stats = db_manager.get_database_stats()
        print(f"Tổng số orders: {stats['total_orders']}")
        print(f"Orders đang mở: {stats['open_orders']}")
    else:
        print("❌ Không thể kết nối database")


if __name__ == "__main__":
    test_db_connection()
