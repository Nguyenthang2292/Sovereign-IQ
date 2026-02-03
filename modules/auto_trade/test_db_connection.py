"""
Database connection test script
"""

from modules.auto_trade.database import get_db_manager

# Lấy database manager
db_manager = get_db_manager()

# Kiểm tra kết nối
if db_manager.check_connection():
    print("✅ Database hoạt động bình thường")

    # Xem thống kê
    stats = db_manager.get_database_stats()
    print(f"Tổng số orders: {stats['total_orders']}")
    print(f"Orders đang mở: {stats['open_orders']}")
else:
    print("❌ Không thể kết nối database")
