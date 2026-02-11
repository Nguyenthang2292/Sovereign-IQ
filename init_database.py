#!/usr/bin/env python3
"""
Initialize Database Script

Creates all required tables for the Auto Trading System.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.auto_trade.database import DatabaseManager


def main():
    """Initialize database with all tables."""
    
    print("=" * 80)
    print("📦 Initializing Database")
    print("=" * 80)
    
    db_path = project_root / "crypto_trading.db"
    
    print(f"\n🎯 Database path: {db_path}")
    
    # Create database manager
    db_manager = DatabaseManager(str(db_path))
    
    # Create all tables
    print("\n🔨 Creating tables...")
    db_manager.create_all_tables()
    
    print("\n✅ Database initialized successfully!")
    print(f"📊 Location: {db_path}")
    print("\n" + "=" * 80)
    print("\nYou can now run:")
    print("  python test_tp_sl_sync.py")
    print("  python run_auto_trade_gui.py")
    print()


if __name__ == "__main__":
    main()
