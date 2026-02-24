"""Backup database before making changes."""

import shutil
from datetime import datetime
from pathlib import Path


def backup_database():
    """Create a backup of the database."""
    print("\n" + "="*80)
    print("DATABASE BACKUP")
    print("="*80 + "\n")

    # Find database file
    db_path = Path("data/crypto_trading.db")

    if not db_path.exists():
        print(f"❌ Database not found at: {db_path}")
        print("   Looking in alternative locations...")

        # Try alternative paths
        alternatives = [
            Path("crypto_trading.db"),
            Path("../data/crypto_trading.db"),
            Path("modules/auto_trade/data/crypto_trading.db"),
        ]

        for alt_path in alternatives:
            if alt_path.exists():
                db_path = alt_path
                print(f"✅ Found database at: {db_path}")
                break
        else:
            print("❌ Could not find database file!")
            return None

    # Create backup directory
    backup_dir = Path("data/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"crypto_trading_backup_{timestamp}.db"
    backup_path = backup_dir / backup_filename

    try:
        # Copy database file
        print(f"📂 Source: {db_path}")
        print(f"📂 Backup: {backup_path}")
        print(f"💾 Size: {db_path.stat().st_size / 1024:.2f} KB")
        print("\nCreating backup...")

        shutil.copy2(db_path, backup_path)

        print("✅ Backup created successfully!")
        print(f"\n💡 Backup location: {backup_path.absolute()}")
        print(f"   Backup size: {backup_path.stat().st_size / 1024:.2f} KB")

        # List recent backups
        print("\n📋 Recent backups:")
        backups = sorted(backup_dir.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
        for i, backup in enumerate(backups, 1):
            mtime = datetime.fromtimestamp(backup.stat().st_mtime)
            size_kb = backup.stat().st_size / 1024
            print(f"   {i}. {backup.name} ({size_kb:.2f} KB) - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

        return backup_path

    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return None
    finally:
        print("\n" + "="*80 + "\n")


def restore_database(backup_path: str):
    """Restore database from backup."""
    print("\n" + "="*80)
    print("DATABASE RESTORE")
    print("="*80 + "\n")

    backup_file = Path(backup_path)
    if not backup_file.exists():
        print(f"❌ Backup file not found: {backup_path}")
        return False

    db_path = Path("data/crypto_trading.db")
    if not db_path.exists():
        db_path = Path("crypto_trading.db")

    print("⚠️  This will restore database from backup!")
    print(f"   Backup: {backup_file}")
    print(f"   Target: {db_path}")
    print("\n   Press Ctrl+C within 3 seconds to cancel...")

    import time
    time.sleep(3)

    try:
        shutil.copy2(backup_file, db_path)
        print("\n✅ Database restored successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Restore failed: {e}")
        return False
    finally:
        print("="*80 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        if len(sys.argv) < 3:
            print("Usage: python backup_database.py restore <backup_file>")
        else:
            restore_database(sys.argv[2])
    else:
        backup_database()
