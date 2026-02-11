#!/usr/bin/env python3
"""
Test Manual Position Sync

Quick test for the position sync service without GUI.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.auto_trade.database import get_db_manager
from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.gui.utils.credential_manager import CredentialManager
from modules.auto_trade.gui.utils.position_sync_service import PositionSyncService
from modules.auto_trade.gui.utils.settings_manager import SettingsManager


def main():
    """Test position sync service."""

    print("=" * 80)
    print("🧪 Testing Manual Position Sync Service")
    print("=" * 80)

    # Load settings
    settings = SettingsManager()
    settings.load()

    # Get credentials
    credential_manager = CredentialManager()
    api_config = credential_manager.load_credentials("binance")

    # Get testnet setting
    api_settings = settings.get("api", {})
    testnet = api_settings.get("mode", "").upper() == "TESTNET"

    print(f"\n🔑 Using API mode: {'TESTNET' if testnet else 'PRODUCTION'}")

    # Create Binance client
    client = BinanceClient(
        api_key=api_config.get("api_key") or "",
        api_secret=api_config.get("api_secret") or "",
        testnet=testnet,
        dry_run=False,
    )

    # Get database manager
    db_manager = get_db_manager()

    # Perform sync
    print("\n🔄 Starting position sync...\n")
    stats = PositionSyncService.sync_all_positions(client, db_manager)

    # Print results
    print("\n" + "=" * 80)
    print("📊 Sync Results")
    print("=" * 80)
    print(f"  Found on Binance: {stats['fetched']}")
    print(f"  ✅ Newly synced:  {stats['synced']}")
    print(f"  📁 Already in DB: {stats['existing']}")
    print(f"  ❌ Failed:        {stats['failed']}")
    print("=" * 80)

    if stats['synced'] > 0:
        print(f"\n✅ Successfully synced {stats['synced']} new position(s) to database!")
    elif stats['fetched'] == 0:
        print("\n⚠️  No open positions found on Binance.")
    elif stats['existing'] > 0:
        print(f"\n✅ All {stats['existing']} position(s) already synced in database.")
    else:
        print("\n❌ Sync completed with issues. Check logs above.")

    print("\nYou can now run the GUI to see the synced positions:")
    print("  python run_auto_trade_gui.py")
    print()


if __name__ == "__main__":
    main()
