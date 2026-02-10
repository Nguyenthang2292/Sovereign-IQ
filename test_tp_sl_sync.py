#!/usr/bin/env python3
"""
Test TP/SL Bidirectional Sync.

Verifies:
- Fetch from Binance Open Orders API
- Update database Order record
- Break Even auto-detection
- Change tracking and logging
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.auto_trade.database import DatabaseManager, Order
from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.gui.utils.credential_manager import CredentialManager
from modules.auto_trade.gui.utils.settings_manager import SettingsManager
from modules.auto_trade.gui.utils.tp_sl_sync import TPSLSyncService


def main():
    """Test TP/SL sync for all open positions."""

    print("=" * 80)
    print("🧪 Testing TP/SL Bidirectional Sync")
    print("=" * 80)

    # Initialize components
    settings = SettingsManager()
    settings.load()

    credential_manager = CredentialManager()
    api_config = credential_manager.load_credentials("binance")

    # Get testnet setting from settings
    api_settings = settings.get("api", {})
    testnet = api_settings.get("mode", "").upper() == "TESTNET"

    db_manager = DatabaseManager(str(project_root / "crypto_trading.db"))

    client = BinanceClient(
        api_key=api_config["api_key"] or "",
        api_secret=api_config["api_secret"] or "",
        testnet=testnet,
        dry_run=False,
    )

    # Get all open positions from DB
    with db_manager.session_scope() as session:
        open_orders = session.query(Order).filter(
            Order.status == "OPEN"
        ).all()

        if not open_orders:
            print("\n⚠️  No open positions found in database.")
            print("   Place some orders first with AutoTrade.\n")
            return

        print(f"\n📊 Found {len(open_orders)} open position(s):\n")

        for order in open_orders:
            print("-" * 80)
            print(f"🔹 Symbol: {order.symbol}")
            print(f"   Side: {order.side}")
            print(f"   Entry: ${order.entry_price}")
            print("\n   Before Sync:")
            print(f"   - TP (DB): {order.take_profit}")
            print(f"   - SL (DB): {order.stop_loss}")
            print(f"   - BE Moved: {order.be_moved}")

            # Perform sync
            try:
                # Get values from order (type: ignore for SQLAlchemy ORM false positives)
                symbol: str = str(order.symbol)  # type: ignore
                side: str = str(order.side)  # type: ignore
                entry_price: float = float(order.entry_price or 0.0)  # type: ignore

                result = TPSLSyncService.sync_position_tp_sl(
                    client=client,
                    session=session,
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price
                )

                # Refresh order from DB to see changes
                session.refresh(order)

                print("\n   After Sync:")
                print(f"   - TP (Binance→DB): ${result['take_profit']}")
                print(f"   - SL (Binance→DB): ${result['stop_loss']}")
                print(f"   - BE (Detected): ${result['break_even']}")
                print("\n   DB State:")
                print(f"   - TP: {order.take_profit}")
                print(f"   - SL: {order.stop_loss}")
                print(f"   - BE Moved: {order.be_moved}")
                print(f"   - Updated At: {order.updated_at}")

                if result['take_profit'] or result['stop_loss']:
                    print("\n   ✅ Sync successful!")
                else:
                    print("\n   ⚠️  No TP/SL orders found on Binance")

            except Exception as e:
                print(f"\n   ❌ Sync failed: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80)
    print("\nNext: Check your database to verify changes:")
    print("  sqlite3 crypto_trading.db")
    print("  SELECT symbol, take_profit, stop_loss, be_moved FROM orders WHERE status='OPEN';")
    print()


if __name__ == "__main__":
    main()
