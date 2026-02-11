#!/usr/bin/env python3
"""
Test TP/SL sync directly with known order in database.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.gui.utils.credential_manager import CredentialManager
from modules.auto_trade.gui.utils.settings_manager import SettingsManager
from modules.auto_trade.gui.utils.tp_sl_sync import TPSLSyncService
from modules.auto_trade.database import get_db_manager, Order


def main():
    """Test TP/SL sync for known order."""

    print("=" * 80)
    print("🧪 Test TP/SL Sync - Direct Test")
    print("=" * 80)

    # Initialize
    settings = SettingsManager()
    settings.load()

    credential_manager = CredentialManager()
    api_config = credential_manager.load_credentials("binance")

    api_settings = settings.get("api", {})
    testnet = api_settings.get("mode", "").upper() == "TESTNET"

    db_manager = get_db_manager()

    client = BinanceClient(
        api_key=api_config.get("api_key") or "",
        api_secret=api_config.get("api_secret") or "",
        testnet=testnet,
        dry_run=False,
    )

    # Check database for open order
    print("\n📊 Step 1: Find OPEN order in database...")
    print("-" * 80)

    with db_manager.session_scope() as session:
        open_order = session.query(Order).filter(
            Order.status == "OPEN"
        ).first()

        if not open_order:
            print("❌ No OPEN orders in database!")
            print("   Run: python test_position_sync_manual.py")
            return

        print(f"\n✅ Found OPEN order:")
        print(f"   ID:           {open_order.id}")
        print(f"   Symbol:       {open_order.symbol}")
        print(f"   Side:         {open_order.side}")
        print(f"   Entry:        ${open_order.entry_price}")
        print(f"   TP (before):  {open_order.take_profit}")
        print(f"   SL (before):  {open_order.stop_loss}")

        # Test symbol formats
        print(f"\n🔍 Step 2: Test symbol normalization...")
        print("-" * 80)
        
        test_symbols = [
            open_order.symbol,  # As stored in DB
            "SKL/USDT",         # Standard format
            "SKLUSDT",          # No separators
        ]
        
        for test_sym in test_symbols:
            print(f"\n  Testing symbol: '{test_sym}'")
            
            # Try to find with this symbol
            found = session.query(Order).filter(
                Order.status == "OPEN",
                Order.symbol == test_sym
            ).first()
            
            if found:
                print(f"    ✅ Match! Found order ID {found.id}")
            else:
                print(f"    ❌ No match")

        # Fetch TP/SL from Binance
        print(f"\n🔍 Step 3: Fetch TP/SL from Binance...")
        print("-" * 80)

        # Try different symbol formats with Binance
        binance_symbol = "SKL/USDT"  # ccxt uses slash format
        
        print(f"   Fetching orders for: {binance_symbol}")
        
        tp, sl, be = TPSLSyncService.fetch_tp_sl_from_binance(client, binance_symbol)
        
        print(f"\n   Results:")
        print(f"   - Take Profit: ${tp}")
        print(f"   - Stop Loss:   ${sl}")
        print(f"   - Break Even:  ${be}")

        # Perform full sync
        print(f"\n🔄 Step 4: Perform full sync...")
        print("-" * 80)

        result = TPSLSyncService.sync_position_tp_sl(
            client=client,
            session=session,
            symbol=binance_symbol,  # Use slash format for Binance API
            side=open_order.side,
            entry_price=float(open_order.entry_price or 0)
        )

        print(f"\n   Sync result:")
        print(f"   - TP: ${result['take_profit']}")
        print(f"   - SL: ${result['stop_loss']}")
        print(f"   - BE: ${result['break_even']}")

        # Refresh and check database
        session.refresh(open_order)
        
        print(f"\n🗄️  Step 5: Check database after sync...")
        print("-" * 80)
        
        print(f"   TP (after):   {open_order.take_profit}")
        print(f"   SL (after):   {open_order.stop_loss}")
        print(f"   BE Moved:     {open_order.be_moved}")
        print(f"   Updated At:   {open_order.updated_at}")

        if result['take_profit'] or result['stop_loss']:
            print(f"\n   ✅ SUCCESS! TP/SL synced to database!")
        else:
            print(f"\n   ⚠️  No TP/SL found on Binance")

    print("\n" + "=" * 80)
    print("✅ Test complete!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
