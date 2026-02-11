#!/usr/bin/env python3
"""
Check all open orders and positions to understand the relationship.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.gui.utils.credential_manager import CredentialManager
from modules.auto_trade.gui.utils.settings_manager import SettingsManager


def main():
    """Check all orders and positions."""

    print("=" * 80)
    print("🔍 Check All Orders and Positions")
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

    print("\n" + "=" * 80)
    print("📊 STEP 1: Fetch All Positions")
    print("=" * 80)

    try:
        positions = client.exchange.fetch_positions()
        open_positions = [p for p in positions if float(p.get("contracts", 0)) != 0]
        
        print(f"\n✅ Found {len(open_positions)} open position(s)\n")
        
        for pos in open_positions:
            symbol = pos.get("symbol")
            side = pos.get("side")
            contracts = pos.get("contracts")
            entry_price = pos.get("entryPrice")
            
            print(f"📍 Position: {symbol}")
            print(f"   Side:         {side}")
            print(f"   Contracts:    {contracts}")
            print(f"   Entry Price:  {entry_price}")
            print()

    except Exception as e:
        print(f"❌ Error fetching positions: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("📋 STEP 2: Fetch ALL Open Orders (all symbols)")
    print("=" * 80)

    try:
        # Fetch ALL open orders (no symbol filter)
        all_orders = client.exchange.fetch_open_orders()
        
        print(f"\n✅ Found {len(all_orders)} open order(s) across all symbols\n")
        
        if not all_orders:
            print("⚠️  No open orders found.")
        else:
            for order in all_orders:
                symbol = order.get("symbol")
                order_type = order.get("type")
                side = order.get("side")
                price = order.get("price")
                stop_price = order.get("stopPrice")
                status = order.get("status")
                
                print(f"📝 Order: {symbol}")
                print(f"   Type:         {order_type}")
                print(f"   Side:         {side}")
                print(f"   Price:        {price}")
                print(f"   Stop Price:   {stop_price}")
                print(f"   Status:       {status}")
                
                # Check if it's a TP/SL order
                if "TAKE_PROFIT" in order_type.upper():
                    print(f"   ✅ This is a TAKE PROFIT order")
                elif "STOP" in order_type.upper():
                    print(f"   ✅ This is a STOP LOSS order")
                print()

    except Exception as e:
        print(f"❌ Error fetching all orders: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("🔍 STEP 3: Check conditional orders via raw API")
    print("=" * 80)
    
    try:
        # Try to fetch conditional orders using raw API
        # This is specific to Binance Futures
        print("\nTrying to fetch conditional orders via fapiPrivateGetOpenOrders...")
        
        # Use ccxt's private method to call Binance Futures API directly
        if hasattr(client.exchange, 'fapiPrivateGetOpenOrders'):
            result = client.exchange.fapiPrivateGetOpenOrders()
            print(f"✅ Raw API response: {result}")
        else:
            print("⚠️  Method not available")
            
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 80)
    print("💡 Analysis")
    print("=" * 80)
    print("""
If you see:
- Open position(s) but NO open orders → TP/SL might be set directly in position
- Orders exist but not detected → Need to check order type format

Possible solutions:
1. TP/SL might be embedded in position object (not separate orders)
2. Need to check position's 'info' field for stopPrice/takeProfitPrice
3. Binance may have changed API structure
""")

    print()


if __name__ == "__main__":
    main()
