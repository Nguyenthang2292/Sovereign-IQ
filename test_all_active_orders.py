#!/usr/bin/env python3
"""
Fetch ALL active orders across all symbols.
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
    """Fetch all active orders."""

    print("=" * 80)
    print("🔍 Fetch ALL Active Orders (All Symbols)")
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

    print("\n📊 Fetching ALL open orders (all symbols)...")
    print("-" * 80)

    try:
        # Suppress warning
        client.exchange.options['warnOnFetchOpenOrdersWithoutSymbol'] = False
        
        # Method 1: ccxt fetch_open_orders (all symbols)
        print("\n🔹 Method 1: ccxt fetch_open_orders()")
        try:
            orders = client.exchange.fetch_open_orders()
            print(f"✅ Found {len(orders)} open orders")
            
            for order in orders:
                symbol = order.get('symbol')
                order_type = order.get('type')
                side = order.get('side')
                price = order.get('price')
                stop_price = order.get('stopPrice')
                status = order.get('status')
                
                print(f"\n   Symbol: {symbol}")
                print(f"   - Type:       {order_type}")
                print(f"   - Side:       {side}")
                print(f"   - Price:      {price}")
                print(f"   - StopPrice:  {stop_price}")
                print(f"   - Status:     {status}")
                
                if 'TAKE_PROFIT' in order_type or 'STOP' in order_type:
                    print(f"   ⭐ TP/SL ORDER!")
                    
        except Exception as e:
            print(f"❌ Error: {e}")

        # Method 2: Direct API call
        print("\n🔹 Method 2: fapiPrivateGetOpenOrders() - Direct API")
        try:
            result = client.exchange.fapiPrivateGetOpenOrders()
            print(f"✅ Found {len(result)} open orders:")
            
            for order in result:
                symbol = order.get('symbol')
                order_type = order.get('type')
                side = order.get('side')
                price = order.get('price')
                stop_price = order.get('stopPrice')
                status = order.get('status')
                
                print(f"\n   Symbol: {symbol}")
                print(f"   - Type:       {order_type}")
                print(f"   - Side:       {side}")
                print(f"   - Price:      {price}")
                print(f"   - StopPrice:  {stop_price}")
                print(f"   - Status:     {status}")
                
                if 'TAKE_PROFIT' in order_type or 'STOP' in order_type:
                    print(f"   ⭐ TP/SL ORDER!")
                    
        except Exception as e:
            print(f"❌ Error: {e}")

        # Method 3: Get ALL recent orders and filter NEW status
        print("\n🔹 Method 3: fapiPrivateGetAllOrders() - Filter NEW status")
        try:
            # Fetch recent orders for SKLUSDT
            result = client.exchange.fapiPrivateGetAllOrders({
                'symbol': 'SKLUSDT',
                'limit': 50  # Last 50 orders
            })
            
            # Filter only NEW (active) orders
            active_orders = [o for o in result if o.get('status') == 'NEW']
            
            print(f"✅ Found {len(active_orders)} NEW (active) orders out of {len(result)} total:")
            
            for order in active_orders:
                symbol = order.get('symbol')
                order_type = order.get('type')
                side = order.get('side')
                price = order.get('price')
                stop_price = order.get('stopPrice')
                status = order.get('status')
                order_id = order.get('orderId')
                
                print(f"\n   Order ID: {order_id}")
                print(f"   Symbol: {symbol}")
                print(f"   - Type:       {order_type}")
                print(f"   - Side:       {side}")
                print(f"   - Price:      {price}")
                print(f"   - StopPrice:  {stop_price}")
                print(f"   - Status:     {status}")
                
                if 'TAKE_PROFIT' in order_type or 'STOP' in order_type:
                    print(f"   ⭐⭐⭐ ACTIVE TP/SL ORDER!")
                    
        except Exception as e:
            print(f"❌ Error: {e}")

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("💡 Conclusion")
    print("=" * 80)
    print("""
If NO active TP/SL orders are found:
→ The orders you see on Binance UI might be:
  1. Already filled/cancelled
  2. From a different account/subaccount
  3. UI cache not refreshed
  
Please double-check on Binance:
- Go to Futures → Orders → Open Orders
- Make sure you're on the right account
- Try refreshing the page
""")
    print()


if __name__ == "__main__":
    main()
