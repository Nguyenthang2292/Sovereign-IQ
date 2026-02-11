#!/usr/bin/env python3
"""
Fetch conditional orders using Binance API directly.

Binance has separate endpoints for conditional orders (TP/SL).
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
    """Fetch conditional orders."""

    print("=" * 80)
    print("🔍 Fetch Conditional Orders (TP/SL)")
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

    symbol = "SKLUSDT"  # Binance API format (no slash)
    
    print(f"\n📊 Fetching conditional orders for {symbol}...")
    print("-" * 80)

    # Try different methods to fetch conditional orders
    
    print("\n🔹 Method 1: Using ccxt's fetch_open_orders()")
    try:
        orders = client.exchange.fetch_open_orders(f"SKL/USDT")
        print(f"✅ Found {len(orders)} orders")
        for order in orders:
            print(f"   - Type: {order.get('type')}, Side: {order.get('side')}, Price: {order.get('price')}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n🔹 Method 2: Using fapiPrivateGetOpenOrders (all orders)")
    try:
        result = client.exchange.fapiPrivateGetOpenOrders({'symbol': symbol})
        print(f"✅ Found {len(result)} orders:")
        for order in result:
            print(f"   - Type: {order.get('type')}, Side: {order.get('side')}, Price: {order.get('price')}, StopPrice: {order.get('stopPrice')}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n🔹 Method 3: Using fapiPrivateGetAllOrders (includes conditional)")
    try:
        # Get all orders including historical
        result = client.exchange.fapiPrivateGetAllOrders({'symbol': symbol, 'limit': 10})
        print(f"✅ Found {len(result)} recent orders:")
        for order in result:
            status = order.get('status')
            order_type = order.get('type')
            side = order.get('side')
            price = order.get('price')
            stop_price = order.get('stopPrice')
            
            print(f"\n   Order ID: {order.get('orderId')}")
            print(f"   - Status:     {status}")
            print(f"   - Type:       {order_type}")
            print(f"   - Side:       {side}")
            print(f"   - Price:      {price}")
            print(f"   - StopPrice:  {stop_price}")
            
            # Highlight active TP/SL orders
            if status == 'NEW' and ('TAKE_PROFIT' in order_type or 'STOP' in order_type):
                print(f"   ⭐ ACTIVE TP/SL ORDER!")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n🔹 Method 4: Check account trades for TP/SL info")
    try:
        # Sometimes TP/SL info is in account's conditional orders
        if hasattr(client.exchange, 'fapiPrivateGetPositionSideDual'):
            mode = client.exchange.fapiPrivateGetPositionSideDual()
            print(f"✅ Position mode: {mode}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 80)
    print("💡 Analysis")
    print("=" * 80)
    print("""
If Method 3 shows orders with:
- Status: NEW
- Type: TAKE_PROFIT_MARKET or STOP_MARKET
→ These are your active TP/SL orders!

The issue is that ccxt's fetch_open_orders() might not include 
conditional orders by default.

Solution: Use fapiPrivateGetAllOrders with status=NEW filter.
""")
    print()


if __name__ == "__main__":
    main()
