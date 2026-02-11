#!/usr/bin/env python3
"""
Debug: Fetch and inspect Binance orders structure.

This script fetches open orders for SKL/USDT and prints their full structure
to help debug why TP/SL detection is not working.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.gui.utils.credential_manager import CredentialManager
from modules.auto_trade.gui.utils.settings_manager import SettingsManager


def main():
    """Debug fetch orders structure."""

    print("=" * 80)
    print("🔍 Debug: Fetch Orders Structure")
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

    # Symbol to check
    symbol = "SKL/USDT"
    
    print(f"\n📊 Fetching open orders for {symbol}...")
    print("-" * 80)

    try:
        # Fetch open orders
        orders = client.exchange.fetch_open_orders(symbol)
        
        print(f"\n✅ Found {len(orders)} open order(s)\n")
        
        if not orders:
            print("⚠️  No open orders found for this symbol.")
            return

        # Print detailed structure for each order
        for i, order in enumerate(orders, 1):
            print(f"\n{'=' * 80}")
            print(f"ORDER #{i}")
            print(f"{'=' * 80}")
            
            # Key fields to check
            print(f"\n📌 Key Fields:")
            print(f"  - ID:          {order.get('id')}")
            print(f"  - Type:        {order.get('type')}")
            print(f"  - Side:        {order.get('side')}")
            print(f"  - Price:       {order.get('price')}")
            print(f"  - StopPrice:   {order.get('stopPrice')}")
            print(f"  - Amount:      {order.get('amount')}")
            print(f"  - Status:      {order.get('status')}")
            
            # Check info field (raw from Binance)
            if 'info' in order:
                info = order['info']
                print(f"\n📋 Info Field (Raw Binance):")
                print(f"  - type:        {info.get('type')}")
                print(f"  - side:        {info.get('side')}")
                print(f"  - price:       {info.get('price')}")
                print(f"  - stopPrice:   {info.get('stopPrice')}")
                print(f"  - workingType: {info.get('workingType')}")
            
            # Full structure (JSON)
            print(f"\n🔬 Full Structure (JSON):")
            print(json.dumps(order, indent=2, default=str))
            print()

        # Test detection logic
        print(f"\n{'=' * 80}")
        print("🧪 Testing Detection Logic")
        print(f"{'=' * 80}\n")
        
        for i, order in enumerate(orders, 1):
            order_type = order.get("type", "").upper()
            stop_price = order.get("stopPrice") or order.get("price", 0)
            
            print(f"Order #{i}:")
            print(f"  Type (uppercase): '{order_type}'")
            print(f"  Stop Price:       {stop_price}")
            
            # Test conditions
            is_tp = "TAKE_PROFIT" in order_type or order_type == "TAKE_PROFIT_MARKET"
            is_sl = ("STOP" in order_type and "MARKET" in order_type) or order_type == "STOP_MARKET" or "STOP_LOSS" in order_type
            
            print(f"  Detected as TP?   {is_tp}")
            print(f"  Detected as SL?   {is_sl}")
            
            if is_tp:
                print(f"  ✅ Would set TP = ${stop_price}")
            elif is_sl:
                print(f"  ✅ Would set SL = ${stop_price}")
            else:
                print(f"  ❌ NOT detected as TP or SL!")
            print()

    except Exception as e:
        print(f"\n❌ Error fetching orders: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ Debug complete!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
