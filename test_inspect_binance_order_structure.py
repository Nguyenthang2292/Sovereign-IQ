#!/usr/bin/env python3
"""
Inspect Binance Order Structure

Script này sẽ:
1. Fetch orders từ Binance (active + recent history)
2. In ra TOÀN BỘ fields và values của mỗi order
3. Hiển thị structure ở nhiều format khác nhau
4. Giúp hiểu rõ data structure để code chính xác
"""

import json
import sys
from pathlib import Path
from pprint import pformat

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.gui.utils.credential_manager import CredentialManager
from modules.auto_trade.gui.utils.settings_manager import SettingsManager


def print_separator(char="=", length=80):
    """Print separator line."""
    print(char * length)


def print_header(title):
    """Print formatted header."""
    print_separator()
    print(f"  {title}")
    print_separator()


def inspect_order_detailed(order, index=1):
    """Inspect and print detailed order information."""

    print(f"\n{'🔷' * 40}")
    print(f"  ORDER #{index}")
    print(f"{'🔷' * 40}\n")

    # === SECTION 1: Main Fields ===
    print("📌 MAIN FIELDS (Top Level):")
    print("-" * 80)

    main_fields = [
        'id', 'clientOrderId', 'timestamp', 'datetime', 'lastTradeTimestamp',
        'symbol', 'type', 'timeInForce', 'postOnly', 'side',
        'price', 'stopPrice', 'triggerPrice', 'amount', 'cost',
        'average', 'filled', 'remaining', 'status', 'fee', 'trades'
    ]

    for field in main_fields:
        value = order.get(field)
        if value is not None:
            print(f"  {field:25s} = {value}")

    # === SECTION 2: Info Field (Raw Binance) ===
    if 'info' in order:
        print("\n📋 INFO FIELD (Raw từ Binance API):")
        print("-" * 80)

        info = order['info']

        # Important fields first
        important_info_fields = [
            'orderId', 'clientOrderId', 'symbol', 'status', 'type', 'side',
            'price', 'stopPrice', 'activatePrice', 'priceRate',
            'origQty', 'executedQty', 'cumQuote', 'cumQty',
            'timeInForce', 'workingType', 'priceProtect',
            'reduceOnly', 'closePosition', 'positionSide',
            'time', 'updateTime', 'avgPrice'
        ]

        print("\n  🔹 Important Info Fields:")
        for field in important_info_fields:
            value = info.get(field)
            if value is not None:
                print(f"    {field:25s} = {value}")

        # All other fields
        print("\n  🔹 All Info Fields (Complete List):")
        for key in sorted(info.keys()):
            value = info[key]
            print(f"    {key:25s} = {value}")

    # === SECTION 3: Order Type Detection ===
    print("\n🎯 ORDER TYPE DETECTION:")
    print("-" * 80)

    order_type = order.get("type", "").upper()
    info_type = order.get("info", {}).get("type", "").upper()
    stop_price = order.get("stopPrice") or order.get("info", {}).get("stopPrice")

    print(f"  Main type field:        '{order_type}'")
    print(f"  Info type field:        '{info_type}'")
    print(f"  Has stopPrice?          {stop_price is not None} (value: {stop_price})")

    # Test detection logic
    print("\n  Detection Results:")
    is_tp = "TAKE_PROFIT" in order_type or order_type == "TAKE_PROFIT_MARKET"
    is_sl = ("STOP" in order_type and "MARKET" in order_type) or order_type == "STOP_MARKET" or "STOP_LOSS" in order_type

    if is_tp:
        print("    ✅ DETECTED AS: Take Profit Order")
    elif is_sl:
        print("    ✅ DETECTED AS: Stop Loss Order")
    elif "LIMIT" in order_type:
        print("    📝 DETECTED AS: Limit Order")
    elif "MARKET" in order_type:
        print("    📝 DETECTED AS: Market Order")
    else:
        print(f"    ❓ UNKNOWN ORDER TYPE: {order_type}")

    # === SECTION 4: Full JSON Structure ===
    print("\n🔬 FULL JSON STRUCTURE:")
    print("-" * 80)
    print(json.dumps(order, indent=2, default=str))

    # === SECTION 5: Python Dict Repr ===
    print("\n🐍 PYTHON DICT REPRESENTATION:")
    print("-" * 80)
    print(pformat(order, indent=2, width=100))

    print("\n" + "=" * 80 + "\n")


def main():
    """Main inspection function."""

    print_header("🔍 BINANCE ORDER STRUCTURE INSPECTOR 🔍")

    # Load settings
    settings = SettingsManager()
    settings.load()

    # Get credentials
    credential_manager = CredentialManager()
    api_config = credential_manager.load_credentials("binance")

    # Get testnet setting
    api_settings = settings.get("api", {})
    testnet = api_settings.get("mode", "").upper() == "TESTNET"

    print(f"\n🔑 API Mode: {'TESTNET' if testnet else 'PRODUCTION'}")
    print("🌐 Connecting to Binance...")

    # Create Binance client
    client = BinanceClient(
        api_key=api_config.get("api_key") or "",
        api_secret=api_config.get("api_secret") or "",
        testnet=testnet,
        dry_run=False,
    )

    symbol = "SKL/USDT"
    symbol_normalized = "SKLUSDT"

    print(f"📊 Symbol: {symbol}")

    # ========================================
    # PART 1: Check Open Orders
    # ========================================

    print_header("📋 PART 1: OPEN ORDERS (Active)")

    try:
        print(f"\nFetching open orders for {symbol}...")
        open_orders = client.exchange.fetch_open_orders(symbol)

        print(f"✅ Found {len(open_orders)} open order(s)\n")

        if open_orders:
            for i, order in enumerate(open_orders, 1):
                inspect_order_detailed(order, i)
        else:
            print("⚠️  No open orders found.\n")

    except Exception as e:
        print(f"❌ Error fetching open orders: {e}\n")

    # ========================================
    # PART 2: Check Recent Order History
    # ========================================

    print_header("📜 PART 2: RECENT ORDER HISTORY (Last 10 orders)")

    try:
        print(f"\nFetching recent orders for {symbol}...")

        # Use direct API to get all recent orders
        all_orders = client.exchange.fapiPrivateGetAllOrders({
            'symbol': symbol_normalized,
            'limit': 10
        })

        print(f"✅ Found {len(all_orders)} recent order(s)\n")

        if all_orders:
            # Show summary first
            print("📊 SUMMARY:")
            print("-" * 80)
            for i, order in enumerate(all_orders, 1):
                order_id = order.get('orderId')
                order_type = order.get('type')
                side = order.get('side')
                status = order.get('status')
                price = order.get('price', 'N/A')
                stop_price = order.get('stopPrice', 'N/A')

                print(f"  #{i} | ID: {order_id} | Type: {order_type:20s} | Side: {side:5s} | Status: {status:10s} | Price: {price} | StopPrice: {stop_price}")

            # Then detailed inspection
            print("\n")
            for i, raw_order in enumerate(all_orders, 1):
                # Convert raw order to ccxt format for consistency
                converted_order = {
                    'id': str(raw_order.get('orderId')),
                    'clientOrderId': raw_order.get('clientOrderId'),
                    'symbol': symbol,
                    'type': raw_order.get('type'),
                    'side': raw_order.get('side'),
                    'price': float(raw_order.get('price', 0)) if raw_order.get('price') else None,
                    'stopPrice': float(raw_order.get('stopPrice', 0)) if raw_order.get('stopPrice') else None,
                    'amount': float(raw_order.get('origQty', 0)) if raw_order.get('origQty') else None,
                    'status': raw_order.get('status'),
                    'timestamp': raw_order.get('time'),
                    'datetime': None,
                    'info': raw_order
                }

                inspect_order_detailed(converted_order, i)
        else:
            print("⚠️  No orders found.\n")

    except Exception as e:
        print(f"❌ Error fetching order history: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # PART 3: Summary & Guide
    # ========================================

    print_header("💡 DETECTION GUIDE")

    print("""
📌 KEY FIELDS TO CHECK:

1. order.get('type')           → Main order type (uppercase)
2. order.get('info').get('type') → Raw Binance type
3. order.get('stopPrice')      → Stop trigger price
4. order.get('status')         → NEW, FILLED, CANCELED, etc.

📌 BINANCE ORDER TYPES:

MARKET ORDERS:
- MARKET                       → Regular market order

LIMIT ORDERS:
- LIMIT                        → Regular limit order
- STOP                         → Stop-limit order
- TAKE_PROFIT                  → Take profit limit order

MARKET STOP ORDERS (TP/SL):
- STOP_MARKET                  → Stop loss market order
- TAKE_PROFIT_MARKET           → Take profit market order
- STOP_LOSS                    → Alternative name for stop
- TRAILING_STOP_MARKET         → Trailing stop order

📌 DETECTION LOGIC:

✅ Take Profit Detection:
   if "TAKE_PROFIT" in order_type:
       → This is a TP order
       → Use stopPrice as TP value

✅ Stop Loss Detection:
   if "STOP" in order_type and ("MARKET" in order_type or "LOSS" in order_type):
       → This is a SL order
       → Use stopPrice as SL value

📌 IMPORTANT NOTES:

⚠️  Nếu không có open orders:
   → TP/SL có thể đã filled/canceled
   → Hoặc không được set

⚠️  Check status field:
   → NEW = Active order
   → FILLED = Đã execute
   → CANCELED = Đã hủy
   → EXPIRED = Hết hạn

⚠️  stopPrice vs price:
   → stopPrice: Trigger price cho TP/SL
   → price: Execution price (có thể khác stopPrice)
""")

    print_separator()
    print("✅ Inspection Complete!")
    print_separator()
    print()


if __name__ == "__main__":
    main()
