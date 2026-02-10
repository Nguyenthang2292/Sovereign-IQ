"""Test fetching open orders from Binance API to get TP/SL."""

import os
from dotenv import load_dotenv

load_dotenv()

def test_binance_open_orders():
    """Test fetching open orders from Binance to find TP/SL orders."""
    print("\n" + "="*80)
    print("TESTING BINANCE OPEN ORDERS API")
    print("="*80 + "\n")
    
    # Get credentials
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    
    if not api_key or not api_secret:
        print("❌ API credentials not found in .env")
        return
    
    try:
        from modules.auto_trade.execution.binance_client import BinanceClient
        
        client = BinanceClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            dry_run=False
        )
        
        # Fetch all open orders
        print("📋 Fetching open orders from Binance...\n")
        open_orders = client.exchange.fetch_open_orders()
        
        print(f"Found {len(open_orders)} open order(s):\n")
        
        # Group by symbol
        from collections import defaultdict
        orders_by_symbol = defaultdict(list)
        
        for order in open_orders:
            symbol = order.get("symbol", "")
            orders_by_symbol[symbol].append(order)
        
        # Display grouped by symbol
        for symbol, orders in orders_by_symbol.items():
            print(f"Symbol: {symbol}")
            print(f"  Total orders: {len(orders)}")
            
            for order in orders:
                order_type = order.get("type", "")
                side = order.get("side", "")
                price = order.get("price", 0)
                stop_price = order.get("stopPrice", 0)
                amount = order.get("amount", 0)
                
                print(f"  - {order_type} {side}")
                print(f"    Price: ${price}")
                print(f"    Stop Price: ${stop_price}")
                print(f"    Amount: {amount}")
                
                # Identify TP/SL orders
                if "TAKE_PROFIT" in order_type.upper():
                    print(f"    → 🎯 This is TAKE PROFIT order!")
                elif "STOP" in order_type.upper() and "MARKET" in order_type.upper():
                    print(f"    → 🛡️ This is STOP LOSS order!")
            
            print()
        
        print("-" * 80)
        print("\n💡 TP/SL Detection Logic:")
        print("  - TAKE_PROFIT_MARKET order → Take Profit price")
        print("  - STOP_MARKET order → Stop Loss price")
        print("  - Match by symbol with position\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*80 + "\n")

if __name__ == "__main__":
    test_binance_open_orders()
