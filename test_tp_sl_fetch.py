"""Test script to verify TP/SL/BE fetching from database for open positions."""

from modules.auto_trade.database import get_open_positions, session_scope


def test_tp_sl_fetch():
    """Test fetching TP/SL from database."""
    print("\n" + "="*80)
    print("TESTING TP/SL/BE FETCH FROM DATABASE")
    print("="*80 + "\n")

    with session_scope() as session:
        # Get all open positions
        all_orders = get_open_positions(session)

        print(f"Found {len(all_orders)} open orders in database:\n")

        for order in all_orders:
            print(f"Symbol: {order.symbol}")
            print(f"  Order ID: {order.order_id}")
            print(f"  Side: {order.side}")
            print(f"  Entry Price: {order.entry_price}")
            print(f"  Size: {order.amount}")
            print(f"  TP: {order.take_profit}")
            print(f"  SL: {order.stop_loss}")
            print(f"  BE Moved: {order.be_moved}")
            print(f"  Status: {order.status}")
            print()

        if not all_orders:
            print("⚠️  No open orders found in database!")
            print("   This means:")
            print("   1. No trades have been placed by the system yet")
            print("   2. Or all trades were closed")
            print("   3. Or database is not synced with Binance")

    print("="*80)
    print("RECOMMENDATION:")
    print("If you have open positions on Binance but not in DB:")
    print("  - They might be MANUAL trades (not tracked by system)")
    print("  - System only tracks PROGRAMMATIC orders")
    print("  - TP/SL will show N/A for manual trades")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_tp_sl_fetch()
