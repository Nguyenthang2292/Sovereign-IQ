"""Check all orders in database and their order_source."""

from modules.auto_trade.database import session_scope
from modules.auto_trade.database.models import Order


def check_order_sources():
    """Check all orders and their sources."""
    print("\n" + "="*80)
    print("CHECKING ALL ORDERS IN DATABASE")
    print("="*80 + "\n")

    with session_scope() as session:
        # Get ALL orders (not just PROGRAMMATIC)
        all_orders = session.query(Order).filter(Order.status == "OPEN").all()

        print(f"Found {len(all_orders)} OPEN orders (all sources):\n")

        for order in all_orders:
            print(f"Symbol: {order.symbol}")
            print(f"  Order ID: {order.order_id}")
            print(f"  Side: {order.side}")
            print(f"  Order Source: '{order.order_source}' ⚠️")
            print(f"  Execution Mode: '{order.execution_mode}'")
            print(f"  Entry Price: {order.entry_price}")
            print(f"  TP: {order.take_profit}")
            print(f"  SL: {order.stop_loss}")
            print(f"  BE Moved: {order.be_moved}")
            print(f"  Status: {order.status}")
            print()

        # Count by source
        print("-" * 80)
        programmatic = [o for o in all_orders if o.order_source == "PROGRAMMATIC"]
        manual = [o for o in all_orders if o.order_source == "MANUAL"]
        other = [o for o in all_orders if o.order_source not in ("PROGRAMMATIC", "MANUAL")]

        print("Summary:")
        print(f"  PROGRAMMATIC: {len(programmatic)}")
        print(f"  MANUAL: {len(manual)}")
        print(f"  OTHER: {len(other)}")

        if manual:
            print(f"\n⚠️  WARNING: {len(manual)} order(s) marked as 'MANUAL' but placed by AutoTrade!")
            print("   These orders will NOT be found by get_open_positions() default query")
            print("   This is why TP/SL shows N/A in GUI\n")

    print("="*80)
    print("SOLUTION:")
    print("Need to fix either:")
    print("  1. Order creation to set order_source='PROGRAMMATIC'")
    print("  2. Query to include MANUAL orders from AutoTrade")
    print("="*80 + "\n")

if __name__ == "__main__":
    check_order_sources()
