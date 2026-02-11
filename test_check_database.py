#!/usr/bin/env python3
"""
Check database content to see what orders exist.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.auto_trade.database import Order, get_db_manager


def main():
    """Check database content."""

    print("=" * 80)
    print("🗄️  Database Content Inspector")
    print("=" * 80)

    db_manager = get_db_manager()

    print("\n📊 Checking Orders table...")
    print("-" * 80)

    with db_manager.session_scope() as session:
        # Get all orders
        all_orders = session.query(Order).all()

        print(f"\n✅ Found {len(all_orders)} total order(s)\n")

        if not all_orders:
            print("⚠️  Database is empty. No orders found.")
            return

        # Group by status
        open_orders = [o for o in all_orders if o.status == "OPEN"]
        closed_orders = [o for o in all_orders if o.status == "CLOSED"]

        print("📁 Status breakdown:")
        print(f"  - OPEN:   {len(open_orders)}")
        print(f"  - CLOSED: {len(closed_orders)}")
        print(f"  - OTHER:  {len(all_orders) - len(open_orders) - len(closed_orders)}")

        # Show open orders in detail
        if open_orders:
            print("\n" + "=" * 80)
            print("🟢 OPEN ORDERS:")
            print("=" * 80)

            for order in open_orders:
                print(f"\n📌 Order ID: {order.id}")
                print(f"   Symbol:           {order.symbol}")
                print(f"   Side:             {order.side}")
                print(f"   Entry Price:      {order.entry_price}")
                print(f"   Amount:           {order.amount}")
                print(f"   Leverage:         {order.leverage}")
                print(f"   Status:           {order.status}")
                print(f"   Order Source:     {order.order_source}")
                print(f"   Execution Mode:   {order.execution_mode}")
                print(f"   Take Profit:      {order.take_profit}")
                print(f"   Stop Loss:        {order.stop_loss}")
                print(f"   BE Moved:         {order.be_moved}")
                print(f"   Created At:       {order.created_at}")
                print(f"   Client Order ID:  {order.client_order_id}")

        # Show recent closed orders
        if closed_orders:
            print("\n" + "=" * 80)
            print("🔴 CLOSED ORDERS (Last 5):")
            print("=" * 80)

            for order in closed_orders[-5:]:
                print(f"\n📌 Order ID: {order.id}")
                print(f"   Symbol:       {order.symbol}")
                print(f"   Side:         {order.side}")
                print(f"   Entry Price:  {order.entry_price}")
                print(f"   Status:       {order.status}")
                print(f"   PnL:          ${order.pnl:.2f} ({order.pnl_percentage:.2f}%)")
                print(f"   Closed At:    {order.closed_at}")

    print("\n" + "=" * 80)
    print("✅ Database inspection complete!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
