"""Fix order_source for existing orders in database.

This script updates orders that were placed by AutoTrade (execution_mode='AUTO')
but incorrectly marked as order_source='MANUAL' to order_source='PROGRAMMATIC'.
"""

from modules.auto_trade.database import session_scope
from modules.auto_trade.database.models import Order


def fix_order_sources(dry_run: bool = True):
    """
    Fix order_source for AutoTrade orders.

    Args:
        dry_run: If True, only show what would be changed without changing it.
                 If False, actually update the database.
    """
    print("\n" + "="*80)
    print("FIX ORDER_SOURCE FOR AUTOTRADE ORDERS")
    print("="*80 + "\n")

    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print("   Run with dry_run=False to actually update\n")
    else:
        print("⚠️  LIVE MODE - Database will be updated!")
        print("   Press Ctrl+C within 3 seconds to cancel...\n")
        import time
        time.sleep(3)
        print("   Proceeding with update...\n")

    with session_scope() as session:
        # Find orders that should be PROGRAMMATIC but are marked as MANUAL
        # Criteria: execution_mode='AUTO' but order_source='MANUAL'
        incorrect_orders = session.query(Order).filter(
            Order.execution_mode == "AUTO",
            Order.order_source == "MANUAL"
        ).all()

        print(f"Found {len(incorrect_orders)} order(s) with incorrect order_source:\n")

        if not incorrect_orders:
            print("✅ No orders need fixing!")
            return

        for order in incorrect_orders:
            print(f"Order ID: {order.id}")
            print(f"  Symbol: {order.symbol}")
            print(f"  Side: {order.side}")
            print(f"  Status: {order.status}")
            print(f"  Current order_source: '{order.order_source}' ❌")
            print(f"  Current execution_mode: '{order.execution_mode}' ✅")
            print("  → Will change to: order_source='PROGRAMMATIC'")
            print()

            if not dry_run:
                order.order_source = "PROGRAMMATIC"

        if not dry_run:
            session.commit()
            print("✅ Database updated successfully!")
            print(f"   {len(incorrect_orders)} order(s) changed to order_source='PROGRAMMATIC'\n")
        else:
            print("💡 To apply these changes, run:")
            print("   python scripts/maintenance/fix_order_source.py --live\n")

    # Verify the fix
    if not dry_run:
        print("-" * 80)
        print("VERIFICATION:")
        with session_scope() as session:
            remaining = session.query(Order).filter(
                Order.execution_mode == "AUTO",
                Order.order_source == "MANUAL"
            ).count()

            if remaining == 0:
                print("✅ All AutoTrade orders now have order_source='PROGRAMMATIC'")
            else:
                print(f"⚠️  Still {remaining} order(s) with incorrect source")

    print("="*80 + "\n")


def show_current_status():
    """Show current status of all orders."""
    print("\n" + "="*80)
    print("CURRENT DATABASE STATUS")
    print("="*80 + "\n")

    with session_scope() as session:
        total = session.query(Order).count()
        programmatic = session.query(Order).filter(Order.order_source == "PROGRAMMATIC").count()
        manual = session.query(Order).filter(Order.order_source == "MANUAL").count()

        auto_mode = session.query(Order).filter(Order.execution_mode == "AUTO").count()
        manual_mode = session.query(Order).filter(Order.execution_mode == "MANUAL").count()

        # The problematic ones
        incorrect = session.query(Order).filter(
            Order.execution_mode == "AUTO",
            Order.order_source == "MANUAL"
        ).count()

        print(f"Total orders: {total}")
        print("\nBy order_source:")
        print(f"  PROGRAMMATIC: {programmatic}")
        print(f"  MANUAL: {manual}")
        print("\nBy execution_mode:")
        print(f"  AUTO: {auto_mode}")
        print(f"  MANUAL: {manual_mode}")
        print(f"\n⚠️  Incorrect (AUTO + MANUAL source): {incorrect}")

        if incorrect > 0:
            print(f"   → These {incorrect} order(s) need fixing!")
        else:
            print("   ✅ All orders have correct source!")

    print("="*80 + "\n")


if __name__ == "__main__":
    import sys

    # Show current status first
    show_current_status()

    # Check if --live flag is passed
    if "--live" in sys.argv or "--apply" in sys.argv:
        print("⚠️  Running in LIVE mode!")
        fix_order_sources(dry_run=False)
    else:
        print("Running in DRY RUN mode (safe)")
        fix_order_sources(dry_run=True)
        print("\n💡 To actually apply changes, run:")
        print("   python scripts/maintenance/fix_order_source.py --live")
