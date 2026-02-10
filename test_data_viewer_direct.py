"""Direct test of DataViewerService to verify query works."""

from modules.auto_trade.database import session_scope
from modules.auto_trade.database.models import Order, Signal
from modules.auto_trade.gui.services.database_service import DataViewerService

def test_direct_query():
    """Test DataViewerService directly without GUI."""
    print("\n" + "="*80)
    print("TESTING DataViewerService DIRECTLY")
    print("="*80)
    
    # Test 1: Get counts
    print("\n1. Testing get_table_count()...")
    for table in ["Orders", "Signals", "Martingale Chains", "Audit Log"]:
        count = DataViewerService.get_table_count(table)
        print(f"   {table}: {count} records")
    
    # Test 2: Get Orders data
    print("\n2. Testing get_table_data() for Orders...")
    orders = DataViewerService.get_table_data("Orders", limit=5, last_id=None)
    print(f"   Retrieved {len(orders)} orders")
    
    if orders:
        print(f"   First order type: {type(orders[0])}")
        print(f"   First order: {orders[0]}")
        
        if hasattr(orders[0], "to_dict"):
            order_dict = orders[0].to_dict()
            print(f"   First order dict keys: {list(order_dict.keys())}")
            print(f"   First order dict: {order_dict}")
        else:
            print("   WARNING: Order does not have to_dict() method!")
    
    # Test 3: Get Signals data
    print("\n3. Testing get_table_data() for Signals...")
    signals = DataViewerService.get_table_data("Signals", limit=5, last_id=None)
    print(f"   Retrieved {len(signals)} signals")
    
    if signals:
        print(f"   First signal type: {type(signals[0])}")
        if hasattr(signals[0], "to_dict"):
            signal_dict = signals[0].to_dict()
            print(f"   First signal dict keys: {list(signal_dict.keys())}")
    
    # Test 4: Direct database query
    print("\n4. Testing direct database query...")
    with session_scope() as session:
        order_count = session.query(Order).count()
        signal_count = session.query(Signal).count()
        print(f"   Direct query - Orders: {order_count}, Signals: {signal_count}")
        
        if order_count > 0:
            first_order = session.query(Order).first()
            print(f"   First order from DB: {first_order}")
            if hasattr(first_order, "to_dict"):
                print(f"   First order dict: {first_order.to_dict()}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_direct_query()
