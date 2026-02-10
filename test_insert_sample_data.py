"""Test script to insert sample data into database for testing Data Viewer."""

from datetime import datetime
from modules.auto_trade.database import session_scope
from modules.auto_trade.database.models import Order, Signal

def insert_sample_data():
    """Insert sample orders and signals."""
    print("[Test] Inserting sample data...")
    
    with session_scope() as session:
        # Insert sample orders
        for i in range(5):
            order = Order(
                symbol=f"BTC/USDT",
                order_id=f"test_order_{i}",
                side="BUY" if i % 2 == 0 else "SELL",
                quantity=0.001 * (i + 1),
                price=50000.0 + (i * 1000),
                status="OPEN" if i < 2 else "CLOSED",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            session.add(order)
            print(f"  Added order: {order.symbol} {order.side} @ {order.price}")
        
        # Insert sample signals
        for i in range(3):
            signal = Signal(
                symbol=f"ETH/USDT",
                side="LONG" if i % 2 == 0 else "SHORT",
                score=0.7 + (i * 0.05),
                confidence=0.8,
                created_at=datetime.now(),
            )
            session.add(signal)
            print(f"  Added signal: {signal.symbol} {signal.side} score={signal.score}")
        
        session.commit()
        print("[Test] Sample data inserted successfully!")
        
        # Verify counts
        order_count = session.query(Order).count()
        signal_count = session.query(Signal).count()
        print(f"[Test] Database now has {order_count} orders, {signal_count} signals")

if __name__ == "__main__":
    insert_sample_data()
