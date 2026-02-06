import traceback
import sys
from unittest.mock import Mock, patch
import time

print("Starting test...")

# Patch BinanceClient
with patch("modules.auto_trade.execution.order_executor.BinanceClient") as mock_bc:
    print("BinanceClient patched")
    mock_instance = Mock()
    mock_instance.exchange.fetch_ticker.return_value = {"last": 50000.0}
    mock_bc.return_value = mock_instance

    # Patch OrderManager
    with patch("modules.auto_trade.execution.order_executor.OrderManager") as mock_om:
        print("OrderManager patched")
        mock_om_instance = Mock()
        mock_om_instance.execute_order.return_value = {
            "success": True,
            "order_id": "TEST_ORDER_123",
            "message": "Order executed successfully",
        }
        mock_om.return_value = mock_om_instance

        from modules.auto_trade.execution.order_executor import OrderExecutor

        print("Imported OrderExecutor")

        executor = OrderExecutor(api_key="test_key", api_secret="test_secret")
        print("Created executor")

        signal_dict = {"symbol": "BTC/USDT", "signal": "LONG", "score": 0.85, "created_at_ts": time.time()}
        tp_sl_settings = {"default_tp": 10.0, "default_sl": 5.0}
        print("About to call execute_from_signal")

        try:
            result = executor.execute_from_signal(signal_dict, tp_sl_settings=tp_sl_settings)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Exception: {type(e).__name__}: {e}")
            traceback.print_exc()

print("Done")
