import traceback
from unittest.mock import Mock, patch
import time

print("Testing DataFetcher with ExchangeManager...")

with patch("modules.auto_trade.execution.order_executor.BinanceClient") as mock_bc:
    mock_instance = Mock()
    mock_instance.exchange.fetch_ticker.return_value = {"last": 50000.0}
    mock_bc.return_value = mock_instance

    from modules.common.core.exchange_manager import ExchangeManager
    from modules.common.core.data_fetcher import DataFetcher

    print("Step 1: Creating ExchangeManager...")
    try:
        exchange_manager = ExchangeManager(api_key="test", api_secret="test", testnet=True)
        print("  SUCCESS")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()

    print("Step 2: Creating DataFetcher...")
    try:
        data_fetcher = DataFetcher(exchange_manager=exchange_manager)
        print("  SUCCESS")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()

    print("Step 3: Testing DataFetcher methods...")
    methods_to_test = ["get_current_price", "fetch_ohlcv", "get_ticker"]
    for method in methods_to_test:
        print(f"  Testing {method}...")
        try:
            if hasattr(data_fetcher, method):
                m = getattr(data_fetcher, method)
                print(f"    Method exists: {m}")
            else:
                print(f"    Method not found")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")

print("Done")
