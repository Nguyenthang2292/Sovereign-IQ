import traceback
from unittest.mock import Mock, patch

with patch("modules.auto_trade.execution.order_executor.BinanceClient"):
    with patch("modules.auto_trade.execution.order_executor.OrderManager"):
        from modules.auto_trade.execution.order_executor import OrderExecutor
        from modules.common.core.exchange_manager import ExchangeManager

        try:
            print("Creating ExchangeManager...")
            em = ExchangeManager(api_key="test", api_secret="test", testnet=True)
            print(f"ExchangeManager created: {em}")
            print(f"ExchangeManager type: {type(em)}")
            attrs = [x for x in dir(em) if not x.startswith("_")][:10]
            print(f"ExchangeManager dir: {attrs}")
        except Exception as e:
            print(f"Exception: {type(e).__name__}: {e}")
            traceback.print_exc()
