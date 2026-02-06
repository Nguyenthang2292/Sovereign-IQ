import traceback
from unittest.mock import Mock, patch, MagicMock
import time

print("Detailed tracing of execute_from_signal...")

# Patch BinanceClient before importing OrderExecutor
with patch("modules.auto_trade.execution.order_executor.BinanceClient") as mock_bc:
    mock_instance = Mock()
    mock_instance.exchange.fetch_ticker.return_value = {"last": 50000.0}
    mock_bc.return_value = mock_instance

    # Patch OrderManager
    with patch("modules.auto_trade.execution.order_executor.OrderManager") as mock_om:
        mock_om_instance = Mock()
        mock_om_instance.execute_order.return_value = {
            "success": True,
            "order_id": "TEST_ORDER_123",
            "message": "Order executed successfully",
        }
        mock_om.return_value = mock_om_instance

        from modules.auto_trade.execution.order_executor import OrderExecutor
        from modules.common.core.exchange_manager import ExchangeManager
        from modules.common.core.data_fetcher import DataFetcher
        from modules.auto_trade.core.signal_selector import FinalSignal

        executor = OrderExecutor(api_key="test_key", api_secret="test_secret")

        # Manually execute the steps in execute_from_signal with debug output
        signal_dict = {"symbol": "BTC/USDT", "signal": "LONG", "score": 0.85, "created_at_ts": time.time()}
        tp_sl_settings = {"default_tp": 10.0, "default_sl": 5.0}

        print("\nStep 1: Check API credentials")
        print(f"  API key present: {bool(executor._api_key)}")
        print(f"  API secret present: {bool(executor._api_secret)}")

        print("\nStep 2: Process symbol and signal")
        symbol = signal_dict.get("symbol", "").replace("USDT", "/USDT")
        if not symbol.endswith("/USDT"):
            symbol = f"{symbol}/USDT"
        signal_type = (signal_dict.get("signal") or "LONG").upper()
        if signal_type not in ("LONG", "SHORT"):
            signal_type = "LONG"
        print(f"  Symbol: {symbol}, Signal: {signal_type}")

        print("\nStep 3: Create ExchangeManager")
        try:
            exchange_manager = ExchangeManager(
                api_key=executor._api_key,
                api_secret=executor._api_secret,
                testnet=executor._testnet,
            )
            print("  SUCCESS")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

        print("\nStep 4: Create DataFetcher")
        try:
            data_fetcher = DataFetcher(exchange_manager=exchange_manager)
            print("  SUCCESS")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

        print("\nStep 5: Create BinanceClient (this should use our mock)")
        try:
            from modules.auto_trade.execution.binance_client import BinanceClient

            client = BinanceClient(
                api_key=executor._api_key,
                api_secret=executor._api_secret,
                testnet=executor._testnet,
                dry_run=executor._dry_run,
            )
            print(f"  Created client: {client}")
            print(f"  Client type: {type(client)}")
            print(f"  Is Mock: {isinstance(client, Mock)}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

        print("\nStep 6: Fetch ticker")
        try:
            ticker = client.exchange.fetch_ticker(symbol)
            print(f"  Ticker: {ticker}")
            entry = float(ticker.get("last", 0) or 0)
            print(f"  Entry price: {entry}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

        print("\nStep 7: Calculate TP/SL")
        try:
            tp_pct = 5.0
            sl_pct = 2.0
            if tp_sl_settings:
                try:
                    tp_pct = float(tp_sl_settings.get("default_tp", tp_pct))
                except (TypeError, ValueError):
                    tp_pct = 5.0
                try:
                    sl_pct = float(tp_sl_settings.get("default_sl", sl_pct))
                except (TypeError, ValueError):
                    sl_pct = 2.0
            print(f"  TP%: {tp_pct}, SL%: {sl_pct}")

            if signal_type == "LONG":
                take_profit = entry * (1 + tp_pct / 100)
                stop_loss = entry * (1 - sl_pct / 100)
            else:
                take_profit = entry * (1 - tp_pct / 100)
                stop_loss = entry * (1 + sl_pct / 100)
            print(f"  TP: {take_profit}, SL: {stop_loss}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

        print("\nStep 8: Create FinalSignal")
        try:
            final_signal = FinalSignal(
                symbol=symbol,
                signal_type=signal_type,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=2,
                score=float(signal_dict.get("score", 0)),
            )
            print(f"  FinalSignal: {final_signal}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

        print("\nStep 9: Create OrderManager")
        try:
            from modules.auto_trade.execution.order_manager import OrderManager

            manager = OrderManager(
                data_fetcher=data_fetcher,
                api_key=executor._api_key,
                api_secret=executor._api_secret,
                testnet=executor._testnet,
                dry_run=executor._dry_run,
            )
            print(f"  OrderManager: {manager}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

        print("\nStep 10: Execute signal")
        try:
            result = manager.execute_signal(final_signal)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

print("\nDone!")
