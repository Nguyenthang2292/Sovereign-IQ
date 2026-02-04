"""
Order Executor – facade for GUI and auto-trade.

Delegates to OrderManager (execute_from_signal) and BinanceClient (place_order).
Resolves credentials from env when not passed.
"""

import os
from typing import Any, Dict, Optional

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.execution.order_builder import OrderTicket
from modules.auto_trade.execution.order_manager import OrderManager
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager


class OrderExecutor:
    """
    Facade for executing orders from the GUI and auto-trade cycle.

    - execute_from_signal(signal_dict): run full pipeline via OrderManager.
    - place_order(symbol, side, amount, ...): place a single market order via BinanceClient.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        dry_run: bool = False,
    ):
        self._api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self._api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        self._testnet = (
            testnet
            if testnet is not None
            else os.getenv("BINANCE_TESTNET", "false").lower() == "true"
        )
        self._dry_run = dry_run

    def execute_from_signal(self, signal_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade from a signal dict (e.g. from get_signals).

        Args:
            signal_dict: Must have "symbol", "signal" (LONG/SHORT). Optional: "score".

        Returns:
            Dict with "success" (bool) and optional "error" or order details.
        """
        try:
            if not self._api_key or not self._api_secret:
                return {"success": False, "error": "API credentials not set"}

            symbol = signal_dict.get("symbol", "").replace("USDT", "/USDT")
            if not symbol.endswith("/USDT"):
                symbol = f"{symbol}/USDT"
            signal_type = (signal_dict.get("signal") or "LONG").upper()
            if signal_type not in ("LONG", "SHORT"):
                signal_type = "LONG"

            exchange_manager = ExchangeManager(
                api_key=self._api_key,
                api_secret=self._api_secret,
                testnet=self._testnet,
            )
            data_fetcher = DataFetcher(exchange_manager=exchange_manager)
            client = BinanceClient(
                api_key=self._api_key,
                api_secret=self._api_secret,
                testnet=self._testnet,
                dry_run=self._dry_run,
            )
            ticker = client.exchange.fetch_ticker(symbol)
            entry = float(ticker.get("last", 0) or 0)
            if entry <= 0:
                return {"success": False, "error": "Could not get current price"}

            tp_pct = 5.0
            sl_pct = 2.0
            if signal_type == "LONG":
                take_profit = entry * (1 + tp_pct / 100)
                stop_loss = entry * (1 - sl_pct / 100)
            else:
                take_profit = entry * (1 - tp_pct / 100)
                stop_loss = entry * (1 + sl_pct / 100)

            final_signal = FinalSignal(
                symbol=symbol,
                signal_type=signal_type,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=2,
                score=float(signal_dict.get("score", 0)),
            )
            manager = OrderManager(
                data_fetcher=data_fetcher,
                api_key=self._api_key,
                api_secret=self._api_secret,
                testnet=self._testnet,
                dry_run=self._dry_run,
            )
            result = manager.execute_signal(final_signal)
            if result is None:
                return {"success": False, "error": "Execution skipped or failed"}
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        leverage: int = 2,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place a single market order (e.g. from trade form).

        Args:
            symbol: e.g. "BTC/USDT" or "BTCUSDT"
            side: "long" or "short" / "buy" or "sell"
            amount: Size in USDT
            leverage: Leverage
            take_profit: TP price (optional)
            stop_loss: SL price (optional)

        Returns:
            Dict with "success" (bool) and optional "error" or order details.
        """
        try:
            if not self._api_key or not self._api_secret:
                return {"success": False, "error": "API credentials not set"}

            sym = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
            side_lower = side.lower()
            if side_lower in ("long", "buy"):
                side_val = "BUY"
            else:
                side_val = "SELL"

            client = BinanceClient(
                api_key=self._api_key,
                api_secret=self._api_secret,
                testnet=self._testnet,
                dry_run=self._dry_run,
            )
            ticket = OrderTicket(
                symbol=sym,
                side=side_val,
                amount=amount,
                leverage=leverage,
                take_profit_price=take_profit,
                stop_loss_price=stop_loss,
            )
            result = client.create_market_order(ticket)
            if result is None:
                return {"success": False, "error": "Order failed"}
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}
