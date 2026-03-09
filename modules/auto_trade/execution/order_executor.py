"""
Order Executor – facade for GUI and auto-trade.

Delegates to OrderManager (execute_from_signal) and BinanceClient (place_order).
Resolves credentials from env when not passed.
"""

import os
from typing import Any, Dict, Literal, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.execution.order_builder import OrderTicket
from modules.auto_trade.execution.order_manager import OrderManager
from modules.auto_trade.security.secret_string import SecretString
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.domain.symbol_codec import SymbolCodec
from modules.common.ui.logging import log_error, log_info, log_warn

_SYMBOL_CODEC = SymbolCodec()


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
        recovery_manager: Optional[Any] = None,
        order_book_imbalance_config: Optional[Dict[str, Any]] = None,
    ):
        resolved_api_key = api_key if api_key is not None else (os.getenv("BINANCE_API_KEY", "") or "")
        resolved_api_secret = api_secret if api_secret is not None else (os.getenv("BINANCE_API_SECRET", "") or "")
        self._api_key = SecretString(resolved_api_key)
        self._api_secret = SecretString(resolved_api_secret)
        self._testnet = testnet if testnet is not None else os.getenv("BINANCE_TESTNET", "false").lower() == "true"
        self._dry_run = dry_run
        self._recovery_manager = recovery_manager

        self._client = BinanceClient(
            api_key=self._api_key.get_secret_value(),
            api_secret=self._api_secret.get_secret_value(),
            testnet=self._testnet,
            dry_run=self._dry_run,
        )

        self._order_book_imbalance_gate: Optional[Any] = None

        if order_book_imbalance_config is not None:
            from modules.order_book.order_book_imbalance_gate import OrderBookImbalanceGate

            gate_config: Dict[str, Any] = dict(order_book_imbalance_config)
            if "depth_limit" not in gate_config:
                for legacy_depth_key in ("depth", "ob_depth"):
                    if legacy_depth_key in gate_config:
                        gate_config["depth_limit"] = gate_config[legacy_depth_key]
                        break
            gate_config.setdefault("testnet", self._testnet)
            # Only keep kwargs that OrderBookImbalanceGate.__init__ accepts;
            # the GUI may inject extra keys that gate does not accept.
            _allowed = {
                "threshold",
                "retry_wait_seconds",
                "max_retries",
                "depth_limit",
                "delta_window_minutes",
                "testnet",
                "enabled",
            }
            gate_config = {k: v for k, v in gate_config.items() if k in _allowed}
            self._order_book_imbalance_gate = OrderBookImbalanceGate(**gate_config)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        return dict(self._client.exchange.fetch_ticker(symbol))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _create_market_order(self, ticket: OrderTicket) -> Optional[dict]:
        return self._client.create_market_order(ticket)

    def execute_from_signal(
        self, signal_dict: Dict[str, Any], tp_sl_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a trade from a signal dict (e.g. from get_signals).

        Args:
            signal_dict: Must have "symbol", "signal" (LONG/SHORT). Optional: "score".
            tp_sl_settings: Optional settings dict from GUI (e.g. {"default_tp": 5.0, "default_sl": 2.5}).

        Returns:
            Dict with "success" (bool) and optional "error" or order details.
        """
        try:
            log_info(f"[OrderExecutor] execute_from_signal called for {signal_dict.get('symbol')}")
            if not self._api_key or not self._api_secret:
                log_error("[OrderExecutor] ERROR: API credentials not set")
                return {"success": False, "error": "API credentials not set"}

            # Normalize symbol to CCXT format (BTC/USDT)
            symbol: str = _SYMBOL_CODEC.to_ccxt(signal_dict.get("symbol", ""))

            signal_type: str = (signal_dict.get("signal") or "LONG").upper()
            if signal_type not in ("LONG", "SHORT"):
                signal_type = "LONG"

            exchange_manager = ExchangeManager(
                api_key=self._api_key.get_secret_value(),
                api_secret=self._api_secret.get_secret_value(),
                testnet=self._testnet,
            )
            data_fetcher = DataFetcher(exchange_manager=exchange_manager)

            log_info(f"[OrderExecutor] Fetching ticker for {symbol}...")
            ticker: Any = self._fetch_ticker(symbol)
            entry: float = float(ticker.get("last", 0) or 0)
            if entry <= 0:
                log_error(f"[OrderExecutor] ERROR: Could not get current price for {symbol}")
                return {"success": False, "error": "Could not get current price"}

            if self._order_book_imbalance_gate is not None:
                from modules.order_book.models import OBIDecision

                decision, combined_result = self._order_book_imbalance_gate.check(symbol, signal_type)
                if decision == OBIDecision.SKIP:
                    score_str = f"{combined_result.combined_score:.3f}" if combined_result is not None else "N/A"
                    log_warn(
                        f"[OrderBookImbalanceGate] {symbol} {signal_type} SKIPPED after retry. "
                        f"Combined Score={score_str} opposes direction."
                    )
                    return {
                        "success": False,
                        "skipped": True,
                        "reason": "ORDER_BOOK_IMBALANCE_CONFLICT",
                    }

            tp_pct: float = 5.0
            sl_pct: float = 2.0
            # Leverage: prefer explicit value in signal_dict, then tp_sl_settings, then default 2
            leverage: int = 2
            if tp_sl_settings:
                try:
                    tp_pct = float(tp_sl_settings.get("default_tp", tp_pct))
                except (TypeError, ValueError):
                    tp_pct = 5.0
                try:
                    sl_pct = float(tp_sl_settings.get("default_sl", sl_pct))
                except (TypeError, ValueError):
                    sl_pct = 2.0
            # Parse leverage from signal_dict first (set by GUI auto_trade cycle)
            raw_lev = signal_dict.get("leverage")
            if raw_lev is not None:
                try:
                    leverage = int(str(raw_lev).replace("x", "").strip())
                except (TypeError, ValueError):
                    leverage = 2
            elif tp_sl_settings:
                raw_lev_cfg = tp_sl_settings.get("default_leverage")
                if raw_lev_cfg is not None:
                    try:
                        leverage = int(str(raw_lev_cfg).replace("x", "").strip())
                    except (TypeError, ValueError):
                        leverage = 2
            log_info(f"[OrderExecutor] Using leverage={leverage}x for {symbol} | tp_roi={tp_pct}% sl_roi={sl_pct}%")

            # Legacy behavior: settings values are direct price-move percentages.
            # Do not scale by leverage here.
            tp_price_pct = tp_pct
            sl_price_pct = sl_pct
            log_info(f"[OrderExecutor] Price-move: tp={tp_price_pct:.4f}% sl={sl_price_pct:.4f}%")

            if signal_type == "LONG":
                take_profit = entry * (1 + tp_price_pct / 100)
                stop_loss = entry * (1 - sl_price_pct / 100)
            else:
                take_profit = entry * (1 - tp_price_pct / 100)
                stop_loss = entry * (1 + sl_price_pct / 100)

            final_signal = FinalSignal(
                symbol=symbol,
                signal_type=signal_type,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=leverage,
                score=float(signal_dict.get("score", 0)),
            )
            manager = OrderManager(
                data_fetcher=data_fetcher,
                api_key=self._api_key.get_secret_value(),
                api_secret=self._api_secret.get_secret_value(),
                testnet=self._testnet,
                dry_run=self._dry_run,
                default_leverage=leverage,  # propagate to OrderBuilder / RiskManager
                recovery_manager=self._recovery_manager,
            )
            log_info(f"[OrderExecutor] Calling OrderManager.execute_signal for {symbol} {signal_type}...")
            result: Optional[dict] = manager.execute_signal(final_signal)
            if result is None:
                log_warn("[OrderExecutor] OrderManager returned None (execution skipped or failed)")
                return {"success": False, "error": "Execution skipped or failed"}
            log_info(f"[OrderExecutor] OrderManager returned success: {result}")
            return {"success": True, **result}
        except Exception as e:
            log_error(f"[OrderExecutor] EXCEPTION in execute_from_signal: {type(e).__name__}: {e}", exc_info=True)
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

            # Normalize symbol to CCXT format
            sym: str = _SYMBOL_CODEC.to_ccxt(symbol)
            side_lower: str = side.lower()
            side_val: Literal["BUY", "SELL"] = "BUY" if side_lower in ("long", "buy") else "SELL"

            ticket = OrderTicket(
                symbol=sym,
                side=side_val,
                amount=amount,
                leverage=leverage,
                take_profit_price=take_profit,
                stop_loss_price=stop_loss,
            )
            result: Optional[dict] = self._create_market_order(ticket)
            if result is None:
                return {"success": False, "error": "Order failed"}

            if not self._dry_run and result.get("market_order"):
                try:
                    from modules.auto_trade.database.repository.context import RepositoryContext

                    market: Dict[str, Any] = result["market_order"]
                    order_id_binance = str(market.get("id") or "")
                    entry_price = float(result.get("entry_price") or market.get("average") or 0.0)

                    order_data: Dict[str, Any] = {
                        "order_id": order_id_binance,
                        "client_order_id": market.get("clientOrderId"),
                        "symbol": _SYMBOL_CODEC.to_db(sym),
                        "side": "LONG" if side_val == "BUY" else "SHORT",
                        "entry_price": entry_price,
                        "amount": float(amount),
                        "leverage": int(leverage),
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "status": "OPEN",
                        "order_source": "PROGRAMMATIC",
                        "execution_mode": "MANUAL",
                    }

                    ctx = RepositoryContext.from_env()
                    ctx.orders.create_order(order_data)

                except Exception as db_err:
                    return {"success": False, "error": f"Order executed but DB persist failed: {db_err}"}

            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}
