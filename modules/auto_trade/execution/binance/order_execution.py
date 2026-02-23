"""
Order Execution Module

Handles market order execution with TP/SL placement.
"""

import time
from typing import Any, Optional, cast

import ccxt

from modules.auto_trade.execution.binance.order_management import _ccxt_futures_symbol, _fetch_all_open_orders
from modules.auto_trade.execution.order_builder import OrderBuilder, OrderTicket
from modules.common.ui.logging import log_error, log_info, log_warn


class OrderExecution:
    """
    Handles order execution operations.
    """

    def __init__(
        self,
        exchange: ccxt.binance,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        dry_run: bool = False,
    ):
        """
        Initialize OrderExecution.

        Args:
            exchange: CCXT exchange instance
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries
            dry_run: Simulate orders without executing
        """
        self.exchange = exchange
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dry_run = dry_run

    def create_market_order(
        self, order: OrderTicket, api_key: Optional[str] = None, api_secret: Optional[str] = None
    ) -> Optional[dict]:
        """
        Create a market order on Binance Futures.

        Args:
            order: Order ticket with all parameters
            api_key: Optional API key override
            api_secret: Optional API secret override

        Returns:
            Order result dict or None if failed
        """
        # Update API keys if provided
        if api_key and api_secret:
            self.exchange.apiKey = api_key
            self.exchange.secret = api_secret

        symbol: str = order.symbol
        side: str = order.side.lower()
        amount_usdt: float = order.amount
        side = cast(Any, side)

        log_info(f"Creating {side.upper()} order for {symbol}: ${amount_usdt:.2f} USDT @ {order.leverage}x leverage")

        # Dry run mode
        if self.dry_run:
            log_info("[DRY RUN] Would execute the following order:")
            log_info(f"  Symbol: {symbol}")
            log_info(f"  Side: {side.upper()}")
            log_info(f"  Amount: ${amount_usdt:.2f} USDT")
            log_info(f"  Leverage: {order.leverage}x")
            log_info(f"  TP: {order.take_profit_price} ({order.take_profit_percentage}%)")
            log_info(f"  SL: {order.stop_loss_price} ({order.stop_loss_percentage}%)")
            return {
                "dry_run": True,
                "symbol": symbol,
                "side": side,
                "amount": amount_usdt,
                "leverage": order.leverage,
            }

        # Step 1: Set leverage
        if not self._set_leverage(symbol, order.leverage):
            log_error(f"Failed to set leverage for {symbol}, aborting order")
            return None

        # Step 2: Get current price to calculate contract amount
        try:
            ticker = cast(dict, self.exchange.fetch_ticker(symbol))
            current_price: float = ticker["last"]
            log_info(f"Current price for {symbol}: ${current_price:,.2f}")

            # Calculate contract amount (notional / price)
            amount_contracts: float = (amount_usdt * order.leverage) / current_price
            log_info(f"Calculated contract amount: {amount_contracts:.4f} contracts")

            # ── Validate against Binance minimum qty / step size ──────────────────
            try:
                import math

                markets = self.exchange.load_markets()
                ccxt_sym = _ccxt_futures_symbol(self.exchange, symbol)
                market_info = markets.get(ccxt_sym) or markets.get(symbol) or {}
                limits = market_info.get("limits") or {}
                precision = market_info.get("precision") or {}

                min_amount: float = float((limits.get("amount") or {}).get("min") or 0.0)
                amount_step: float = float(precision.get("amount") or 0.0)

                # Round DOWN to valid step size to avoid exchange precision rejection
                if amount_step > 0:
                    amount_contracts = math.floor(amount_contracts / amount_step) * amount_step
                    dp = len(str(amount_step).rstrip("0").split(".")[-1]) if "." in str(amount_step) else 0
                    amount_contracts = round(amount_contracts, dp)
                    log_info(f"Rounded contract amount (step={amount_step}): {amount_contracts} contracts")

                # Reject below minimum BEFORE submitting to exchange
                if min_amount > 0 and amount_contracts < min_amount:
                    notional_needed = (min_amount * current_price) / order.leverage
                    log_warn(
                        f"[{symbol}] Order skipped: calculated {amount_contracts:.6f} contracts "
                        f"is below exchange minimum {min_amount}. "
                        f"Need ≥ ${notional_needed:.2f} USDT balance at {order.leverage}x leverage "
                        f"(current usable: ${amount_usdt:.2f} USDT)."
                    )
                    return None

            except Exception as min_exc:
                log_warn(f"Could not validate minimum qty for {symbol}: {min_exc} — proceeding anyway")

        except Exception as e:
            log_error(f"Failed to fetch ticker for {symbol}: {e}")
            return None

        # Step 3: Create market order
        market_order_result: Optional[dict] = None
        params: dict = {}
        if getattr(order, "client_order_id", None):
            params["newClientOrderId"] = order.client_order_id

        for attempt in range(self.max_retries):
            try:
                market_order_result = cast(
                    dict,
                    self.exchange.create_order(
                        symbol=symbol,
                        type="market",
                        side=cast(Any, side),
                        amount=amount_contracts,
                        params=params,
                    ),
                )

                log_info(
                    f"✅ Market order executed: {market_order_result.get('id') if market_order_result else 'Unknown'}"
                )
                break

            except Exception as e:
                log_error(f"Market order failed (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    delay: float = self.retry_delay * (2**attempt)
                    log_warn(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    log_error(f"Market order failed after {self.max_retries} attempts")
                    return None

        if not market_order_result:
            return None

        # Step 4: Get filled price
        filled_price: float = market_order_result.get("average") or current_price
        log_info(f"Order filled at price: ${filled_price:,.2f}")

        # Update order with entry price
        builder = OrderBuilder()
        order = builder.update_order_with_entry(order, filled_price)

        # Step 5: Cancel any existing TP/SL conditional orders before placing new ones
        # This prevents duplicate conditional orders (the root cause of the 10-order bug)
        ccxt_symbol: str = _ccxt_futures_symbol(self.exchange, order.symbol)
        self._cancel_existing_tp_sl(ccxt_symbol)

        # Step 6: Place TP/SL orders using the correct futures symbol
        tp_order_result: Optional[dict] = self._place_take_profit(order, amount_contracts, ccxt_symbol)
        sl_order_result: Optional[dict] = self._place_stop_loss(order, amount_contracts, ccxt_symbol)

        # Step 7: Return combined result
        return {
            "market_order": market_order_result,
            "entry_price": filled_price,
            "take_profit_order": tp_order_result,
            "stop_loss_order": sl_order_result,
            "order_ticket": order.to_dict(),
        }

    def _set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading symbol
            leverage: Leverage multiplier

        Returns:
            True if successful
        """
        if self.dry_run:
            log_info(f"[DRY RUN] Would set leverage {leverage}x for {symbol}")
            return True

        for attempt in range(self.max_retries):
            try:
                self.exchange.set_leverage(leverage, symbol)
                log_info(f"✅ Set leverage {leverage}x for {symbol}")
                return True
            except Exception as e:
                log_warn(f"Failed to set leverage (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay: float = self.retry_delay * (2**attempt)
                    time.sleep(delay)

        log_error(f"Failed to set leverage after {self.max_retries} attempts")
        return False

    def _cancel_existing_tp_sl(self, ccxt_symbol: str) -> None:
        """
        Cancel all existing TP/SL conditional orders for a symbol.
        This prevents duplicate conditional orders when placing new TP/SL.

        Args:
            ccxt_symbol: CCXT-formatted futures symbol (e.g. DOGE/USDT:USDT)
        """
        if self.dry_run:
            return

        try:
            open_orders: list = _fetch_all_open_orders(self.exchange, ccxt_symbol)
            for order in open_orders:
                info = order.get("info") or {}
                order_type = (info.get("type") or info.get("origType") or order.get("type") or "").upper()
                # Cancel TP and SL conditional orders
                if "TAKE_PROFIT" in order_type or "STOP" in order_type:
                    try:
                        self.exchange.cancel_order(order["id"], ccxt_symbol)
                        log_info(f"Cancelled existing conditional order: {order['id']} ({order_type})")
                    except Exception as e:
                        log_warn(f"Failed to cancel conditional order {order['id']}: {e}")
        except Exception as e:
            log_warn(f"Could not fetch/cancel existing orders for {ccxt_symbol}: {e}")

    def _place_take_profit(
        self, order: OrderTicket, amount: float, ccxt_symbol: Optional[str] = None
    ) -> Optional[dict]:
        """
        Place take profit order.

        Args:
            order: Order ticket with TP price
            amount: Contract amount
            ccxt_symbol: Optional pre-resolved CCXT futures symbol

        Returns:
            TP order result or None
        """
        if not order.take_profit_price:
            log_warn("No TP price set, skipping TP order")
            return None

        tp_side: str = "sell" if order.side == "BUY" else "buy"
        symbol: str = ccxt_symbol or _ccxt_futures_symbol(self.exchange, order.symbol)

        for attempt in range(self.max_retries):
            try:
                tp_order = cast(
                    dict,
                    self.exchange.create_order(
                        symbol=symbol,
                        type=cast(Any, "TAKE_PROFIT_MARKET"),
                        side=cast(Any, tp_side),
                        amount=amount,
                        params={
                            "stopPrice": order.take_profit_price,
                            "reduceOnly": True,
                            "workingType": "MARK_PRICE",
                        },
                    ),
                )

                log_info(f"✅ Take Profit order placed at ${order.take_profit_price:,.2f}")
                return tp_order

            except Exception as e:
                log_warn(f"TP order failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))

        log_error("Failed to place TP order")
        return None

    def _place_stop_loss(self, order: OrderTicket, amount: float, ccxt_symbol: Optional[str] = None) -> Optional[dict]:
        """
        Place stop loss order.

        Args:
            order: Order ticket with SL price
            amount: Contract amount
            ccxt_symbol: Optional pre-resolved CCXT futures symbol

        Returns:
            SL order result or None
        """
        if not order.stop_loss_price:
            log_warn("No SL price set, skipping SL order")
            return None

        sl_side: str = "sell" if order.side == "BUY" else "buy"
        symbol: str = ccxt_symbol or _ccxt_futures_symbol(self.exchange, order.symbol)

        for attempt in range(self.max_retries):
            try:
                sl_order = cast(
                    dict,
                    self.exchange.create_order(
                        symbol=symbol,
                        type=cast(Any, "STOP_MARKET"),
                        side=cast(Any, sl_side),
                        amount=amount,
                        params={
                            "stopPrice": order.stop_loss_price,
                            "reduceOnly": True,
                            "workingType": "MARK_PRICE",
                        },
                    ),
                )

                log_info(f"✅ Stop Loss order placed at ${order.stop_loss_price:,.2f}")
                return sl_order

            except Exception as e:
                log_warn(f"SL order failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))

        log_error("Failed to place SL order")
        return None

    def fetch_ticker(self, symbol: str) -> Optional[dict]:
        """
        Fetch current ticker for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            CCXT ticker dict or None on error
        """
        try:
            return cast(dict, self.exchange.fetch_ticker(symbol))
        except Exception as e:
            log_error(f"Failed to fetch ticker for {symbol}: {e}")
            return None

    def verify_order(self, order_id: str, symbol: str) -> Optional[dict]:
        """
        Verify order execution by fetching order details.

        Args:
            order_id: Order ID to verify
            symbol: Trading symbol

        Returns:
            Order details or None if not found
        """
        try:
            order_details = cast(dict, self.exchange.fetch_order(order_id, symbol))
            log_info(f"Order verification: {order_details.get('status')}")
            return order_details
        except Exception as e:
            log_error(f"Failed to verify order {order_id}: {e}")
            return None
