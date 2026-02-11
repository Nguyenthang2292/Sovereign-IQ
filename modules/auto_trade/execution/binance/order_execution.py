"""
Order Execution Module

Handles market order execution with TP/SL placement.
"""

import time
from typing import Any, Optional, cast

import ccxt

from modules.auto_trade.execution.order_builder import OrderTicket, OrderBuilder
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

            # Calculate contract amount
            amount_contracts: float = (amount_usdt * order.leverage) / current_price
            log_info(f"Calculated contract amount: {amount_contracts:.4f} contracts")

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
                        side=side,
                        amount=amount_contracts,
                        params=params,
                    ),
                )

                log_info(f"✅ Market order executed: {market_order_result.get('id') if market_order_result else 'Unknown'}")
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

        # Step 5: Place TP/SL orders
        tp_order_result: Optional[dict] = self._place_take_profit(order, amount_contracts)
        sl_order_result: Optional[dict] = self._place_stop_loss(order, amount_contracts)

        # Step 6: Return combined result
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

    def _place_take_profit(self, order: OrderTicket, amount: float) -> Optional[dict]:
        """
        Place take profit order.

        Args:
            order: Order ticket with TP price
            amount: Contract amount

        Returns:
            TP order result or None
        """
        if not order.take_profit_price:
            log_warn("No TP price set, skipping TP order")
            return None

        tp_side: str = "sell" if order.side == "BUY" else "buy"

        for attempt in range(self.max_retries):
            try:
                tp_order = cast(
                    dict,
                    self.exchange.create_order(
                        symbol=order.symbol,
                        type=cast(Any, "take_profit_market"),
                        side=tp_side,
                        amount=amount,
                        params={
                            "stopPrice": order.take_profit_price,
                            "reduceOnly": True,
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

    def _place_stop_loss(self, order: OrderTicket, amount: float) -> Optional[dict]:
        """
        Place stop loss order.

        Args:
            order: Order ticket with SL price
            amount: Contract amount

        Returns:
            SL order result or None
        """
        if not order.stop_loss_price:
            log_warn("No SL price set, skipping SL order")
            return None

        sl_side: str = "sell" if order.side == "BUY" else "buy"

        for attempt in range(self.max_retries):
            try:
                sl_order = cast(
                    dict,
                    self.exchange.create_order(
                        symbol=order.symbol,
                        type=cast(Any, "stop_market"),
                        side=sl_side,
                        amount=amount,
                        params={
                            "stopPrice": order.stop_loss_price,
                            "reduceOnly": True,
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
