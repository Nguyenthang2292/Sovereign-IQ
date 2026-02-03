"""
Binance Client Module

Handles Binance Futures API integration for order execution.
Implements market orders with TP/SL, rate limiting, error handling.
"""

import time
from typing import Optional

import ccxt

from modules.auto_trade.execution.order_builder import OrderTicket
from modules.common.ui.logging import log_error, log_info, log_warn


class BinanceClient:
    """
    CCXT-based Binance Futures client for order execution.

    Example:
        >>> client = BinanceClient(api_key, api_secret, testnet=True)
        >>> order_result = client.create_market_order(order_ticket)
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        enable_rate_limiting: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        dry_run: bool = False,
    ):
        """
        Initialize BinanceClient.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use demo environment if True (uses demo-fapi.binance.com)
                    Note: Binance now uses demo API endpoints instead of old testnet
            enable_rate_limiting: Enable CCXT rate limiting
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Initial delay between retries (exponential backoff)
            dry_run: If True, simulate orders without executing
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dry_run = dry_run

        # Initialize CCXT exchange
        self.exchange = self._initialize_exchange(enable_rate_limiting)

    def _initialize_exchange(self, enable_rate_limiting: bool) -> ccxt.binance:
        """
        Initialize CCXT Binance exchange instance.

        Args:
            enable_rate_limiting: Enable rate limiting

        Returns:
            CCXT Binance exchange instance
        """
        config = {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": enable_rate_limiting,
            "options": {
                "defaultType": "future",  # Use USDT-M futures
                "adjustForTimeDifference": True,
            },
        }

        if self.testnet:
            # Binance Futures Demo Account (NEW - replaces old testnet)
            # REST base URL for demo: https://demo-fapi.binance.com
            # CRITICAL: Must override ALL futures endpoints (fapiPublic, fapiPrivate, fapiPrivateV2, etc.)
            # because the balance/position calls use fapiPrivateV2, not just "private"
            config["urls"] = {
                "api": {
                    # Override ALL futures endpoints for demo
                    "fapiPublic": "https://demo-fapi.binance.com/fapi/v1",
                    "fapiPublicV2": "https://demo-fapi.binance.com/fapi/v2",
                    "fapiPublicV3": "https://demo-fapi.binance.com/fapi/v3",
                    "fapiPrivate": "https://demo-fapi.binance.com/fapi/v1",
                    "fapiPrivateV2": "https://demo-fapi.binance.com/fapi/v2",
                    "fapiPrivateV3": "https://demo-fapi.binance.com/fapi/v3",
                    "fapiData": "https://demo-fapi.binance.com/futures/data",
                }
            }
            log_info("Initialized Binance Demo client (uses demo-fapi.binance.com)")
        else:
            # Production or Demo Account
            # Note: Binance demo accounts use production endpoints (fapi.binance.com)
            # with special demo API keys. No special URL configuration needed.
            log_info("Initialized Binance Live/Demo client (uses production endpoints)")

        exchange = ccxt.binance(config)
        return exchange

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            leverage: Leverage multiplier

        Returns:
            True if successful, False otherwise
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
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)

        log_error(f"Failed to set leverage after {self.max_retries} attempts")
        return False

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

        Order Flow:
            1. Set leverage
            2. Create market order
            3. Verify order execution
            4. Place TP/SL orders
            5. Return order details
        """
        # Update API keys if provided
        if api_key and api_secret:
            self.exchange.apiKey = api_key
            self.exchange.secret = api_secret

        symbol = order.symbol
        side = order.side.lower()  # 'buy' or 'sell'
        amount_usdt = order.amount

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
        if not self.set_leverage(symbol, order.leverage):
            log_error(f"Failed to set leverage for {symbol}, aborting order")
            return None

        # Step 2: Get current price to calculate contract amount
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker["last"]
            log_info(f"Current price for {symbol}: ${current_price:,.2f}")

            # Calculate contract amount (for futures)
            # amount_contracts = (amount_usdt × leverage) / current_price
            amount_contracts = (amount_usdt * order.leverage) / current_price

            log_info(f"Calculated contract amount: {amount_contracts:.4f} contracts")

        except Exception as e:
            log_error(f"Failed to fetch ticker for {symbol}: {e}", exc_info=True)
            return None

        # Step 3: Create market order
        market_order_result = None
        for attempt in range(self.max_retries):
            try:
                market_order_result = self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=amount_contracts,
                )

                log_info(f"✅ Market order executed: {market_order_result.get('id')}")
                break

            except Exception as e:
                log_error(
                    f"Market order failed (attempt {attempt + 1}/{self.max_retries}): {e}",
                    exc_info=True,
                )
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    log_warn(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    log_error(f"Market order failed after {self.max_retries} attempts")
                    return None

        if not market_order_result:
            return None

        # Step 4: Get filled price
        filled_price = market_order_result.get("average") or current_price
        log_info(f"Order filled at price: ${filled_price:,.2f}")

        # Update order with entry price
        from modules.auto_trade.execution.order_builder import OrderBuilder

        builder = OrderBuilder()
        order = builder.update_order_with_entry(order, filled_price)

        # Step 5: Place TP/SL orders
        tp_order_result = self._place_take_profit(order, amount_contracts)
        sl_order_result = self._place_stop_loss(order, amount_contracts)

        # Step 6: Return combined result
        return {
            "market_order": market_order_result,
            "entry_price": filled_price,
            "take_profit_order": tp_order_result,
            "stop_loss_order": sl_order_result,
            "order_ticket": order.to_dict(),
        }

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

        tp_side = "sell" if order.side == "BUY" else "buy"  # Opposite side to close position

        for attempt in range(self.max_retries):
            try:
                tp_order = self.exchange.create_order(
                    symbol=order.symbol,
                    type="take_profit_market",  # TP Market order
                    side=tp_side,
                    amount=amount,
                    params={
                        "stopPrice": order.take_profit_price,
                        "reduceOnly": True,
                    },
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

        sl_side = "sell" if order.side == "BUY" else "buy"  # Opposite side to close position

        for attempt in range(self.max_retries):
            try:
                sl_order = self.exchange.create_order(
                    symbol=order.symbol,
                    type="stop_market",  # SL Market order
                    side=sl_side,
                    amount=amount,
                    params={
                        "stopPrice": order.stop_loss_price,
                        "reduceOnly": True,
                    },
                )

                log_info(f"✅ Stop Loss order placed at ${order.stop_loss_price:,.2f}")
                return sl_order

            except Exception as e:
                log_warn(f"SL order failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))

        log_error("Failed to place SL order")
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
            order_details = self.exchange.fetch_order(order_id, symbol)
            log_info(f"Order verification: {order_details.get('status')}")
            return order_details
        except Exception as e:
            log_error(f"Failed to verify order {order_id}: {e}")
            return None
