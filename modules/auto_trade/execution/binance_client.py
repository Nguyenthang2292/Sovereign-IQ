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
                "recvWindow": 60000,  # 60 seconds tolerance for timestamp difference
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

        # CRITICAL: Force time synchronization with the server BEFORE any authenticated request
        # This resolves Binance -1021 timestamp errors
        try:
            exchange.load_time_difference()
        except Exception:
            pass  # Ignore errors - adjustForTimeDifference will handle it

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
                    delay: float = self.retry_delay * (2**attempt)
                    time.sleep(delay)

        log_error(f"Failed to set leverage after {self.max_retries} attempts")
        return False

    def fetch_ticker(self, symbol: str) -> Optional[dict]:
        """
        Fetch current ticker (last price) for a symbol.
        Used by trailing stop and negative breakeven jobs for mark price.

        Args:
            symbol: Trading symbol (e.g. 'BTCUSDT' or 'BTC/USDT')

        Returns:
            CCXT ticker dict with 'last' and other keys, or None on error.
        """
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            log_error(f"Failed to fetch ticker for {symbol}: {e}")
            return None

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

        symbol: str = order.symbol
        side: str = order.side.lower()  # 'buy' or 'sell'
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
        if not self.set_leverage(symbol, order.leverage):
            log_error(f"Failed to set leverage for {symbol}, aborting order")
            return None

        # Step 2: Get current price to calculate contract amount
        try:
            ticker: dict = self.exchange.fetch_ticker(symbol)
            current_price: float = ticker["last"]
            log_info(f"Current price for {symbol}: ${current_price:,.2f}")

            # Calculate contract amount (for futures)
            # amount_contracts = (amount_usdt × leverage) / current_price
            amount_contracts: float = (amount_usdt * order.leverage) / current_price

            log_info(f"Calculated contract amount: {amount_contracts:.4f} contracts")

        except Exception as e:
            log_error(f"Failed to fetch ticker for {symbol}: {e}")
            return None

        # Step 3: Create market order (pass client_order_id so Binance returns AT_ for DB sync)
        market_order_result: Optional[dict] = None
        params: dict = {}
        if getattr(order, "client_order_id", None):
            params["newClientOrderId"] = order.client_order_id
        for attempt in range(self.max_retries):
            try:
                market_order_result = self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=amount_contracts,
                    params=params,
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
        from modules.auto_trade.execution.order_builder import OrderBuilder

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

        tp_side: str = "sell" if order.side == "BUY" else "buy"  # Opposite side to close position

        for attempt in range(self.max_retries):
            try:
                tp_order: dict = self.exchange.create_order(
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

        sl_side: str = "sell" if order.side == "BUY" else "buy"  # Opposite side to close position

        for attempt in range(self.max_retries):
            try:
                sl_order: dict = self.exchange.create_order(
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
            order_details: dict = self.exchange.fetch_order(order_id, symbol)
            log_info(f"Order verification: {order_details.get('status')}")
            return order_details
        except Exception as e:
            log_error(f"Failed to verify order {order_id}: {e}")
            return None

    def close_position(
        self, symbol: str, side: str, size: float, order_type: str = "market", limit_price: Optional[float] = None
    ) -> Optional[dict]:
        """
        Close a position (full or partial).

        Args:
            symbol: Trading symbol
            side: Position side ('long' or 'short')
            size: Amount to close
            order_type: 'market' or 'limit'
            limit_price: Limit price (only for limit orders)

        Returns:
            Order result dict or None if failed
        """
        if self.dry_run:
            log_info(f"[DRY RUN] Would close {size} of {symbol} {side} position ({order_type})")
            if order_type == "limit" and limit_price:
                log_info(f"  Limit price: ${limit_price:,.2f}")
            return {
                "dry_run": True,
                "symbol": symbol,
                "side": side,
                "size": size,
                "type": order_type,
            }

        # Calculate order side (opposite to position side)
        close_side: str = "sell" if side.lower() == "long" else "buy"

        # Get current price for limit orders
        if order_type == "limit" and not limit_price:
            log_error("Limit price required for limit orders")
            return None

        try:
            log_info(f"Closing {size} of {symbol} {side} position ({order_type})")

            result: dict
            if order_type == "market":
                # Market order
                result = self.exchange.create_order(
                    symbol=symbol, type="market", side=close_side, amount=size, params={"reduceOnly": True}
                )
            else:
                # Limit order
                result = self.exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=close_side,
                    amount=size,
                    price=limit_price,
                    params={"reduceOnly": True},
                )

            log_info(f"✅ Position close order executed: {result.get('id')}")
            return result

        except Exception as e:
            log_error(f"Failed to close position: {e}")
            return None

    def get_position(self, symbol: str) -> Optional[dict]:
        """
        Fetch current position for a symbol.

        Args:
            symbol: Trading symbol (any format: BTCUSDT, BTC/USDT, BTC/USDT:USDT)

        Returns:
            Position dict or None if not found
        """
        if self.dry_run:
            return {"symbol": symbol, "contracts": 0, "side": "long", "notional": 0}

        try:
            from modules.common.domain.symbols import normalize_symbol_key

            # Normalize input symbol for comparison (BTCUSDT, BTC/USDT, BTC/USDT:USDT -> BTCUSDT)
            normalized_input = normalize_symbol_key(symbol)

            # fetch_positions might return a list of all positions or filtered by symbols depending on exchange
            positions: list = self.exchange.fetch_positions([symbol])
            for pos in positions:
                pos_symbol = pos.get("symbol", "")
                # CCXT futures returns "BTC/USDT:USDT", normalize for comparison
                if normalize_symbol_key(pos_symbol) == normalized_input:
                    return pos
            return None
        except Exception as e:
            log_error(f"Failed to fetch position for {symbol}: {e}")
            return None

    def modify_take_profit(
        self, symbol: str, position_id: Optional[str], take_profit_price: Optional[float] = None
    ) -> Optional[dict]:
        """
        Modify take profit order for a position.

        Args:
            symbol: Trading symbol
            position_id: Position ID (if available)
            take_profit_price: New TP price (None to cancel existing TP)

        Returns:
            Order result dict or None if failed
        """
        if self.dry_run:
            if take_profit_price:
                log_info(f"[DRY RUN] Would modify TP for {symbol} to ${take_profit_price:,.2f}")
            else:
                log_info(f"[DRY RUN] Would cancel TP for {symbol}")
            return {"dry_run": True, "symbol": symbol}

        try:
            # 1. Get current position to determine side and amount
            position: Optional[dict] = self.get_position(symbol)
            if not position:
                log_error(f"No open position found for {symbol}")
                return None

            amount: float = abs(float(position.get("contracts", 0) or position.get("info", {}).get("positionAmt", 0)))
            if amount == 0:
                log_warn(f"Position size is 0 for {symbol}, cannot modify TP")
                return None

            side: str = str(position.get("side") or "")  # 'long' or 'short'
            if not side:
                # Fallback if CCXT doesn't normalize side
                amt: float = float(position.get("info", {}).get("positionAmt", 0))
                side = "long" if amt > 0 else "short"

            tp_side: str = "sell" if side == "long" else "buy"

            # 2. Fetch open orders to find existing TP
            open_orders: list = self.exchange.fetch_open_orders(symbol)
            cancelled_count: int = 0

            # 3. Cancel existing TP orders
            for order in open_orders:
                order_type: str = order.get("type", "").lower()
                # TP orders are usually TAKE_PROFIT or TAKE_PROFIT_MARKET
                if "take_profit" in order_type:
                    try:
                        self.exchange.cancel_order(order["id"], symbol)
                        cancelled_count += 1
                        log_info(f"Cancelled existing TP order: {order['id']}")
                    except Exception as e:
                        log_warn(f"Failed to cancel TP order {order['id']}: {e}")

            # 4. Place new TP order if price provided
            if take_profit_price:
                log_info(f"Setting new TP for {symbol} at ${take_profit_price:,.2f}")

                tp_order: dict = self.exchange.create_order(
                    symbol=symbol,
                    type="take_profit_market",
                    side=tp_side,
                    amount=amount,
                    params={
                        "stopPrice": take_profit_price,
                        "reduceOnly": True,
                    },
                )
                log_info(f"✅ Take Profit order updated at ${take_profit_price:,.2f}")
                return tp_order
            else:
                log_info(f"TP cancelled for {symbol}")
                return {"symbol": symbol, "cancelled_tp_count": cancelled_count}

        except Exception as e:
            log_error(f"Failed to modify TP: {e}")
            return None

    def modify_stop_loss(
        self, symbol: str, position_id: Optional[str], stop_loss_price: Optional[float] = None
    ) -> Optional[dict]:
        """
        Modify stop loss order for a position.

        Args:
            symbol: Trading symbol
            position_id: Position ID (if available)
            stop_loss_price: New SL price (None to cancel existing SL)

        Returns:
            Order result dict or None if failed
        """
        if self.dry_run:
            if stop_loss_price:
                log_info(f"[DRY RUN] Would modify SL for {symbol} to ${stop_loss_price:,.2f}")
            else:
                log_info(f"[DRY RUN] Would cancel SL for {symbol}")
            return {"dry_run": True, "symbol": symbol}

        try:
            # 1. Get current position to determine side and amount
            position: Optional[dict] = self.get_position(symbol)
            if not position:
                log_error(f"No open position found for {symbol}")
                return None

            amount: float = abs(float(position.get("contracts", 0) or position.get("info", {}).get("positionAmt", 0)))
            if amount == 0:
                log_warn(f"Position size is 0 for {symbol}, cannot modify SL")
                return None

            side: str = str(position.get("side") or "")  # 'long' or 'short'
            if not side:
                # Fallback
                amt: float = float(position.get("info", {}).get("positionAmt", 0))
                side = "long" if amt > 0 else "short"

            sl_side: str = "sell" if side == "long" else "buy"

            # 2. Fetch open orders to find existing SL
            open_orders: list = self.exchange.fetch_open_orders(symbol)
            cancelled_count: int = 0

            # 3. Cancel existing SL orders
            for order in open_orders:
                order_type: str = order.get("type", "").lower()
                # SL orders are usually STOP or STOP_MARKET
                if "stop" in order_type:
                    try:
                        self.exchange.cancel_order(order["id"], symbol)
                        cancelled_count += 1
                        log_info(f"Cancelled existing SL order: {order['id']}")
                    except Exception as e:
                        log_warn(f"Failed to cancel SL order {order['id']}: {e}")

            # 4. Place new SL order if price provided
            if stop_loss_price:
                log_info(f"Setting new SL for {symbol} at ${stop_loss_price:,.2f}")

                sl_order: dict = self.exchange.create_order(
                    symbol=symbol,
                    type="stop_market",
                    side=sl_side,
                    amount=amount,
                    params={
                        "stopPrice": stop_loss_price,
                        "reduceOnly": True,
                    },
                )
                log_info(f"✅ Stop Loss order updated at ${stop_loss_price:,.2f}")
                return sl_order
            else:
                log_info(f"SL cancelled for {symbol}")
                return {"symbol": symbol, "cancelled_sl_count": cancelled_count}

        except Exception as e:
            log_error(f"Failed to modify SL: {e}")
            return None

    def modify_tp_sl(
        self,
        symbol: str,
        position_id: Optional[str] = None,
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Modify both take profit and stop loss for a position.

        Args:
            symbol: Trading symbol
            position_id: Position ID (if available)
            take_profit_price: New TP price (None to keep current)
            stop_loss_price: New SL price (None to keep current)

        Returns:
            Combined result dict or None if failed
        """
        results: dict = {}

        if take_profit_price is not None:
            tp_result: Optional[dict] = self.modify_take_profit(symbol, position_id, take_profit_price)
            results["tp_result"] = tp_result

        if stop_loss_price is not None:
            sl_result: Optional[dict] = self.modify_stop_loss(symbol, position_id, stop_loss_price)
            results["sl_result"] = sl_result

        return results if results else None

    def modify_margin(self, symbol: str, amount: float, type: int = 1, position_side: str = "BOTH") -> Optional[dict]:
        """
        Modify position margin (for Isolated Margin).

        Args:
            symbol: Trading symbol
            amount: Amount of margin to add (or remove)
            type: 1 = Add Position Margin, 2 = Reduce Position Margin
            position_side: 'BOTH', 'LONG', or 'SHORT' (Default 'BOTH' for One-way Mode)

        Returns:
            Result dict or None if failed
        """
        if self.dry_run:
            action: str = "Add" if type == 1 else "Reduce"
            log_info(f"[DRY RUN] Would {action} margin for {symbol} by ${amount:,.2f}")
            return {"dry_run": True, "symbol": symbol, "amount": amount, "type": type}

        try:
            log_info(f"Modifying margin for {symbol}: amount=${amount}, type={type}")

            # Note: CCXT may not have a unified method for this, so we use the implicit API
            # fapiPrivatePostPositionMargin maps to POST /fapi/v1/positionMargin
            params: dict = {
                "symbol": self.exchange.market_id(symbol),
                "amount": amount,
                "type": type,
                "positionSide": position_side,
            }

            response: dict = self.exchange.fapiPrivatePostPositionMargin(params)
            log_info(f"✅ Margin modified for {symbol}. New amount: {response.get('amount')}")
            return response

        except Exception as e:
            log_error(f"Failed to modify margin: {e}")
            return None

    def cancel_open_orders(self, symbol: str) -> Optional[dict]:
        """
        Cancel all open orders for a symbol (TP, SL, limit orders).

        Args:
            symbol: Trading symbol

        Returns:
            Cancel result dict with success count
        """
        if self.dry_run:
            log_info(f"[DRY RUN] Would cancel all open orders for {symbol}")
            return {"dry_run": True, "symbol": symbol, "cancelled_count": 0}

        try:
            log_info(f"Cancelling all open orders for {symbol}")

            # Get all open orders
            open_orders: list = self.exchange.fetch_open_orders(symbol)

            cancelled_count: int = 0
            for order in open_orders:
                try:
                    self.exchange.cancel_order(order["id"], symbol)
                    cancelled_count += 1
                    log_info(f"  Cancelled order: {order['id']} ({order.get('type', 'N/A')})")
                except Exception as e:
                    log_warn(f"Failed to cancel order {order['id']}: {e}")

            log_info(f"✅ Cancelled {cancelled_count} open orders for {symbol}")
            return {"symbol": symbol, "cancelled_count": cancelled_count, "success": True}

        except Exception as e:
            log_error(f"Failed to cancel open orders: {e}")
            return {"symbol": symbol, "cancelled_count": 0, "success": False}
