"""
Order Validator Module

Validates orders before and after execution.
Ensures orders meet all safety and regulatory requirements.
"""

from typing import Optional

from modules.auto_trade.execution.order_builder import OrderTicket
from modules.common.ui.logging import log_error, log_info, log_warn


class OrderValidator:
    """
    Validates orders before and after execution.

    Example:
        >>> validator = OrderValidator()
        >>> is_valid = validator.validate_pre_order(order, balance=1000.0, current_price=50000.0)
    """

    def __init__(
        self,
        min_position_size: float = 0.0,
        max_position_size: Optional[float] = None,
        max_leverage: int = 125,
        max_slippage_pct: float = 2.0,
    ):
        """
        Initialize OrderValidator.

        Args:
            min_position_size: Minimum position size in USDT (default: 0.0, disabled)
            max_position_size: Optional maximum position size in USDT
            max_leverage: Maximum allowed leverage
            max_slippage_pct: Maximum acceptable slippage percentage
        """
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_slippage_pct = max_slippage_pct

    def validate_pre_order(
        self,
        order: OrderTicket,
        balance: float,
        current_price: float,
        market_info: Optional[dict] = None,
    ) -> bool:
        """
        Validate order before execution.

        Args:
            order: Order ticket to validate
            balance: Available account balance
            current_price: Current market price
            market_info: Optional market information

        Returns:
            True if order is valid, False otherwise

        Checks:
            1. Sufficient balance
            2. Valid leverage
            3. Market is open (if market_info provided)
            4. Symbol exists (if market_info provided)
            5. Price sanity check
            6. Position size limits
        """
        log_info(f"Validating pre-order for {order.symbol}...")

        # Check 1: Sufficient balance
        if not self._validate_balance(order, balance):
            return False

        # Check 2: Valid leverage
        if not self._validate_leverage(order):
            return False

        # Check 3: Price sanity check
        if not self._validate_price(current_price):
            return False

        # Check 4: Position size limits
        if not self._validate_position_size(order):
            return False

        # Check 5: Market is open (if market_info available)
        if market_info and not self._validate_market_open(market_info):
            return False

        # Check 6: Symbol exists (if market_info available)
        if market_info and not self._validate_symbol_exists(order.symbol, market_info):
            return False

        log_info(f"✅ Pre-order validation passed for {order.symbol}")
        return True

    def validate_post_order(
        self,
        order_result: dict,
        expected_order: OrderTicket,
        max_slippage_pct: Optional[float] = None,
    ) -> bool:
        """
        Validate order after execution.

        Args:
            order_result: Order result from exchange
            expected_order: Expected order parameters
            max_slippage_pct: Optional slippage override

        Returns:
            True if order executed correctly, False otherwise

        Checks:
            1. Order was filled
            2. TP/SL orders placed
            3. Slippage within acceptable range
            4. Position opened correctly
        """
        log_info("Validating post-order execution...")

        # Check 1: Market order filled
        market_order = order_result.get("market_order")
        if not market_order:
            log_error("No market order result found")
            return False

        status = market_order.get("status")
        if status not in ["closed", "filled"]:
            log_error(f"Market order not filled, status: {status}")
            return False

        # Check 2: Entry price exists
        entry_price = order_result.get("entry_price")
        if not entry_price or entry_price <= 0:
            log_error(f"Invalid entry price: {entry_price}")
            return False

        # Check 3: TP order placed
        tp_order = order_result.get("take_profit_order")
        if expected_order.take_profit_price and not tp_order:
            log_warn("Take profit order was not placed")

        # Check 4: SL order placed
        sl_order = order_result.get("stop_loss_order")
        if expected_order.stop_loss_price and not sl_order:
            log_warn("Stop loss order was not placed")

        # Check 5: Slippage check (if expected price available)
        if expected_order.entry_price:
            if not self._validate_slippage(
                expected_order.entry_price,
                entry_price,
                max_slippage_pct or self.max_slippage_pct,
            ):
                return False

        log_info("✅ Post-order validation passed")
        return True

    def _validate_balance(self, order: OrderTicket, balance: float) -> bool:
        """Validate sufficient balance."""
        required_margin = order.amount / order.leverage

        if balance < required_margin:
            log_error(f"Insufficient balance: required ${required_margin:.2f}, available ${balance:.2f}")
            return False

        return True

    def _validate_leverage(self, order: OrderTicket) -> bool:
        """Validate leverage is within limits."""
        if order.leverage < 1:
            log_error(f"Leverage must be >= 1, got {order.leverage}")
            return False

        if order.leverage > self.max_leverage:
            log_error(f"Leverage {order.leverage}x exceeds maximum {self.max_leverage}x")
            return False

        return True

    def _validate_price(self, price: float) -> bool:
        """Validate price is positive and reasonable."""
        if price <= 0:
            log_error(f"Invalid price: {price}")
            return False

        # TODO: Add price range checks (e.g., not 1000x expected price)
        return True

    def _validate_position_size(self, order: OrderTicket) -> bool:
        """Validate position size is within limits."""
        if order.amount < self.min_position_size:
            log_error(f"Position size ${order.amount:.2f} is below minimum ${self.min_position_size:.2f}")
            return False

        if self.max_position_size and order.amount > self.max_position_size:
            log_error(f"Position size ${order.amount:.2f} exceeds maximum ${self.max_position_size:.2f}")
            return False

        return True

    def _validate_market_open(self, market_info: dict) -> bool:
        """Validate market is open for trading."""
        is_active = market_info.get("active", True)
        if not is_active:
            log_error(f"Market is not active: {market_info.get('symbol')}")
            return False

        return True

    def _validate_symbol_exists(self, symbol: str, market_info: dict) -> bool:
        """Validate symbol exists in market."""
        market_symbol = market_info.get("symbol")
        if market_symbol != symbol:
            log_error(f"Symbol mismatch: expected {symbol}, got {market_symbol}")
            return False

        return True

    def _validate_slippage(self, expected_price: float, actual_price: float, max_slippage_pct: float) -> bool:
        """Validate slippage is within acceptable range."""
        slippage_pct: float = abs(actual_price - expected_price) / expected_price * 100

        if slippage_pct > max_slippage_pct:
            log_error(f"Slippage {slippage_pct:.2f}% exceeds maximum {max_slippage_pct:.2f}%")
            return False

        log_info(f"Slippage check passed: {slippage_pct:.2f}%")
        return True
