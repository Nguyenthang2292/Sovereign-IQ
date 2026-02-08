"""
Order Builder Module

Responsible for building properly formatted order tickets for Binance Futures.
Calculates amount, TP, SL based on risk manager input and signal.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from modules.auto_trade.core.signal_selector import FinalSignal


@dataclass
class OrderTicket:
    """Order ticket structure containing all order parameters."""

    # Required fields (no defaults)
    symbol: str
    side: Literal["BUY", "SELL"]  # BUY = LONG, SELL = SHORT
    amount: float  # Position size in quote currency (USDT)
    leverage: int

    # Optional fields (with defaults)
    order_type: Literal["MARKET"] = "MARKET"
    entry_price: Optional[float] = None  # Will be set after execution
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_percentage: float = 5.0  # Default 5%
    stop_loss_percentage: float = 50.0  # Default 50%
    client_order_id: Optional[str] = None  # AT_ prefix for DB sync and Binance identification

    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "amount": self.amount,
            "leverage": self.leverage,
            "entry_price": self.entry_price,
            "take_profit_price": self.take_profit_price,
            "stop_loss_price": self.stop_loss_price,
            "tp_percentage": self.take_profit_percentage,
            "sl_percentage": self.stop_loss_percentage,
        }


class OrderBuilder:
    """
    Builds order tickets from signals and risk parameters.

    Example:
        >>> signal = FinalSignal(symbol="BTC/USDT", signal_type="LONG", ...)
        >>> builder = OrderBuilder(default_tp_pct=5.0, default_sl_pct=50.0, default_leverage=2)
        >>> order = builder.build_order(signal, position_size=100.0)
    """

    def __init__(
        self,
        default_tp_pct: float = 5.0,
        default_sl_pct: float = 50.0,
        default_leverage: int = 2,
    ):
        """
        Initialize OrderBuilder.

        Args:
            default_tp_pct: Default take profit percentage (default: 5.0%)
            default_sl_pct: Default stop loss percentage (default: 50.0%)
            default_leverage: Default leverage (default: 2x)
        """
        self.default_tp_pct = default_tp_pct
        self.default_sl_pct = default_sl_pct
        self.default_leverage = default_leverage

    def build_order(
        self,
        signal: FinalSignal,
        position_size: float,
        leverage: Optional[int] = None,
        tp_pct: Optional[float] = None,
        sl_pct: Optional[float] = None,
    ) -> OrderTicket:
        """
        Build an order ticket from a signal.

        Args:
            signal: Final signal from signal selector
            position_size: Position size in USDT (calculated by RiskManager)
            leverage: Optional leverage override
            tp_pct: Optional take profit percentage override
            sl_pct: Optional stop loss percentage override

        Returns:
            OrderTicket ready for execution

        Raises:
            ValueError: If position_size <= 0
        """
        if position_size <= 0:
            raise ValueError(f"Position size must be positive, got {position_size}")

        # Determine side
        side: Literal["BUY", "SELL"] = "BUY" if signal.signal_type == "LONG" else "SELL"

        # Use defaults if not overridden
        leverage = leverage or self.default_leverage
        tp_pct = tp_pct or self.default_tp_pct
        sl_pct = sl_pct or self.default_sl_pct

        # Build order ticket
        order = OrderTicket(
            symbol=signal.symbol,
            side=side,
            order_type="MARKET",
            amount=position_size,
            leverage=leverage,
            take_profit_percentage=tp_pct,
            stop_loss_percentage=sl_pct,
        )

        return order

    def calculate_tp_sl_prices(self, order: OrderTicket, entry_price: float) -> tuple[float, float]:
        """
        Calculate take profit and stop loss prices based on entry price.

        Args:
            order: Order ticket
            entry_price: Entry price (current market price)

        Returns:
            Tuple of (take_profit_price, stop_loss_price)

        Formulas:
            - LONG:
                TP = entry_price × (1 + tp_percentage / 100)
                SL = entry_price × (1 - sl_percentage / 100)
            - SHORT:
                TP = entry_price × (1 - tp_percentage / 100)
                SL = entry_price × (1 + sl_percentage / 100)
        """
        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {entry_price}")

        tp_percentage = order.take_profit_percentage
        sl_percentage = order.stop_loss_percentage

        if order.side == "BUY":  # LONG
            tp_price = entry_price * (1 + tp_percentage / 100.0)
            sl_price = entry_price * (1 - sl_percentage / 100.0)
        else:  # SELL (SHORT)
            tp_price = entry_price * (1 - tp_percentage / 100.0)
            sl_price = entry_price * (1 + sl_percentage / 100.0)

        # Ensure SL > 0
        if sl_price <= 0:
            raise ValueError(
                f"Calculated SL price is invalid: {sl_price} (entry={entry_price}, sl_pct={sl_percentage})"
            )

        return tp_price, sl_price

    def update_order_with_entry(self, order: OrderTicket, entry_price: float) -> OrderTicket:
        """
        Update order ticket with entry price and calculated TP/SL prices.

        Args:
            order: Order ticket to update
            entry_price: Entry price (current market price)

        Returns:
            Updated order ticket
        """
        tp_price, sl_price = self.calculate_tp_sl_prices(order, entry_price)

        order.entry_price = entry_price
        order.take_profit_price = tp_price
        order.stop_loss_price = sl_price

        return order
