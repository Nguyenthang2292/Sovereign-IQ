"""
Binance Client Module

Main client that orchestrates all Binance operations.
"""

from typing import Optional

import ccxt

from modules.auto_trade.execution.binance.exchange_setup import ExchangeSetup
from modules.auto_trade.execution.binance.order_execution import OrderExecution
from modules.auto_trade.execution.binance.order_management import OrderManagement
from modules.auto_trade.execution.binance.position_management import PositionManagement
from modules.auto_trade.execution.order_builder import OrderTicket
from modules.auto_trade.security.secret_string import SecretString


class BinanceClient:
    """
    CCXT-based Binance Futures client for order execution.

    Refactored into sub-modules for better maintainability:
    - ExchangeSetup: Initialize CCXT exchange
    - OrderExecution: Market orders with TP/SL
    - PositionManagement: Position operations
    - OrderManagement: TP/SL modification, order cancellation

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
            testnet: Use demo environment if True
            enable_rate_limiting: Enable CCXT rate limiting
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Initial delay between retries (exponential backoff)
            dry_run: If True, simulate orders without executing
        """
        self.api_key = SecretString(api_key)
        self.api_secret = SecretString(api_secret)
        self.testnet = testnet
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dry_run = dry_run

        # Initialize exchange
        self.exchange: ccxt.binance = ExchangeSetup.initialize_exchange(
            api_key=self.api_key.get_secret_value(),
            api_secret=self.api_secret.get_secret_value(),
            testnet=testnet,
            enable_rate_limiting=enable_rate_limiting,
        )

        # Initialize sub-modules
        self.order_execution = OrderExecution(
            exchange=self.exchange,
            max_retries=max_retries,
            retry_delay=retry_delay,
            dry_run=dry_run,
        )
        self.position_management = PositionManagement(
            exchange=self.exchange,
            dry_run=dry_run,
        )
        self.order_management = OrderManagement(
            exchange=self.exchange,
            max_retries=max_retries,
            retry_delay=retry_delay,
            dry_run=dry_run,
        )

    # ========== Delegation Methods ==========
    # Delegate to sub-modules for backward compatibility

    def create_market_order(
        self, order: OrderTicket, api_key: Optional[str] = None, api_secret: Optional[str] = None
    ) -> Optional[dict]:
        """Create a market order. Delegates to OrderExecution."""
        return self.order_execution.create_market_order(order, api_key, api_secret)

    def fetch_ticker(self, symbol: str) -> Optional[dict]:
        """Fetch ticker. Delegates to OrderExecution."""
        return self.order_execution.fetch_ticker(symbol)

    def verify_order(self, order_id: str, symbol: str) -> Optional[dict]:
        """Verify order. Delegates to OrderExecution."""
        return self.order_execution.verify_order(order_id, symbol)

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage. Delegates to OrderExecution."""
        return self.order_execution._set_leverage(symbol, leverage)

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position. Delegates to PositionManagement."""
        return self.position_management.get_position(symbol)

    def close_position(
        self, symbol: str, side: str, size: float, order_type: str = "market", limit_price: Optional[float] = None
    ) -> Optional[dict]:
        """Close position. Delegates to PositionManagement."""
        return self.position_management.close_position(symbol, side, size, order_type, limit_price)

    def modify_margin(self, symbol: str, amount: float, type: int = 1, position_side: str = "BOTH") -> Optional[dict]:
        """Modify margin. Delegates to PositionManagement."""
        return self.position_management.modify_margin(symbol, amount, type, position_side)

    def modify_take_profit(
        self, symbol: str, position_id: Optional[str], take_profit_price: Optional[float] = None
    ) -> Optional[dict]:
        """Modify TP. Delegates to OrderManagement."""
        return self.order_management.modify_take_profit(symbol, position_id, take_profit_price)

    def modify_stop_loss(
        self, symbol: str, position_id: Optional[str], stop_loss_price: Optional[float] = None
    ) -> Optional[dict]:
        """Modify SL. Delegates to OrderManagement."""
        return self.order_management.modify_stop_loss(symbol, position_id, stop_loss_price)

    def modify_tp_sl(
        self,
        symbol: str,
        position_id: Optional[str] = None,
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
    ) -> Optional[dict]:
        """Modify TP/SL. Delegates to OrderManagement."""
        return self.order_management.modify_tp_sl(symbol, position_id, take_profit_price, stop_loss_price)

    def cancel_open_orders(self, symbol: str) -> Optional[dict]:
        """Cancel open orders. Delegates to OrderManagement."""
        return self.order_management.cancel_open_orders(symbol)
