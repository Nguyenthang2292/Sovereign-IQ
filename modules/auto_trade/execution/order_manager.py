"""
Order Manager Module

Orchestrates the complete order execution flow.
Integrates all components: builder, validator, risk manager, and Binance client.
"""

from typing import Optional

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.auto_trade.execution.binance_client import BinanceClient
from modules.auto_trade.execution.order_builder import OrderBuilder, OrderTicket
from modules.auto_trade.execution.order_validator import OrderValidator
from modules.auto_trade.execution.risk_manager import RiskManager
from modules.common.core.data_fetcher import DataFetcher
from modules.common.ui.logging import log_error, log_info, log_warn


class OrderManager:
    """
    Order Manager orchestrates the complete order execution flow.

    Flow:
        1. Check if any positions are open (via DataFetcher)
        2. If no positions → proceed with order execution
        3. Calculate position size (RiskManager)
        4. Build order ticket (OrderBuilder)
        5. Validate pre-order (OrderValidator)
        6. Execute order (BinanceClient)
        7. Validate post-order (OrderValidator)
        8. Return order result

    Example:
        >>> manager = OrderManager(data_fetcher, api_key, api_secret)
        >>> result = manager.execute_signal(signal)
    """

    def __init__(
        self,
        data_fetcher: DataFetcher,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        dry_run: bool = False,
        balance_percentage: float = 0.95,
        default_leverage: int = 2,
        default_tp_pct: float = 5.0,
        default_sl_pct: float = 50.0,
    ):
        """
        Initialize OrderManager.

        Args:
            data_fetcher: DataFetcher instance
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True
            dry_run: Simulate orders without execution
            balance_percentage: Percentage of balance to use (default: 0.95)
            default_leverage: Default leverage (default: 2x)
            default_tp_pct: Default take profit percentage (default: 5%)
            default_sl_pct: Default stop loss percentage (default: 50%)
        """
        self.data_fetcher = data_fetcher
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.dry_run = dry_run

        # Initialize components
        self.risk_manager = RiskManager(
            data_fetcher=data_fetcher,
            balance_percentage=balance_percentage,
            default_leverage=default_leverage,
        )

        self.order_builder = OrderBuilder(
            default_tp_pct=default_tp_pct,
            default_sl_pct=default_sl_pct,
            default_leverage=default_leverage,
        )

        self.order_validator = OrderValidator()

        self.binance_client = BinanceClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            dry_run=dry_run,
        )

        log_info(
            f"OrderManager initialized ({'DRY RUN' if dry_run else 'LIVE'} mode, {'testnet' if testnet else 'mainnet'})"
        )

    def check_open_positions(self) -> Optional[list]:
        """
        Check if there are any open positions.

        Returns:
            List of open positions or None if error/no positions
        """
        try:
            positions = self.data_fetcher.fetch_binance_futures_positions(
                api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
            )

            if not positions:
                log_info("No open positions found")
                return None

            # Filter for positions with non-zero amount
            open_positions = [p for p in positions if float(p.get("positionAmt", 0)) != 0]

            if open_positions:
                log_info(f"Found {len(open_positions)} open position(s)")
                for pos in open_positions:
                    symbol = pos.get("symbol")
                    amount = pos.get("positionAmt")
                    entry_price = pos.get("entryPrice")
                    unrpc = pos.get("unRealizedProfit")
                    log_info(f"  - {symbol}: amount={amount}, entry=${entry_price}, PnL=${unrpc}")
            else:
                log_info("No active positions (all positions have zero amount)")
                return None

            return open_positions

        except Exception as e:
            log_error(f"Failed to check open positions: {e}", exc_info=True)
            return None

    def execute_signal(
        self,
        signal: FinalSignal,
        force_execution: bool = False,
        leverage_override: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Execute a trading signal.

        Args:
            signal: Final signal from signal pipeline
            force_execution: Force execution even if position exists
            leverage_override: Override default leverage

        Returns:
            Order result dict or None if failed/skipped

        Flow:
            1. Check open positions
            2. Calculate position size
            3. Build order ticket
            4. Fetch current price
            5. Pre-order validation
            6. Execute order
            7. Post-order validation
            8. Return result
        """
        log_info(f"🚀 Executing signal: {signal.symbol} {signal.signal_type}")

        # Step 1: Check for open positions
        if not force_execution:
            open_positions = self.check_open_positions()
            if open_positions:
                log_warn("Open position(s) detected, skipping new order execution")
                log_warn("Use force_execution=True to override this check")
                return None
        else:
            log_warn("Force execution enabled, skipping position check")

        # Step 2: Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
        )

        if not position_size:
            log_error("Failed to calculate position size, aborting")
            return None

        # Step 3: Build order ticket
        order = self.order_builder.build_order(
            signal=signal,
            position_size=position_size,
            leverage=leverage_override,
        )

        log_info(f"Built order ticket: {order.symbol} {order.side} ${order.amount:.2f} @ {order.leverage}x")

        # Step 4: Fetch current price
        try:
            # Use DataFetcher or directly from CCXT
            ticker = self.binance_client.exchange.fetch_ticker(signal.symbol)
            current_price = ticker["last"]
            log_info(f"Current price for {signal.symbol}: ${current_price:,.2f}")
        except Exception as e:
            log_error(f"Failed to fetch current price: {e}", exc_info=True)
            return None

        # Step 5: Pre-order validation
        balance = self.risk_manager.fetch_account_balance(
            api_key=self.api_key, api_secret=self.api_secret, testnet=self.testnet
        )

        if balance is None:
            log_error("Failed to fetch balance for validation, aborting")
            return None

        if not self.order_validator.validate_pre_order(order, balance, current_price):
            log_error("Pre-order validation failed, aborting")
            return None

        # Step 6: Execute order
        log_info(f"Executing order on Binance...")
        order_result = self.binance_client.create_market_order(order)

        if not order_result:
            log_error("Order execution failed")
            return None

        # Step 7: Post-order validation
        if not self.order_validator.validate_post_order(order_result, order):
            log_warn("Post-order validation failed, but order was executed")

        # Step 8: Return result
        log_info(f"✅ Order executed successfully for {signal.symbol}")
        return order_result

    def emergency_stop(self, reason: str):
        """
        Trigger emergency stop to halt all trading.

        Args:
            reason: Reason for emergency stop
        """
        self.risk_manager.trigger_emergency_stop(reason)

    def reset_emergency_stop(self):
        """Reset emergency stop."""
        self.risk_manager.reset_emergency_stop()

    @property
    def is_emergency_stop_active(self) -> bool:
        """Check if emergency stop is active."""
        return self.risk_manager.is_emergency_stop_active
