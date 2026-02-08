"""
Break-Even Manager Module (WebSocket Version)

Monitors position drawdown and moves TP to break-even when drawdown reaches 30% of account.
Uses real-time mark prices from WebSocket for accurate, instant drawdown detection.

Key improvements over REST polling:
- Instant drawdown detection (no 5s delay)
- Real-time mark price updates (~3s on Binance)
- Prevents slippage losses from delayed detection
"""

import asyncio
import logging
from typing import Optional

from modules.auto_trade.monitoring.position_monitor import PositionSnapshot
from modules.auto_trade.websocket.client import BinanceWebSocketClient

logger = logging.getLogger(__name__)


class BreakEvenManager:
    """
    Manages break-even protection for positions using WebSocket.

    When position drawdown reaches 30% of account, moves TP to break-even price
    to protect capital.

    Uses real-time mark prices for instant detection.

    Example:
        >>> be_mgr = BreakEvenManager(ws_client, drawdown_threshold=30.0)
        >>> await be_mgr.check_and_move_breakeven(position, account_balance)
    """

    def __init__(
        self,
        ws_client: BinanceWebSocketClient,
        drawdown_threshold_percent: float = 30.0,
        database=None,  # Optional database for tracking
    ):
        """
        Initialize BreakEvenManager.

        Args:
            ws_client: WebSocket client instance
            drawdown_threshold_percent: Drawdown % to trigger BE move (default: 30%)
            database: Optional database instance for tracking BE moves
        """
        self.ws_client = ws_client
        self.drawdown_threshold = drawdown_threshold_percent
        self.database = database
        self._be_moved_positions: set[str] = set()  # Track symbols where BE was moved

        logger.info(f"BreakEvenManager initialized (threshold={drawdown_threshold_percent}%, WebSocket mode)")

    async def check_and_move_breakeven(
        self,
        position: PositionSnapshot,
        account_balance: float,
        dry_run: bool = False,
    ) -> bool:
        """
        Check if position should move to break-even and execute if needed.

        Args:
            position: Current position snapshot
            account_balance: Total account balance in USDT
            dry_run: If True, simulate without actually moving TP

        Returns:
            True if BE was moved (or would be in dry-run), False otherwise
        """
        symbol = position.symbol

        # Check if BE already moved for this position
        if symbol in self._be_moved_positions:
            return False

        # Calculate drawdown relative to account balance
        drawdown_usdt = abs(position.unrealized_pnl)
        drawdown_percent_of_account = (drawdown_usdt / account_balance) * 100

        # Check if drawdown threshold reached
        if drawdown_percent_of_account < self.drawdown_threshold:
            return False

        # Threshold reached, move TP to break-even
        logger.warning(
            f"⚠️  Drawdown threshold reached for {symbol}: "
            f"${drawdown_usdt:.2f} ({drawdown_percent_of_account:.1f}% of account)"
        )

        if dry_run:
            logger.info(f"[DRY RUN] Would move TP to break-even for {symbol}")
            return True

        # Move TP to break-even
        success = await self._move_tp_to_breakeven(position)

        if success:
            # Mark as moved
            self._be_moved_positions.add(symbol)

            # Update database if available
            if self.database:
                try:
                    self.database.mark_be_moved(symbol)
                except Exception as e:
                    logger.error(f"Failed to update database for BE move: {e}")

            logger.info(f"✅ Moved TP to break-even for {symbol}")
            return True
        else:
            logger.error(f"❌ Failed to move TP to break-even for {symbol}")
            return False

    async def _move_tp_to_breakeven(self, position: PositionSnapshot) -> bool:
        """
        Move take profit order to break-even price.

        Args:
            position: Position snapshot

        Returns:
            True if successful, False otherwise
        """
        symbol = position.symbol
        entry_price = position.entry_price
        position_amt = position.position_amt

        try:
            # Cancel existing TP orders first
            logger.info(f"Canceling existing TP orders for {symbol}...")
            await self._cancel_tp_orders(symbol)

            # Determine side for TP order (opposite of position)
            tp_side = "sell" if position.side == "long" else "buy"

            # Create new TP order at break-even price using WebSocket API
            logger.info(f"Placing new TP order at break-even price ${entry_price:.2f}...")

            # Create order via REST (ccxt.pro doesn't support order creation via WS yet)
            tp_order = await self.ws_client.exchange.create_order(
                symbol=symbol,
                type="TAKE_PROFIT_MARKET",
                side=tp_side,
                amount=position_amt,
                params={"stopPrice": entry_price, "reduceOnly": True},
            )

            logger.info(f"✅ TP order placed: {tp_order.get('id')}")
            return True

        except Exception as e:
            logger.error(f"Error moving TP to break-even: {e}", exc_info=True)
            return False

    async def _cancel_tp_orders(self, symbol: str) -> bool:
        """
        Cancel all open take profit orders for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            True if successful, False otherwise
        """
        try:
            # Fetch open orders via REST
            open_orders = await self.ws_client.exchange.fetch_open_orders(symbol)

            # Find and cancel TP orders
            tp_orders = [
                order
                for order in open_orders
                if order.get("type", "").upper() in ["TAKE_PROFIT", "TAKE_PROFIT_MARKET"]
            ]

            for order in tp_orders:
                order_id = order.get("id")
                try:
                    await self.ws_client.exchange.cancel_order(order_id, symbol)
                    logger.info(f"Cancelled TP order: {order_id}")
                except Exception as e:
                    logger.warning(f"Failed to cancel order {order_id}: {e}")

            return True

        except Exception as e:
            logger.error(f"Error canceling TP orders: {e}")
            return False

    def reset_position(self, symbol: str):
        """
        Reset BE moved flag for a symbol (e.g., when position closes).

        Args:
            symbol: Trading symbol
        """
        if symbol in self._be_moved_positions:
            self._be_moved_positions.remove(symbol)
            logger.info(f"Reset BE flag for {symbol}")

    def is_be_moved(self, symbol: str) -> bool:
        """Check if BE was moved for a symbol."""
        return symbol in self._be_moved_positions

    @property
    def moved_positions(self) -> set:
        """Get set of symbols where BE was moved."""
        return self._be_moved_positions.copy()


class BreakEvenMonitor:
    """
    Automated break-even monitor that watches positions and moves TP automatically.

    This continuously monitors positions via callbacks and automatically triggers
    break-even protection when threshold is reached.

    Example:
        >>> monitor = BreakEvenMonitor(ws_client, position_monitor, account_balance)
        >>> await monitor.start()
    """

    def __init__(
        self,
        ws_client: BinanceWebSocketClient,
        position_monitor,  # PositionMonitor instance
        account_balance: float,
        drawdown_threshold_percent: float = 30.0,
        dry_run: bool = False,
    ):
        """
        Initialize BreakEvenMonitor.

        Args:
            ws_client: WebSocket client instance
            position_monitor: PositionMonitor instance
            account_balance: Initial account balance
            drawdown_threshold_percent: Drawdown threshold (default: 30%)
            dry_run: Dry run mode
        """
        self.ws_client = ws_client
        self.position_monitor = position_monitor
        self.account_balance = account_balance
        self.drawdown_threshold = drawdown_threshold_percent
        self.dry_run = dry_run

        self.be_manager = BreakEvenManager(ws_client, drawdown_threshold_percent)

        self._running = False

        logger.info(f"BreakEvenMonitor initialized (threshold={drawdown_threshold_percent}%)")

    async def start(self):
        """Start monitoring for break-even conditions."""
        if self._running:
            logger.warning("BreakEvenMonitor is already running")
            return

        self._running = True

        # Register callback with position monitor
        self.position_monitor.add_callback(self._handle_position_update)

        # Register balance callback to keep account balance updated
        self.ws_client.on_balance_update(self._handle_balance_update)

        logger.info("✅ BreakEvenMonitor started (WebSocket mode)")

    async def stop(self):
        """Stop monitoring."""
        self._running = False
        logger.info("⏹️  BreakEvenMonitor stopped")

    def _handle_position_update(self, position: PositionSnapshot):
        """
        Handle position update from position monitor.

        Args:
            position: Position snapshot
        """
        if not self._running:
            return

        # Check break-even condition asynchronously
        asyncio.create_task(self._check_breakeven(position))

    async def _check_breakeven(self, position: PositionSnapshot):
        """
        Check and trigger break-even if needed.

        Args:
            position: Position snapshot
        """
        try:
            await self.be_manager.check_and_move_breakeven(position, self.account_balance, self.dry_run)
        except Exception as e:
            logger.error(f"Error checking break-even for {position.symbol}: {e}")

    def _handle_balance_update(self, balance: dict):
        """
        Handle balance update from WebSocket.

        Args:
            balance: Balance dict from ccxt.pro
        """
        if not self._running:
            return

        # Update account balance
        usdt_balance = balance.get("USDT", {})
        total = usdt_balance.get("total", 0)

        if total > 0:
            self.account_balance = total
            logger.debug(f"Updated account balance: ${total:.2f}")

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running
