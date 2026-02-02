"""
Break-Even Manager Module

Monitors position drawdown and moves TP to break-even when drawdown reaches 30% of account.
Integrates with position monitor and order management system.
"""

from typing import Optional

import ccxt

from modules.auto_trade.monitoring.position_monitor import PositionSnapshot
from modules.common.ui.logging import log_error, log_info, log_warn


class BreakEvenManager:
    """
    Manages break-even protection for positions.

    When position drawdown reaches 30% of account, moves TP to break-even price
    to protect capital.

    Example:
        >>> be_mgr = BreakEvenManager(exchange, drawdown_threshold=30.0)
        >>> be_mgr.check_and_move_breakeven(position, account_balance)
    """

    def __init__(
        self,
        exchange: ccxt.binance,
        drawdown_threshold_percent: float = 30.0,
        database=None,  # Optional database for tracking
    ):
        """
        Initialize BreakEvenManager.

        Args:
            exchange: CCXT Binance exchange instance
            drawdown_threshold_percent: Drawdown % to trigger BE move (default: 30%)
            database: Optional database instance for tracking BE moves
        """
        self.exchange = exchange
        self.drawdown_threshold = drawdown_threshold_percent
        self.database = database
        self._be_moved_positions = set()  # Track symbols where BE was moved

        log_info(f"BreakEvenManager initialized (threshold={drawdown_threshold_percent}%)")

    def check_and_move_breakeven(
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
        log_warn(
            f"⚠️ Drawdown threshold reached for {symbol}: "
            f"${drawdown_usdt:.2f} ({drawdown_percent_of_account:.1f}% of account)"
        )

        if dry_run:
            log_info(f"[DRY RUN] Would move TP to break-even for {symbol}")
            return True

        # Move TP to break-even
        success = self._move_tp_to_breakeven(position)

        if success:
            # Mark as moved
            self._be_moved_positions.add(symbol)

            # Update database if available
            if self.database:
                try:
                    self.database.mark_be_moved(symbol)
                except Exception as e:
                    log_error(f"Failed to update database for BE move: {e}")

            log_info(f"✅ Moved TP to break-even for {symbol}")
            return True
        else:
            log_error(f"❌ Failed to move TP to break-even for {symbol}")
            return False

    def _move_tp_to_breakeven(self, position: PositionSnapshot) -> bool:
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
            log_info(f"Canceling existing TP orders for {symbol}...")
            self._cancel_tp_orders(symbol)

            # Determine side for TP order (opposite of position)
            tp_side = "sell" if position.side == "LONG" else "buy"

            # Create new TP order at break-even price
            log_info(f"Placing new TP order at break-even price ${entry_price:.2f}...")

            tp_order = self.exchange.create_order(
                symbol=symbol,
                type="take_profit_market",
                side=tp_side,
                amount=position_amt,
                params={"stopPrice": entry_price, "reduceOnly": True},
            )

            log_info(f"✅ TP order placed: {tp_order.get('id')}")
            return True

        except Exception as e:
            log_error(f"Error moving TP to break-even: {e}", exc_info=True)
            return False

    def _cancel_tp_orders(self, symbol: str) -> bool:
        """
        Cancel all open take profit orders for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            True if successful, False otherwise
        """
        try:
            # Fetch open orders
            open_orders = self.exchange.fetch_open_orders(symbol)

            # Find and cancel TP orders
            tp_orders = [order for order in open_orders if order.get("type") in ["take_profit", "take_profit_market"]]

            for order in tp_orders:
                order_id = order.get("id")
                try:
                    self.exchange.cancel_order(order_id, symbol)
                    log_info(f"Canceled TP order: {order_id}")
                except Exception as e:
                    log_warn(f"Failed to cancel order {order_id}: {e}")

            return True

        except Exception as e:
            log_error(f"Error canceling TP orders: {e}")
            return False

    def reset_position(self, symbol: str):
        """
        Reset BE moved flag for a symbol (e.g., when position closes).

        Args:
            symbol: Trading symbol
        """
        if symbol in self._be_moved_positions:
            self._be_moved_positions.remove(symbol)
            log_info(f"Reset BE flag for {symbol}")

    def is_be_moved(self, symbol: str) -> bool:
        """Check if BE was moved for a symbol."""
        return symbol in self._be_moved_positions

    @property
    def moved_positions(self) -> set:
        """Get set of symbols where BE was moved."""
        return self._be_moved_positions.copy()
