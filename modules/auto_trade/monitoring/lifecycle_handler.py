"""
Position Lifecycle Handler Module

Handles position lifecycle events: opening, closing, profit, loss.
Integrates with Martingale strategy, EventBus, and database for tracking.
"""

from datetime import datetime
from typing import Optional

from modules.auto_trade.monitoring.event_system import EventSystem, EventType
from modules.auto_trade.monitoring.position_monitor import PositionSnapshot
from modules.auto_trade.strategies.martingale import MartingaleStrategy
from modules.common.ui.logging import log_error, log_info, log_warn


class PositionLifecycleHandler:
    """
    Handles position lifecycle events and state transitions.

    Example:
        >>> handler = PositionLifecycleHandler(martingale, database, event_bus)
        >>> handler.on_position_opened(symbol, entry_price, leverage)
        >>> handler.on_position_closed(symbol, pnl, is_profit)
    """

    def __init__(
        self,
        martingale: MartingaleStrategy,
        database=None,  # Optional database for tracking
        event_bus: Optional[EventSystem] = None,  # EventBus for publishing events
    ):
        """
        Initialize PositionLifecycleHandler.

        Args:
            martingale: MartingaleStrategy instance
            database: Optional database for persisting lifecycle events
            event_bus: Optional EventSystem for publishing position events
        """
        self.martingale = martingale
        self.database = database
        self.event_bus = event_bus

        self._open_positions = {}  # symbol -> position data
        self._win_count = 0
        self._loss_count = 0
        self._total_profit = 0.0
        self._total_loss = 0.0

        log_info("PositionLifecycleHandler initialized")

    def on_position_opened(
        self,
        symbol: str,
        entry_price: float,
        leverage: int,
        amount: float,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ):
        """
        Handle position opened event.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            leverage: Leverage used
            amount: Position size in USDT
            tp_price: Take profit price
            sl_price: Stop loss price
        """
        position_data = {
            "symbol": symbol,
            "entry_price": entry_price,
            "leverage": leverage,
            "amount": amount,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "opened_at": datetime.now(),
        }

        self._open_positions[symbol] = position_data

        log_info(f"📈 Position opened: {symbol} @ ${entry_price:.2f}, {leverage}x leverage, ${amount:.2f} USDT")

        if tp_price:
            log_info(f"   TP: ${tp_price:.2f}")
        if sl_price:
            log_info(f"   SL: ${sl_price:.2f}")

        # Persist to database
        if self.database:
            try:
                self.database.record_position_opened(position_data)
            except Exception as e:
                log_error(f"Failed to persist position open to database: {e}")

    def on_position_closed(
        self,
        symbol: str,
        exit_price: float,
        pnl: float,
        is_profit: bool,
    ):
        """
        Handle position closed event.

        Args:
            symbol: Trading symbol
            exit_price: Exit price
            pnl: Realized P&L in USDT
            is_profit: True if profit, False if loss
        """
        # Get position data
        position = self._open_positions.get(symbol)
        if not position:
            log_warn(f"Position closed event for unknown symbol: {symbol}")
            return

        leverage = position.get("leverage", 1)
        entry_price = position.get("entry_price")
        opened_at = position.get("opened_at")
        closed_at = datetime.now()
        duration = (closed_at - opened_at).total_seconds() if opened_at else 0

        log_info(
            f"📊 Position closed: {symbol} @ ${exit_price:.2f}, PnL=${pnl:+.2f} ({'PROFIT' if is_profit else 'LOSS'})"
        )
        log_info(f"   Entry: ${entry_price:.2f}, Duration: {duration / 60:.1f} minutes")

        # Update statistics
        if is_profit:
            self._win_count += 1
            self._total_profit += abs(pnl)
            log_info(f"✅ Profit trade (#{self._win_count})")

            # Reset Martingale on profit
            self.martingale.record_profit(abs(pnl))

        else:
            self._loss_count += 1
            self._total_loss += abs(pnl)
            log_warn(f"❌ Loss trade (#{self._loss_count})")

            # Record loss in Martingale
            self.martingale.record_loss(abs(pnl), leverage)

            # Check if should continue Martingale
            if self.martingale.should_stop():
                log_error("⚠️ Martingale safety limits reached! Consider manual intervention.")

        # Calculate win rate
        total_trades = self._win_count + self._loss_count
        win_rate = (self._win_count / total_trades * 100) if total_trades > 0 else 0

        log_info(
            f"📊 Overall stats: {self._win_count}W / {self._loss_count}L "
            f"({win_rate:.1f}% win rate), "
            f"Total P&L: ${self._total_profit - self._total_loss:+.2f}"
        )

        # Persist to database
        if self.database:
            try:
                self.database.record_position_closed(
                    symbol=symbol,
                    exit_price=exit_price,
                    pnl=pnl,
                    is_profit=is_profit,
                    duration_seconds=duration,
                )
            except Exception as e:
                log_error(f"Failed to persist position close to database: {e}")

        # Publish POSITION_CLOSED event to EventBus for RecoveryManager
        if self.event_bus:
            try:
                self.event_bus.publish(
                    EventType.POSITION_CLOSED,
                    {
                        "symbol": symbol,
                        "pnl": pnl,
                        "is_profit": is_profit,
                        "exit_price": exit_price,
                        "entry_price": entry_price,
                        "leverage": leverage,
                        "duration_seconds": duration,
                        "is_programmatic": True,
                    },
                    source="PositionLifecycleHandler",
                )
            except Exception as e:
                log_error(f"Failed to publish POSITION_CLOSED event: {e}")

        # Remove from open positions
        del self._open_positions[symbol]

    def on_position_update(self, position: PositionSnapshot):
        """
        Handle position update event (from monitor).

        Args:
            position: Current position snapshot
        """
        # Update tracked position data
        if position.symbol in self._open_positions:
            self._open_positions[position.symbol]["last_update"] = datetime.now()
            self._open_positions[position.symbol]["current_pnl"] = position.unrealized_pnl

    def prepare_next_order(self) -> dict:
        """
        Prepare parameters for next order based on Martingale state.

        Returns:
            Dict with recommended leverage and other parameters
        """
        if self.martingale.is_active:
            next_leverage = self.martingale.get_next_leverage()
            recovery_amount = self.martingale.calculate_recovery_amount()

            log_info(f"Next order (Martingale): {next_leverage}x leverage (need to recover ${recovery_amount:.2f})")

            return {
                "leverage": next_leverage,
                "recovery_amount": recovery_amount,
                "martingale_step": self.martingale.current_step + 1,
                "should_stop": self.martingale.should_stop(),
            }
        else:
            # No Martingale active, use initial leverage
            log_info(f"Next order: {self.martingale.initial_leverage}x leverage (initial)")

            return {
                "leverage": self.martingale.initial_leverage,
                "recovery_amount": 0.0,
                "martingale_step": 0,
                "should_stop": False,
            }

    def get_stats(self) -> dict:
        """
        Get lifecycle statistics.

        Returns:
            Dict with win/loss counts, P&L, win rate, etc.
        """
        total_trades = self._win_count + self._loss_count
        win_rate = (self._win_count / total_trades * 100) if total_trades > 0 else 0
        net_pnl = self._total_profit - self._total_loss

        return {
            "total_trades": total_trades,
            "wins": self._win_count,
            "losses": self._loss_count,
            "win_rate": win_rate,
            "total_profit": self._total_profit,
            "total_loss": self._total_loss,
            "net_pnl": net_pnl,
            "open_positions": len(self._open_positions),
            "martingale_active": self.martingale.is_active,
            "martingale_step": self.martingale.current_step,
        }

    def reset_stats(self):
        """Reset all statistics."""
        log_info("Resetting lifecycle statistics")
        self._win_count = 0
        self._loss_count = 0
        self._total_profit = 0.0
        self._total_loss = 0.0
        self.martingale.reset()

    @property
    def win_rate(self) -> float:
        """Get current win rate as percentage."""
        total = self._win_count + self._loss_count
        return (self._win_count / total * 100) if total > 0 else 0.0

    @property
    def net_pnl(self) -> float:
        """Get net P&L (profit - loss)."""
        return self._total_profit - self._total_loss

    @property
    def has_open_positions(self) -> bool:
        """Check if there are any open positions."""
        return len(self._open_positions) > 0
