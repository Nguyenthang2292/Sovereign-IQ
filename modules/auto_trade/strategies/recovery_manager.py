"""
Recovery Manager Module

Orchestrates Gradual Recovery across all trading - GLOBAL scope.
Subscribes to POSITION_CLOSED events via EventBus and automatically
manages the GradualRecoveryStrategy lifecycle.

When a position closes with a loss:
- If recovery is enabled and no active recovery: start new recovery
- If active recovery: add loss to remaining loss

When a position closes with profit:
- If active recovery: record profit toward recovery

Created: 2026-02-06
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from modules.auto_trade.monitoring.event_system import Event, EventSystem, EventType
from modules.auto_trade.strategies.gradual_recovery import (
    GradualRecoveryStrategy,
    RecoveryConfig,
    RecoveryState,
)


class RecoveryManager:
    """
    Manages Gradual Recovery lifecycle for the auto_trade system.

    GLOBAL scope: One recovery sequence applies to ALL trading.
    This simplifies the system and tracks total portfolio recovery.

    Example:
        >>> recovery_manager = RecoveryManager(event_bus, config, enabled=True)
        >>> recovery_manager.start()
        >>> # ... trading happens ...
        >>> recovery_manager.stop()
    """

    def __init__(
        self,
        event_bus: Optional[EventSystem] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = False,
        database=None,
    ):
        """
        Initialize RecoveryManager.

        Args:
            event_bus: EventSystem for subscribing to position events
            config: Recovery configuration dictionary
            enabled: Whether auto-recovery is enabled
            database: Optional database session factory
        """
        self.event_bus = event_bus
        self.config = config or {}
        self._enabled = enabled
        self.database = database

        self._strategy: Optional[GradualRecoveryStrategy] = None
        self._recovery_id: Optional[str] = None
        self._subscribed = False

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"RecoveryManager initialized (enabled={enabled}, "
            f"config keys: {list(self.config.keys())})"
        )

    def start(self):
        """
        Start the RecoveryManager.

        Subscribes to EventBus POSITION_CLOSED events and loads
        any active recovery from database.
        """
        if self._subscribed:
            self.logger.warning("RecoveryManager already started")
            return

        if self.event_bus:
            self.event_bus.subscribe(EventType.POSITION_CLOSED, self._on_position_closed)
            self._subscribed = True
            self.logger.info("RecoveryManager subscribed to POSITION_CLOSED events")

        # Load active recovery from database
        self._load_active_recovery()

    def stop(self):
        """
        Stop the RecoveryManager.

        Unsubscribes from EventBus events.
        """
        if self._subscribed and self.event_bus:
            self.event_bus.unsubscribe(EventType.POSITION_CLOSED, self._on_position_closed)
            self._subscribed = False
            self.logger.info("RecoveryManager unsubscribed from events")

    def set_enabled(self, enabled: bool):
        """
        Enable or disable auto-recovery.

        Args:
            enabled: True to enable, False to disable
        """
        self._enabled = enabled
        self.logger.info(f"RecoveryManager enabled={enabled}")

    def update_config(self, config: Dict[str, Any]):
        """
        Update recovery configuration.

        Args:
            config: New configuration dictionary
        """
        self.config = config
        self.logger.info(f"RecoveryManager config updated: {list(config.keys())}")

        # If strategy exists, update it with new config
        if self._strategy and config:
            self._strategy.config = self._strategy._validate_config(config)

    def get_recovery_parameters(self) -> Dict[str, Any]:
        """
        Get recovery parameters for order execution.

        Returns:
            Dict with:
                - active: Whether recovery is active
                - leverage: Recommended leverage
                - position_size: Recommended position size (margin)
                - remaining_loss: Remaining loss to recover
                - recovery_percentage: Recovery progress
        """
        if not self._strategy or not self._strategy.is_active:
            return {
                "active": False,
                "leverage": self.config.get("min_leverage", 2),
                "position_size": None,
                "remaining_loss": 0.0,
                "recovery_percentage": 0.0,
            }

        return {
            "active": True,
            "leverage": self._strategy.calculate_next_leverage(),
            "position_size": self._strategy.calculate_next_position_size(),
            "remaining_loss": self._strategy._state["remaining_loss"],
            "recovery_percentage": self._strategy.recovery_percentage,
        }

    def get_state(self) -> Optional[RecoveryState]:
        """
        Get current recovery state.

        Returns:
            RecoveryState if active recovery, None otherwise
        """
        if self._strategy:
            return self._strategy.get_state()
        return None

    def manual_start_recovery(self, initial_loss: float) -> bool:
        """
        Manually start a new recovery sequence.

        Args:
            initial_loss: Initial loss amount to recover

        Returns:
            True if started successfully
        """
        if self._strategy and self._strategy.is_active:
            self.logger.warning("Recovery already active, cannot start new one")
            return False

        return self._start_new_recovery(initial_loss)

    def manual_record_profit(self, profit: float):
        """
        Manually record a profit (for testing or manual trades).

        Args:
            profit: Profit amount
        """
        if self._strategy:
            self._strategy.record_profit(profit)
            self._persist_state()
            self.logger.info(f"Manually recorded profit: ${profit:.2f}")

    def manual_record_loss(self, loss: float):
        """
        Manually record a loss (for testing or manual trades).

        Args:
            loss: Loss amount (positive value)
        """
        if self._strategy:
            self._strategy.record_loss(loss)
            self._persist_state()
            self.logger.info(f"Manually recorded loss: ${loss:.2f}")

    def reset(self):
        """Reset the recovery strategy."""
        if self._strategy:
            self._strategy.reset()
            self._persist_state()
            self.logger.info("Recovery reset")

    def cancel(self):
        """Cancel the current recovery."""
        if self._strategy:
            self._strategy = None
            self._recovery_id = None
            self._cancel_in_database()
            self.logger.info("Recovery cancelled")

    @property
    def is_active(self) -> bool:
        """Check if recovery is currently active."""
        return self._strategy is not None and self._strategy.is_active

    @property
    def is_enabled(self) -> bool:
        """Check if auto-recovery is enabled."""
        return self._enabled

    @property
    def recovery_id(self) -> Optional[str]:
        """Get the current recovery ID."""
        return self._recovery_id

    # ==================== Event Handlers ====================

    def _on_position_closed(self, event: Event):
        """
        Handle POSITION_CLOSED event from EventBus.

        Args:
            event: Event containing position close data
        """
        try:
            data = event.data
            symbol = data.get("symbol")
            pnl = data.get("pnl", 0.0)
            is_programmatic = data.get("is_programmatic", True)

            # Only handle programmatic orders
            if not is_programmatic:
                self.logger.debug(f"Ignoring non-programmatic position close: {symbol}")
                return

            self.logger.info(
                f"Position closed event received: {symbol}, PnL=${pnl:.2f}, "
                f"enabled={self._enabled}, active_recovery={self.is_active}"
            )

            if pnl >= 0:
                self._handle_profit(pnl)
            else:
                self._handle_loss(abs(pnl))

        except Exception as e:
            self.logger.error(f"Error handling position closed event: {e}", exc_info=True)

    def _handle_profit(self, profit: float):
        """
        Handle profit from a closed position.

        If recovery is active, record profit toward recovery.

        Args:
            profit: Profit amount
        """
        if not self._strategy:
            self.logger.debug(f"No active recovery, profit ${profit:.2f} not recorded")
            return

        self._strategy.record_profit(profit)
        self._persist_state()

        state = self._strategy.get_state()
        self.logger.info(
            f"Recovery profit recorded: ${profit:.2f}, "
            f"progress: {state.recovery_percentage:.1f}%, "
            f"remaining: ${state.remaining_loss:.2f}"
        )

        if state.is_complete:
            self.logger.info("Recovery COMPLETE! All losses recovered.")
            self._mark_recovery_complete()

    def _handle_loss(self, loss: float):
        """
        Handle loss from a closed position.

        If enabled and no active recovery: start new recovery.
        If active recovery: add loss to remaining loss.

        Args:
            loss: Loss amount (positive value)
        """
        if not self._enabled:
            self.logger.debug(f"Auto-recovery disabled, loss ${loss:.2f} not recorded")
            return

        if not self._strategy:
            # Start new recovery
            self.logger.info(f"Starting new recovery for loss: ${loss:.2f}")
            self._start_new_recovery(loss)
        else:
            # Add to existing recovery
            self._strategy.record_loss(loss)
            self._persist_state()

            state = self._strategy.get_state()
            self.logger.warning(
                f"Recovery setback: ${loss:.2f} added, "
                f"remaining: ${state.remaining_loss:.2f}"
            )

            if self._strategy.should_stop():
                self.logger.error("Recovery safety limits reached!")

    # ==================== Recovery Lifecycle ====================

    def _start_new_recovery(self, initial_loss: float) -> bool:
        """
        Start a new recovery sequence.

        Args:
            initial_loss: Initial loss amount

        Returns:
            True if started successfully
        """
        try:
            self._recovery_id = f"REC_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            recovery_config: RecoveryConfig = {
                "target_profit_per_trade": self.config.get("target_profit_per_trade", 5.0),
                "max_recovery_trades": self.config.get("max_recovery_trades", 20),
                "margin_scaling_mode": self.config.get("margin_scaling_mode", "fixed"),
                "leverage_scaling_mode": self.config.get("leverage_scaling_mode", "fixed"),
                "min_leverage": self.config.get("min_leverage", 2),
                "max_leverage": self.config.get("max_leverage", 10),
                "enable_streak_bonus": self.config.get("enable_streak_bonus", False),
            }

            self._strategy = GradualRecoveryStrategy(
                initial_loss=initial_loss,
                config=recovery_config,
            )

            self._persist_state(create=True)

            self.logger.info(
                f"New recovery started: ID={self._recovery_id}, "
                f"initial_loss=${initial_loss:.2f}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to start recovery: {e}", exc_info=True)
            return False

    def _load_active_recovery(self):
        """Load active recovery from database on startup."""
        if not self.database:
            return

        try:
            from modules.auto_trade.database import get_session
            from modules.auto_trade.database.queries import get_active_gradual_recovery

            with get_session() as session:
                recovery = get_active_gradual_recovery(session, symbol=None)  # GLOBAL
                if recovery:
                    self._recovery_id = recovery.recovery_id
                    config = recovery.get_config() or {}

                    recovery_config: RecoveryConfig = {
                        "target_profit_per_trade": config.get("target_profit_per_trade", 5.0),
                        "max_recovery_trades": config.get("max_recovery_trades", 20),
                        "margin_scaling_mode": config.get("margin_scaling_mode", "fixed"),
                        "leverage_scaling_mode": config.get("leverage_scaling_mode", "fixed"),
                        "min_leverage": config.get("min_leverage", 2),
                        "max_leverage": config.get("max_leverage", 10),
                        "enable_streak_bonus": config.get("enable_streak_bonus", False),
                    }

                    self._strategy = GradualRecoveryStrategy(
                        initial_loss=recovery.initial_loss,
                        config=recovery_config,
                    )

                    # Restore state
                    self._strategy._state["remaining_loss"] = recovery.remaining_loss
                    self._strategy._state["total_profit_accumulated"] = recovery.total_profit_accumulated
                    self._strategy._state["trades_count"] = recovery.trades_count
                    self._strategy._state["win_streak"] = recovery.win_streak
                    self._strategy._state["is_complete"] = recovery.status == "COMPLETE"

                    self.logger.info(
                        f"Loaded active recovery: ID={self._recovery_id}, "
                        f"remaining=${recovery.remaining_loss:.2f}, "
                        f"progress={recovery.recovery_percentage:.1f}%"
                    )

        except Exception as e:
            self.logger.error(f"Error loading active recovery: {e}", exc_info=True)

    def _persist_state(self, create: bool = False):
        """
        Persist recovery state to database.

        Args:
            create: If True, create new record; otherwise update
        """
        if not self.database or not self._strategy or not self._recovery_id:
            return

        try:
            from modules.auto_trade.database import get_session
            from modules.auto_trade.database.queries import (
                create_gradual_recovery,
                update_gradual_recovery,
            )

            state = self._strategy.get_state()

            with get_session() as session:
                if create:
                    create_gradual_recovery(
                        session=session,
                        recovery_id=self._recovery_id,
                        initial_loss=state.initial_loss,
                        config=self._strategy.config,
                        symbol=None,  # GLOBAL
                    )
                else:
                    update_gradual_recovery(
                        session=session,
                        recovery_id=self._recovery_id,
                        remaining_loss=state.remaining_loss,
                        total_profit_accumulated=state.total_profit_accumulated,
                        recovery_percentage=state.recovery_percentage,
                        trades_count=state.trades_count,
                        win_streak=state.win_streak,
                        estimated_trades_remaining=state.estimated_trades_remaining,
                    )

        except Exception as e:
            self.logger.error(f"Error persisting recovery state: {e}", exc_info=True)

    def _mark_recovery_complete(self):
        """Mark recovery as complete in database."""
        if not self.database or not self._recovery_id:
            return

        try:
            from modules.auto_trade.database import get_session
            from modules.auto_trade.database.queries import update_gradual_recovery

            with get_session() as session:
                update_gradual_recovery(
                    session=session,
                    recovery_id=self._recovery_id,
                    status="COMPLETE",
                )

            self.logger.info(f"Recovery {self._recovery_id} marked as COMPLETE")

        except Exception as e:
            self.logger.error(f"Error marking recovery complete: {e}", exc_info=True)

    def _cancel_in_database(self):
        """Cancel recovery in database."""
        if not self.database or not self._recovery_id:
            return

        try:
            from modules.auto_trade.database import get_session
            from modules.auto_trade.database.queries import cancel_gradual_recovery

            with get_session() as session:
                cancel_gradual_recovery(session, self._recovery_id)

            self.logger.info(f"Recovery {self._recovery_id} cancelled in database")

        except Exception as e:
            self.logger.error(f"Error cancelling recovery: {e}", exc_info=True)
