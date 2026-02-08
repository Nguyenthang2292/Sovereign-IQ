"""
Martingale Strategy Module

Implements Martingale position sizing strategy for loss recovery.
Doubles leverage after each loss to recover previous losses.
"""

from dataclasses import dataclass
from typing import Any, Optional

from modules.common.ui.logging import log_error, log_info, log_warn


@dataclass
class MartingaleState:
    """Current state of Martingale progression."""

    current_step: int
    total_loss: float
    last_leverage: int
    next_leverage: int
    max_steps_reached: bool


class MartingaleStrategy:
    """
    Implements Martingale strategy for position sizing.

    After a loss, doubles the leverage for the next trade to recover losses.
    Has built-in safety limits to prevent excessive risk.

    Example:
        >>> martingale = MartingaleStrategy(max_steps=4, max_leverage=16)
        >>> martingale.record_loss(100.0, leverage=2)
        >>> next_leverage = martingale.get_next_leverage()  # Returns 4
    """

    def __init__(
        self,
        initial_leverage: int = 2,
        max_steps: int = 4,
        max_leverage: int = 16,
        max_total_loss: Optional[float] = None,
        database: Any = None,  # Optional database for persistence
    ) -> None:
        """
        Initialize MartingaleStrategy.

        Args:
            initial_leverage: Starting leverage (default: 2x)
            max_steps: Maximum Martingale steps (default: 4)
            max_leverage: Maximum allowed leverage (default: 16x)
            max_total_loss: Optional max total loss before stopping (in USDT)
            database: Optional database for persisting state
        """
        if max_steps < 1 or max_steps > 10:
            raise ValueError(f"max_steps must be 1-10, got {max_steps}")

        self.initial_leverage: int = initial_leverage
        self.max_steps: int = max_steps
        self.max_leverage: int = max_leverage
        self.max_total_loss: Optional[float] = max_total_loss
        self.database: Any = database

        # State
        self._current_step: int = 0
        self._total_loss: float = 0.0
        self._last_leverage: int = initial_leverage
        self._loss_history: list[dict[str, Any]] = []

        log_info(
            f"MartingaleStrategy initialized: "
            f"initial={initial_leverage}x, max_steps={max_steps}, max_leverage={max_leverage}x"
        )

    def record_loss(self, loss_amount: float, leverage: int) -> None:
        """
        Record a loss and update Martingale state.

        Args:
            loss_amount: Loss amount in USDT (positive number)
            leverage: Leverage used for the losing trade
        """
        if loss_amount < 0:
            raise ValueError(f"Loss amount must be positive, got {loss_amount}")

        self._current_step += 1
        self._total_loss += loss_amount
        self._last_leverage = leverage
        self._loss_history.append({"step": self._current_step, "loss": loss_amount, "leverage": leverage})

        log_warn(
            f"📉 Martingale loss recorded: step={self._current_step}, "
            f"loss=${loss_amount:.2f} @ {leverage}x, total_loss=${self._total_loss:.2f}"
        )

        # Check if max limits reached
        if self._current_step >= self.max_steps:
            log_error(f"⚠️ Max Martingale steps ({self.max_steps}) reached! Consider stopping.")

        if self.max_total_loss and self._total_loss >= self.max_total_loss:
            log_error(f"⚠️ Max total loss ${self.max_total_loss:.2f} reached! Stopping Martingale.")

        # Persist to database if available
        if self.database:
            try:
                self.database.record_martingale_loss(
                    step=self._current_step,
                    loss=loss_amount,
                    leverage=leverage,
                    total_loss=self._total_loss,
                )
            except Exception as e:
                log_error(f"Failed to persist Martingale state to database: {e}")

    def record_profit(self, profit_amount: float) -> None:
        """
        Record a profit and reset Martingale state.

        Args:
            profit_amount: Profit amount in USDT
        """
        log_info(f"💰 Profit recorded: ${profit_amount:.2f}. Resetting Martingale counter.")

        recovered_loss: float = min(profit_amount, self._total_loss)
        log_info(f"Recovered ${recovered_loss:.2f} of ${self._total_loss:.2f} total loss")

        # Reset state
        self.reset()

    def get_next_leverage(self) -> int:
        """
        Calculate next leverage for Martingale progression.

        Returns:
            Next leverage to use (doubles from last leverage)

        Formula:
            next_leverage = last_leverage × 2
            Capped at max_leverage
        """
        if self._current_step == 0:
            # No losses yet, use initial leverage
            return self.initial_leverage

        # Double the last leverage
        next_leverage = self._last_leverage * 2

        # Cap at max leverage
        next_leverage = min(next_leverage, self.max_leverage)

        log_info(f"Next Martingale leverage: {next_leverage}x (step {self._current_step + 1})")

        return next_leverage

    def should_stop(self) -> bool:
        """
        Check if Martingale should stop (safety limits reached).

        Returns:
            True if should stop, False if can continue
        """
        # Check max steps
        if self._current_step >= self.max_steps:
            log_warn(f"Max steps ({self.max_steps}) reached")
            return True

        # Check max total loss
        if self.max_total_loss and self._total_loss >= self.max_total_loss:
            log_warn(f"Max total loss ${self.max_total_loss:.2f} reached")
            return True

        # Check if next leverage would exceed max
        next_lev = self.get_next_leverage()
        if next_lev > self.max_leverage:
            log_warn(f"Next leverage {next_lev}x would exceed max {self.max_leverage}x")
            return True

        return False

    def calculate_recovery_amount(self) -> float:
        """
        Calculate amount needed to recover all losses.

        Returns:
            USDT amount needed to break even
        """
        return self._total_loss

    def get_state(self) -> MartingaleState:
        """
        Get current Martingale state.

        Returns:
            MartingaleState with current statistics
        """
        return MartingaleState(
            current_step=self._current_step,
            total_loss=self._total_loss,
            last_leverage=self._last_leverage,
            next_leverage=self.get_next_leverage(),
            max_steps_reached=self._current_step >= self.max_steps,
        )

    def reset(self) -> None:
        """Reset Martingale state to initial."""
        log_info("Resetting Martingale strategy to initial state")

        self._current_step = 0
        self._total_loss = 0.0
        self._last_leverage = self.initial_leverage
        self._loss_history = []

        if self.database:
            try:
                self.database.reset_martingale()
            except Exception as e:
                log_error(f"Failed to reset Martingale in database: {e}")

    @property
    def current_step(self) -> int:
        """Get current Martingale step."""
        return self._current_step

    @property
    def total_loss(self) -> float:
        """Get total accumulated loss."""
        return self._total_loss

    @property
    def is_active(self) -> bool:
        """Check if Martingale is currently active (has losses to recover)."""
        return self._current_step > 0

    @property
    def loss_history(self) -> list[dict[str, Any]]:
        """Get loss history."""
        return self._loss_history.copy()
