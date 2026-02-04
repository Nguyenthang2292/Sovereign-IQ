from dataclasses import dataclass
from typing import TypedDict, Optional, Dict, Any
import logging


@dataclass
class RecoveryState:
    initial_loss: float
    remaining_loss: float
    total_profit_accumulated: float
    recovery_percentage: float
    trades_count: int
    win_streak: int
    is_complete: bool
    estimated_trades_remaining: int


class RecoveryConfig(TypedDict, total=False):
    target_profit_per_trade: float
    max_recovery_trades: int
    max_total_loss: float
    margin_scaling_mode: str
    leverage_scaling_mode: str
    min_leverage: int
    max_leverage: int
    enable_streak_bonus: bool


class GradualRecoveryStrategy:
    def __init__(self, initial_loss: float, config: RecoveryConfig, database: Optional[object] = None):
        self.initial_loss = initial_loss
        self.config = self._validate_config(config)
        self.database = database

        self._state: Dict[str, Any] = {
            "remaining_loss": initial_loss,
            "total_profit_accumulated": 0.0,
            "trades_count": 0,
            "win_streak": 0,
            "is_complete": False,
        }

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Gradual Recovery initialized with ${initial_loss} loss")

    def _validate_config(self, config: RecoveryConfig) -> Dict[str, Any]:
        defaults = {
            "target_profit_per_trade": 5.0,
            "max_recovery_trades": 20,
            "max_total_loss": 2.0 * self.initial_loss,
            "margin_scaling_mode": "fixed",
            "leverage_scaling_mode": "fixed",
            "min_leverage": 2,
            "max_leverage": 10,
            "enable_streak_bonus": False,
        }

        for key, value in defaults.items():
            if key not in config:
                config[key] = value  # type: ignore

        if config.get("margin_scaling_mode") not in ["fixed", "progressive", "adaptive"]:
            raise ValueError(f"Invalid margin_scaling_mode: {config.get('margin_scaling_mode')}")

        if config.get("leverage_scaling_mode") not in ["fixed", "progressive", "adaptive"]:
            raise ValueError(f"Invalid leverage_scaling_mode: {config.get('leverage_scaling_mode')}")

        return dict(config)

    def record_profit(self, profit_amount: float):
        self._state["remaining_loss"] -= profit_amount
        self._state["total_profit_accumulated"] += profit_amount
        self._state["trades_count"] += 1
        self._state["win_streak"] += 1

        if self._state["remaining_loss"] <= 0:
            self._state["remaining_loss"] = 0
            self._state["is_complete"] = True
            self.logger.info("Recovery complete!")

        progress_pct = self.recovery_percentage
        self.logger.info(f"Profit recorded: ${profit_amount:.2f}. Progress: {progress_pct:.1f}%")

        self._persist_state()

    def record_loss(self, loss_amount: float):
        self._state["remaining_loss"] += loss_amount
        self._state["win_streak"] = 0

        self.logger.warning(f"Setback: ${loss_amount:.2f} added to remaining loss")

        max_loss = self.config.get("max_total_loss", 2.0 * self.initial_loss)
        if self._state["remaining_loss"] >= max_loss:
            self.logger.warning(f"Max total loss reached: ${max_loss:.2f}")

        self._persist_state()

    def calculate_next_position_size(self) -> float:
        mode = self.config.get("margin_scaling_mode", "fixed")
        remaining = self._state["remaining_loss"]

        if mode == "fixed":
            return self._calculate_fixed_margin()
        elif mode == "progressive":
            return self._calculate_progressive_margin()
        elif mode == "adaptive":
            return self._calculate_adaptive_margin()

        target_pct = self.config.get("target_profit_per_trade", 5.0) / 100
        return remaining / target_pct if target_pct > 0 else remaining

    def calculate_next_leverage(self) -> int:
        mode = self.config.get("leverage_scaling_mode", "fixed")

        if mode == "fixed":
            return self._calculate_fixed_leverage()
        elif mode == "progressive":
            return self._calculate_progressive_leverage()
        elif mode == "adaptive":
            return self._calculate_adaptive_leverage()

        return self.config.get("min_leverage", 2)

    def estimate_remaining_trades(self) -> int:
        target_pct = self.config.get("target_profit_per_trade", 5.0) / 100
        trades = self._state["trades_count"]

        if trades > 0 and self._state["total_profit_accumulated"] > 0:
            avg_profit = self._state["total_profit_accumulated"] / trades
            estimated_profit = avg_profit
        else:
            estimated_profit = self._state["remaining_loss"] * target_pct

        return int(self._state["remaining_loss"] / estimated_profit) if estimated_profit > 0 else 0

    def should_stop(self) -> bool:
        max_trades = self.config.get("max_recovery_trades", 20)
        max_loss = self.config.get("max_total_loss", 2.0 * self.initial_loss)

        if self._state["trades_count"] >= max_trades:
            self.logger.warning(f"Max trades reached: {max_trades}")
            return True

        if self._state["remaining_loss"] >= max_loss:
            self.logger.warning(f"Max total loss reached: ${max_loss:.2f}")
            return True

        return False

    def get_state(self) -> RecoveryState:
        return RecoveryState(
            initial_loss=self.initial_loss,
            remaining_loss=self._state["remaining_loss"],
            total_profit_accumulated=self._state["total_profit_accumulated"],
            recovery_percentage=self.recovery_percentage,
            trades_count=self._state["trades_count"],
            win_streak=self._state["win_streak"],
            is_complete=self._state["is_complete"],
            estimated_trades_remaining=self.estimate_remaining_trades(),
        )

    def reset(self):
        self._state = {
            "remaining_loss": self.initial_loss,
            "total_profit_accumulated": 0.0,
            "trades_count": 0,
            "win_streak": 0,
            "is_complete": False,
        }
        self.logger.info("Recovery state reset")
        self._clear_state()

    @property
    def is_active(self) -> bool:
        return not self._state["is_complete"]

    @property
    def recovery_percentage(self) -> float:
        if self.initial_loss == 0:
            return 100.0
        recovered = self.initial_loss - self._state["remaining_loss"]
        return (recovered / self.initial_loss) * 100

    @property
    def progress_bar(self) -> str:
        pct = self.recovery_percentage
        filled = int(pct / 10)
        empty = 10 - filled
        return "█" * filled + "░" * empty + f" {pct:.0f}%"

    def _calculate_fixed_margin(self) -> float:
        return self.initial_loss / 10

    def _calculate_progressive_margin(self) -> float:
        base_margin = self.initial_loss / 10
        recovery_pct = self.recovery_percentage / 100
        scaling_factor = 0.5
        return base_margin * (1 + recovery_pct * scaling_factor)

    def _calculate_adaptive_margin(self) -> float:
        base_margin = self.initial_loss / 10
        win_streak = self._state["win_streak"]
        streak_bonus = 0.1 * min(win_streak, 5)

        recovery_pct = self.recovery_percentage / 100
        progress_boost = 0.3 * recovery_pct

        return base_margin * (1 + streak_bonus + progress_boost)

    def _calculate_fixed_leverage(self) -> int:
        return self.config.get("min_leverage", 2)

    def _calculate_progressive_leverage(self) -> int:
        min_lev = self.config.get("min_leverage", 2)
        max_lev = self.config.get("max_leverage", 10)
        recovery_pct = self.recovery_percentage / 100

        leverage = min_lev + int((max_lev - min_lev) * recovery_pct)
        return max(min_lev, min(max_lev, leverage))

    def _calculate_adaptive_leverage(self) -> int:
        min_lev = self.config.get("min_leverage", 2)
        max_lev = self.config.get("max_leverage", 10)
        win_streak = self._state["win_streak"]

        base_lev = min_lev + int((max_lev - min_lev) * (self.recovery_percentage / 100))
        streak_bonus = min(win_streak, 3)

        leverage = base_lev + streak_bonus
        return max(min_lev, min(max_lev, leverage))

    def _persist_state(self):
        if self.database:
            pass

    def _load_state(self):
        if self.database:
            pass

    def _clear_state(self):
        if self.database:
            pass


def create_recovery_plan(initial_loss: float, config: RecoveryConfig) -> Dict[str, Any]:
    target_pct = config.get("target_profit_per_trade", 5.0) / 100
    profit_per_trade = initial_loss * target_pct
    estimated_trades = int(initial_loss / profit_per_trade) if profit_per_trade > 0 else 0

    return {
        "initial_loss": initial_loss,
        "target_profit_per_trade": target_pct * 100,
        "estimated_trades_needed": estimated_trades,
        "suggested_margin_per_trade": initial_loss / 10,
        "suggested_leverage_start": config.get("min_leverage", 2),
        "suggested_leverage_end": config.get("max_leverage", 10),
        "max_total_loss": config.get("max_total_loss", 2.0 * initial_loss),
        "risk_assessment": "Moderate" if estimated_trades < 15 else "High",
    }
