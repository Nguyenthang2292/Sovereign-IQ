
from typing import TYPE_CHECKING, Optional

import pandas as pd
import pandas as pd

"""
High-Order HMM Strategy Implementation.

This module contains the TrueHighOrderHMMStrategy class that implements the HMMStrategy interface.
"""



if TYPE_CHECKING:
    from modules.hmm.signals.strategy import HMMStrategyResult

from config import (
    HMM_HIGH_ORDER_MAX_ORDER_DEFAULT,
    HMM_HIGH_ORDER_MIN_ORDER_DEFAULT,
    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
)
from modules.hmm.core.high_order.workflow import true_high_order_hmm
from modules.hmm.core.swings.models import BEARISH, BULLISH, NEUTRAL
from modules.hmm.signals.strategy import HMMStrategy


class TrueHighOrderHMMStrategy(HMMStrategy):
    """
    HMM Strategy wrapper for True High-Order HMM.

    Implements HMMStrategy interface to enable registry-based management.
    """

    def __init__(
        self,
        name: str = "true_high_order",
        weight: float = 1.0,
        enabled: bool = True,
        orders_argrelextrema: Optional[int] = None,
        strict_mode: Optional[bool] = None,
        min_order: int = HMM_HIGH_ORDER_MIN_ORDER_DEFAULT,
        max_order: int = HMM_HIGH_ORDER_MAX_ORDER_DEFAULT,
        **kwargs,
    ):
        """
        Initialize True High-Order HMM Strategy.

        Args:
            name: Strategy name (default: "true_high_order")
            weight: Strategy weight for voting (default: 1.0)
            enabled: Whether strategy is enabled (default: True)
            orders_argrelextrema: Order parameter for swing detection
            strict_mode: Use strict mode for swing-to-state conversion
            min_order: Minimum order k for optimization
            max_order: Maximum order k for optimization
            **kwargs: Additional parameters (train_ratio, eval_mode, etc.)
        """

        super().__init__(name=name, weight=weight, enabled=enabled, **kwargs)
        self.orders_argrelextrema = (
            orders_argrelextrema if orders_argrelextrema is not None else HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
        )
        self.strict_mode = strict_mode if strict_mode is not None else HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
        self.min_order = min_order
        self.max_order = max_order
        # Update params with strategy-specific parameters
        self.params.update(
            {
                "orders_argrelextrema": self.orders_argrelextrema,
                "strict_mode": self.strict_mode,
                "min_order": self.min_order,
                "max_order": self.max_order,
            }
        )

    def analyze(self, df: pd.DataFrame, **kwargs) -> "HMMStrategyResult":
        """
        Analyze market data using True High-Order HMM.

        Args:
            df: DataFrame containing OHLCV data
            **kwargs: Additional parameters (may override self.params)

        Returns:
            HMMStrategyResult with signal, probability, state, and metadata
        """
        from modules.hmm.signals.resolution import HOLD, LONG, SHORT
        from modules.hmm.signals.strategy import HMMStrategyResult

        # Merge params with kwargs (kwargs take precedence)
        params = {**self.params, **kwargs}
        train_ratio = params.get("train_ratio", 0.8)
        eval_mode = params.get("eval_mode", True)

        # Run True High-Order HMM analysis
        result = true_high_order_hmm(
            df,
            train_ratio=train_ratio,
            eval_mode=eval_mode,
            orders_argrelextrema=self.orders_argrelextrema,
            strict_mode=self.strict_mode,
            min_order=self.min_order,
            max_order=self.max_order,
        )

        # Convert to HMMStrategyResult
        # Map BULLISH/NEUTRAL/BEARISH to LONG/HOLD/SHORT
        # Note: Values are the same (-1, 0, 1) but explicit mapping improves clarity
        raw_signal = result.next_state_with_high_order_hmm
        signal_map_internal = {BULLISH: LONG, NEUTRAL: HOLD, BEARISH: SHORT}
        signal = signal_map_internal.get(raw_signal, HOLD)

        probability = result.next_state_probability
        state = result.next_state_with_high_order_hmm

        metadata = {
            "duration": result.next_state_duration,
            "train_ratio": train_ratio,
            "eval_mode": eval_mode,
            "min_order": self.min_order,
            "max_order": self.max_order,
        }

        return HMMStrategyResult(signal=signal, probability=probability, state=state, metadata=metadata)
