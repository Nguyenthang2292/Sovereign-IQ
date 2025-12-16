"""
HMM-Swings Strategy Implementation.

This module contains the SwingsHMMStrategy class that implements the HMMStrategy interface.
"""

from typing import Optional, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from modules.hmm.signals.strategy import HMMStrategyResult

from modules.hmm.core.swings.workflow import hmm_swings
from modules.hmm.core.swings.models import HMM_SWINGS, BULLISH, NEUTRAL, BEARISH
from modules.hmm.signals.strategy import HMMStrategy
from config import (
    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
)


class SwingsHMMStrategy(HMMStrategy):
    """
    HMM Strategy wrapper for Basic HMM with swings.
    
    Implements HMMStrategy interface to enable registry-based management.
    """
    
    def __init__(
        self,
        name: str = "swings",
        weight: float = 1.0,
        enabled: bool = True,
        orders_argrelextrema: Optional[int] = None,
        strict_mode: Optional[bool] = None,
        **kwargs
    ):
        """
        Initialize Swings HMM Strategy.
        
        Args:
            name: Strategy name (default: "swings")
            weight: Strategy weight for voting (default: 1.0)
            enabled: Whether strategy is enabled (default: True)
            orders_argrelextrema: Order parameter for swing detection
            strict_mode: Use strict mode for swing-to-state conversion
            **kwargs: Additional parameters (train_ratio, eval_mode, etc.)
        """
        from modules.hmm.signals.resolution import LONG, HOLD, SHORT
        
        super().__init__(name=name, weight=weight, enabled=enabled, **kwargs)
        self.orders_argrelextrema = (
            orders_argrelextrema
            if orders_argrelextrema is not None
            else HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
        )
        self.strict_mode = (
            strict_mode
            if strict_mode is not None
            else HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
        )
        # Update params with strategy-specific parameters
        self.params.update({
            'orders_argrelextrema': self.orders_argrelextrema,
            'strict_mode': self.strict_mode
        })
    
    def analyze(self, df: pd.DataFrame, **kwargs) -> 'HMMStrategyResult':
        """
        Analyze market data using Basic HMM with swings.
        
        Args:
            df: DataFrame containing OHLCV data
            **kwargs: Additional parameters (may override self.params)
            
        Returns:
            HMMStrategyResult with signal, probability, state, and metadata
        """
        from modules.hmm.signals.strategy import HMMStrategyResult
        from modules.hmm.signals.resolution import LONG, HOLD, SHORT
        
        # Merge params with kwargs (kwargs take precedence)
        params = {**self.params, **kwargs}
        train_ratio = params.get("train_ratio", 0.8)
        eval_mode = params.get("eval_mode", True)
        
        # Run HMM swings analysis
        result = hmm_swings(
            df,
            train_ratio=train_ratio,
            eval_mode=eval_mode,
            orders_argrelextrema=self.orders_argrelextrema,
            strict_mode=self.strict_mode,
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
        }
        
        return HMMStrategyResult(
            signal=signal,
            probability=probability,
            state=state,
            metadata=metadata
        )

