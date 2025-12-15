"""
HMM-KAMA Strategy Implementation.

This module contains the KamaHMMStrategy class that implements the HMMStrategy interface.
"""

from typing import Optional, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from modules.hmm.signals.strategy import HMMStrategyResult

from modules.hmm.core.kama.workflow import hmm_kama
from modules.hmm.core.kama.models import HMM_KAMA
from modules.common.utils import log_info, log_error, log_warn, log_model, log_analysis
from config import (
    HMM_WINDOW_KAMA_DEFAULT,
    HMM_FAST_KAMA_DEFAULT,
    HMM_SLOW_KAMA_DEFAULT,
    HMM_WINDOW_SIZE_DEFAULT,
)


class KamaHMMStrategy:
    """
    HMM Strategy wrapper for HMM-KAMA.
    
    Implements HMMStrategy interface to enable registry-based management.
    """
    
    def __init__(
        self,
        name: str = "kama",
        weight: float = 1.5,
        enabled: bool = True,
        window_kama: Optional[int] = None,
        fast_kama: Optional[int] = None,
        slow_kama: Optional[int] = None,
        window_size: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize KAMA HMM Strategy.
        
        Args:
            name: Strategy name (default: "kama")
            weight: Strategy weight for voting (default: 1.5)
            enabled: Whether strategy is enabled (default: True)
            window_kama: KAMA window size
            fast_kama: Fast KAMA parameter
            slow_kama: Slow KAMA parameter
            window_size: Rolling window size
            **kwargs: Additional parameters
        """
        from modules.hmm.signals.strategy import HMMStrategy, HMMStrategyResult
        from modules.hmm.signals.resolution import LONG, HOLD, SHORT
        
        self.name = name
        self.weight = weight
        self.enabled = enabled
        self.window_kama = (
            window_kama if window_kama is not None else HMM_WINDOW_KAMA_DEFAULT
        )
        self.fast_kama = (
            fast_kama if fast_kama is not None else HMM_FAST_KAMA_DEFAULT
        )
        self.slow_kama = (
            slow_kama if slow_kama is not None else HMM_SLOW_KAMA_DEFAULT
        )
        self.window_size = (
            window_size if window_size is not None else HMM_WINDOW_SIZE_DEFAULT
        )
        self.params = kwargs
    
    def analyze(self, df: pd.DataFrame, **kwargs) -> 'HMMStrategyResult':
        """
        Analyze market data using HMM-KAMA.
        
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
        
        # Run HMM-KAMA analysis
        result = hmm_kama(
            df,
            window_kama=self.window_kama,
            fast_kama=self.fast_kama,
            slow_kama=self.slow_kama,
            window_size=self.window_size,
        )
        
        # Map HMM_KAMA state to Signal
        # States: 0,1,2,3 -> 0,2 are bearish, 1,3 are bullish
        primary_state = result.next_state_with_hmm_kama
        
        # Convert state to signal
        # States 1,3 = bullish (LONG), States 0,2 = bearish (SHORT)
        if primary_state in {1, 3}:
            signal = LONG
        elif primary_state in {0, 2}:
            signal = SHORT
        else:
            signal = HOLD
        
        # Calculate confidence from transition states and ARM states
        # Use a simple heuristic: count bullish/bearish indicators
        bullish_indicators = 0
        bearish_indicators = 0
        
        # Transition states
        # Note: These functions return 0 or 1, never -1
        # - 0 typically means "normal/stable" (within std dev for std, cluster 0 for others)
        # - 1 typically means "abnormal/volatile" (outside std dev for std, cluster 1 for others)
        # We interpret: 1 = bullish volatility/change, 0 = bearish stability/no change
        
        if result.current_state_of_state_using_std == 1:
            bullish_indicators += 1
        elif result.current_state_of_state_using_std == 0:
            bearish_indicators += 1
        # -1 case is handled for completeness, though it should never occur
        elif result.current_state_of_state_using_std == -1:
            bearish_indicators += 1
        
        if result.current_state_of_state_using_hmm == 1:
            bullish_indicators += 1
        elif result.current_state_of_state_using_hmm == 0:
            bearish_indicators += 1
        # -1 case is handled for completeness, though it should never occur
        elif result.current_state_of_state_using_hmm == -1:
            bearish_indicators += 1
        
        if result.current_state_of_state_using_kmeans == 1:
            bullish_indicators += 1
        elif result.current_state_of_state_using_kmeans == 0:
            bearish_indicators += 1
        # -1 case is handled for completeness, though it should never occur
        elif result.current_state_of_state_using_kmeans == -1:
            bearish_indicators += 1
        
        # ARM states
        if result.state_high_probabilities_using_arm_apriori in {1, 3}:
            bullish_indicators += 1
        elif result.state_high_probabilities_using_arm_apriori in {0, 2}:
            bearish_indicators += 1
        
        if result.state_high_probabilities_using_arm_fpgrowth in {1, 3}:
            bullish_indicators += 1
        elif result.state_high_probabilities_using_arm_fpgrowth in {0, 2}:
            bearish_indicators += 1
        
        # Calculate probability based on indicator agreement
        total_indicators = bullish_indicators + bearish_indicators
        if total_indicators > 0:
            if signal == LONG:
                probability = max(0.5, bullish_indicators / total_indicators)
            elif signal == SHORT:
                probability = max(0.5, bearish_indicators / total_indicators)
            else:
                probability = 0.5
        else:
            probability = 0.5
        
        metadata = {
            "primary_state": primary_state,
            "transition_std": result.current_state_of_state_using_std,
            "transition_hmm": result.current_state_of_state_using_hmm,
            "transition_kmeans": result.current_state_of_state_using_kmeans,
            "arm_apriori": result.state_high_probabilities_using_arm_apriori,
            "arm_fpgrowth": result.state_high_probabilities_using_arm_fpgrowth,
            "bullish_indicators": bullish_indicators,
            "bearish_indicators": bearish_indicators,
        }
        
        return HMMStrategyResult(
            signal=signal,
            probability=probability,
            state=primary_state,
            metadata=metadata
        )

