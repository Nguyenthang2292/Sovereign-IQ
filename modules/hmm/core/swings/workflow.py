"""
HMM-Swings Main Workflow.

This module contains the main hmm_swings function that orchestrates the entire workflow.
"""

from typing import Optional
import pandas as pd

from modules.hmm.core.swings.models import HMM_SWINGS, SwingsHMM, NEUTRAL
from config import HMM_HIGH_ORDER_USE_DATA_DRIVEN_INIT


def hmm_swings(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    eval_mode: bool = True,
    orders_argrelextrema: Optional[int] = None,
    strict_mode: Optional[bool] = None,
) -> HMM_SWINGS:
    """
    Generates and trains a Hidden Markov Model (HMM) using swing points extracted from market price data.
    
    This is a wrapper function that uses the SwingsHMM class internally.
    
    Parameters:
        df: DataFrame containing price data with at least the columns 'open', 'high', 'low', 'close'.
        train_ratio: The ratio of data to use for training (default: 0.8).
        eval_mode: If True, evaluates model performance on the test set.
        orders_argrelextrema: Order parameter for swing detection (default: from config).
        strict_mode: Whether to use strict mode for swing-to-state conversion (default: from config).
    
        Returns:
            HMM_SWINGS: Instance containing the predicted market state.
    """
    analyzer = SwingsHMM(
        orders_argrelextrema=orders_argrelextrema,
        strict_mode=strict_mode,
        use_data_driven=HMM_HIGH_ORDER_USE_DATA_DRIVEN_INIT,
        train_ratio=train_ratio,
    )
    return analyzer.analyze(df, eval_mode=eval_mode)

