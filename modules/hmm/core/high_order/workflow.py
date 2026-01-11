
from typing import Optional

import pandas as pd

from config import (

from config import (

"""
High-Order HMM Main Workflow.

This module contains the main true_high_order_hmm function that orchestrates the entire workflow.
"""



    HMM_HIGH_ORDER_MAX_ORDER_DEFAULT,
    HMM_HIGH_ORDER_MIN_ORDER_DEFAULT,
    HMM_HIGH_ORDER_USE_DATA_DRIVEN_INIT,
)
from modules.hmm.core.high_order.models import TrueHighOrderHMM
from modules.hmm.core.swings.models import HMM_SWINGS


def true_high_order_hmm(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    eval_mode: bool = True,
    orders_argrelextrema: Optional[int] = None,
    strict_mode: Optional[bool] = None,
    min_order: int = HMM_HIGH_ORDER_MIN_ORDER_DEFAULT,
    max_order: int = HMM_HIGH_ORDER_MAX_ORDER_DEFAULT,
) -> HMM_SWINGS:
    """
    Generates and trains a true High-Order Hidden Markov Model using swing points.

    This is a wrapper function that uses the TrueHighOrderHMM class internally.

    Parameters:
        df: DataFrame containing price data with at least the columns 'open', 'high', 'low', 'close'.
        train_ratio: The ratio of data to use for training (default: 0.8).
        eval_mode: If True, evaluates model performance on the test set.
        orders_argrelextrema: Order parameter for swing detection (default: from config).
        strict_mode: Whether to use strict mode for swing-to-state conversion (default: from config).
        min_order: Minimum order k to try during optimization (default: 2).
        max_order: Maximum order k to try during optimization (default: 4).

    Returns:
        HMM_SWINGS: Instance containing the predicted market state.
    """
    analyzer = TrueHighOrderHMM(
        orders_argrelextrema=orders_argrelextrema,
        strict_mode=strict_mode,
        use_data_driven=HMM_HIGH_ORDER_USE_DATA_DRIVEN_INIT,
        train_ratio=train_ratio,
        min_order=min_order,
        max_order=max_order,
    )
    return analyzer.analyze(df, eval_mode=eval_mode)
