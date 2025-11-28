"""
Pair transformation utilities for pairs trading analysis.

This module provides functions for transforming and manipulating trading pairs data,
such as reversing pair positions and adjusting hedge ratios.
"""

import pandas as pd
from typing import Optional


def reverse_pairs(pairs_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Reverse pairs by swapping long and short positions.
    
    This function swaps the long and short positions in a pairs DataFrame,
    adjusting hedge ratios accordingly. Useful for viewing the opposite
    trading direction or analyzing pairs from different perspectives.
    
    Transformations applied:
    - long_symbol ↔ short_symbol
    - long_score ↔ short_score (if present)
    - hedge_ratio = 1 / original_hedge_ratio (if present)
    - kalman_hedge_ratio = 1 / original_kalman_hedge_ratio (if present)
    
    Args:
        pairs_df: DataFrame with pairs data containing columns:
            - long_symbol, short_symbol (required)
            - long_score, short_score (optional)
            - hedge_ratio, kalman_hedge_ratio (optional)
    
    Returns:
        DataFrame with reversed pairs. Returns empty DataFrame if input is None/empty.
        
    Example:
        >>> pairs = pd.DataFrame({
        ...     'long_symbol': ['BTC/USDT'],
        ...     'short_symbol': ['ETH/USDT'],
        ...     'hedge_ratio': [2.0]
        ... })
        >>> reversed = reverse_pairs(pairs)
        >>> reversed['long_symbol'][0]
        'ETH/USDT'
        >>> reversed['hedge_ratio'][0]
        0.5
    
    Note:
        Hedge ratios are inverted because the relationship between the pairs
        is reversed. If originally long 1 unit of A and short β units of B,
        after reversal you long 1 unit of B and short (1/β) units of A.
    """
    if pairs_df is None or pairs_df.empty:
        return pd.DataFrame(columns=pairs_df.columns if pairs_df is not None else [])
    
    reversed_df = pairs_df.copy()
    
    # Swap long and short symbols
    if "long_symbol" in reversed_df.columns and "short_symbol" in reversed_df.columns:
        reversed_df["long_symbol"], reversed_df["short_symbol"] = (
            reversed_df["short_symbol"],
            reversed_df["long_symbol"],
        )
    
    # Swap long_score and short_score if they exist
    if "long_score" in reversed_df.columns and "short_score" in reversed_df.columns:
        reversed_df["long_score"], reversed_df["short_score"] = (
            reversed_df["short_score"],
            reversed_df["long_score"],
        )
    
    # Invert hedge ratios (since we're reversing the pair relationship)
    if "hedge_ratio" in reversed_df.columns:
        # Replace 0 with inf to avoid division by zero, then invert
        reversed_df["hedge_ratio"] = 1.0 / reversed_df["hedge_ratio"].replace(0, float('inf'))
    
    if "kalman_hedge_ratio" in reversed_df.columns:
        # Replace 0 with inf to avoid division by zero, then invert
        reversed_df["kalman_hedge_ratio"] = 1.0 / reversed_df["kalman_hedge_ratio"].replace(0, float('inf'))
    
    return reversed_df


