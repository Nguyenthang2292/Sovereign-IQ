"""
Candidate pool management utilities for pairs trading analysis.

This module provides functions for managing candidate pools of best and worst
performing symbols, ensuring target symbols are included in appropriate pools.
"""

import pandas as pd
from typing import List, Tuple


def ensure_symbols_in_candidate_pools(
    performance_df: pd.DataFrame,
    best_df: pd.DataFrame,
    worst_df: pd.DataFrame,
    target_symbols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure target symbols are present in candidate pools based on their score direction.
    
    This function ensures that user-specified target symbols are included in the
    appropriate candidate pools (best or worst performers) based on their performance scores.
    Symbols with positive scores are added to the best performers pool, while symbols
    with negative scores are added to the worst performers pool.
    
    Args:
        performance_df: DataFrame with all performance data containing columns:
            - symbol: Trading symbol (e.g., 'BTC/USDT')
            - score: Performance score (positive = good, negative = bad)
        best_df: DataFrame of best performers (top performers with high scores)
        worst_df: DataFrame of worst performers (bottom performers with low scores)
        target_symbols: List of symbols that must be included in candidate pools
        
    Returns:
        Tuple of (updated_best_df, updated_worst_df) with target symbols added
        and sorted by score:
        - best_df: Sorted descending by score (highest first)
        - worst_df: Sorted ascending by score (lowest first)
        
    Example:
        >>> performance = pd.DataFrame({
        ...     'symbol': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        ...     'score': [0.5, -0.3, 0.2]
        ... })
        >>> best = pd.DataFrame({'symbol': ['BTC/USDT'], 'score': [0.5]})
        >>> worst = pd.DataFrame({'symbol': ['ETH/USDT'], 'score': [-0.3]})
        >>> best, worst = ensure_symbols_in_candidate_pools(
        ...     performance, best, worst, ['SOL/USDT']
        ... )
        >>> 'SOL/USDT' in best['symbol'].values
        True
    
    Note:
        - Symbols already in pools are not duplicated
        - Symbols not found in performance_df are silently skipped
        - Both returned DataFrames are re-sorted and have reset indices
    """
    if not target_symbols:
        return best_df, worst_df

    # Track which symbols are already in each pool
    best_symbols = set(best_df["symbol"].tolist())
    worst_symbols = set(worst_df["symbol"].tolist())

    # Add each target symbol to appropriate pool
    for symbol in target_symbols:
        # Find symbol in performance data
        row = performance_df[performance_df["symbol"] == symbol]
        if row.empty:
            # Symbol not found in performance data, skip
            continue
            
        score = row.iloc[0]["score"]
        
        # Add to best pool if score >= 0
        if score >= 0:
            if symbol not in best_symbols:
                best_df = pd.concat([best_df, row], ignore_index=True)
                best_symbols.add(symbol)
        # Add to worst pool if score < 0
        else:
            if symbol not in worst_symbols:
                worst_df = pd.concat([worst_df, row], ignore_index=True)
                worst_symbols.add(symbol)

    # Re-sort pools and reset indices
    best_df = (
        best_df.sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    worst_df = (
        worst_df.sort_values("score", ascending=True)
        .reset_index(drop=True)
    )
    
    return best_df, worst_df

