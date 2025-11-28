"""
Pair selection utilities for pairs trading analysis.

This module provides functions for selecting and filtering trading pairs
based on various criteria such as uniqueness, target symbols, and scores.
"""

import pandas as pd
from typing import Optional, List


def select_top_unique_pairs(pairs_df: pd.DataFrame, target_pairs: int) -> pd.DataFrame:
    """
    Select up to target_pairs trading pairs, prioritizing unique symbols for diversification.
    
    This function maximizes portfolio diversification by selecting pairs with non-overlapping
    symbols when possible. It uses a two-pass selection strategy:
    
    1. **First pass**: Selects pairs where both long_symbol and short_symbol are unique
       (not used in previously selected pairs). This ensures maximum symbol diversity.
    
    2. **Second pass**: If the target number hasn't been reached, fills remaining slots
       with any available pairs from the DataFrame.
    
    3. **Fallback**: If no pairs are selected in the first two passes, returns the top N
       pairs directly from the input DataFrame (assumes pairs_df is pre-sorted by score).
    
    The function preserves the original order of pairs_df, selecting the first matching
    pairs that meet the uniqueness criteria.
    
    Args:
        pairs_df: DataFrame containing pairs data with required columns:
            - long_symbol: Symbol to go long on (str)
            - short_symbol: Symbol to go short on (str)
            Optional columns (e.g., 'score', 'opportunity_score') are preserved in output.
        target_pairs: Maximum number of pairs to select. If fewer unique pairs are
            available, returns fewer pairs (not guaranteed to return exactly target_pairs).
        
    Returns:
        DataFrame with selected pairs, preserving all original columns.
        - Index is reset (0-based sequential)
        - Pairs are in selection order (first pass pairs, then second pass pairs)
        - Returns original pairs_df unchanged if pairs_df is empty or None
        
    Example:
        >>> pairs = pd.DataFrame({
        ...     'long_symbol': ['BTC/USDT', 'ETH/USDT', 'BTC/USDT', 'SOL/USDT'],
        ...     'short_symbol': ['ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT'],
        ...     'score': [0.9, 0.8, 0.7, 0.6]
        ... })
        >>> selected = select_top_unique_pairs(pairs, target_pairs=2)
        >>> len(selected)
        2
        >>> # First pair: BTC/USDT - ETH/USDT (both unique)
        >>> # Second pair: ETH/USDT - BNB/USDT (ETH already used, but selected in second pass)
        >>> selected['long_symbol'].tolist()
        ['BTC/USDT', 'ETH/USDT']
    
    Note:
        - This function assumes pairs_df is already sorted by desirability (e.g., score)
          as it selects pairs in order of appearance
        - Symbols are considered unique at the pair level - if a symbol appears in multiple
          pairs, only the first occurrence (in order) is selected in the first pass
        - Returns fewer than target_pairs if insufficient unique pairs are available
    """
    if pairs_df is None or pairs_df.empty:
        return pairs_df

    selected_indices = []
    used_symbols = set()

    # First pass: Select pairs with completely unique symbols
    for idx, row in pairs_df.iterrows():
        long_symbol = row["long_symbol"]
        short_symbol = row["short_symbol"]
        if long_symbol in used_symbols or short_symbol in used_symbols:
            continue
        selected_indices.append(idx)
        used_symbols.update([long_symbol, short_symbol])
        if len(selected_indices) == target_pairs:
            break

    # Second pass: Fill remaining slots if needed
    if len(selected_indices) < target_pairs:
        for idx in pairs_df.index:
            if idx in selected_indices:
                continue
            selected_indices.append(idx)
            if len(selected_indices) == target_pairs:
                break

    # Fallback: If still no selections, just take top N
    if not selected_indices:
        return pairs_df.head(target_pairs).reset_index(drop=True)

    return pairs_df.loc[selected_indices].reset_index(drop=True)


def select_pairs_for_symbols(
    pairs_df: pd.DataFrame, 
    target_symbols: List[str], 
    max_pairs: Optional[int] = None
) -> pd.DataFrame:
    """
    Select the best pair (highest score) for each requested symbol.
    
    This function finds the best pair opportunity for each symbol in target_symbols,
    where the symbol appears as either the long or short side of the pair.
    
    Args:
        pairs_df: DataFrame containing pairs data with 'long_symbol' and 'short_symbol' columns
        target_symbols: List of symbols to find pairs for
        max_pairs: Maximum number of pairs to return (None = no limit)
        
    Returns:
        DataFrame with selected pairs for the requested symbols
        Returns empty DataFrame if no matches found
        
    Example:
        >>> pairs = pd.DataFrame({
        ...     'long_symbol': ['BTC/USDT', 'ETH/USDT'],
        ...     'short_symbol': ['ETH/USDT', 'BNB/USDT'],
        ...     'score': [0.9, 0.8]
        ... })
        >>> selected = select_pairs_for_symbols(pairs, ['BTC/USDT'])
        >>> len(selected)
        1
    """
    if pairs_df is None or pairs_df.empty or not target_symbols:
        return pd.DataFrame(columns=pairs_df.columns if pairs_df is not None else [])

    selected_rows = []
    for symbol in target_symbols:
        # Find pairs where symbol appears on either side
        matches = pairs_df[
            (pairs_df["long_symbol"] == symbol) | (pairs_df["short_symbol"] == symbol)
        ]
        if matches.empty:
            continue
        # Take the best match (first row, assuming sorted by score)
        selected_rows.append(matches.iloc[0])
        if max_pairs is not None and len(selected_rows) >= max_pairs:
            break

    if not selected_rows:
        return pd.DataFrame(columns=pairs_df.columns)

    return pd.DataFrame(selected_rows).reset_index(drop=True)

