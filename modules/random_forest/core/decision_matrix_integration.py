"""
Random Forest Decision Matrix Integration.

This module provides helper functions to integrate Random Forest predictions
into the Decision Matrix voting system.
"""

from typing import Optional, Tuple
from pathlib import Path
import pandas as pd

from modules.random_forest.core.model import load_random_forest_model
from modules.random_forest.core.signals import get_latest_random_forest_signal
from modules.common.core.data_fetcher import DataFetcher


def calculate_random_forest_vote(
    data_fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    limit: int,
    expected_signal: int,
    model_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[int, float]:
    """
    Calculate Random Forest vote for Decision Matrix.
    
    Args:
        data_fetcher: DataFetcher instance
        symbol: Trading pair symbol
        timeframe: Timeframe for data
        limit: Number of candles to fetch
        expected_signal: Expected signal direction (1 for LONG, -1 for SHORT)
        model_path: Optional path to model file (default: uses default path)
        df: Optional DataFrame to use instead of fetching from API
    
    Returns:
        Tuple of (vote, confidence) where:
        - vote: 1 if signal matches expected_signal, 0 otherwise
        - confidence: Signal confidence (0.0 to 1.0)
    """
    try:
        # Load model
        model = load_random_forest_model(Path(model_path) if model_path else None)
        if model is None:
            return (0, 0.0)
        
        # Use provided DataFrame if available, otherwise fetch from API
        if df is None:
            df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=limit,
                timeframe=timeframe,
                check_freshness=False,
            )
        
        if df is None or df.empty:
            return (0, 0.0)
        
        # Validate required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return (0, 0.0)
        
        # Get signal from Random Forest model
        signal_str, confidence = get_latest_random_forest_signal(df, model)
        
        # Convert signal string to int: "LONG" -> 1, "SHORT" -> -1, "NEUTRAL" -> 0
        if signal_str == "LONG":
            signal = 1
        elif signal_str == "SHORT":
            signal = -1
        else:
            signal = 0
        
        # Calculate vote: 1 if signal matches expected_signal, 0 otherwise
        vote = 1 if signal == expected_signal else 0
        
        return (vote, confidence)
    
    except Exception as e:
        return (0, 0.0)


def get_random_forest_signal_for_decision_matrix(
    data_fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    limit: int,
    model_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[int, float]]:
    """
    Get Random Forest signal for Decision Matrix integration.
    
    This is a wrapper around the core signal calculator that returns
    signal in the format expected by Decision Matrix.
    
    Args:
        data_fetcher: DataFetcher instance
        symbol: Trading pair symbol
        timeframe: Timeframe for data
        limit: Number of candles to fetch
        model_path: Optional path to model file (default: uses default path)
        df: Optional DataFrame to use instead of fetching from API
    
    Returns:
        Tuple of (signal, confidence) where:
        - signal: 1 (LONG), -1 (SHORT), or 0 (NEUTRAL)
        - confidence: Signal confidence (0.0 to 1.0)
        Returns None on error
    """
    try:
        # Load model
        model = load_random_forest_model(Path(model_path) if model_path else None)
        if model is None:
            return None
        
        # Use provided DataFrame if available, otherwise fetch from API
        if df is None:
            df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=limit,
                timeframe=timeframe,
                check_freshness=False,
            )
        
        if df is None or df.empty:
            return None
        
        # Validate required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return None
        
        # Get signal from Random Forest model
        signal_str, confidence = get_latest_random_forest_signal(df, model)
        
        # Convert signal string to int: "LONG" -> 1, "SHORT" -> -1, "NEUTRAL" -> 0
        if signal_str == "LONG":
            signal = 1
        elif signal_str == "SHORT":
            signal = -1
        else:
            signal = 0
        
        return (signal, confidence)
    
    except Exception as e:
        return None


__all__ = [
    "calculate_random_forest_vote",
    "get_random_forest_signal_for_decision_matrix",
]

