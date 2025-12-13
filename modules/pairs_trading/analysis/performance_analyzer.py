"""
Performance analyzer for calculating symbol performance scores across multiple timeframes.
"""

import math
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.common.DataFetcher import DataFetcher
    import threading

try:
    from config import (
        PAIRS_TRADING_WEIGHTS,
        PAIRS_TRADING_TOP_N,
        PAIRS_TRADING_MIN_CANDLES,
        PAIRS_TRADING_TIMEFRAME,
        PAIRS_TRADING_LIMIT,
    )
    from modules.common.utils import (
        color_text,
        normalize_symbol,
        timeframe_to_minutes,
        log_info,
        log_success,
        log_error,
        log_warn,
        log_progress,
    )
    from modules.common.ProgressBar import ProgressBar, NullProgressBar
except ImportError:
    PAIRS_TRADING_WEIGHTS = {'1d': 0.5, '3d': 0.3, '1w': 0.2}
    PAIRS_TRADING_TOP_N = 5
    PAIRS_TRADING_MIN_CANDLES = 168
    PAIRS_TRADING_TIMEFRAME = "1h"
    PAIRS_TRADING_LIMIT = 200
    color_text = None
    normalize_symbol = None
    
    def log_info(message: str) -> None:
        print(f"[INFO] {message}")
    
    def log_success(message: str) -> None:
        print(f"[SUCCESS] {message}")
    
    def log_error(message: str) -> None:
        print(f"[ERROR] {message}")
    
    def log_warn(message: str) -> None:
        print(f"[WARN] {message}")
    
    def log_progress(message: str) -> None:
        print(f"[PROGRESS] {message}")

    def timeframe_to_minutes(timeframe: str) -> int:
        match = re.match(r"^\s*(\d+)\s*([mhdw])\s*$", timeframe.lower())
        if not match:
            return 60
        value, unit = match.groups()
        value = int(value)
        if unit == "m":
            return value
        if unit == "h":
            return value * 60
        if unit == "d":
            return value * 60 * 24
        if unit == "w":
            return value * 60 * 24 * 7
        return 60

    ProgressBar = None
    
    class NullProgressBar:
        """Null object pattern for ProgressBar when ProgressBar is not available."""
        def update(self, step: int = 1) -> None:
            pass
        def finish(self) -> None:
            pass


class PerformanceAnalyzer:
    """
    Analyzes performance of trading symbols across multiple timeframes.
    
    Calculates weighted performance scores from 1 day, 3 days, and 1 week returns.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_candles: int = PAIRS_TRADING_MIN_CANDLES,
        timeframe: str = PAIRS_TRADING_TIMEFRAME,
        limit: int = PAIRS_TRADING_LIMIT,
    ):
        """
        Initialize PerformanceAnalyzer.

        Args:
            weights: Dictionary with weights for '1d', '3d', '1w' timeframes.
                     Default: {'1d': 0.5, '3d': 0.3, '1w': 0.2}
            min_candles: Minimum number of candles required for analysis (default: 168)
            timeframe: Timeframe for OHLCV data (default: '1h')
            limit: Number of candles to fetch (default: 200)
            
        Raises:
            ValueError: If parameter values are invalid
        """
        # Validate and set weights
        self.weights = weights or PAIRS_TRADING_WEIGHTS.copy()
        if not isinstance(self.weights, dict):
            raise ValueError(f"weights must be a dictionary, got {type(self.weights)}")
        
        # Validate weight keys
        required_keys = {'1d', '3d', '1w'}
        if not required_keys.issubset(self.weights.keys()):
            raise ValueError(f"weights must contain keys {required_keys}, got {set(self.weights.keys())}")
        
        # Validate weight values
        for key, value in self.weights.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"weights[{key}] must be numeric, got {type(value)}")
            if np.isnan(value) or np.isinf(value):
                raise ValueError(f"weights[{key}] must be finite, got {value}")
            if value < 0:
                raise ValueError(f"weights[{key}] must be non-negative, got {value}")
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight:.6f}. "
                f"Current weights: {self.weights}"
            )
        
        # Validate other parameters
        if min_candles < 1:
            raise ValueError(f"min_candles must be >= 1, got {min_candles}")
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if min_candles > limit:
            raise ValueError(
                f"min_candles ({min_candles}) must be <= limit ({limit})"
            )
        if not isinstance(timeframe, str) or not timeframe:
            raise ValueError(f"timeframe must be a non-empty string, got {timeframe}")
        
        self.min_candles = min_candles
        self.timeframe = timeframe
        self.limit = limit

    def _check_consecutive_nan_chunks(
        self, 
        values: np.ndarray, 
        symbol: str,
        max_gap_ratio: float = 0.1,
        max_gap_candles: int = 24
    ) -> bool:
        """
        Check for consecutive NaN/Inf chunks that may cause timestamp alignment issues.
        
        Args:
            values: Array of values to check
            symbol: Symbol name for logging
            max_gap_ratio: Maximum allowed gap as ratio of total data (default: 10%)
            max_gap_candles: Maximum allowed consecutive NaN candles (default: 24)
            
        Returns:
            True if data is acceptable, False if gaps are too large
        """
        invalid_mask = np.isnan(values) | np.isinf(values)
        
        if not invalid_mask.any():
            return True
        
        # Find consecutive chunks of invalid values
        # Detect transitions: False->True (start of chunk) and True->False (end of chunk)
        padded_mask = np.concatenate(([False], invalid_mask, [False]))
        diff_mask = np.diff(padded_mask.astype(int))
        
        # Find start indices (where diff is +1) and end indices (where diff is -1)
        start_indices = np.where(diff_mask == 1)[0]
        end_indices = np.where(diff_mask == -1)[0]
        
        # Pair up starts and ends
        if len(start_indices) != len(end_indices):
            # Edge case: chunks at boundaries
            if len(start_indices) > len(end_indices):
                end_indices = np.concatenate((end_indices, [len(invalid_mask)]))
            elif len(end_indices) > len(start_indices):
                start_indices = np.concatenate(([0], start_indices))
        
        # Check each chunk
        total_size = len(values)
        for start_idx, end_idx in zip(start_indices, end_indices):
            chunk_size = end_idx - start_idx
            gap_ratio = chunk_size / total_size if total_size > 0 else 0.0
            
            # Log warning for large gaps
            if chunk_size > max_gap_candles or gap_ratio > max_gap_ratio:
                log_warn(
                    f"{symbol}: Large data gap detected - {chunk_size} consecutive NaN/Inf values "
                    f"(positions {start_idx}-{end_idx}, {gap_ratio*100:.1f}% of data). "
                    "This may affect timestamp alignment and return calculations."
                )
                
                # Reject if gap is extremely large
                if chunk_size > max_gap_candles * 2 or gap_ratio > max_gap_ratio * 2:
                    log_warn(
                        f"{symbol}: Gap too large ({chunk_size} candles, {gap_ratio*100:.1f}%). "
                        "Rejecting symbol to preserve data quality."
                    )
                    return False
        
        return True

    def _calculate_timeframe_returns(
        self,
        df: pd.DataFrame,
        current_price: float,
        current_timestamp: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Calculate returns for 1d, 3d, and 1w timeframes based on actual timestamps.
        
        This method ensures timestamp alignment by using actual time periods rather than
        array indices, which is critical when there are gaps in the data.
        
        Args:
            df: DataFrame with 'close' and 'timestamp' columns, sorted by timestamp
            current_price: Current closing price
            current_timestamp: Current timestamp
            
        Returns:
            Dictionary with '1d', '3d', '1w' return percentages
        """
        returns = {'1d': 0.0, '3d': 0.0, '1w': 0.0}
        
        # Calculate target timestamps for each timeframe
        target_timestamps = {
            '1d': current_timestamp - pd.Timedelta(days=1),
            '3d': current_timestamp - pd.Timedelta(days=3),
            '1w': current_timestamp - pd.Timedelta(days=7),
        }
        
        # For each timeframe, find the closest price before the target timestamp
        for timeframe, target_ts in target_timestamps.items():
            # Find rows before or at the target timestamp
            mask = df['timestamp'] <= target_ts
            
            if not mask.any():
                # No data before target timestamp, return 0.0
                returns[timeframe] = 0.0
                continue
            
            # Get the closest price before target timestamp
            df_before = df[mask]
            if df_before.empty:
                returns[timeframe] = 0.0
                continue
            
            # Get the most recent price before target timestamp
            past_price = float(df_before['close'].iloc[-1])
            
            # Validate past_price
            if past_price <= 0 or np.isnan(past_price) or np.isinf(past_price):
                returns[timeframe] = 0.0
                continue
            
            # Calculate return
            ret = (current_price - past_price) / past_price
            
            # Validate return
            if not (np.isnan(ret) or np.isinf(ret)):
                returns[timeframe] = ret
            else:
                returns[timeframe] = 0.0
        
        return returns

    def calculate_performance_score(
        self, symbol: str, df: pd.DataFrame
    ) -> Optional[Dict[str, float]]:
        """
        Calculate performance score for a symbol from OHLCV DataFrame.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            df: DataFrame with OHLCV data, must have 'close' column and be sorted by timestamp

        Returns:
            Dictionary with keys:
                - 'symbol': Symbol name
                - 'score': Weighted performance score
                - '1d_return': 1-day return percentage
                - '3d_return': 3-day return percentage
                - '1w_return': 1-week return percentage
                - 'current_price': Current closing price
            Returns None if insufficient data or calculation fails.
            
        Raises:
            ValueError: If symbol is None/empty or df is invalid type
        """
        # Validate inputs
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"symbol must be a non-empty string, got {symbol}")
        
        if df is None:
            return None
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"df must be a pd.DataFrame, got {type(df)}")
        
        if df.empty:
            return None

        if 'close' not in df.columns:
            return None

        # Ensure DataFrame is sorted by timestamp (ascending)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        # Check if we have enough data
        if len(df) < self.min_candles:
            return None
        
        # Check for infinite values in close column
        if np.isinf(df['close']).any():
            return None

        try:
            # Check for consecutive NaN chunks that may cause timestamp alignment issues
            close_prices = df['close'].values
            if not self._check_consecutive_nan_chunks(close_prices, symbol):
                return None
            
            # Use DataFrame with timestamp index for accurate time-based calculations
            # This ensures returns are calculated based on actual time periods, not array indices
            df_clean = df.copy()
            
            # Filter out NaN/Inf values but keep timestamp alignment
            invalid_mask = np.isnan(df_clean['close']) | np.isinf(df_clean['close'])
            invalid_count = invalid_mask.sum()
            
            # Log warning if significant data loss
            if invalid_count > 0:
                invalid_ratio = invalid_count / len(df_clean) if len(df_clean) > 0 else 0.0
                if invalid_ratio > 0.05:  # More than 5% invalid
                    log_warn(
                        f"{symbol}: Filtered {invalid_count}/{len(df_clean)} invalid values "
                        f"({invalid_ratio*100:.1f}%). Timestamp alignment preserved."
                    )
            
            # Remove invalid rows but preserve timestamp column
            df_clean = df_clean[~invalid_mask].copy()
            
            if len(df_clean) < self.min_candles:
                return None
            
            # Ensure we have a timestamp column for time-based calculations
            if 'timestamp' not in df_clean.columns:
                # If no timestamp, create one from index (assume sequential)
                df_clean['timestamp'] = pd.to_datetime(df_clean.index, errors='coerce')
            
            # Sort by timestamp to ensure chronological order
            df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
            
            # Get current price (most recent valid price)
            current_price = float(df_clean['close'].iloc[-1])
            
            # Validate current_price
            if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
                return None

            # Get current timestamp
            current_timestamp = df_clean['timestamp'].iloc[-1]
            
            # Calculate returns for different timeframes using actual timestamps
            returns = self._calculate_timeframe_returns(df_clean, current_price, current_timestamp)

            # Calculate weighted score
            score = (
                returns['1d'] * self.weights.get('1d', 0.0) +
                returns['3d'] * self.weights.get('3d', 0.0) +
                returns['1w'] * self.weights.get('1w', 0.0)
            )
            
            # Validate final score
            if np.isnan(score) or np.isinf(score):
                return None

            return {
                'symbol': str(symbol),
                'score': float(score),
                '1d_return': float(returns['1d']),
                '3d_return': float(returns['3d']),
                '1w_return': float(returns['1w']),
                'current_price': current_price,
            }

        except (ValueError, IndexError, KeyError, TypeError, AttributeError) as e:
            # Log calculation errors for debugging
            log_warn(f"{symbol}: Calculation error in performance score: {e}")
            return None
        except Exception as e:
            # Log unexpected errors
            log_error(f"{symbol}: Unexpected error in performance score calculation: {e}")
            return None

    def analyze_all_symbols(
        self,
        symbols: List[str],
        data_fetcher: Optional["DataFetcher"],
        verbose: bool = True,
        shutdown_event: Optional["threading.Event"] = None,
    ) -> pd.DataFrame:
        """
        Analyze performance for all symbols.

        Args:
            symbols: List of trading symbols to analyze
            data_fetcher: DataFetcher instance for fetching OHLCV data
            verbose: If True, print progress messages
            shutdown_event: Optional threading.Event to signal shutdown

        Returns:
            DataFrame with columns: ['symbol', 'score', '1d_return', '3d_return', 
                                   '1w_return', 'current_price']
            Sorted by score (descending). Symbols with insufficient data are excluded.
        """
        # Validate inputs
        if not symbols:
            return pd.DataFrame(
                columns=['symbol', 'score', '1d_return', '3d_return', '1w_return', 'current_price']
            )
        
        if not isinstance(symbols, list):
            raise ValueError(f"symbols must be a list, got {type(symbols)}")
        
        if data_fetcher is None:
            raise ValueError("data_fetcher must not be None")

        results = []

        if verbose and ProgressBar:
            progress = ProgressBar(len(symbols), "Performance Analysis")
        else:
            progress = NullProgressBar()

        for symbol in symbols:
            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                progress.update()
                continue
            
            # Check for shutdown signal
            if shutdown_event and hasattr(shutdown_event, 'is_set') and shutdown_event.is_set():
                if verbose:
                    log_warn("Analysis aborted due to shutdown signal.")
                break

            try:
                # Fetch OHLCV data
                df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                    symbol,
                    limit=self.limit,
                    timeframe=self.timeframe,
                    check_freshness=False,
                )

                if df is None or df.empty:
                    if verbose:
                        log_warn(f"Skipping {symbol}: No data available")
                    progress.update()
                    continue

                # Calculate performance score
                result = self.calculate_performance_score(symbol, df)

                if result is None:
                    if verbose:
                        log_warn(f"Skipping {symbol}: Insufficient data or calculation failed")
                    progress.update()
                    continue

                results.append(result)

                if verbose:
                    score_pct = result['score'] * 100
                    # Validate score_pct is finite
                    if not (np.isnan(score_pct) or np.isinf(score_pct)):
                        try:
                            message = (
                                f"{symbol}: Score {score_pct:+.2f}% "
                                f"(1d: {result['1d_return']*100:+.2f}%, "
                                f"3d: {result['3d_return']*100:+.2f}%, "
                                f"1w: {result['1w_return']*100:+.2f}%)"
                            )
                            if score_pct > 0:
                                log_success(message)
                            else:
                                log_info(message)
                        except (UnicodeEncodeError, ValueError, KeyError):
                            # Fallback for encoding issues or missing keys
                            log_info(f"{symbol}: Score {score_pct:+.2f}%")

            except (AttributeError, TypeError, ValueError) as e:
                if verbose:
                    log_error(f"Error analyzing {symbol}: {type(e).__name__}: {e}")
            except Exception as e:
                if verbose:
                    log_error(f"Unexpected error analyzing {symbol}: {type(e).__name__}: {e}")
            finally:
                progress.update()

        progress.finish()

        if not results:
            if verbose:
                log_warn("No valid performance data found for any symbols.")
            return pd.DataFrame(
                columns=['symbol', 'score', '1d_return', '3d_return', '1w_return', 'current_price']
            )

        # Create DataFrame and sort by score (descending)
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('score', ascending=False).reset_index(drop=True)

        if verbose:
            log_success(f"Successfully analyzed {len(df_results)}/{len(symbols)} symbols.")

        return df_results

    def get_top_performers(
        self, df: pd.DataFrame, top_n: int = PAIRS_TRADING_TOP_N
    ) -> pd.DataFrame:
        """
        Get top N best performing symbols.

        Args:
            df: DataFrame from analyze_all_symbols()
            top_n: Number of top performers to return (default: 5, must be > 0)

        Returns:
            DataFrame with top N performers (sorted by score, descending)
            
        Raises:
            ValueError: If top_n <= 0
        """
        if top_n <= 0:
            raise ValueError(f"top_n must be > 0, got {top_n}")
        
        if df is None or df.empty:
            return pd.DataFrame(
                columns=['symbol', 'score', '1d_return', '3d_return', '1w_return', 'current_price']
            )

        # DataFrame is already sorted by score descending
        return df.head(top_n).copy()

    def get_worst_performers(
        self, df: pd.DataFrame, top_n: int = PAIRS_TRADING_TOP_N
    ) -> pd.DataFrame:
        """
        Get top N worst performing symbols.

        Args:
            df: DataFrame from analyze_all_symbols()
            top_n: Number of worst performers to return (default: 5, must be > 0)

        Returns:
            DataFrame with top N worst performers (sorted by score, ascending)
            
        Raises:
            ValueError: If top_n <= 0
        """
        if top_n <= 0:
            raise ValueError(f"top_n must be > 0, got {top_n}")
        
        if df is None or df.empty:
            return pd.DataFrame(
                columns=['symbol', 'score', '1d_return', '3d_return', '1w_return', 'current_price']
            )

        # Sort by score ascending and take top N
        df_sorted = df.sort_values('score', ascending=True).reset_index(drop=True)
        return df_sorted.head(top_n).copy()