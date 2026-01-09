"""
Market Batch Scanner for scanning entire market with Gemini.

Orchestrates the workflow: get symbols → batch → generate charts → analyze → aggregate results.
"""

import os
import sys
import json
import glob
import time
import argparse
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from datetime import datetime
from pathlib import Path

from modules.common.core.exchange_manager import ExchangeManager, PublicExchangeManager
from modules.common.core.data_fetcher import DataFetcher
from modules.gemini_chart_analyzer.core.generators.chart_batch_generator import ChartBatchGenerator
from modules.gemini_chart_analyzer.core.analyzers.gemini_batch_chart_analyzer import GeminiBatchChartAnalyzer
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_analysis_results_dir
from modules.common.ui.logging import log_info, log_error, log_success, log_warn
from core.voting_analyzer import VotingAnalyzer
from config import (
    DEFAULT_TIMEFRAME,
    DECISION_MATRIX_VOTING_THRESHOLD,
    DECISION_MATRIX_MIN_VOTES,
    SPC_LOOKBACK,
    SPC_P_LOW,
    SPC_P_HIGH,
    SPC_STRATEGY_PARAMETERS,
    RANGE_OSCILLATOR_LENGTH,
    RANGE_OSCILLATOR_MULTIPLIER,
    HMM_WINDOW_SIZE_DEFAULT,
    HMM_WINDOW_KAMA_DEFAULT,
    HMM_FAST_KAMA_DEFAULT,
    HMM_SLOW_KAMA_DEFAULT,
    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
)


class SymbolFetchError(Exception):
    """Custom exception for symbol fetching errors."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None, is_retryable: bool = False):
        super().__init__(message)
        self.original_exception = original_exception
        self.is_retryable = is_retryable


class MarketBatchScanner:
    """Scan entire market by batching symbols and analyzing with Gemini."""
    
    # Minimum number of candles required for reliable technical analysis
    MIN_CANDLES: int = 20
    
    # Batch size for multi-timeframe charts (reduced because each symbol has multiple TFs)
    MULTI_TF_CHARTS_PER_BATCH: int = 25
    
    def __init__(
        self,
        charts_per_batch: int = 100,
        cooldown_seconds: float = 2.5,
        quote_currency: str = 'USDT',
        exchange_name: str = 'binance',
        min_candles: Optional[int] = None
    ):
        """
        Initialize MarketBatchScanner.
        
        Args:
            charts_per_batch: Number of charts per batch (default: 100)
            cooldown_seconds: Cooldown between batch requests (default: 2.5s)
            quote_currency: Quote currency to filter symbols (default: 'USDT')
            exchange_name: Exchange name to connect to (default: 'binance')
            min_candles: Minimum number of candles required for reliable technical analysis (default: 20)
        
        Raises:
            ValueError: If min_candles is less than or equal to 0
        """
        self.charts_per_batch = charts_per_batch
        self.cooldown_seconds = cooldown_seconds
        self.quote_currency = quote_currency
        self.exchange_name = exchange_name
        
        # Set min_candles with validation
        self.min_candles = min_candles if min_candles is not None else self.MIN_CANDLES
        if self.min_candles <= 0:
            raise ValueError(f"min_candles must be greater than 0, got {self.min_candles}")
        
        
        # Initialize components (except GeminiBatchChartAnalyzer - lazy init)
        try:
            self.exchange_manager = ExchangeManager()
            
            self.public_exchange_manager = PublicExchangeManager()  # For load_markets (no credentials needed)
            
            self.data_fetcher = DataFetcher(self.exchange_manager)
            
            self.batch_chart_generator = ChartBatchGenerator(charts_per_batch=charts_per_batch)
            
            
            # Use lazy initialization for GeminiBatchChartAnalyzer to avoid stdin issues
            # The genai library initialization can close stdin on Windows, so we delay it
            # until the analyzer is actually needed (when analyze_batch_chart is called)
            self._gemini_analyzer_cooldown = cooldown_seconds
            self._gemini_analyzer = None  # Will be initialized lazily
        except Exception as e:
            raise
    
    @property
    def batch_gemini_analyzer(self):
        """
        Lazy initialization property for GeminiBatchChartAnalyzer.
        Initializes the analyzer only when first accessed, after all user input is collected.
        """
        if self._gemini_analyzer is None:
            
            # Protect stdin before initializing GeminiBatchChartAnalyzer
            saved_stdin = None
            if sys.platform == 'win32' and sys.stdin is not None:
                try:
                    saved_stdin = sys.stdin
                    if hasattr(sys.stdin, 'closed') and sys.stdin.closed:
                        try:
                            sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
                            saved_stdin = sys.stdin
                        except (OSError, IOError):
                            pass
                except (AttributeError, ValueError, OSError, IOError):
                    pass
            
            try:
                self._gemini_analyzer = GeminiBatchChartAnalyzer(cooldown_seconds=self._gemini_analyzer_cooldown)
            finally:
                # Restore stdin after initialization
                if sys.platform == 'win32' and saved_stdin is not None:
                    try:
                        if sys.stdin is None or (hasattr(sys.stdin, 'closed') and sys.stdin.closed):
                            if saved_stdin is not None and not (hasattr(saved_stdin, 'closed') and saved_stdin.closed):
                                sys.stdin = saved_stdin
                            else:
                                try:
                                    sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
                                except (OSError, IOError):
                                    pass
                    except (AttributeError, ValueError, OSError, IOError):
                        pass
            
        
        return self._gemini_analyzer
    
    def cleanup(self, force_gc: bool = False):
        """
        Cleanup resources and free memory by clearing caches and forcing garbage collection.
        
        This method:
        - Clears cached data in exchange managers
        - Always triggers garbage collection to free memory
        - If force_gc is True, performs an additional GC cycle for more aggressive cleanup
        
        Call this after scan_market() completes to free exchange connections and other resources.
        
        Args:
            force_gc: If True, perform an additional garbage collection cycle (default: False)
        """
        import gc
        
        # Cleanup exchange connections and clear their caches
        # Use separate try-except blocks so one failure doesn't prevent the other from running
        try:
            if hasattr(self.exchange_manager, 'cleanup_unused_exchanges'):
                self.exchange_manager.cleanup_unused_exchanges()
            if hasattr(self.exchange_manager, 'clear'):
                self.exchange_manager.clear()
        except Exception as e:
            log_warn(f"Error cleaning up exchange manager: {e}")
        
        try:
            if hasattr(self.public_exchange_manager, 'cleanup_unused_exchanges'):
                self.public_exchange_manager.cleanup_unused_exchanges()
            if hasattr(self.public_exchange_manager, 'clear'):
                self.public_exchange_manager.clear()
        except Exception as e:
            log_warn(f"Error cleaning up public exchange manager: {e}")
        
        # Force garbage collection
        # Always call gc.collect() to free memory and resources
        gc.collect()
        if force_gc:
            # If force_gc is True, call it again for more aggressive cleanup
            gc.collect()
            log_info("Forced garbage collection")
        else:
            log_info("Garbage collection completed")
        
        log_info("Cleaned up MarketBatchScanner resources")
    
    def scan_market(
        self,
        timeframe: Optional[str] = '1h',
        timeframes: Optional[List[str]] = None,
        max_symbols: Optional[int] = None,
        limit: int = 500,
        cancelled_callback: Optional[Callable[[], bool]] = None,
        initial_symbols: Optional[List[str]] = None
    ) -> Dict:
        """
        Scan entire market and return LONG/SHORT signals.
        
        Args:
            timeframe: Single timeframe for charts (default: '1h', ignored if timeframes provided)
            timeframes: List of timeframes for multi-timeframe analysis (enables multi-TF mode)
            max_symbols: Maximum number of symbols to scan (None = all). Applied after initial_symbols if provided.
            limit: Number of candles to fetch per symbol (default: 500)
            cancelled_callback: Optional callable that returns bool; True indicates cancellation and stops processing (default: None)
            initial_symbols: Optional list of symbols already pre-filtered from external pre-filter (default: None)
            
        Returns:
            Dictionary with:
            - 'long_symbols': List of LONG symbols
            - 'short_symbols': List of SHORT symbols
            - 'none_symbols': List of symbols with no signal
            - 'all_results': Full results dict {symbol: signal}
            - 'summary': Summary statistics
        """
        
        log_info("=" * 60)
        log_info("MARKET BATCH SCANNER")
        log_info("=" * 60)
        
        # Determine if multi-timeframe mode
        is_multi_tf = timeframes is not None and len(timeframes) > 0
        if is_multi_tf:
            from modules.gemini_chart_analyzer.core.utils import normalize_timeframes
            from modules.gemini_chart_analyzer.core.generators.chart_multi_timeframe_batch_generator import ChartMultiTimeframeBatchGenerator
            from modules.gemini_chart_analyzer.core.aggregators.signal_aggregator import SignalAggregator
            
            normalized_tfs = normalize_timeframes(timeframes)
            log_info(f"Multi-timeframe mode: {', '.join(normalized_tfs)}")
            
            # Use multi-TF batch chart generator
            multi_tf_generator = ChartMultiTimeframeBatchGenerator(
                charts_per_batch=self.MULTI_TF_CHARTS_PER_BATCH,
                timeframes_per_symbol=len(normalized_tfs)
            )
            signal_aggregator = SignalAggregator()
        else:
            normalized_tfs = [timeframe] if timeframe else ['1h']
            log_info(f"Single timeframe mode: {normalized_tfs[0]}")
        
        # Step 0: Cleanup old batch scan results
        self._cleanup_old_results()
        
        # Step 1: Get all symbols
        if initial_symbols is not None:
            # Use pre-filtered symbols from external pre-filter
            log_info("Step 1: Using pre-filtered symbols from external pre-filter...")
            all_symbols = initial_symbols
            log_info(f"Using {len(all_symbols)} pre-filtered symbols")
        else:
            # Get symbols from exchange
            log_info("Step 1: Getting all symbols from exchange...")
            try:
                all_symbols = self.get_all_symbols()
            except SymbolFetchError as e:
                log_error(f"Failed to fetch symbols from exchange: {e}")
                if e.is_retryable:
                    log_error("This was a retryable error (network/rate limit). Please check your connection and try again.")
                else:
                    log_error("This was a non-retryable error. Please check exchange configuration and API access.")
                # Re-raise to let caller handle the failure appropriately
                raise
            
            if not all_symbols:
                log_warn("No symbols found matching the criteria. This may indicate:")
                log_warn(f"  - No active spot markets for {self.quote_currency} on {self.exchange_name}")
                log_warn(f"  - Exchange API returned empty market list")
                log_warn("Continuing with empty symbol list...")
        
        # Apply max_symbols (after initial_symbols if provided)
        if max_symbols and all_symbols:
            all_symbols = all_symbols[:max_symbols]
            log_info(f"Limited to {max_symbols} symbols")
        
        log_success(f"Found {len(all_symbols)} symbols to scan")
        
        # Step 2: Split into batches
        batch_size = self.MULTI_TF_CHARTS_PER_BATCH if is_multi_tf else self.charts_per_batch
        batches = self._split_into_batches(all_symbols, batch_size=batch_size)
        
        log_info(f"Split into {len(batches)} batches ({batch_size} symbols per batch)")
        
        # Step 3: Process each batch
        all_results = {}
        batch_results = []
        
        for batch_idx, batch_symbols in enumerate(batches, 1):
            # Check if cancelled before processing batch
            if cancelled_callback and cancelled_callback():
                log_warn("Scan cancelled by user")
                log_info(f"Processed {batch_idx - 1}/{len(batches)} batches before cancellation")
                
                # Extract and categorize partial results
                (
                    long_symbols,
                    short_symbols,
                    none_symbols,
                    long_symbols_with_confidence,
                    short_symbols_with_confidence,
                    none_symbols_with_confidence
                ) = self._categorize_and_sort_results(all_results)
                
                # Return partial results
                return {
                    'status': 'cancelled',
                    'long_symbols': long_symbols,
                    'short_symbols': short_symbols,
                    'none_symbols': none_symbols,
                    'long_symbols_with_confidence': long_symbols_with_confidence,
                    'short_symbols_with_confidence': short_symbols_with_confidence,
                    'none_symbols_with_confidence': none_symbols_with_confidence,
                    'all_results': all_results,
                    'summary': {
                        'total': len(all_results),
                        'long': len([r for r in all_results.values() if r.get('signal') == 'LONG']),
                        'short': len([r for r in all_results.values() if r.get('signal') == 'SHORT']),
                        'none': len([r for r in all_results.values() if r.get('signal') == 'NONE']),
                    },
                    'results_file': '',
                    'batches_processed': batch_idx - 1,
                    'total_batches': len(batches)
                }            
            log_info(f"\n{'='*60}")
            log_info(f"Processing batch {batch_idx}/{len(batches)}")
            log_info(f"{'='*60}")
            
            try:
                if is_multi_tf:
                    # Multi-timeframe: Fetch data for all timeframes
                    log_info(f"Fetching OHLCV data for {len(batch_symbols)} symbols across {len(normalized_tfs)} timeframes...")
                    symbols_tf_data = {}  # {symbol: {timeframe: df}}
                    
                    for symbol in batch_symbols:
                        symbols_tf_data[symbol] = {}
                        for tf in normalized_tfs:
                            try:
                                df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                                    symbol=symbol,
                                    timeframe=tf,
                                    limit=limit,
                                    check_freshness=False
                                )
                                if df is not None and not df.empty and len(df) >= self.min_candles:
                                    symbols_tf_data[symbol][tf] = df
                            except Exception as e:
                                log_error(f"Error fetching {symbol} {tf}: {e}")
                    
                    # Filter symbols that have data for at least one timeframe
                    valid_symbols = {sym for sym, tf_data in symbols_tf_data.items() if tf_data}
                    
                    if not valid_symbols:
                        log_warn(f"No valid data for batch {batch_idx}, skipping...")
                        for symbol in batch_symbols:
                            all_results[symbol] = {"signal": "NONE", "confidence": 0.0}
                        continue
                    
                    log_success(f"Fetched data for {len(valid_symbols)} symbols")
                    
                    # Generate multi-TF batch chart
                    log_info("Generating multi-timeframe batch chart image...")
                    batch_chart_path, truncated = multi_tf_generator.create_multi_tf_batch_chart(
                        symbols_data=symbols_tf_data,
                        timeframes=normalized_tfs,
                        batch_id=batch_idx
                    )
                    
                    # Analyze with Gemini (multi-TF prompt)
                    log_info("Sending to Gemini for multi-timeframe analysis...")
                    parsed_results = self.batch_gemini_analyzer.analyze_multi_tf_batch_chart(
                        batch_chart_path=batch_chart_path,
                        symbols=list(valid_symbols),
                        normalized_timeframes=normalized_tfs
                    )
                    
                    # Aggregate signals for each symbol (if aggregated not provided by Gemini)
                    batch_result = {}
                    for symbol in valid_symbols:
                        if symbol in parsed_results:
                            symbol_result = parsed_results[symbol]
                            tf_signals = symbol_result.get('timeframes', {})
                            aggregated = symbol_result.get('aggregated')
                            
                            # If aggregated signal not provided by Gemini, calculate it
                            if aggregated is None:
                                aggregated = signal_aggregator.aggregate_signals(tf_signals)
                            
                            batch_result[symbol] = {
                                'timeframes': tf_signals,
                                'aggregated': aggregated
                            }
                        else:
                            # Symbol not found in parsed results (shouldn't happen, but handle gracefully)
                            log_warn(f"Symbol {symbol} not found in parsed multi-TF results")
                            batch_result[symbol] = {
                                'timeframes': {tf: {'signal': 'NONE', 'confidence': 0.0} for tf in normalized_tfs},
                                'aggregated': {'signal': 'NONE', 'confidence': 0.0}
                            }
                    
                else:
                    # Single timeframe: Original logic
                    # Fetch OHLCV data for batch
                    log_info(f"Fetching OHLCV data for {len(batch_symbols)} symbols...")
                    symbols_data = self._fetch_batch_data(batch_symbols, normalized_tfs[0], limit)
                    
                    if not symbols_data:
                        log_warn(f"No data fetched for batch {batch_idx}, skipping...")
                        # Mark all as NONE
                        for symbol in batch_symbols:
                            all_results[symbol] = {"signal": "NONE", "confidence": 0.0}
                        continue
                    
                    log_success(f"Fetched data for {len(symbols_data)} symbols")
                    
                    # Generate batch chart
                    log_info("Generating batch chart image...")
                    batch_chart_path, truncated = self.batch_chart_generator.create_batch_chart(
                        symbols_data=symbols_data,
                        timeframe=normalized_tfs[0],
                        batch_id=batch_idx
                    )
                    if truncated:
                        log_warn(f"Batch {batch_idx}: Input symbols list was truncated to {self.charts_per_batch} items")
                    
                    # Analyze with Gemini
                    log_info("Sending to Gemini for analysis...")
                    batch_result = self.batch_gemini_analyzer.analyze_batch_chart(
                        image_path=batch_chart_path,
                        batch_id=batch_idx,
                        total_batches=len(batches),
                        symbols=[sd['symbol'] for sd in symbols_data]
                    )
                
                # Merge results
                if is_multi_tf:
                    # For multi-TF, extract aggregated signals
                    for symbol, result in batch_result.items():
                        if isinstance(result, dict) and 'aggregated' in result:
                            all_results[symbol] = result['aggregated']
                        else:
                            all_results[symbol] = {"signal": "NONE", "confidence": 0.0}
                    
                    batch_results.append({
                        'batch_id': batch_idx,
                        'symbols': list(valid_symbols),
                        'results': batch_result
                    })
                    
                    # Handle symbols that failed
                    for symbol in batch_symbols:
                        if symbol not in valid_symbols:
                            all_results[symbol] = {"signal": "NONE", "confidence": 0.0}
                else:
                    # Single timeframe: Original logic
                    all_results.update(batch_result)
                    batch_results.append({
                        'batch_id': batch_idx,
                        'symbols': [sd['symbol'] for sd in symbols_data],
                        'results': batch_result
                    })
                    
                    # Handle symbols that failed to fetch data
                    fetched_symbols = {sd['symbol'] for sd in symbols_data}
                    for symbol in batch_symbols:
                        if symbol not in fetched_symbols:
                            all_results[symbol] = {"signal": "NONE", "confidence": 0.0}
                
            except Exception as e:
                log_error(f"Error processing batch {batch_idx}: {e}")
                # Mark all symbols in this batch as NONE
                for symbol in batch_symbols:
                    if symbol not in all_results:
                        all_results[symbol] = {"signal": "NONE", "confidence": 0.0}
        
        # Step 4: Aggregate and sort results by confidence
        log_info(f"\n{'='*60}")
        log_info("Aggregating and sorting results by confidence...")
        log_info(f"{'='*60}")
        
        # Extract signals and confidence
        (
            long_symbols,
            short_symbols,
            none_symbols,
            long_symbols_with_confidence,
            short_symbols_with_confidence,
            none_symbols_with_confidence
        ) = self._categorize_and_sort_results(all_results)
        
        summary = {
            'total_symbols': len(all_symbols),
            'scanned_symbols': len(all_results),
            'long_count': len(long_symbols),
            'short_count': len(short_symbols),
            'none_count': len(none_symbols),
            'long_percentage': (len(long_symbols) / len(all_results) * 100) if all_results else 0,
            'short_percentage': (len(short_symbols) / len(all_results) * 100) if all_results else 0,
            'avg_long_confidence': sum(c for _, c in long_symbols_with_confidence) / len(long_symbols_with_confidence) if long_symbols_with_confidence else 0.0,
            'avg_short_confidence': sum(c for _, c in short_symbols_with_confidence) / len(short_symbols_with_confidence) if short_symbols_with_confidence else 0.0,
        }
        
        # Save results
        # For single TF: pass string, for multi-TF: pass first TF as primary, timeframes list separately
        primary_timeframe = normalized_tfs[0] if normalized_tfs else '1h'
        results_file = self._save_results(
            all_results, long_symbols, short_symbols, summary, primary_timeframe,
            long_symbols_with_confidence, short_symbols_with_confidence,
            timeframes=normalized_tfs if is_multi_tf else None
        )
        
        log_success(f"\n{'='*60}")
        log_success("SCAN COMPLETED")
        log_success(f"{'='*60}")
        log_success(f"Total symbols: {summary['total_symbols']}")
        log_success(f"LONG signals: {summary['long_count']} ({summary['long_percentage']:.1f}%)")
        if summary.get('avg_long_confidence', 0) > 0:
            log_success(f"  Average LONG confidence: {summary['avg_long_confidence']:.2f}")
        log_success(f"SHORT signals: {summary['short_count']} ({summary['short_percentage']:.1f}%)")
        if summary.get('avg_short_confidence', 0) > 0:
            log_success(f"  Average SHORT confidence: {summary['avg_short_confidence']:.2f}")
        log_success(f"Results saved to: {results_file}")
        
        return {
            'long_symbols': long_symbols,  # Sorted by confidence (high to low)
            'short_symbols': short_symbols,  # Sorted by confidence (high to low)
            'none_symbols': none_symbols,
            'long_symbols_with_confidence': long_symbols_with_confidence,  # [(symbol, confidence), ...]
            'short_symbols_with_confidence': short_symbols_with_confidence,  # [(symbol, confidence), ...]
            'all_results': all_results,
            'summary': summary,
            'results_file': results_file
        }
    
    def get_all_symbols(self, max_retries: int = 3, retry_delay: float = 1.0) -> List[str]:
        """
        Get all trading symbols from exchange with retry logic for transient errors.
        
        Args:
            max_retries: Maximum number of retry attempts for transient errors (default: 3)
            retry_delay: Initial delay in seconds for exponential backoff (default: 1.0)
        
        Returns:
            List of symbol strings (e.g., ['BTC/USDT', 'ETH/USDT', ...])
            Empty list if no symbols found (but no error occurred)
        
        Raises:
            SymbolFetchError: If symbol fetching fails after all retries or encounters
                           a non-retryable error. The exception includes information
                           about whether the error is retryable and the original exception.
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Use public exchange manager (no credentials needed for load_markets)
                exchange = self.public_exchange_manager.connect_to_exchange_with_no_credentials(self.exchange_name)
                
                # Load markets
                markets = exchange.load_markets()
                
                # Filter by quote currency and active status
                symbols = []
                for symbol, market in markets.items():
                    if (market.get('quote') == self.quote_currency and 
                        market.get('active', True) and
                        market.get('type') == 'spot'):  # Only spot markets
                        symbols.append(symbol)
                
                # Sort alphabetically
                symbols.sort()
                
                # Success - return symbols (empty list is valid if no symbols match criteria)
                return symbols
                
            except Exception as e:
                last_exception = e
                error_message = str(e)
                
                # Determine if error is retryable (network errors, rate limits, temporary unavailability)
                error_code = None
                if hasattr(e, 'status_code'):
                    error_code = e.status_code
                elif hasattr(e, 'code'):
                    error_code = e.code
                elif '503' in error_message or 'UNAVAILABLE' in error_message.upper():
                    error_code = 503
                elif '429' in error_message or 'RATE_LIMIT' in error_message.upper():
                    error_code = 429
                
                is_retryable = (
                    error_code in [503, 429] or
                    'overloaded' in error_message.lower() or
                    'rate limit' in error_message.lower() or
                    'unavailable' in error_message.lower() or
                    'timeout' in error_message.lower() or
                    'connection' in error_message.lower() or
                    'network' in error_message.lower()
                )
                
                # Log the error
                if attempt < max_retries - 1 and is_retryable:
                    wait_time = retry_delay * (2 ** attempt)
                    log_warn(
                        f"Retryable error getting symbols (attempt {attempt + 1}/{max_retries}): {error_message}. "
                        f"Waiting {wait_time}s before retrying..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error or final attempt failed
                    log_error(f"Error getting symbols: {error_message}")
                    if is_retryable:
                        # Final retry failed
                        raise SymbolFetchError(
                            f"Failed to fetch symbols after {max_retries} attempts: {error_message}",
                            original_exception=e,
                            is_retryable=True
                        ) from e
                    else:
                        # Non-retryable error
                        raise SymbolFetchError(
                            f"Failed to fetch symbols (non-retryable error): {error_message}",
                            original_exception=e,
                            is_retryable=False
                        ) from e
        
        # This should never be reached, but just in case
        if last_exception:
            raise SymbolFetchError(
                f"Failed to fetch symbols after {max_retries} attempts",
                original_exception=last_exception,
                is_retryable=True
            ) from last_exception
    
    def _split_into_batches(self, symbols: List[str], batch_size: Optional[int] = None) -> List[List[str]]:
        """
        Split symbols into batches.
        
        Args:
            symbols: List of all symbols
            batch_size: Optional batch size (defaults to self.charts_per_batch if not provided)
            
        Returns:
            List of batches, each containing up to batch_size symbols
        """
        if batch_size is None:
            batch_size = self.charts_per_batch
        
        batches = []
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def _categorize_and_sort_results(
        self, 
        all_results: Dict[str, Any]
    ) -> Tuple[
        List[str], List[str], List[str],
        List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]]
    ]:
        """
        Categorize results into LONG/SHORT/NONE and sort by confidence.
        
        Args:
            all_results: Dictionary mapping symbol to result (dict with 'signal' and 'confidence' or legacy string)
            
        Returns:
            Tuple of 6 lists:
            - long_symbols: List of LONG symbols (sorted by confidence, high to low)
            - short_symbols: List of SHORT symbols (sorted by confidence, high to low)
            - none_symbols: List of NONE symbols (sorted by confidence, high to low)
            - long_symbols_with_confidence: List of (symbol, confidence) tuples for LONG
            - short_symbols_with_confidence: List of (symbol, confidence) tuples for SHORT
            - none_symbols_with_confidence: List of (symbol, confidence) tuples for NONE
        """
        long_symbols_with_confidence = []
        short_symbols_with_confidence = []
        none_symbols_with_confidence = []
        legacy_format_symbols = []
        
        for symbol, result in all_results.items():
            if isinstance(result, dict):
                signal = result.get('signal', 'NONE')
                confidence = result.get('confidence', 0.0)
            else:
                # Backward compatibility: if result is string
                # Use 0.0 confidence to denote fallback/untrusted confidence for legacy string results
                # This ensures real confidence scores remain distinguishable from fallbacks
                signal = result if isinstance(result, str) else 'NONE'
                confidence = 0.0
                legacy_format_symbols.append(symbol)
            
            if signal == 'LONG':
                long_symbols_with_confidence.append((symbol, confidence))
            elif signal == 'SHORT':
                short_symbols_with_confidence.append((symbol, confidence))
            else:
                none_symbols_with_confidence.append((symbol, confidence))
        
        # Log legacy format warning as summary if any found
        if legacy_format_symbols:
            count = len(legacy_format_symbols)
            examples = legacy_format_symbols[:5]  # Show first 5 as examples
            examples_str = ', '.join(examples)
            if count > 5:
                log_warn(f"Legacy format detected for {count} symbol(s) (e.g., {examples_str}, ...): using fallback confidence 0.0")
            else:
                log_warn(f"Legacy format detected for {count} symbol(s) ({examples_str}): using fallback confidence 0.0")
        
        # Sort by confidence (descending)
        long_symbols_with_confidence.sort(key=lambda x: x[1], reverse=True)
        short_symbols_with_confidence.sort(key=lambda x: x[1], reverse=True)
        none_symbols_with_confidence.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just symbols (sorted by confidence)
        long_symbols = [s for s, _ in long_symbols_with_confidence]
        short_symbols = [s for s, _ in short_symbols_with_confidence]
        none_symbols = [s for s, _ in none_symbols_with_confidence]
        
        return (
            long_symbols,
            short_symbols,
            none_symbols,
            long_symbols_with_confidence,
            short_symbols_with_confidence,
            none_symbols_with_confidence
        )
    
    def _fetch_batch_data(
        self,
        symbols: List[str],
        timeframe: str,
        limit: int
    ) -> List[Dict[str, Any]]:        
        """
        Fetch OHLCV data for a batch of symbols.
        
        Args:
            symbols: List of symbols to fetch
            timeframe: Timeframe string
            limit: Number of candles
            
        Returns:
            List of dicts with 'symbol' and 'df' keys
        """
        symbols_data = []
        
        for symbol in symbols:
            try:
                df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    check_freshness=False
                )
                
                if df is not None and not df.empty and len(df) >= self.min_candles:
                    symbols_data.append({
                        'symbol': symbol,
                        'df': df
                    })
                else:
                    log_warn(f"Insufficient data for {symbol}, skipping...")
                    
            except Exception as e:
                log_warn(f"Error fetching {symbol}: {e}, skipping...")
                continue
        
        return symbols_data
    
    def _save_results(
        self,
        all_results: Dict[str, Any],
        long_symbols: List[str],
        short_symbols: List[str],
        summary: Dict,
        timeframe: str,
        long_with_confidence: Optional[List[Tuple[str, float]]] = None,
        short_with_confidence: Optional[List[Tuple[str, float]]] = None,
        timeframes: Optional[List[str]] = None
    ) -> str:
        """
        Save scan results to JSON file.
        
        Args:
            all_results: Full results dictionary
            long_symbols: List of LONG symbols
            short_symbols: List of SHORT symbols
            summary: Summary statistics
            timeframe: Timeframe string (or primary timeframe for multi-TF)
            long_with_confidence: List of (symbol, confidence) tuples for LONG
            short_with_confidence: List of (symbol, confidence) tuples for SHORT
            timeframes: Optional list of timeframes (for multi-TF mode)
            
        Returns:
            Path to saved results file
        """
        results_base_dir = get_analysis_results_dir()
        output_dir = os.path.join(results_base_dir, "batch_scan")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if timeframes:
            tf_str = "_".join(timeframes)
            results_file = os.path.join(output_dir, f"batch_scan_multi_tf_{tf_str}_{timestamp}.json")
        else:
            results_file = os.path.join(output_dir, f"batch_scan_{timeframe}_{timestamp}.json")
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'timeframe': timeframe,
            'timeframes': timeframes if timeframes else [timeframe],
            'summary': summary,
            'long_symbols': long_symbols,  # Sorted by confidence
            'short_symbols': short_symbols,  # Sorted by confidence
            'long_symbols_with_confidence': long_with_confidence or [],
            'short_symbols_with_confidence': short_with_confidence or [],
            'all_results': all_results
        }
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            log_error(f"Failed to save results file {results_file}: {e}")
            raise IOError(f"Failed to save results file: {e}") from e
        except Exception as e:
            log_error(f"Unexpected error saving results file: {e}")
            raise RuntimeError(f"Unexpected error saving results file: {e}") from e
        
        # Generate HTML report
        html_path = None
        try:
            from modules.gemini_chart_analyzer.core.reporting.html_report import generate_batch_html_report
            html_path = generate_batch_html_report(results_data, output_dir)
            log_success(f"Generated HTML report: {html_path}")
            
            # Open HTML report in browser
            try:
                import webbrowser
                html_uri = Path(html_path).resolve().as_uri()
                webbrowser.open(html_uri)
                log_success("Opened HTML report in browser")
            except Exception as e:
                log_warn(f"Could not open browser automatically: {e}")
                log_info(f"Please open file manually: {html_path}")
        except Exception as e:
            log_warn(f"Could not generate HTML report: {e}")
            # Don't fail the whole operation if HTML generation fails
        
        return results_file
    
    def _cleanup_old_results(self):
        """
        Cleanup old batch scan results and charts before starting new scan.
        Deletes all JSON and HTML files in batch_scan directory and PNG files in charts/batch directory.
        """
        # Cleanup old batch scan results
        try:
            results_base_dir = get_analysis_results_dir()
            batch_scan_dir = os.path.join(results_base_dir, "batch_scan")
            
            if os.path.exists(batch_scan_dir):
                # Find all JSON files in batch_scan directory
                json_files = glob.glob(os.path.join(batch_scan_dir, "batch_scan_*.json"))
                
                deleted_count = 0
                for file_path in json_files:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        log_warn(f"Could not delete file {os.path.basename(file_path)}: {e}")
                
                if deleted_count > 0:
                    log_info(f"Deleted {deleted_count} old batch scan result file(s)")
                
                # Find all HTML files in batch_scan directory
                html_files = glob.glob(os.path.join(batch_scan_dir, "batch_scan_*.html"))
                
                deleted_count = 0
                for file_path in html_files:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        log_warn(f"Could not delete file {os.path.basename(file_path)}: {e}")
                
                if deleted_count > 0:
                    log_info(f"Deleted {deleted_count} old batch scan HTML report file(s)")
                
        except Exception as e:
            log_warn(f"Error cleaning up batch scan results: {e}")
        
        # Cleanup old batch charts
        try:
            from modules.gemini_chart_analyzer.core.utils.chart_paths import get_charts_dir
            charts_dir = get_charts_dir()
            batch_charts_dir = os.path.join(str(charts_dir), "batch")
            
            if os.path.exists(batch_charts_dir):
                # Find all PNG files in batch directory
                png_files = glob.glob(os.path.join(batch_charts_dir, "batch_chart_*.png"))
                
                deleted_count = 0
                for file_path in png_files:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        log_warn(f"Could not delete file {os.path.basename(file_path)}: {e}")
                
                if deleted_count > 0:
                    log_info(f"Deleted {deleted_count} old batch chart file(s)")
                
                # Find all HTML files in batch directory (if any)
                html_files = glob.glob(os.path.join(batch_charts_dir, "batch_chart_*.html"))
                
                deleted_count = 0
                for file_path in html_files:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        log_warn(f"Could not delete file {os.path.basename(file_path)}: {e}")
                
                if deleted_count > 0:
                    log_info(f"Deleted {deleted_count} old batch chart HTML file(s)")
                
        except Exception as e:
            log_warn(f"Error cleaning up batch charts: {e}")


