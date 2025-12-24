"""
Market Batch Scanner for scanning entire market with Gemini.

Orchestrates the workflow: get symbols → batch → generate charts → analyze → aggregate results.
"""

import os
import json
import glob
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

from modules.common.core.exchange_manager import ExchangeManager, PublicExchangeManager
from modules.common.core.data_fetcher import DataFetcher
from modules.gemini_chart_analyzer.core.batch_chart_generator import BatchChartGenerator
from modules.gemini_chart_analyzer.core.batch_gemini_analyzer import BatchGeminiAnalyzer
from modules.common.ui.logging import log_info, log_error, log_success, log_warn


def _get_analysis_results_dir() -> str:
    """Get the analysis results directory path relative to module root."""
    module_root = Path(__file__).parent.parent
    results_dir = module_root / "analysis_results"
    return str(results_dir)


class MarketBatchScanner:
    """Scan entire market by batching symbols and analyzing with Gemini."""
    
    # Minimum number of candles required for reliable technical analysis
    MIN_CANDLES: int = 20
    
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
        
        # Validate MIN_CANDLES constant before using as fallback
        if self.MIN_CANDLES <= 0:
            raise ValueError(f"MIN_CANDLES class constant must be greater than 0, got {self.MIN_CANDLES}")
        
        # Set min_candles with validation
        self.min_candles = min_candles if min_candles is not None else self.MIN_CANDLES
        if self.min_candles <= 0:
            raise ValueError(f"min_candles must be greater than 0, got {self.min_candles}")
        
        # Initialize components
        self.exchange_manager = ExchangeManager()
        self.public_exchange_manager = PublicExchangeManager()  # For load_markets (no credentials needed)
        self.data_fetcher = DataFetcher(self.exchange_manager)
        self.batch_chart_generator = BatchChartGenerator(charts_per_batch=charts_per_batch)
        self.batch_gemini_analyzer = BatchGeminiAnalyzer(cooldown_seconds=cooldown_seconds)
    
    def scan_market(
        self,
        timeframe: Optional[str] = '1h',
        timeframes: Optional[List[str]] = None,
        max_symbols: Optional[int] = None,
        limit: int = 200
    ) -> Dict:
        """
        Scan entire market and return LONG/SHORT signals.
        
        Args:
            timeframe: Single timeframe for charts (default: '1h', ignored if timeframes provided)
            timeframes: List of timeframes for multi-timeframe analysis (enables multi-TF mode)
            max_symbols: Maximum number of symbols to scan (None = all)
            limit: Number of candles to fetch per symbol (default: 200)
            
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
            from modules.gemini_chart_analyzer.core.multi_tf_batch_chart_generator import MultiTFBatchChartGenerator
            from modules.gemini_chart_analyzer.core.signal_aggregator import SignalAggregator
            
            normalized_tfs = normalize_timeframes(timeframes)
            log_info(f"Multi-timeframe mode: {', '.join(normalized_tfs)}")
            
            # Use multi-TF batch chart generator
            multi_tf_generator = MultiTFBatchChartGenerator(
                charts_per_batch=25,  # Reduced because each symbol has multiple TFs
                timeframes_per_symbol=len(normalized_tfs)
            )
            signal_aggregator = SignalAggregator()
        else:
            normalized_tfs = [timeframe] if timeframe else ['1h']
            log_info(f"Single timeframe mode: {normalized_tfs[0]}")
        
        # Step 0: Cleanup old batch scan results
        self._cleanup_old_results()
        
        # Step 1: Get all symbols
        log_info("Step 1: Getting all symbols from exchange...")
        all_symbols = self.get_all_symbols()
        
        if max_symbols:
            all_symbols = all_symbols[:max_symbols]
            log_info(f"Limited to {max_symbols} symbols")
        
        log_success(f"Found {len(all_symbols)} symbols to scan")
        
        # Step 2: Split into batches
        if is_multi_tf:
            # For multi-TF, reduce batch size because each symbol takes more space
            original_batch_size = self.charts_per_batch
            self.charts_per_batch = 25  # Temporary adjustment
            batches = self._split_into_batches(all_symbols)
            self.charts_per_batch = original_batch_size
        else:
            batches = self._split_into_batches(all_symbols)
        
        log_info(f"Split into {len(batches)} batches ({self.charts_per_batch if not is_multi_tf else 25} symbols per batch)")
        
        # Step 3: Process each batch
        all_results = {}
        batch_results = []
        
        for batch_idx, batch_symbols in enumerate(batches, 1):
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
                    prompt = self.batch_gemini_analyzer._create_multi_tf_batch_prompt(list(valid_symbols), normalized_tfs)
                    response_text = self.batch_gemini_analyzer._analyze_with_custom_prompt(batch_chart_path, prompt)
                    
                    # Parse multi-TF JSON response
                    import json
                    import re
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                        json_str = json_match.group(0) if json_match else '{}'
                    
                    try:
                        batch_result_raw = json.loads(json_str)
                    except:
                        batch_result_raw = {}
                    
                    # Aggregate signals for each symbol
                    batch_result = {}
                    for symbol in valid_symbols:
                        if symbol in batch_result_raw:
                            symbol_data = batch_result_raw[symbol]
                            # Extract timeframe signals
                            tf_signals = {}
                            for tf in normalized_tfs:
                                if tf in symbol_data:
                                    tf_signals[tf] = symbol_data[tf]
                            
                            # Aggregate
                            aggregated = signal_aggregator.aggregate_signals(tf_signals)
                            batch_result[symbol] = {
                                'timeframes': tf_signals,
                                'aggregated': aggregated
                            }
                        else:
                            batch_result[symbol] = {
                                'timeframes': {},
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
        long_symbols_with_confidence = []
        short_symbols_with_confidence = []
        none_symbols_with_confidence = []
        
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
                log_warn(f"Legacy format detected for {symbol}: using fallback confidence 0.0")
            
            if signal == 'LONG':
                long_symbols_with_confidence.append((symbol, confidence))
            elif signal == 'SHORT':
                short_symbols_with_confidence.append((symbol, confidence))
            else:
                none_symbols_with_confidence.append((symbol, confidence))
        
        # Sort by confidence (descending)
        long_symbols_with_confidence.sort(key=lambda x: x[1], reverse=True)
        short_symbols_with_confidence.sort(key=lambda x: x[1], reverse=True)
        none_symbols_with_confidence.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just symbols (sorted by confidence)
        long_symbols = [s for s, _ in long_symbols_with_confidence]
        short_symbols = [s for s, _ in short_symbols_with_confidence]
        none_symbols = [s for s, _ in none_symbols_with_confidence]
        
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
        tf_for_save = normalized_tfs if is_multi_tf else normalized_tfs[0]
        results_file = self._save_results(
            all_results, long_symbols, short_symbols, summary, tf_for_save,
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
    
    def get_all_symbols(self) -> List[str]:
        """
        Get all trading symbols from exchange.
        
        Returns:
            List of symbol strings (e.g., ['BTC/USDT', 'ETH/USDT', ...])
        """
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
            
            return symbols
            
        except Exception as e:
            log_error(f"Error getting symbols: {e}")
            # Fallback: return empty list or try other exchanges
            return []
    
    def _split_into_batches(self, symbols: List[str]) -> List[List[str]]:
        """
        Split symbols into batches.
        
        Args:
            symbols: List of all symbols
            
        Returns:
            List of batches, each containing up to charts_per_batch symbols
        """
        batches = []
        for i in range(0, len(symbols), self.charts_per_batch):
            batch = symbols[i:i + self.charts_per_batch]
            batches.append(batch)
        return batches
    
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
        results_base_dir = _get_analysis_results_dir()
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
        
        return results_file
    
    def _cleanup_old_results(self):
        """
        Cleanup old batch scan results and charts before starting new scan.
        Deletes all JSON files in batch_scan directory and PNG files in charts/batch directory.
        """
        # Cleanup old batch scan results
        try:
            results_base_dir = _get_analysis_results_dir()
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
                
        except Exception as e:
            log_warn(f"Error cleaning up batch scan results: {e}")
        
        # Cleanup old batch charts
        try:
            from modules.gemini_chart_analyzer.core.batch_chart_generator import _get_charts_dir
            charts_dir = _get_charts_dir()
            batch_charts_dir = os.path.join(charts_dir, "batch")
            
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
                
        except Exception as e:
            log_warn(f"Error cleaning up batch charts: {e}")

