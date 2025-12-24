"""
Multi-timeframe Analyzer for coordinating analysis across multiple timeframes.

Điều phối phân tích multi-timeframe cho cả Deep Analysis và Batch Analysis modes.
"""

from typing import List, Dict, Optional, Any, Callable
from modules.gemini_chart_analyzer.core.signal_aggregator import SignalAggregator, DEFAULT_TIMEFRAME_WEIGHTS
from modules.gemini_chart_analyzer.core.utils import normalize_timeframes, validate_timeframes
from modules.common.ui.logging import log_info, log_error, log_success


class MultiTimeframeAnalyzer:
    """Điều phối multi-timeframe analysis."""
    
    def __init__(
        self,
        timeframe_weights: Optional[Dict[str, float]] = None
    ):
        """
        Khởi tạo MultiTimeframeAnalyzer.
        
        Args:
            timeframe_weights: Optional custom weights dict
        """
        self.signal_aggregator = SignalAggregator(timeframe_weights)
        self.timeframe_weights = timeframe_weights or DEFAULT_TIMEFRAME_WEIGHTS.copy()
    
    def _validate_timeframes(self, timeframes: List[str]) -> List[str]:
        """
        Validate và normalize timeframes.
        
        Args:
            timeframes: List of timeframe strings
        
        Returns:
            Normalized and validated timeframes list
        
        Raises:
            ValueError: Nếu timeframes không hợp lệ
        """
        if not timeframes:
            raise ValueError("Timeframes list cannot be empty")
        
        is_valid, error_msg = validate_timeframes(timeframes)
        if not is_valid:
            raise ValueError(error_msg or "Invalid timeframes")
        
        return normalize_timeframes(timeframes)
    
    def _calculate_timeframe_weights(self, timeframes: List[str]) -> Dict[str, float]:
        """
        Tính weights cho từng timeframe.
        
        Args:
            timeframes: List of normalized timeframes
        
        Returns:
            Dict mapping timeframe -> weight
        """
        weights = {}
        for tf in timeframes:
            weights[tf] = self.timeframe_weights.get(tf, 0.1)
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {tf: w / total for tf, w in weights.items()}
        
        return weights
    
    def analyze_deep(
        self,
        symbol: str,
        timeframes: List[str],
        fetch_data_func: Callable,
        generate_chart_func: Callable,
        analyze_chart_func: Callable
    ) -> Dict[str, Any]:
        """
        Deep analysis mode: Phân tích riêng từng timeframe.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to analyze
            fetch_data_func: Function to fetch OHLCV data (symbol, timeframe) -> DataFrame
            generate_chart_func: Function to generate chart (df, symbol, timeframe) -> chart_path
            analyze_chart_func: Function to analyze chart (chart_path, symbol, timeframe) -> analysis_result
        
        Returns:
            Dict với kết quả:
            {
                'symbol': str,
                'timeframes': {
                    '15m': {'signal': 'LONG', 'confidence': 0.7, 'analysis': '...'},
                    ...
                },
                'aggregated': {...}
            }
        """
        # Validate timeframes
        normalized_tfs = self._validate_timeframes(timeframes)
        
        log_info(f"Starting deep analysis for {symbol} across {len(normalized_tfs)} timeframes: {', '.join(normalized_tfs)}")
        
        timeframe_results = {}
        
        # Analyze each timeframe
        for tf in normalized_tfs:
            try:
                log_info(f"Analyzing {symbol} on {tf} timeframe...")
                
                # Fetch data
                df = fetch_data_func(symbol, tf)
                if df is None or df.empty:
                    log_error(f"No data for {symbol} on {tf}, skipping...")
                    timeframe_results[tf] = {
                        'signal': 'NONE',
                        'confidence': 0.0,
                        'analysis': f'No data available for {tf}',
                        'error': 'No data'
                    }
                    continue
                
                # Generate chart
                chart_path = generate_chart_func(df, symbol, tf)
                
                # Analyze with Gemini
                analysis_result = analyze_chart_func(chart_path, symbol, tf)
                
                # Store analysis result
                # Note: Signal parsing from Gemini text analysis is complex and would require
                # NLP or structured prompts. For now, we store the analysis text.
                # The signal aggregation will use default values, but the analysis text
                # is available for manual review or future enhancement.
                timeframe_results[tf] = {
                    'signal': 'NONE',  # Default - can be enhanced with signal parsing
                    'confidence': 0.5,  # Default - can be enhanced with confidence extraction
                    'analysis': analysis_result  # Full analysis text from Gemini
                }
                
                log_success(f"Completed analysis for {symbol} on {tf}")
                
            except Exception as e:
                log_error(f"Error analyzing {symbol} on {tf}: {e}")
                timeframe_results[tf] = {
                    'signal': 'NONE',
                    'confidence': 0.0,
                    'analysis': f'Error: {str(e)}',
                    'error': str(e)
                }
        
        # Aggregate signals
        timeframe_signals = {
            tf: {
                'signal': result.get('signal', 'NONE'),
                'confidence': result.get('confidence', 0.0)
            }
            for tf, result in timeframe_results.items()
        }
        
        aggregated = self.signal_aggregator.aggregate_signals(timeframe_signals)
        
        return {
            'symbol': symbol,
            'timeframes': timeframe_results,
            'aggregated': aggregated
        }
    
    def analyze_batch(
        self,
        symbols: List[str],
        timeframes: List[str],
        fetch_data_func: Callable,
        generate_batch_chart_func: Callable,
        analyze_batch_chart_func: Callable
    ) -> Dict[str, Any]:
        """
        Batch analysis mode: Gộp nhiều timeframes vào batch chart.
        
        Args:
            symbols: List of symbols to analyze
            timeframes: List of timeframes to analyze
            fetch_data_func: Function to fetch OHLCV data (symbol, timeframe) -> DataFrame
            generate_batch_chart_func: Function to generate multi-TF batch chart (symbols_data, timeframes) -> chart_path
            analyze_batch_chart_func: Function to analyze batch chart (chart_path, symbols, timeframes) -> results
        
        Returns:
            Dict với kết quả cho từng symbol:
            {
                'BTC/USDT': {
                    'timeframes': {
                        '15m': {'signal': 'LONG', 'confidence': 0.7},
                        ...
                    },
                    'aggregated': {...}
                },
                ...
            }
        """
        # Validate timeframes
        normalized_tfs = self._validate_timeframes(timeframes)
        
        log_info(f"Starting batch analysis for {len(symbols)} symbols across {len(normalized_tfs)} timeframes")
        
        # Fetch data for all symbols and timeframes
        symbols_data = {}
        for symbol in symbols:
            symbols_data[symbol] = {}
            for tf in normalized_tfs:
                try:
                    df = fetch_data_func(symbol, tf)
                    if df is not None and not df.empty:
                        symbols_data[symbol][tf] = df
                except Exception as e:
                    log_error(f"Error fetching {symbol} on {tf}: {e}")
        
        # Generate multi-TF batch chart
        chart_path = generate_batch_chart_func(symbols_data, normalized_tfs)
        
        # Analyze with Gemini
        batch_results = analyze_batch_chart_func(chart_path, symbols, normalized_tfs)
        
        # Aggregate signals for each symbol
        final_results = {}
        for symbol in symbols:
            if symbol not in batch_results:
                # No result for this symbol
                final_results[symbol] = {
                    'timeframes': {},
                    'aggregated': {
                        'signal': 'NONE',
                        'confidence': 0.0
                    }
                }
                continue
            
            symbol_tf_results = batch_results[symbol]
            
            # Extract timeframe signals
            timeframe_signals = {}
            for tf in normalized_tfs:
                if tf in symbol_tf_results:
                    timeframe_signals[tf] = symbol_tf_results[tf]
            
            # Aggregate
            aggregated = self.signal_aggregator.aggregate_signals(timeframe_signals)
            
            final_results[symbol] = {
                'timeframes': symbol_tf_results,
                'aggregated': aggregated
            }
        
        return final_results

