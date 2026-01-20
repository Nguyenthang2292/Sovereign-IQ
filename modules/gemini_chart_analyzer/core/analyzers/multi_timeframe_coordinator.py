"""
Multi-timeframe Analyzer for coordinating analysis across multiple timeframes.

Coordinates multi-timeframe analysis for both Deep Analysis and Batch Analysis modes.
"""

import os
from typing import Any, Callable, Dict, List, Optional

# Import timeframe weights from config
from config.gemini_chart_analyzer import TIMEFRAME_WEIGHTS
from modules.common.ui.logging import log_debug, log_error, log_info, log_success, log_warn
from modules.gemini_chart_analyzer.core.aggregators.signal_aggregator import SignalAggregator
from modules.gemini_chart_analyzer.core.exceptions import (
    ChartGenerationError,
    DataFetchError,
    GeminiAnalysisError,
    ScanConfigurationError,
)
from modules.gemini_chart_analyzer.core.utils import normalize_timeframes, validate_timeframes


class MultiTimeframeCoordinator:
    """Coordinator for multi-timeframe analysis."""

    def __init__(self, timeframe_weights: Optional[Dict[str, float]] = None):
        """
        Initialize MultiTimeframeCoordinator.

        Args:
            timeframe_weights: Optional custom weights dict
        """
        self.signal_aggregator = SignalAggregator(timeframe_weights)
        self.timeframe_weights = timeframe_weights or TIMEFRAME_WEIGHTS.copy()

    def _validate_timeframes(self, timeframes: List[str]) -> List[str]:
        """
        Validate and normalize timeframes.

        Args:
            timeframes: List of timeframe strings

        Returns:
            Normalized and validated list of timeframes

        Raises:
            ValueError: If the timeframes are invalid
        """
        if not timeframes:
            raise ScanConfigurationError("Timeframes list cannot be empty")

        is_valid, error_msg = validate_timeframes(timeframes)
        if not is_valid:
            raise ScanConfigurationError(error_msg or "Invalid timeframes")

        return normalize_timeframes(timeframes)

    def _calculate_timeframe_weights(self, timeframes: List[str]) -> Dict[str, float]:
        """
        Calculate weights for each timeframe.

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
        else:
            # If total = 0 (all weights = 0), assign equal weights
            if weights:
                equal_weight = 1.0 / len(weights)
                weights = {tf: equal_weight for tf in weights.keys()}

        return weights

    def _create_empty_symbol_result(self) -> Dict[str, Any]:
        """
        Create an empty result structure for a symbol.

        Returns:
            Dict with empty timeframes and default aggregated signal
        """
        return {"timeframes": {}, "aggregated": {"signal": "NONE", "confidence": 0.0}}

    def _safe_remove_chart(self, chart_path: Optional[str]) -> None:
        """
        Safely remove a chart file if it exists.

        This helper ensures consistent chart cleanup lifecycle by validating
        the chart_path before attempting deletion. Charts are temporary files
        created for analysis and should be cleaned up after use to prevent
        resource leaks.

        Args:
            chart_path: Path to chart file (may be None or non-string)
        """
        if not chart_path:
            return

        if not isinstance(chart_path, str):
            log_debug(f"Chart path is not a string, skipping cleanup: {type(chart_path)}")
            return

        if not os.path.exists(chart_path):
            log_debug(f"Chart file does not exist, skipping cleanup: {chart_path}")
            return

        try:
            os.remove(chart_path)
            log_debug(f"Successfully removed chart file: {chart_path}")

        except Exception as e:
            # Sanitize error message to prevent information leakage
            log_error(f"Failed to remove chart file {chart_path}: {type(e).__name__}")

    def analyze_deep(
        self,
        symbol: str,
        timeframes: List[str],
        fetch_data_func: Callable,
        generate_chart_func: Callable,
        analyze_chart_func: Callable,
    ) -> Dict[str, Any]:
        """
        Deep analysis mode: Analyze each timeframe independently.

        **IMPORTANT LIMITATION**: This mode does NOT automatically parse signals from Gemini's
        text analysis. All timeframe results will have signal='NONE' and confidence=0.5 by default.
        The full analysis text is available in the 'analysis' field for manual review.

        To get actual signals, consider using:
        - Batch analysis mode (analyze_batch) which uses structured JSON prompts
        - Or implement signal parsing logic using NLP/structured prompts

        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to analyze
            fetch_data_func: Function to fetch OHLCV data (symbol, timeframe) -> DataFrame
            generate_chart_func: Function to generate chart (df, symbol, timeframe) -> chart_path
            analyze_chart_func: Function to analyze chart (chart_path, symbol, timeframe) -> analysis_result

        Returns:
            Dict with results:
            {
                'symbol': str,
                'timeframes': {
                    '15m': {'signal': 'NONE', 'confidence': 0.5, 'analysis': '...'},
                    ...
                },
                'aggregated': {'signal': 'NONE', 'confidence': 0.5, ...}
            }

        Note:
            The aggregated signal will always be 'NONE' since all timeframe signals are 'NONE'.
            Use this mode primarily for detailed text analysis, not for signal-based trading decisions.
        """
        # Validate timeframes
        normalized_tfs = self._validate_timeframes(timeframes)

        log_info(
            f"Starting deep analysis for {symbol} across {len(normalized_tfs)} timeframes: {', '.join(normalized_tfs)}"
        )
        log_warn(
            "Deep analysis mode: Signals will be 'NONE' by default. Full analysis text is available for manual review."
        )

        timeframe_results = {}

        # Analyze each timeframe
        for tf in normalized_tfs:
            chart_path = None
            try:
                log_info(f"Analyzing {symbol} on {tf} timeframe...")

                # Fetch data
                df = fetch_data_func(symbol, tf)
                if df is None or df.empty:
                    log_error(f"No data for {symbol} on {tf}, skipping...")
                    timeframe_results[tf] = {
                        "signal": "NONE",
                        "confidence": 0.0,
                        "analysis": f"No data available for {tf}",
                        "error": "No data",
                    }
                    continue

                # Generate chart
                # Chart files are temporary and should be cleaned up after analysis
                chart_path = generate_chart_func(df, symbol, tf)

                # Analyze with Gemini
                analysis_result = analyze_chart_func(chart_path, symbol, tf)

                # Store analysis result
                # Note: Signal parsing from Gemini text analysis is complex and would require
                # NLP or structured prompts. For now, we store the analysis text.
                # The signal aggregation will use default values, but the analysis text
                # is available for manual review or future enhancement.
                timeframe_results[tf] = {
                    "signal": "NONE",  # Default - can be enhanced with signal parsing
                    "confidence": 0.5,  # Default - can be enhanced with confidence extraction
                    "analysis": analysis_result,  # Full analysis text from Gemini
                }

                log_success(f"Completed analysis for {symbol} on {tf}")

            except GeminiAnalysisError as e:
                log_error(f"Gemini analysis error for {symbol} on {tf}: {e}")
                timeframe_results[tf] = {
                    "signal": "NONE",
                    "confidence": 0.0,
                    "analysis": "Gemini analysis error",
                    "error": str(e),
                }
            except ChartGenerationError as e:
                log_error(f"Chart generation error for {symbol} on {tf}: {e}")
                timeframe_results[tf] = {
                    "signal": "NONE",
                    "confidence": 0.0,
                    "analysis": "Chart generation error",
                    "error": str(e),
                }
            except DataFetchError as e:
                log_error(f"Data fetch error for {symbol} on {tf}: {e}")
                timeframe_results[tf] = {
                    "signal": "NONE",
                    "confidence": 0.0,
                    "analysis": "Data fetch error",
                    "error": str(e),
                }
            except (OSError, IOError) as e:
                log_error(f"File I/O error analyzing {symbol} on {tf}: {e}")
                timeframe_results[tf] = {
                    "signal": "NONE",
                    "confidence": 0.0,
                    "analysis": "File I/O error",
                    "error": type(e).__name__,
                }
            except Exception as e:
                # Sanitize error message to prevent information leakage
                log_error(f"Unexpected error analyzing {symbol} on {tf}: {type(e).__name__}")
                timeframe_results[tf] = {
                    "signal": "NONE",
                    "confidence": 0.0,
                    "analysis": "Unexpected error",
                    "error": type(e).__name__,
                }
            finally:
                # Ensure chart cleanup happens on both success and failure paths
                # This prevents resource leaks and ensures consistent lifecycle management
                self._safe_remove_chart(chart_path)

        # Aggregate signals
        timeframe_signals = {
            tf: {"signal": result.get("signal", "NONE"), "confidence": result.get("confidence", 0.0)}
            for tf, result in timeframe_results.items()
        }

        aggregated = self.signal_aggregator.aggregate_signals(timeframe_signals)

        return {"symbol": symbol, "timeframes": timeframe_results, "aggregated": aggregated}

    def analyze_batch(
        self,
        symbols: List[str],
        timeframes: List[str],
        fetch_data_func: Callable,
        generate_batch_chart_func: Callable,
        analyze_batch_chart_func: Callable,
    ) -> Dict[str, Any]:
        """
        Batch analysis mode: Combine multiple timeframes into a batch chart.

        Args:
            symbols: List of symbols to analyze
            timeframes: List of timeframes to analyze
            fetch_data_func: Function to fetch OHLCV data (symbol, timeframe) -> DataFrame
            generate_batch_chart_func: Function to generate a multi-timeframe batch chart
                (symbols_data, timeframes) -> chart_path
            analyze_batch_chart_func: Function to analyze the batch chart (chart_path, symbols, timeframes) -> results

        Returns:
            Dict with results for each symbol:
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
                    # Sanitize error message to prevent information leakage
                    log_error(f"Error fetching {symbol} on {tf}: {type(e).__name__}")

        # Generate multi-TF batch chart
        chart_path = None
        try:
            chart_path = generate_batch_chart_func(symbols_data, normalized_tfs)
        except (ChartGenerationError, OSError, IOError) as e:
            # Sanitize error message to prevent information leakage
            log_error(f"Error generating batch chart: {e}")
            # Return empty batch_results structure for all symbols
            return {symbol: self._create_empty_symbol_result() for symbol in symbols}
        except Exception as e:
            # Sanitize error message to prevent information leakage
            log_error(
                f"Unexpected error generating batch chart for {len(symbols)} symbols "
                f"across {len(normalized_tfs)} timeframes: {type(e).__name__}"
            )
            # Return empty batch_results structure for all symbols
            return {symbol: self._create_empty_symbol_result() for symbol in symbols}

        # Analyze with Gemini
        # Chart files are temporary and should be cleaned up after analysis
        # to prevent resource leaks. Cleanup happens in finally block to ensure
        # it occurs on both success and failure paths.
        batch_results = None
        try:
            batch_results = analyze_batch_chart_func(chart_path, symbols, normalized_tfs)
        except GeminiAnalysisError as e:
            log_error(f"Gemini analysis error for batch: {e}")
            batch_results = {symbol: self._create_empty_symbol_result() for symbol in symbols}
        except FileNotFoundError as e:
            log_error(f"Chart file not found for batch analysis: {e}")
            batch_results = {symbol: self._create_empty_symbol_result() for symbol in symbols}
        except Exception as e:
            log_error(f"Unexpected error in batch analysis: {e}")
            batch_results = {symbol: self._create_empty_symbol_result() for symbol in symbols}
        finally:
            # Ensure chart cleanup happens on both success and failure paths
            # This prevents resource leaks and ensures consistent lifecycle management
            self._safe_remove_chart(chart_path)

        # Ensure batch_results is a dict (defensive check)
        if batch_results is None or not isinstance(batch_results, dict):
            log_warn(
                f"batch_results is not a dict for {len(symbols)} symbols "
                f"across {len(normalized_tfs)} timeframes, using empty results"
            )
            batch_results = {}

        # Aggregate signals for each symbol
        final_results = {}
        for symbol in symbols:
            if symbol not in batch_results:
                # No result for this symbol
                final_results[symbol] = self._create_empty_symbol_result()
                continue

            symbol_tf_results = batch_results[symbol]

            # Extract timeframe signals
            timeframe_signals = {}
            for tf in normalized_tfs:
                if tf in symbol_tf_results:
                    timeframe_signals[tf] = symbol_tf_results[tf]

            # Aggregate
            aggregated = self.signal_aggregator.aggregate_signals(timeframe_signals)

            final_results[symbol] = {"timeframes": symbol_tf_results, "aggregated": aggregated}

        return final_results
