"""
Market Batch Scanner for scanning entire market with Gemini.

Orchestrates the workflow: get symbols → batch → generate charts → analyze → aggregate results.

This refactored version uses sub-modules for better code organization:
- SymbolFetcher: Symbol retrieval from exchanges
- DataFetcherAdapter: OHLCV data fetching wrapper
- ResultManager: Result categorization and persistence
- CleanupManager: Old file cleanup operations
- protect_stdin_windows: Windows stdin protection utility
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add project root to sys.path
if "__file__" in globals():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.ui.logging import log_error, log_info, log_success, log_warn
from modules.gemini_chart_analyzer.core.analyzers.gemini_batch_chart_analyzer import GeminiBatchChartAnalyzer
from modules.gemini_chart_analyzer.core.exceptions import (
    DataFetchError,
    ScanConfigurationError,
)
from modules.gemini_chart_analyzer.core.generators.chart_batch_generator import ChartBatchGenerator
from modules.gemini_chart_analyzer.core.protocols import BatchChartAnalyzerProtocol
from modules.gemini_chart_analyzer.core.scanner_types import BatchScanResult

# Import sub-modules
from .batch_processor import BatchProcessor
from .batch_scanner_components import (
    CleanupManager,
    DataFetcherAdapter,
    ResultManager,
    SymbolFetcher,
    protect_stdin_windows,
)


@dataclass
class ScanConfig:
    """Configuration for market scan operation."""

    timeframe: Optional[str] = "1h"
    timeframes: Optional[List[str]] = None
    max_symbols: Optional[int] = None
    limit: int = 500
    cancelled_callback: Optional[Callable[[], bool]] = None
    initial_symbols: Optional[List[str]] = None
    skip_cleanup: bool = False


class MarketBatchScanner:
    """
    Scan entire market by batching symbols and analyzing with Gemini.

    Refactored version with modular components for improved maintainability.
    """

    # Minimum number of candles required for reliable technical analysis
    MIN_CANDLES: int = 20

    # Batch size for multi-timeframe charts (reduced because each symbol has multiple TFs)
    MULTI_TF_CHARTS_PER_BATCH: int = 25

    def __init__(
        self,
        charts_per_batch: int = 100,
        cooldown_seconds: float = 2.5,
        quote_currency: str = "USDT",
        exchange_name: str = "binance",
        min_candles: Optional[int] = None,
        rf_model_path: Optional[str] = None,
    ):
        """
        Initialize MarketBatchScanner.

        Args:
            charts_per_batch: Number of charts per batch (default: 100)
            cooldown_seconds: Cooldown between batch requests (default: 2.5s)
            quote_currency: Quote currency to filter symbols (default: 'USDT')
            exchange_name: Exchange name to connect to (default: 'binance')
            min_candles: Minimum number of candles required for reliable technical analysis (default: 20)
            rf_model_path: Path to Random Forest model for pre-filtering (default: None)

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

        self.rf_model_path = rf_model_path

        # Initialize core components
        self.exchange_manager = ExchangeManager()
        self.data_fetcher = DataFetcher(self.exchange_manager)
        self.batch_chart_generator = ChartBatchGenerator(charts_per_batch=charts_per_batch)
        self._gemini_analyzer_cooldown = cooldown_seconds
        self._gemini_analyzer = None  # Lazy initialization

        # Initialize sub-modules
        self.symbol_fetcher = SymbolFetcher(exchange_name=exchange_name, quote_currency=quote_currency)
        self.data_fetcher_adapter = DataFetcherAdapter(data_fetcher=self.data_fetcher, min_candles=self.min_candles)
        self.result_manager = ResultManager()
        self.cleanup_manager = CleanupManager()

    @property
    def batch_gemini_analyzer(self) -> BatchChartAnalyzerProtocol:
        """
        Lazy initialization property for GeminiBatchChartAnalyzer.

        Initializes the analyzer only when first accessed, after all user input is collected.
        This lazy initialization prevents stdin issues during interactive menu setup.

        The analyzer initialization is protected with stdin handling to prevent
        "I/O operation on closed file" errors on Windows caused by Google SDK initialization.

        Returns:
            BatchChartAnalyzerProtocol: The initialized batch analyzer instance
        """
        if self._gemini_analyzer is None:
            # Use context manager to protect stdin during initialization
            with protect_stdin_windows():
                self._gemini_analyzer = GeminiBatchChartAnalyzer(cooldown_seconds=self._gemini_analyzer_cooldown)

        return self._gemini_analyzer

    def cleanup(self, force_gc: bool = False):
        """
        Cleanup resources and free memory by clearing caches and forcing garbage collection.

        This method:
        - Clears cached data in exchange managers
        - Clears symbol fetcher resources
        - Always triggers garbage collection to free memory
        - If force_gc is True, performs an additional GC cycle for more aggressive cleanup

        Call this after scan_market() completes to free exchange connections and other resources.

        Args:
            force_gc: If True, perform an additional garbage collection cycle (default: False)
        """
        import gc

        # Cleanup exchange manager
        try:
            if hasattr(self.exchange_manager, "cleanup_unused_exchanges"):
                self.exchange_manager.cleanup_unused_exchanges()
            if hasattr(self.exchange_manager, "clear"):
                self.exchange_manager.clear()
        except Exception as e:
            log_warn(f"Error cleaning up exchange manager: {e}")

        # Cleanup symbol fetcher
        try:
            self.symbol_fetcher.cleanup()
        except Exception as e:
            log_warn(f"Error cleaning up symbol fetcher: {e}")

        # Force garbage collection
        gc.collect()
        if force_gc:
            gc.collect()
            log_info("Forced garbage collection")
        else:
            log_info("Garbage collection completed")

        log_info("Cleaned up MarketBatchScanner resources")

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
            DataFetchError: If symbol fetching fails after all retries
        """
        return self.symbol_fetcher.get_all_symbols(max_retries=max_retries, retry_delay=retry_delay)

    def scan_market(self, config: ScanConfig) -> BatchScanResult:
        """
        Scan entire market and return LONG/SHORT signals.

        Args:
            config: ScanConfig containing all scan parameters

        Returns:
            BatchScanResult with signals, confidence scores, and summary statistics
        """
        log_info("=" * 60)
        log_info("MARKET BATCH SCANNER")
        log_info("=" * 60)

        # Determine if multi-timeframe mode
        is_multi_tf = config.timeframes is not None and len(config.timeframes) > 0
        if is_multi_tf:
            from modules.gemini_chart_analyzer.core.aggregators.signal_aggregator import SignalAggregator
            from modules.gemini_chart_analyzer.core.generators.chart_multi_timeframe_batch_generator import (
                ChartMultiTimeframeBatchGenerator,
            )
            from modules.gemini_chart_analyzer.core.utils import normalize_timeframes

            normalized_tfs = normalize_timeframes(config.timeframes or [])
            if not normalized_tfs:
                raise ScanConfigurationError("No valid timeframes provided for multi-timeframe scan")
            log_info(f"Multi-timeframe mode: {', '.join(normalized_tfs)}")

            # Use multi-TF batch chart generator
            multi_tf_generator = ChartMultiTimeframeBatchGenerator(
                charts_per_batch=self.MULTI_TF_CHARTS_PER_BATCH, timeframes_per_symbol=len(normalized_tfs)
            )
            signal_aggregator = SignalAggregator()
        else:
            normalized_tfs = [config.timeframe] if config.timeframe else ["1h"]
            log_info(f"Single timeframe mode: {normalized_tfs[0]}")

        # Step 0: Cleanup old batch scan results
        if not config.skip_cleanup:
            self._cleanup_old_results()

        # Step 1: Get symbols (from initial_symbols or fetch from exchange)
        all_symbols = self._get_symbols_for_scan(config.initial_symbols)

        # Apply max_symbols
        if config.max_symbols and all_symbols:
            all_symbols = all_symbols[: config.max_symbols]
            log_info(f"Limited to {config.max_symbols} symbols")

        log_success(f"Found {len(all_symbols)} symbols to scan")

        # Step 2: Split into batches
        batch_size = self.MULTI_TF_CHARTS_PER_BATCH if is_multi_tf else self.charts_per_batch
        batches = self._split_into_batches(all_symbols, batch_size=batch_size)
        log_info(f"Split into {len(batches)} batches ({batch_size} symbols per batch)")

        # Step 3: Process batches
        all_results, batch_results = self._process_batches(
            batches=batches,
            is_multi_tf=is_multi_tf,
            normalized_tfs=normalized_tfs,
            limit=config.limit,
            cancelled_callback=config.cancelled_callback,
            multi_tf_generator=multi_tf_generator if is_multi_tf else None,
            signal_aggregator=signal_aggregator if is_multi_tf else None,
        )

        # Step 4: Aggregate and sort results
        return self._finalize_results(all_results, all_symbols, normalized_tfs, is_multi_tf)

    # ========================================
    # Private Helper Methods
    # ========================================

    def _cleanup_old_results(self):
        """Cleanup old batch scan results and charts."""
        self.cleanup_manager.cleanup_old_results()
        self.cleanup_manager.cleanup_old_charts()

    def _get_symbols_for_scan(self, initial_symbols: Optional[List[str]]) -> List[str]:
        """
        Get symbols for scanning (from initial_symbols or fetch from exchange).

        Args:
            initial_symbols: Optional pre-filtered symbols

        Returns:
            List of symbols to scan

        Raises:
            DataFetchError: If symbol fetching fails
        """
        if initial_symbols is not None:
            log_info("Step 1: Using pre-filtered symbols from external pre-filter...")
            log_info(f"Using {len(initial_symbols)} pre-filtered symbols")
            return initial_symbols

        log_info("Step 1: Getting all symbols from exchange...")
        try:
            symbols = self.get_all_symbols()
        except DataFetchError as e:
            log_error(f"Failed to fetch symbols from exchange: {e}")
            raise

        if not symbols:
            log_warn("No symbols found matching the criteria. This may indicate:")
            log_warn(f"  - No active spot markets for {self.quote_currency} on {self.exchange_name}")
            log_warn("  - Exchange API returned empty market list")
            log_warn("Continuing with empty symbol list...")

        return symbols

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
            batch = symbols[i : i + batch_size]
            batches.append(batch)
        return batches

    def _process_batches(
        self,
        batches: List[List[str]],
        is_multi_tf: bool,
        normalized_tfs: List[str],
        limit: int,
        cancelled_callback: Optional[Callable[[], bool]],
        multi_tf_generator: Optional[Any],
        signal_aggregator: Optional[Any],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process all batches and return aggregated results by delegating to BatchProcessor.
        """
        processor = BatchProcessor(self)
        return processor.process_batches(
            batches=batches,
            is_multi_tf=is_multi_tf,
            normalized_tfs=normalized_tfs,
            limit=limit,
            cancelled_callback=cancelled_callback,
            multi_tf_generator=multi_tf_generator,
            signal_aggregator=signal_aggregator,
        )

    def _process_single_tf_batch(self, batch_symbols: List[str], timeframe: str, limit: int, batch_idx: int) -> Dict[str, Any]:
        """
        Backward-compatible wrapper delegating single-timeframe batch processing to BatchProcessor.
        """
        processor = BatchProcessor(self)
        return processor._process_single_tf_batch(
            batch_symbols=batch_symbols,
            timeframe=timeframe,
            limit=limit,
            batch_idx=batch_idx,
        )

    def _process_multi_tf_batch(
        self,
        batch_symbols: List[str],
        normalized_tfs: List[str],
        limit: int,
        batch_idx: int,
        multi_tf_generator: Any,
        signal_aggregator: Any,
    ) -> Dict[str, Any]:
        """
        Backward-compatible wrapper delegating multi-timeframe batch processing to BatchProcessor.
        """
        processor = BatchProcessor(self)
        return processor._process_multi_tf_batch(
            batch_symbols=batch_symbols,
            normalized_tfs=normalized_tfs,
            limit=limit,
            batch_idx=batch_idx,
            multi_tf_generator=multi_tf_generator,
            signal_aggregator=signal_aggregator,
        )

    def _finalize_results(
        self, all_results: Dict[str, Any], all_symbols: List[str], normalized_tfs: List[str], is_multi_tf: bool
    ) -> BatchScanResult:
        """
        Finalize results: categorize, sort, save, and return BatchScanResult.

        Args:
            all_results: All scan results
            all_symbols: All symbols scanned
            normalized_tfs: List of timeframes
            is_multi_tf: Whether multi-TF mode

        Returns:
            BatchScanResult with categorized and sorted signals
        """
        log_info(f"\n{'=' * 60}")
        log_info("Aggregating and sorting results by confidence...")
        log_info(f"{'=' * 60}")

        # Categorize and sort results
        (
            long_symbols,
            short_symbols,
            none_symbols,
            long_symbols_with_confidence,
            short_symbols_with_confidence,
            none_symbols_with_confidence,
        ) = self.result_manager.categorize_and_sort_results(all_results)

        # Build summary
        summary = {
            "total_symbols": len(all_symbols),
            "scanned_symbols": len(all_results),
            "long_count": len(long_symbols),
            "short_count": len(short_symbols),
            "none_count": len(none_symbols),
            "long_percentage": (len(long_symbols) / len(all_results) * 100) if all_results else 0,
            "short_percentage": (len(short_symbols) / len(all_results) * 100) if all_results else 0,
            "avg_long_confidence": sum(c for _, c in long_symbols_with_confidence) / len(long_symbols_with_confidence)
            if long_symbols_with_confidence
            else 0.0,
            "avg_short_confidence": sum(c for _, c in short_symbols_with_confidence)
            / len(short_symbols_with_confidence)
            if short_symbols_with_confidence
            else 0.0,
        }

        # Save results
        primary_timeframe = normalized_tfs[0] if normalized_tfs else "1h"
        results_file = self.result_manager.save_results(
            all_results,
            long_symbols,
            short_symbols,
            summary,
            primary_timeframe,
            long_symbols_with_confidence,
            short_symbols_with_confidence,
            timeframes=normalized_tfs if is_multi_tf else None,
        )

        # Log summary
        log_success(f"\n{'=' * 60}")
        log_success("SCAN COMPLETED")
        log_success(f"{'=' * 60}")
        log_success(f"Total symbols: {summary['total_symbols']}")
        log_success(f"LONG signals: {summary['long_count']} ({summary['long_percentage']:.1f}%)")
        if summary.get("avg_long_confidence", 0) > 0:
            log_success(f"  Average LONG confidence: {summary['avg_long_confidence']:.2f}")
        log_success(f"SHORT signals: {summary['short_count']} ({summary['short_percentage']:.1f}%)")
        if summary.get("avg_short_confidence", 0) > 0:
            log_success(f"  Average SHORT confidence: {summary['avg_short_confidence']:.2f}")
        log_success(f"Results saved to: {results_file}")

        return BatchScanResult(
            long_symbols=long_symbols,
            short_symbols=short_symbols,
            none_symbols=none_symbols,
            long_symbols_with_confidence=long_symbols_with_confidence,
            short_symbols_with_confidence=short_symbols_with_confidence,
            none_symbols_with_confidence=none_symbols_with_confidence,
            all_results=all_results,
            summary=summary,
            results_file=results_file,
        )
