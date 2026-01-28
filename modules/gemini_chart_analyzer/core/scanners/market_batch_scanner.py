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
    ChartGenerationError,
    DataFetchError,
    GeminiAnalysisError,
    ScanConfigurationError,
)
from modules.gemini_chart_analyzer.core.generators.chart_batch_generator import ChartBatchGenerator
from modules.gemini_chart_analyzer.core.prefilter.workflow import run_prefilter_worker
from modules.gemini_chart_analyzer.core.scanner_types import BatchScanResult, SignalResult, SymbolScanResult

# Import sub-modules
from .batch_scanner_components import (
    CleanupManager,
    DataFetcherAdapter,
    ResultManager,
    SymbolFetcher,
    protect_stdin_windows,
)


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
    def batch_gemini_analyzer(self):
        """
        Lazy initialization property for GeminiBatchChartAnalyzer.

        Initializes the analyzer only when first accessed, after all user input is collected.
        This lazy initialization prevents stdin issues during interactive menu setup.

        The analyzer initialization is protected with stdin handling to prevent
        "I/O operation on closed file" errors on Windows caused by Google SDK initialization.

        Returns:
            GeminiBatchChartAnalyzer: The initialized batch analyzer instance
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

    def scan_market(
        self,
        timeframe: Optional[str] = "1h",
        timeframes: Optional[List[str]] = None,
        max_symbols: Optional[int] = None,
        limit: int = 500,
        cancelled_callback: Optional[Callable[[], bool]] = None,
        initial_symbols: Optional[List[str]] = None,
        enable_pre_filter: bool = False,
        pre_filter_mode: str = "voting",
        pre_filter_percentage: Optional[float] = None,
        pre_filter_auto_skip_threshold: int = 10,
        fast_mode: bool = True,
        spc_config: Optional[Dict[str, Any]] = None,
        skip_cleanup: bool = False,
        stage0_sample_percentage: Optional[float] = None,
        stage0_sampling_strategy: str = "random",
        stage0_stratified_strata_count: int = 3,
        stage0_hybrid_top_percentage: float = 50.0,
        atc_performance: Optional[Dict[str, Any]] = None,
        approximate_ma_scanner: Optional[Dict[str, Any]] = None,
        use_atc_performance: bool = True,
    ) -> BatchScanResult:
        """
        Scan entire market and return LONG/SHORT signals.

        Args:
            timeframe: Single timeframe for charts (default: '1h', ignored if timeframes provided)
            timeframes: List of timeframes for multi-timeframe analysis (enables multi-TF mode)
            max_symbols: Maximum number of symbols to scan (None = all)
            limit: Number of candles to fetch per symbol (default: 500)
            cancelled_callback: Optional callable that returns bool; True indicates cancellation
            initial_symbols: Optional pre-filtered symbols from external pre-filter
            enable_pre_filter: Whether to run internal pre-filtering using VotingAnalyzer
            pre_filter_mode: Mode for pre-filtering ('voting' or 'hybrid')
            pre_filter_percentage: Percentage of symbols to select via pre-filter (0-100)
            pre_filter_auto_skip_threshold: Auto-skip percentage filter if Stage 3 returns fewer symbols
            fast_mode: Whether to run pre-filter in fast mode
            spc_config: Optional configuration for SPC analyzer
            skip_cleanup: If True, skip automatic cleanup of old results/charts
            stage0_sample_percentage: Stage 0 sampling percentage
            stage0_sampling_strategy: Stage 0 sampling strategy
            stage0_stratified_strata_count: Number of strata for stratified sampling
            stage0_hybrid_top_percentage: Top percentage for hybrid sampling
            atc_performance: ATC high-performance parameters
            approximate_ma_scanner: Approximate MA scanner configuration
            use_atc_performance: Switch between LTS (True) and Legacy (False) ATC modules

        Returns:
            BatchScanResult with signals, confidence scores, and summary statistics
        """
        log_info("=" * 60)
        log_info("MARKET BATCH SCANNER")
        log_info("=" * 60)

        # Determine if multi-timeframe mode
        is_multi_tf = timeframes is not None and len(timeframes) > 0
        if is_multi_tf:
            from modules.gemini_chart_analyzer.core.aggregators.signal_aggregator import SignalAggregator
            from modules.gemini_chart_analyzer.core.generators.chart_multi_timeframe_batch_generator import (
                ChartMultiTimeframeBatchGenerator,
            )
            from modules.gemini_chart_analyzer.core.utils import normalize_timeframes

            normalized_tfs = normalize_timeframes(timeframes)
            if not normalized_tfs:
                raise ScanConfigurationError("No valid timeframes provided for multi-timeframe scan")
            log_info(f"Multi-timeframe mode: {', '.join(normalized_tfs)}")

            # Use multi-TF batch chart generator
            multi_tf_generator = ChartMultiTimeframeBatchGenerator(
                charts_per_batch=self.MULTI_TF_CHARTS_PER_BATCH, timeframes_per_symbol=len(normalized_tfs)
            )
            signal_aggregator = SignalAggregator()
        else:
            normalized_tfs = [timeframe] if timeframe else ["1h"]
            log_info(f"Single timeframe mode: {normalized_tfs[0]}")

        # Step 0: Cleanup old batch scan results
        if not skip_cleanup:
            self._cleanup_old_results()

        # Step 1: Get symbols (from initial_symbols or fetch from exchange)
        all_symbols = self._get_symbols_for_scan(initial_symbols)

        # Step 1.5: Apply internal pre-filter if enabled
        if enable_pre_filter and all_symbols:
            all_symbols = self._apply_pre_filter(
                all_symbols=all_symbols,
                pre_filter_percentage=pre_filter_percentage,
                normalized_tfs=normalized_tfs,
                limit=limit,
                pre_filter_mode=pre_filter_mode,
                fast_mode=fast_mode,
                spc_config=spc_config,
                stage0_sample_percentage=stage0_sample_percentage,
                stage0_sampling_strategy=stage0_sampling_strategy,
                stage0_stratified_strata_count=stage0_stratified_strata_count,
                stage0_hybrid_top_percentage=stage0_hybrid_top_percentage,
                atc_performance=atc_performance,
                approximate_ma_scanner=approximate_ma_scanner,
                pre_filter_auto_skip_threshold=pre_filter_auto_skip_threshold,
                use_atc_performance=use_atc_performance,
            )

        # Apply max_symbols
        if max_symbols and all_symbols:
            all_symbols = all_symbols[:max_symbols]
            log_info(f"Limited to {max_symbols} symbols")

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
            limit=limit,
            cancelled_callback=cancelled_callback,
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

    def _apply_pre_filter(
        self,
        all_symbols: List[str],
        pre_filter_percentage: Optional[float],
        normalized_tfs: List[str],
        limit: int,
        pre_filter_mode: str,
        fast_mode: bool,
        spc_config: Optional[Dict[str, Any]],
        stage0_sample_percentage: Optional[float],
        stage0_sampling_strategy: str,
        stage0_stratified_strata_count: int,
        stage0_hybrid_top_percentage: float,
        atc_performance: Optional[Dict[str, Any]],
        approximate_ma_scanner: Optional[Dict[str, Any]],
        pre_filter_auto_skip_threshold: int,
        use_atc_performance: bool,
    ) -> List[str]:
        """
        Apply internal pre-filter to symbol list.

        Args:
            all_symbols: List of all symbols
            pre_filter_percentage: Percentage to select
            (Additional args for pre-filter configuration)

        Returns:
            Filtered symbol list
        """
        log_info(f"Step 1.5: Running internal pre-filter ({pre_filter_mode})...")

        # Use provided percentage or default to 10%
        if pre_filter_percentage is None:
            pre_filter_percentage = 10.0
        elif pre_filter_percentage <= 0.0 or pre_filter_percentage > 100.0:
            log_warn(f"Invalid pre_filter_percentage {pre_filter_percentage}, using default 10.0")
            pre_filter_percentage = 10.0

        log_info(f"Pre-filter will select top {pre_filter_percentage}% of symbols by weighted score")

        try:
            pre_filtered = self._run_pre_filter(
                symbols=all_symbols,
                percentage=pre_filter_percentage,
                timeframe=normalized_tfs[0],
                limit=limit,
                mode=pre_filter_mode,
                fast_mode=fast_mode,
                spc_config=spc_config,
                stage0_sample_percentage=stage0_sample_percentage,
                stage0_sampling_strategy=stage0_sampling_strategy,
                stage0_stratified_strata_count=stage0_stratified_strata_count,
                stage0_hybrid_top_percentage=stage0_hybrid_top_percentage,
                atc_performance=atc_performance,
                approximate_ma_scanner=approximate_ma_scanner,
                auto_skip_threshold=pre_filter_auto_skip_threshold,
                use_atc_performance=use_atc_performance,
            )
            if pre_filtered:
                log_success(f"Internal pre-filter selected {len(pre_filtered)}/{len(all_symbols)} symbols")
                return pre_filtered
            else:
                log_warn("Internal pre-filter returned no symbols, continuing with original list")
                return all_symbols
        except Exception as e:
            log_error(f"Error during internal pre-filtering: {e}")
            log_warn("Continuing with original symbol list due to pre-filter error")
            return all_symbols

    def _run_pre_filter(
        self,
        symbols: List[str],
        percentage: float,
        timeframe: str,
        limit: int,
        mode: str = "voting",
        fast_mode: bool = True,
        spc_config: Optional[Dict[str, Any]] = None,
        stage0_sample_percentage: Optional[float] = None,
        stage0_sampling_strategy: str = "random",
        stage0_stratified_strata_count: int = 3,
        stage0_hybrid_top_percentage: float = 50.0,
        atc_performance: Optional[Dict[str, Any]] = None,
        approximate_ma_scanner: Optional[Dict[str, Any]] = None,
        auto_skip_threshold: int = 10,
        use_atc_performance: bool = True,
    ) -> List[str]:
        """
        Run internal pre-filter using 3-stage sequential filtering workflow.

        Args:
            symbols: List of symbols to filter
            percentage: Percentage of symbols to select (0-100)
            timeframe: Timeframe for analysis
            limit: Number of candles per symbol
            mode: Pre-filter mode ('voting' or 'hybrid')
            fast_mode: Whether to run in fast mode
            spc_config: Optional SPC configuration
            (Additional stage0 and atc_performance args)

        Returns:
            List of pre-filtered symbols from Stage 3
        """
        try:
            log_info(f"Pre-filter processing {len(symbols)} symbols (this may take several minutes)...")

            # Call pre-filter function directly in the same process
            filtered_symbols = run_prefilter_worker(
                all_symbols=symbols,
                percentage=percentage,
                timeframe=timeframe,
                limit=limit,
                mode=mode,
                fast_mode=fast_mode,
                spc_config=spc_config,
                rf_model_path=self.rf_model_path,
                stage0_sample_percentage=stage0_sample_percentage,
                stage0_sampling_strategy=stage0_sampling_strategy,
                stage0_stratified_strata_count=stage0_stratified_strata_count,
                stage0_hybrid_top_percentage=stage0_hybrid_top_percentage,
                atc_performance=atc_performance,
                auto_skip_threshold=auto_skip_threshold,
                use_atc_performance=use_atc_performance,
            )

            log_success(f"Pre-filter completed: Selected {len(filtered_symbols)}/{len(symbols)} symbols")
            return filtered_symbols

        except Exception as e:
            log_error(f"Failed to run pre-filter: {e}")
            import traceback

            traceback.print_exc()
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
        Process all batches and return aggregated results.

        This is a placeholder that delegates to the original complex batch processing logic.
        In a full refactoring, this would be moved to a separate batch_processor.py module.

        Args:
            batches: List of symbol batches
            is_multi_tf: Whether multi-timeframe mode is enabled
            normalized_tfs: List of normalized timeframes
            limit: Number of candles per symbol
            cancelled_callback: Optional cancellation callback
            multi_tf_generator: Optional multi-TF chart generator
            signal_aggregator: Optional signal aggregator

        Returns:
            Tuple of (all_results dict, batch_results list)
        """
        all_results = {}
        batch_results = []

        for batch_idx, batch_symbols in enumerate(batches, 1):
            # Check for cancellation
            if cancelled_callback and cancelled_callback():
                log_warn("Scan cancelled by user")
                log_info(f"Processed {batch_idx - 1}/{len(batches)} batches before cancellation")
                break

            log_info(f"\n{'=' * 60}")
            log_info(f"Processing batch {batch_idx}/{len(batches)}")
            log_info(f"{'=' * 60}")

            try:
                if is_multi_tf:
                    batch_result = self._process_multi_tf_batch(
                        batch_symbols, normalized_tfs, limit, batch_idx, multi_tf_generator, signal_aggregator
                    )

                    # Extract aggregated signals
                    for symbol, result in batch_result.items():
                        if hasattr(result, "aggregated") and result.aggregated:
                            all_results[symbol] = result.aggregated
                        else:
                            all_results[symbol] = SignalResult(signal="NONE", confidence=0.0)

                    valid_symbols = [s for s in batch_result.keys()]
                    batch_results.append(
                        {"batch_id": batch_idx, "symbols": valid_symbols, "results": batch_result}
                    )

                    # Handle failed symbols
                    for symbol in batch_symbols:
                        if symbol not in valid_symbols:
                            all_results[symbol] = SignalResult(signal="NONE", confidence=0.0)
                else:
                    batch_result = self._process_single_tf_batch(
                        batch_symbols, normalized_tfs[0], limit, batch_idx
                    )

                    all_results.update(batch_result)
                    symbols_data_keys = [s for s in batch_result.keys()]
                    batch_results.append(
                        {"batch_id": batch_idx, "symbols": symbols_data_keys, "results": batch_result}
                    )

                    # Handle failed symbols
                    fetched_symbols = set(batch_result.keys())
                    for symbol in batch_symbols:
                        if symbol not in fetched_symbols:
                            all_results[symbol] = SignalResult(signal="NONE", confidence=0.0)

            except (GeminiAnalysisError, ChartGenerationError) as e:
                log_error(f"Error processing batch {batch_idx}: {e}")
                for symbol in batch_symbols:
                    if symbol not in all_results:
                        all_results[symbol] = SignalResult(signal="NONE", confidence=0.0)
            except Exception as e:
                log_error(f"Unexpected error processing batch {batch_idx}: {e}")
                for symbol in batch_symbols:
                    if symbol not in all_results:
                        all_results[symbol] = SignalResult(signal="NONE", confidence=0.0)

        return all_results, batch_results

    def _process_single_tf_batch(
        self, batch_symbols: List[str], timeframe: str, limit: int, batch_idx: int
    ) -> Dict[str, Any]:
        """
        Process a single-timeframe batch.

        Args:
            batch_symbols: List of symbols in this batch
            timeframe: Timeframe string
            limit: Number of candles
            batch_idx: Batch index

        Returns:
            Dict mapping symbol to result
        """
        log_info(f"Fetching OHLCV data for {len(batch_symbols)} symbols...")
        symbols_data = self.data_fetcher_adapter.fetch_batch_data(batch_symbols, timeframe, limit)

        if not symbols_data:
            log_warn(f"No data fetched for batch {batch_idx}, skipping...")
            return {}

        log_success(f"Fetched data for {len(symbols_data)} symbols")

        # Generate batch chart
        log_info("Generating batch chart image...")
        batch_chart_path, truncated = self.batch_chart_generator.create_batch_chart(
            symbols_data=symbols_data, timeframe=timeframe, batch_id=batch_idx
        )
        if truncated:
            log_warn(f"Batch {batch_idx}: Input symbols list was truncated to {self.charts_per_batch} items")

        # Analyze with Gemini
        log_info("Sending to Gemini for analysis...")
        batch_result = self.batch_gemini_analyzer.analyze_batch_chart(
            image_path=batch_chart_path,
            batch_id=batch_idx,
            total_batches=None,
            symbols=[sd["symbol"] for sd in symbols_data],
        )

        return batch_result

    def _process_multi_tf_batch(
        self,
        batch_symbols: List[str],
        normalized_tfs: List[str],
        limit: int,
        batch_idx: int,
        multi_tf_generator: Any,
        signal_aggregator: Any,
    ) -> Dict[str, SymbolScanResult]:
        """
        Process a multi-timeframe batch.

        Args:
            batch_symbols: List of symbols in this batch
            normalized_tfs: List of normalized timeframes
            limit: Number of candles
            batch_idx: Batch index
            multi_tf_generator: Multi-TF chart generator
            signal_aggregator: Signal aggregator

        Returns:
            Dict mapping symbol to SymbolScanResult
        """
        msg = f"Fetching OHLCV data for {len(batch_symbols)} symbols across {len(normalized_tfs)} timeframes..."
        log_info(msg)
        symbols_tf_data = {}  # {symbol: {timeframe: df}}

        for symbol in batch_symbols:
            symbols_tf_data[symbol] = {}
            for tf in normalized_tfs:
                try:
                    df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                        symbol=symbol, timeframe=tf, limit=limit, check_freshness=False
                    )
                    if df is not None and not df.empty and len(df) >= self.min_candles:
                        symbols_tf_data[symbol][tf] = df
                except Exception as e:
                    log_error(f"Error fetching {symbol} {tf}: {e}")

        # Filter symbols that have data for at least one timeframe
        valid_symbols = {sym for sym, tf_data in symbols_tf_data.items() if tf_data}

        if not valid_symbols:
            log_warn(f"No valid data for batch {batch_idx}, skipping...")
            return {}

        log_success(f"Fetched data for {len(valid_symbols)} symbols")

        # Generate multi-TF batch chart
        log_info("Generating multi-timeframe batch chart image...")
        batch_chart_path, truncated = multi_tf_generator.create_multi_tf_batch_chart(
            symbols_data=symbols_tf_data, timeframes=normalized_tfs, batch_id=batch_idx
        )

        # Analyze with Gemini (multi-TF prompt)
        log_info("Sending to Gemini for multi-timeframe analysis...")
        parsed_results = self.batch_gemini_analyzer.analyze_multi_tf_batch_chart(
            batch_chart_path=batch_chart_path,
            symbols=sorted(valid_symbols),
            normalized_timeframes=normalized_tfs,
        )

        # Handle empty/None results
        if parsed_results is None:
            log_error(f"Gemini analysis failed for batch {batch_idx}: No results object returned. Skipping batch.")
            return {}
        elif isinstance(parsed_results, dict) and not parsed_results:
            log_info(f"Gemini analyzed batch {batch_idx}, but found no signals (empty result set). Skipping batch.")
            return {}

        log_success(f"Parsed {len(parsed_results)} results from Gemini")

        # Aggregate signals for each symbol
        batch_result = {}
        for symbol in valid_symbols:
            if symbol in parsed_results:
                symbol_result = parsed_results[symbol]

                # Support both new dataclass format (SymbolScanResult) and legacy dict format
                if isinstance(symbol_result, SymbolScanResult):
                    tf_signals = symbol_result.timeframes
                    aggregated = symbol_result.aggregated
                elif isinstance(symbol_result, dict):
                    # Backward compatibility: dict format
                    tf_signals = symbol_result.get("timeframes", {})
                    aggregated = symbol_result.get("aggregated")
                else:
                    log_warn(
                        f"Unexpected result format for {symbol}: {type(symbol_result).__name__}, "
                        f"expected SymbolScanResult or dict"
                    )
                    tf_signals = {}
                    aggregated = None

                # If aggregated signal not provided by Gemini, calculate it
                if aggregated is None:
                    aggregated = signal_aggregator.aggregate_signals(tf_signals)

                batch_result[symbol] = SymbolScanResult(timeframes=tf_signals, aggregated=aggregated)
            else:
                # Symbol not found in parsed results
                log_warn(f"Symbol {symbol} not found in parsed multi-TF results")
                batch_result[symbol] = SymbolScanResult(
                    timeframes={tf: SignalResult(signal="NONE", confidence=0.0) for tf in normalized_tfs},
                    aggregated=SignalResult(signal="NONE", confidence=0.0),
                )

        return batch_result

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
