from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from modules.common.ui.logging import log_error, log_info, log_success, log_warn
from modules.gemini_chart_analyzer.core.exceptions import ChartGenerationError, GeminiAnalysisError
from modules.gemini_chart_analyzer.core.scanner_types import SignalResult, SymbolScanResult

if TYPE_CHECKING:
    from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner


class BatchProcessor:
    """Processes batches of symbols for market scanning."""

    def __init__(self, scanner: "MarketBatchScanner"):
        self.scanner = scanner

    def process_batches(
        self,
        batches: List[List[str]],
        is_multi_tf: bool,
        normalized_tfs: List[str],
        limit: int,
        cancelled_callback: Optional[Callable[[], bool]],
        multi_tf_generator: Optional[Any],
        signal_aggregator: Optional[Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Process all batches and return aggregated results."""
        all_results = {}
        batch_results = []

        for batch_idx, batch_symbols in enumerate(batches, 1):
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

                    for symbol, result in batch_result.items():
                        if hasattr(result, "aggregated") and result.aggregated:
                            all_results[symbol] = result.aggregated
                        else:
                            all_results[symbol] = SignalResult(signal="NONE", confidence=0.0)

                    valid_symbols = [s for s in batch_result.keys()]
                    batch_results.append({"batch_id": batch_idx, "symbols": valid_symbols, "results": batch_result})

                    for symbol in batch_symbols:
                        if symbol not in valid_symbols:
                            all_results[symbol] = SignalResult(signal="NONE", confidence=0.0)
                else:
                    batch_result = self._process_single_tf_batch(batch_symbols, normalized_tfs[0], limit, batch_idx)

                    all_results.update(batch_result)  # type: ignore[arg-type]
                    symbols_data_keys = [s for s in batch_result.keys()]
                    batch_results.append({"batch_id": batch_idx, "symbols": symbols_data_keys, "results": batch_result})

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
        """Process a single-timeframe batch."""
        log_info(f"Fetching OHLCV data for {len(batch_symbols)} symbols...")
        symbols_data = self.scanner.data_fetcher_adapter.fetch_batch_data(batch_symbols, timeframe, limit)

        if not symbols_data:
            log_warn(f"No data fetched for batch {batch_idx}, skipping...")
            return {}

        log_success(f"Fetched data for {len(symbols_data)} symbols")

        log_info("Generating batch chart image...")
        batch_chart_path, truncated = self.scanner.batch_chart_generator.create_batch_chart(
            symbols_data=symbols_data, timeframe=timeframe, batch_id=batch_idx
        )
        if truncated:
            log_warn(f"Batch {batch_idx}: Input symbols list was truncated to {self.scanner.charts_per_batch} items")

        log_info("Sending to Gemini for analysis...")
        batch_result = self.scanner.batch_gemini_analyzer.analyze_batch_chart(
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
        """Process a multi-timeframe batch."""
        msg = f"Fetching OHLCV data for {len(batch_symbols)} symbols across {len(normalized_tfs)} timeframes..."
        log_info(msg)
        symbols_tf_data: Dict[str, Dict[str, Any]] = {}

        for symbol in batch_symbols:
            symbols_tf_data[symbol] = {}
            for tf in normalized_tfs:
                try:
                    df, _ = self.scanner.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                        symbol=symbol, timeframe=tf, limit=limit, check_freshness=False
                    )
                    if df is not None and not df.empty and len(df) >= self.scanner.min_candles:
                        symbols_tf_data[symbol][tf] = df
                except Exception as e:
                    log_error(f"Error fetching {symbol} {tf}: {e}")

        valid_symbols = {sym for sym, tf_data in symbols_tf_data.items() if tf_data}

        if not valid_symbols:
            log_warn(f"No valid data for batch {batch_idx}, skipping...")
            return {}

        log_success(f"Fetched data for {len(valid_symbols)} symbols")

        log_info("Generating multi-timeframe batch chart image...")
        batch_chart_path, truncated = multi_tf_generator.create_multi_tf_batch_chart(
            symbols_data=symbols_tf_data, timeframes=normalized_tfs, batch_id=batch_idx
        )

        log_info("Sending to Gemini for multi-timeframe analysis...")
        parsed_results = self.scanner.batch_gemini_analyzer.analyze_multi_tf_batch_chart(
            batch_chart_path=batch_chart_path,
            symbols=sorted(valid_symbols),
            normalized_timeframes=normalized_tfs,
        )

        if parsed_results is None:
            log_error(f"Gemini analysis failed for batch {batch_idx}: No results object returned. Skipping batch.")
            return {}
        elif isinstance(parsed_results, dict) and not parsed_results:
            log_info(f"Gemini analyzed batch {batch_idx}, but found no signals (empty result set). Skipping batch.")
            return {}

        log_success(f"Parsed {len(parsed_results)} results from Gemini")

        batch_result = {}
        for symbol in valid_symbols:
            if symbol in parsed_results:
                symbol_result = parsed_results[symbol]

                if isinstance(symbol_result, SymbolScanResult):
                    tf_signals = symbol_result.timeframes
                    aggregated = symbol_result.aggregated
                elif isinstance(symbol_result, dict):
                    tf_signals = symbol_result.get("timeframes", {})
                    aggregated = symbol_result.get("aggregated")
                else:
                    log_warn(
                        f"Unexpected result format for {symbol}: {type(symbol_result).__name__}, "
                        f"expected SymbolScanResult or dict"
                    )
                    tf_signals = {}
                    aggregated = None

                if aggregated is None:
                    aggregated = signal_aggregator.aggregate_signals(tf_signals)

                batch_result[symbol] = SymbolScanResult(timeframes=tf_signals, aggregated=aggregated)
            else:
                log_warn(f"Symbol {symbol} not found in parsed multi-TF results")
                batch_result[symbol] = SymbolScanResult(
                    timeframes={tf: SignalResult(signal="NONE", confidence=0.0) for tf in normalized_tfs},
                    aggregated=SignalResult(signal="NONE", confidence=0.0),
                )

        return batch_result
