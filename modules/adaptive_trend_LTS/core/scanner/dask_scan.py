"""Dask-based scanner for processing large symbol lists out-of-core."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import dask.bag as db
import numpy as np
import pandas as pd

from modules.adaptive_trend_enhance.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_enhance.core.process_layer1 import trend_sign
from modules.adaptive_trend_enhance.utils.config import ATCConfig

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher

try:
    from modules.common.utils import log_progress, log_warn
except ImportError:

    def log_warn(message: str) -> None:
        print(f"[WARN] {message}")

    def log_progress(message: str) -> None:
        print(f"[PROGRESS] {message}")


try:
    from dask.callbacks import Callback

    HAS_DASK_CALLBACKS = True
except ImportError:

    class Callback:
        pass

    HAS_DASK_CALLBACKS = False


def _process_single_symbol_dask(
    symbol_data: Tuple[str, pd.DataFrame, str],
    atc_config: ATCConfig,
    min_signal: float,
) -> Optional[Dict[str, Any]]:
    """Process a single symbol (for Dask map operation)."""
    symbol, df, exchange = symbol_data

    try:
        if df is None or df.empty:
            return None

        calculation_source = atc_config.calculation_source.lower()
        valid_sources = ["close", "open", "high", "low"]

        if calculation_source not in valid_sources:
            calculation_source = "close"

        if calculation_source not in df.columns:
            return None

        raw_values = df[calculation_source].values
        target_dtype = np.float32 if atc_config.precision == "float32" else np.float64
        if raw_values.dtype != target_dtype:
            raw_values = raw_values.astype(target_dtype)

        if not raw_values.flags["C_CONTIGUOUS"]:
            raw_values = np.ascontiguousarray(raw_values)

        price_series = pd.Series(raw_values, index=df.index, name=calculation_source)

        if len(price_series) < atc_config.limit:
            return None

        current_price = price_series.iloc[-1]

        if pd.isna(current_price) or current_price <= 0:
            return None

        atc_results = compute_atc_signals(
            prices=price_series,
            src=None,
            ema_len=atc_config.ema_len,
            hull_len=atc_config.hma_len,
            wma_len=atc_config.wma_len,
            dema_len=atc_config.dema_len,
            lsma_len=atc_config.lsma_len,
            kama_len=atc_config.kama_len,
            ema_w=atc_config.ema_w,
            hma_w=atc_config.hma_w,
            wma_w=atc_config.wma_w,
            dema_w=atc_config.dema_w,
            lsma_w=atc_config.lsma_w,
            kama_w=atc_config.kama_w,
            robustness=atc_config.robustness,
            La=atc_config.lambda_param,
            De=atc_config.decay,
            cutout=atc_config.cutout,
            long_threshold=atc_config.long_threshold,
            short_threshold=atc_config.short_threshold,
            precision=atc_config.precision,
        )

        average_signal = atc_results.get("Average_Signal")
        if average_signal is None or average_signal.empty:
            return None

        latest_signal = average_signal.iloc[-1]

        if pd.isna(latest_signal):
            return None

        latest_trend = trend_sign(average_signal)
        latest_trend_value = latest_trend.iloc[-1] if not latest_trend.empty else 0

        if abs(latest_signal) < min_signal:
            return None

        return {
            "symbol": symbol,
            "signal": latest_signal,
            "trend": latest_trend_value,
            "price": current_price,
            "exchange": exchange or "UNKNOWN",
        }
    except Exception as e:
        log_warn(f"Error processing {symbol}: {type(e).__name__}: {e}")
        return None


def _fetch_partition_lazy(
    partition_symbols: list,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
) -> list:
    """Fetch data for a partition only when needed."""
    partition_data = []
    for symbol in partition_symbols:
        try:
            df, exchange = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=atc_config.limit,
                timeframe=atc_config.timeframe,
                check_freshness=True,
            )
            partition_data.append((symbol, df, exchange))
        except Exception as e:
            log_warn(f"Failed to fetch {symbol}: {type(e).__name__}: {e}")
            partition_data.append((symbol, None, None))
    return partition_data


def _process_partition_with_gc(
    partition_data: list,
    atc_config: ATCConfig,
    min_signal: float,
) -> list:
    """Process partition and force GC."""
    results = []
    for symbol_data in partition_data:
        result = _process_single_symbol_dask(symbol_data, atc_config, min_signal)
        if result:
            results.append(result)
    gc.collect()
    return results


class ProgressCallback(Callback):
    def __init__(self, total_symbols: int):
        self.total = total_symbols
        self.processed = 0
        self.last_logged = 0

    def _start(self, dsk):
        log_progress(f"Starting Dask computation for {self.total} symbols")

    def _posttask(self, key, result, dsk, state, id):
        self.processed += 1
        if self.processed - self.last_logged >= 10 or self.processed == self.total:
            log_progress(f"Progress: {self.processed}/{self.total} symbols processed")
            self.last_logged = self.processed


def _scan_dask(
    symbols: list,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float = 0.01,
    npartitions: Optional[int] = None,
    batch_size: int = 100,
) -> Tuple[list, int, int, list]:
    """Scan symbols using Dask for out-of-core processing."""
    if not symbols:
        return [], 0, 0, []

    if npartitions is None:
        npartitions = max(1, len(symbols) // batch_size)

    log_progress(f"Starting Dask scan for {len(symbols)} symbols with {npartitions} partitions")

    def fetch_and_process_partition(partition_symbols: list) -> list:
        """Fetch and process a single partition."""
        partition_data = _fetch_partition_lazy(partition_symbols, data_fetcher, atc_config)
        results = _process_partition_with_gc(partition_data, atc_config, min_signal)
        return results

    symbols_bag = db.from_sequence(symbols, npartitions=npartitions)
    results_bag = symbols_bag.map_partitions(fetch_and_process_partition)

    if HAS_DASK_CALLBACKS:
        with ProgressCallback(len(symbols)):
            results_list = results_bag.compute()
    else:
        results_list = results_bag.compute()

    results = [r for r in results_list if isinstance(r, dict) and r]

    processed = len(results)
    total_skipped_errors = len(symbols) - processed
    skipped_count = 0
    error_count = total_skipped_errors

    all_result_symbols = {r["symbol"] for r in results} if results else set()
    skipped_symbols = [s for s in symbols if s not in all_result_symbols]

    return results, skipped_count, error_count, skipped_symbols
