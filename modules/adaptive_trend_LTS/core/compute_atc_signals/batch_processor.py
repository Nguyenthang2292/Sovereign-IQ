import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np
import pandas as pd

from modules.common.utils import log_error, log_info, log_warn

# Import relative to package, assuming this file is in modules/adaptive_trend_LTS/core/compute_atc_signals/
from .compute_atc_signals import compute_atc_signals


def process_symbols_batch_with_approximate_filter(
    symbols_data: Dict[str, pd.Series],
    config: dict,
    approximate_threshold: float = 0.1,  # Filter threshold
    min_signal_candidate: float = 0.05,  # Minimum signal to be candidate
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Process symbols with two-stage filtering:
    1. Approximate MAs for initial filtering
    2. Full precision for candidates only
    """
    # Stage 1: Approximate filtering
    candidates = {}

    log_info(f"Starting approximate filtering for {len(symbols_data)} symbols...")

    for symbol, prices in symbols_data.items():
        try:
            # Fast approximate calculation
            approx_results = compute_atc_signals(prices, use_approximate=True, **config)

            if "Average_Signal" in approx_results and not approx_results["Average_Signal"].empty:
                approx_signal = approx_results["Average_Signal"].iloc[-1]

                # Filter candidates
                if abs(approx_signal) >= min_signal_candidate:
                    candidates[symbol] = prices
        except Exception as e:
            log_warn(f"Approximate calculation failed for {symbol}: {e}")

    log_info(f"Approximate filtering: {len(candidates)}/{len(symbols_data)} candidates")

    # Stage 2: Full precision for candidates
    if not candidates:
        return {}

    # Use existing batch processing for candidates
    return process_symbols_batch_rust(candidates, config)


def process_symbols_batch_cuda(symbols_data, config, num_threads=4):
    """
    Process symbols using True Batch CUDA processing.
    """
    if not symbols_data:
        return {}

    try:
        import atc_rust

        # Clone and prepare config for Rust call
        params = config.copy()

        la = params.get("La", params.get("la", 0.02))
        de = params.get("De", params.get("de", 0.03))

        # Scaling logic: Rust expects scaled values matching compute_atc_signals.py
        la_scaled = la / 1000.0
        de_scaled = de / 100.0

        log_info(f"Launching True Batch CUDA processing for {len(symbols_data)} symbols...")

        # IMPORTANT: PyO3 extract expects numpy arrays, not pandas Series
        # Convert all series values to numpy arrays
        symbols_numpy = {}
        for s, v in symbols_data.items():
            if v is not None:
                # Ensure it's a numpy array of float64
                if isinstance(v, pd.Series):
                    symbols_numpy[s] = v.values.astype(np.float64)
                elif isinstance(v, np.ndarray):
                    symbols_numpy[s] = v.astype(np.float64)
                else:
                    symbols_numpy[s] = np.array(v, dtype=np.float64)

        # Call the batch Rust function
        batch_results = atc_rust.compute_atc_signals_batch(
            symbols_numpy,
            ema_len=params.get("ema_len", 28),
            hull_len=params.get("hull_len", 28),
            wma_len=params.get("wma_len", 28),
            dema_len=params.get("dema_len", 28),
            lsma_len=params.get("lsma_len", 28),
            kama_len=params.get("kama_len", 28),
            robustness=params.get("robustness", "Medium"),
            la=la_scaled,
            de=de_scaled,
            long_threshold=params.get("long_threshold", 0.1),
            short_threshold=params.get("short_threshold", -0.1),
        )

        formatted_results = {}
        for symbol, classified_array in batch_results.items():
            # Convert back to Series if needed by the caller
            # Typically benchmark_comparison handles either array or Series
            # but we restore the index if we have it
            orig_series = symbols_data.get(symbol)
            if orig_series is not None and hasattr(orig_series, "index"):
                formatted_results[symbol] = {"Average_Signal": pd.Series(classified_array, index=orig_series.index)}
            else:
                formatted_results[symbol] = {"Average_Signal": pd.Series(classified_array)}

        log_info(f"True Batch CUDA completed: {len(formatted_results)} symbols processed.")
        return formatted_results

    except Exception as e:
        log_error(f"True Batch CUDA failed: {e}. Falling back to ThreadPool/Per-symbol.")
        import traceback

        traceback.print_exc()

        # Fallback to original ThreadPool logic
        results = {}

        def process_one(symbol, prices):
            try:
                result = compute_atc_signals(prices=prices, **config)
                return symbol, result
            except Exception as e_inner:
                log_error(f"Error processing {symbol} in fallback CUDA batch: {e_inner}")
                return symbol, None

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_symbol = {
                executor.submit(process_one, symbol, prices): symbol for symbol, prices in symbols_data.items()
            }
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    sym, res = future.result()
                    if res is not None:
                        results[sym] = res
                except Exception as e_inner:
                    log_error(f"Exception in fallback worker for {symbol}: {e_inner}")

        return results


def process_symbols_batch_rust(symbols_data, config, num_threads=None):
    """
    Process symbols using Rust Rayon (CPU Multi-threaded) batch processing.

    Args:
        symbols_data: Dictionary of symbol -> price Series/ndarray
        config: ATC configuration parameters
        num_threads: Number of threads (optional, Rayon uses default if None)

    Returns:
        Dictionary mapping symbol -> {"Average_Signal": pd.Series}
    """
    if not symbols_data:
        return {}

    try:
        import atc_rust

        # Prepare params
        params = config.copy()

        la = params.get("La", params.get("la", 0.02))
        de = params.get("De", params.get("de", 0.03))

        # Scaling logic: Rust expects scaled values
        la_scaled = la / 1000.0
        de_scaled = de / 100.0

        log_info(f"Launching Rayon CPU Batch processing for {len(symbols_data)} symbols...")

        # Convert all series values to numpy arrays
        symbols_numpy = {}
        for s, v in symbols_data.items():
            if v is not None:
                if isinstance(v, pd.Series):
                    symbols_numpy[s] = v.values.astype(np.float64)
                elif isinstance(v, np.ndarray):
                    symbols_numpy[s] = v.astype(np.float64)
                else:
                    symbols_numpy[s] = np.array(v, dtype=np.float64)

        # Call the batch Rust function (CPU/Rayon version)
        batch_results = atc_rust.compute_atc_signals_batch_cpu(
            symbols_numpy,
            ema_len=params.get("ema_len", 28),
            hull_len=params.get("hull_len", 28),
            wma_len=params.get("wma_len", 28),
            dema_len=params.get("dema_len", 28),
            lsma_len=params.get("lsma_len", 28),
            kama_len=params.get("kama_len", 28),
            robustness=params.get("robustness", "Medium"),
            la=la_scaled,
            de=de_scaled,
            long_threshold=params.get("long_threshold", 0.1),
            short_threshold=params.get("short_threshold", -0.1),
        )

        formatted_results = {}
        for symbol, classified_array in batch_results.items():
            orig_series = symbols_data.get(symbol)
            if orig_series is not None and hasattr(orig_series, "index"):
                formatted_results[symbol] = {"Average_Signal": pd.Series(classified_array, index=orig_series.index)}
            else:
                formatted_results[symbol] = {"Average_Signal": pd.Series(classified_array)}

        log_info(f"Rayon CPU Batch completed: {len(formatted_results)} symbols processed.")
        return formatted_results

    except Exception as e:
        log_error(f"Rayon CPU Batch failed: {e}. Falling back to sequential.")
        import traceback

        traceback.print_exc()

        # Fallback to original sequential logic or ThreadPool if preferred
        results = {}
        for symbol, prices in symbols_data.items():
            try:
                result = compute_atc_signals(prices=prices, **config)
                results[symbol] = result
            except Exception as e_inner:
                log_error(f"Error processing {symbol} in fallback: {e_inner}")
                results[symbol] = None

        return results


def process_symbols_batch_with_dask(
    symbols_data: Dict[str, pd.Series], config: dict, use_dask: bool = True, npartitions: Optional[int] = None, **kwargs
) -> Dict[str, Dict[str, pd.Series]]:
    """Process symbols with optional Dask for out-of-core processing.

    Args:
        symbols_data: Dictionary of symbol -> price Series
        config: ATC configuration
        use_dask: Use Dask if data is large (default: True)
        npartitions: Number of Dask partitions
        **kwargs: Passed to batch processor (use_cuda, use_rust, etc.)

    Returns:
        Dictionary mapping symbol -> {"Average_Signal": pd.Series}
    """
    # Auto-detect if Dask is needed (e.g., >1000 symbols)
    if use_dask and len(symbols_data) > 1000:
        from .dask_batch_processor import process_symbols_batch_dask

        return process_symbols_batch_dask(
            symbols_data,
            config,
            use_rust=kwargs.get("use_rust", True),
            use_cuda=kwargs.get("use_cuda", False),
            npartitions=npartitions,
            partition_size=kwargs.get("partition_size", 50),
            use_fallback=kwargs.get("use_fallback", True),
        )
    else:
        # Use existing batch processor
        if kwargs.get("use_cuda", False):
            return process_symbols_batch_cuda(symbols_data, config, **kwargs)
        else:
            return process_symbols_batch_rust(symbols_data, config, **kwargs)
