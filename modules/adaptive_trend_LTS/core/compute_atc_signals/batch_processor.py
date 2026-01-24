import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import atc_rust
import numpy as np
import pandas as pd

from modules.common.utils import log_error, log_info, log_warn

# Import relative to package, assuming this file is in modules/adaptive_trend_LTS/core/compute_atc_signals/
from .compute_atc_signals import compute_atc_signals


def process_symbols_batch_cuda(symbols_data, config, num_threads=4):
    """
    Process symbols using True Batch CUDA processing.
    """
    if not symbols_data:
        return {}

    try:
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
            La=la_scaled,
            De=de_scaled,
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
