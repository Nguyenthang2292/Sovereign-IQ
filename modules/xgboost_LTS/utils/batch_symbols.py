"""
Batch symbol training for XGBoost module.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Optional, Callable

import pandas as pd

from modules.common.utils import log_error, log_warn


def _train_one(symbol: str, df: pd.DataFrame, train_fn: Callable[..., Any], use_cache: bool) -> tuple:
    """
    Helper function to train one symbol.
    Must be top-level for pickling support in multiprocessing.
    """
    try:
        # Note: Assuming train_and_predict_fn accepts use_cache kwargs
        # If not, it might need to be wrapped or arguments adjusted
        out = train_fn(df, use_cache=use_cache)
        return (symbol, {"ok": True, "result": out})
    except Exception as e:
        # We can't use log_error here easily if it's not picklable or if logging is not configured in worker
        # But normally logging works if configured properly.
        # Printing might interfere with progress bars.
        return (symbol, {"ok": False, "error": str(e)})


def batch_train_symbols(
    symbols_data: Dict[str, pd.DataFrame],
    train_and_predict_fn: Callable[..., Any],
    max_workers: Optional[int] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Train models for multiple symbols in parallel.

    Args:
        symbols_data: Dict mapping symbol -> DataFrame (with features and Target).
        train_and_predict_fn: Function with signature (df, **kwargs) -> result (e.g. train_and_predict).
        max_workers: Max parallel processes; default os.cpu_count() - 1 or 1.
        use_cache: Passed to labeling/training if supported.

    Returns:
        Dict mapping symbol -> result of train_and_predict_fn (or exception info on failure).
    """
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    results: Dict[str, Any] = {}

    # If only 1 worker, run sequentially to avoid overhead/pickling issues
    if max_workers == 1:
        for symbol, df in symbols_data.items():
            results[symbol] = _train_one(symbol, df.copy(), train_and_predict_fn, use_cache)[1]
        return results

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        # We copy dataframe to ensure thread safety if passed by reference (though processes use pickle)
        # Note: train_and_predict_fn must be picklable
        futures = {
            executor.submit(_train_one, symbol, df.copy(), train_and_predict_fn, use_cache): symbol
            for symbol, df in symbols_data.items()
        }

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                symbol_res, data = future.result()
                results[symbol_res] = data

                # Log error in main process if failed
                if not data["ok"]:
                    log_error(f"Failed to train {symbol_res}: {data.get('error')}")

            except Exception as e:
                log_error(f"Worker failure for {symbol}: {e}")
                results[symbol] = {"ok": False, "error": str(e)}

    return results
