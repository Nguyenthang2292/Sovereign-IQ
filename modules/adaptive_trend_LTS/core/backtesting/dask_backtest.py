"""Dask-based backtesting for large historical datasets."""

from __future__ import annotations

import gc
from typing import List, Optional

import dask.dataframe as dd
import pandas as pd

try:
    from modules.common.utils import log_error, log_info, log_warn
except ImportError:

    def log_info(message: str) -> None:
        print(f"[INFO] {message}")

    def log_error(message: str) -> None:
        print(f"[ERROR] {message}")

    def log_warn(message: str) -> None:
        print(f"[WARN] {message}")


def _process_symbol_group(
    group_df: pd.DataFrame,
    symbol_column: str,
    price_column: str,
    atc_config: dict,
) -> pd.DataFrame:
    """Process a single symbol's historical data.

    Args:
        group_df: DataFrame with symbol's data
        symbol_column: Column name for symbol
        price_column: Column name for price
        atc_config: ATC configuration parameters

    Returns:
        DataFrame with computed signals
    """
    try:
        from modules.adaptive_trend_enhance.core.compute_atc_signals import compute_atc_signals

        symbol = group_df[symbol_column].iloc[0] if not group_df.empty else "UNKNOWN"
        prices = group_df[price_column].sort_index()

        if prices.empty or len(prices) < atc_config.get("ema_len", 28):
            return pd.DataFrame()

        result = compute_atc_signals(prices=prices, **atc_config)

        avg_signal = result.get("Average_Signal", pd.Series())

        if avg_signal.empty:
            return pd.DataFrame()

        return pd.DataFrame(
            {
                "symbol": [symbol] * len(avg_signal),
                "signal": avg_signal.values,
                "price": prices.values,
                "timestamp": prices.index,
            }
        )
    except Exception as e:
        log_error(f"Error processing symbol group: {e}")
        return pd.DataFrame()


def backtest_with_dask(
    historical_data_path: str,
    atc_config: dict,
    chunksize: str = "100MB",
    symbol_column: str = "symbol",
    price_column: str = "close",
) -> pd.DataFrame:
    """Backtest ATC signals on large historical data using Dask.

    Args:
        historical_data_path: Path to CSV/Parquet file
        atc_config: ATC configuration parameters
        chunksize: Size of each chunk (e.g., "100MB")
        symbol_column: Column name for symbol
        price_column: Column name for price

    Returns:
        DataFrame with backtest results
    """
    log_info(f"Loading historical data from {historical_data_path}")

    try:
        ddf = dd.read_csv(
            historical_data_path,
            blocksize=chunksize,
            dtype={symbol_column: "string", price_column: "float64"},
        )
    except Exception as e:
        log_error(f"Failed to read CSV file: {e}")
        return pd.DataFrame()

    grouped = ddf.groupby(symbol_column)

    meta = {
        "symbol": "string",
        "signal": "float64",
        "price": "float64",
        "timestamp": "datetime64[ns]",
    }

    def process_with_gc(group_df: pd.DataFrame) -> pd.DataFrame:
        result = _process_symbol_group(group_df, symbol_column, price_column, atc_config)
        gc.collect()
        return result

    try:
        results_ddf = grouped.apply(process_with_gc, meta=meta)
    except Exception as e:
        log_error(f"Error in Dask apply: {e}")
        return pd.DataFrame()

    try:
        results_df = results_ddf.compute()
    except Exception as e:
        log_error(f"Error computing Dask results: {e}")
        return pd.DataFrame()

    log_info(f"Completed backtesting with {len(results_df)} records")

    return results_df


def backtest_from_dataframe(
    df: pd.DataFrame,
    atc_config: dict,
    symbol_column: str = "symbol",
    price_column: str = "close",
    npartitions: Optional[int] = None,
    partition_size: int = 10,
) -> pd.DataFrame:
    """Backtest ATC signals on an existing DataFrame using Dask.

    Args:
        df: Input DataFrame with historical data
        atc_config: ATC configuration parameters
        symbol_column: Column name for symbol
        price_column: Column name for price
        npartitions: Number of Dask partitions (auto if None)
        partition_size: Symbols per partition

    Returns:
        DataFrame with backtest results
    """
    if df.empty:
        log_warn("Empty DataFrame provided")
        return pd.DataFrame()

    log_info(f"Backtesting {len(df)} records for {df[symbol_column].nunique()} symbols")

    import dask.dataframe as dd

    ddf = dd.from_pandas(df, npartitions=npartitions)

    grouped = ddf.groupby(symbol_column)

    meta = {
        "symbol": "string",
        "signal": "float64",
        "price": "float64",
        "timestamp": "datetime64[ns]",
    }

    def process_with_gc(group_df: pd.DataFrame) -> pd.DataFrame:
        result = _process_symbol_group(group_df, symbol_column, price_column, atc_config)
        gc.collect()
        return result

    try:
        results_ddf = grouped.apply(process_with_gc, meta=meta)
    except Exception as e:
        log_error(f"Error in Dask apply: {e}")
        return pd.DataFrame()

    try:
        results_df = results_ddf.compute()
    except Exception as e:
        log_error(f"Error computing Dask results: {e}")
        return pd.DataFrame()

    log_info(f"Completed backtesting with {len(results_df)} records")

    return results_df


def backtest_multiple_files_dask(
    file_paths: List[str],
    atc_config: dict,
    chunksize: str = "100MB",
    symbol_column: str = "symbol",
    price_column: str = "close",
) -> pd.DataFrame:
    """Backtest across multiple historical data files.

    Args:
        file_paths: List of file paths
        atc_config: ATC configuration parameters
        chunksize: Size of each chunk
        symbol_column: Column name for symbol
        price_column: Column name for price

    Returns:
        Combined DataFrame with all results
    """
    import dask.bag as db

    if not file_paths:
        log_warn("No file paths provided")
        return pd.DataFrame()

    log_info(f"Backtesting {len(file_paths)} files")

    results_list = []

    for file_path in file_paths:
        log_info(f"Processing file: {file_path}")
        try:
            result = backtest_with_dask(
                file_path, atc_config, chunksize, symbol_column, price_column
            )
            if not result.empty:
                results_list.append(result)
        except Exception as e:
            log_error(f"Error processing file {file_path}: {e}")

    if not results_list:
        log_warn("No results from any file")
        return pd.DataFrame()

    try:
        combined_df = pd.concat(results_list, ignore_index=True)
        log_info(f"Combined results: {len(combined_df)} records")
        return combined_df
    except Exception as e:
        log_error(f"Error combining results: {e}")
        return pd.DataFrame()
