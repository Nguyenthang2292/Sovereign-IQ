"""Dask-based backtesting for large historical datasets."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import List, Optional

import dask.dataframe as dd
import pandas as pd
from dask.delayed import delayed

try:
    from modules.common.utils import log_error, log_info, log_warn
except ImportError:

    def log_info(msg: str) -> None:
        print(f"[INFO] {msg}")

    def log_error(msg: str) -> None:
        print(f"[ERROR] {msg}")

    def log_warn(msg: str) -> None:
        print(f"[WARN] {msg}")


def _create_error_result(symbol: str, error_msg: str) -> pd.DataFrame:
    """Create a standardized error result DataFrame.

    Args:
        symbol: The symbol being processed
        error_msg: The error message

    Returns:
        DataFrame with error tracking columns
    """
    return pd.DataFrame(
        {
            "symbol": [symbol],
            "signal": [float("nan")],
            "price": [float("nan")],
            "timestamp": [pd.NaT],
            "status": ["error"],
            "error_msg": [error_msg],
        }
    )


def _create_dask_from_memmap(
    mmap_array,
    descriptor,
    symbol_column: str,
    price_column: str,
    chunksize: str = "100MB",
) -> dd.DataFrame:
    """Create a Dask DataFrame from memory-mapped data.

    Args:
        mmap_array: Memory-mapped numpy array
        descriptor: MemmapDescriptor with metadata
        symbol_column: Column name for symbol
        price_column: Column name for price
        chunksize: Size of each Dask partition

    Returns:
        Dask DataFrame
    """
    import numpy as np

    if symbol_column not in descriptor.columns:
        raise ValueError(f"symbol_column '{symbol_column}' not in descriptor columns: {list(descriptor.columns)}")
    if price_column not in descriptor.columns:
        raise ValueError(f"price_column '{price_column}' not in descriptor columns: {list(descriptor.columns)}")

    total_rows = descriptor.shape[0]
    target_rows_per_partition = 100_000

    match = re.search(r"(\d+)", str(chunksize))
    if match:
        n = int(match.group(1))
        if "GB" in str(chunksize).upper():
            target_rows_per_partition = max(1000, n * 1_000_000)
        elif "MB" in str(chunksize).upper():
            target_rows_per_partition = max(1000, n * 1_000)
        else:
            target_rows_per_partition = max(1000, n)
    num_partitions = max(1, total_rows // target_rows_per_partition)
    rows_per_partition = total_rows // num_partitions

    def read_memmap_partition(partition_idx):
        start = partition_idx * rows_per_partition
        end = start + rows_per_partition if partition_idx < num_partitions - 1 else total_rows
        partition_slice = slice(start, end)

        partition_data = mmap_array[partition_slice]

        data_dict = {}
        for col in descriptor.columns:
            data_dict[col] = np.array(partition_data[col])

        return pd.DataFrame(data_dict, index=range(start, end))

    delayed_objs = [delayed(read_memmap_partition)(i) for i in range(num_partitions)]

    ddf = dd.from_delayed(
        delayed_objs,
        meta={col: descriptor.dtypes.get(col, "float64") for col in descriptor.columns},
    )

    return ddf


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
    symbol = group_df[symbol_column].iloc[0] if not group_df.empty else "UNKNOWN"

    try:
        from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals

        prices = group_df[price_column].sort_index()

        if prices.empty or len(prices) < atc_config.get("ema_len", 28):
            return _create_error_result(symbol, "Insufficient data: empty or below ema_len threshold")

        # Map config parameters to function arguments
        func_args = atc_config.copy()
        if "lambda_param" in func_args:
            func_args["La"] = func_args.pop("lambda_param")
        if "decay" in func_args:
            func_args["De"] = func_args.pop("decay")

        # Remove parameters not accepted by compute_atc_signals using inspect
        valid_params = set(inspect.signature(compute_atc_signals).parameters.keys())
        func_args = {k: v for k, v in func_args.items() if k in valid_params}

        result = compute_atc_signals(prices=prices, **func_args)

        avg_signal = result.get("Average_Signal", pd.Series())

        if avg_signal.empty:
            return _create_error_result(symbol, "No average signal computed")

        return pd.DataFrame(
            {
                "symbol": [symbol] * len(avg_signal),
                "signal": avg_signal.values,
                "price": prices.values,
                "timestamp": prices.index,
                "status": ["success"] * len(avg_signal),
                "error_msg": [None] * len(avg_signal),
            }
        )
    except Exception as e:
        log_error(f"Error processing symbol group: {e}")
        return _create_error_result(symbol, str(e))


def _execute_dask_backtest(
    ddf: dd.DataFrame,
    atc_config: dict,
    symbol_column: str = "symbol",
    price_column: str = "close",
    show_progress: bool = False,
    scheduler: Optional[str] = None,
) -> pd.DataFrame:
    """Execute Dask backtesting on loaded data.

    Args:
        ddf: Dask DataFrame with historical data
        atc_config: ATC configuration parameters
        symbol_column: Column name for symbol
        price_column: Column name for price
        show_progress: Show progress bar during computation
        scheduler: Dask scheduler to use (None, 'threads', 'processes', or address)

    Returns:
        DataFrame with backtest results
    """
    grouped = ddf.groupby(symbol_column)

    meta = {
        "symbol": "string",
        "signal": "float64",
        "price": "float64",
        "timestamp": "datetime64[ns]",
    }

    try:
        # Add include_groups=False to avoid FutureWarning about grouping columns
        results_ddf = grouped.apply(
            _process_symbol_group,
            symbol_column=symbol_column,
            price_column=price_column,
            atc_config=atc_config,
            meta=meta,
            include_groups=False,
        )
    except Exception as e:
        log_error(f"Error in Dask apply: {e}")
        return pd.DataFrame()

    try:
        if show_progress:
            from dask.diagnostics.progress import ProgressBar

            with ProgressBar():
                results_df = results_ddf.compute(scheduler=scheduler) if scheduler else results_ddf.compute()
        else:
            results_df = results_ddf.compute(scheduler=scheduler) if scheduler else results_ddf.compute()
    except Exception as e:
        log_error(f"Error computing Dask results: {e}")
        return pd.DataFrame()

    log_info(f"Completed backtesting with {len(results_df)} records")

    return results_df


def _validate_file_path(file_path: str) -> bool:
    """Validate that a file path exists and is accessible.

    Args:
        file_path: Path to validate

    Returns:
        True if path is valid, False otherwise
    """
    try:
        path = Path(file_path)
        if not path.exists():
            log_error(f"File does not exist: {file_path}")
            return False

        if not path.is_file():
            log_error(f"Path is not a file: {file_path}")
            return False

        valid_extensions = {".csv", ".parquet", ".pq"}
        if path.suffix.lower() not in valid_extensions:
            log_warn(f"Unexpected file extension: {path.suffix}. Expected: {valid_extensions}")

        return True
    except Exception as e:
        log_error(f"Invalid file path {file_path}: {e}")
        return False


def backtest_with_dask(
    historical_data_path: str,
    atc_config: dict,
    chunksize: str = "100MB",
    symbol_column: str = "symbol",
    price_column: str = "close",
    use_memory_mapped: bool = False,
) -> pd.DataFrame:
    """Backtest ATC signals on large historical data using Dask.

    Args:
        historical_data_path: Path to CSV/Parquet file
        atc_config: ATC configuration parameters
        chunksize: Size of each chunk (e.g., "100MB")
        symbol_column: Column name for symbol
        price_column: Column name for price
        use_memory_mapped: Use memory-mapped files to reduce RAM usage

    Returns:
        DataFrame with backtest results
    """
    log_info(f"Loading historical data from {historical_data_path}")

    if not _validate_file_path(historical_data_path):
        return pd.DataFrame()

    try:
        if use_memory_mapped:
            from modules.adaptive_trend_LTS_mini.utils.memory_mapped_data import (
                load_memory_mapped_from_csv,
            )

            descriptor, mmap_array = load_memory_mapped_from_csv(historical_data_path, symbol_column, price_column)

            if descriptor is None or mmap_array is None:
                log_error("Failed to load memory-mapped data")
                return pd.DataFrame()

            log_info(f"Using memory-mapped data: {descriptor.mmap_path}")

            ddf = _create_dask_from_memmap(mmap_array, descriptor, symbol_column, price_column, chunksize)
        else:
            ddf = dd.read_csv(
                historical_data_path,
                blocksize=chunksize,
                dtype={symbol_column: "string", price_column: "float64"},
            )
    except Exception as e:
        log_error(f"Failed to read CSV file: {e}")
        return pd.DataFrame()

    return _execute_dask_backtest(ddf, atc_config, symbol_column, price_column)


def backtest_from_dataframe(
    df: pd.DataFrame,
    atc_config: dict,
    symbol_column: str = "symbol",
    price_column: str = "close",
    npartitions: Optional[int] = None,
) -> pd.DataFrame:
    """Backtest ATC signals on an existing DataFrame using Dask.

    Args:
        df: Input DataFrame with historical data
        atc_config: ATC configuration parameters
        symbol_column: Column name for symbol
        price_column: Column name for price
        npartitions: Number of Dask partitions (auto if None)

    Returns:
        DataFrame with backtest results
    """
    if df.empty:
        log_warn("Empty DataFrame provided")
        return pd.DataFrame()

    # Validate required columns exist
    if symbol_column not in df.columns:
        log_error(f"Column '{symbol_column}' not found in DataFrame. Available columns: {list(df.columns)}")
        return pd.DataFrame()

    if price_column not in df.columns:
        log_error(f"Column '{price_column}' not found in DataFrame. Available columns: {list(df.columns)}")
        return pd.DataFrame()

    log_info(f"Backtesting {len(df)} records for {df[symbol_column].nunique()} symbols")

    ddf = dd.from_pandas(df, npartitions=npartitions)

    return _execute_dask_backtest(ddf, atc_config, symbol_column, price_column)


def backtest_multiple_files_dask(
    file_paths: List[str],
    atc_config: dict,
    chunksize: str = "100MB",
    symbol_column: str = "symbol",
    price_column: str = "close",
) -> pd.DataFrame:
    """Backtest across multiple historical data files in parallel.

    Args:
        file_paths: List of file paths
        atc_config: ATC configuration parameters
        chunksize: Size of each chunk
        symbol_column: Column name for symbol
        price_column: Column name for price

    Returns:
        Combined DataFrame with all results
    """

    if not file_paths:
        log_warn("No file paths provided")
        return pd.DataFrame()

    log_info(f"Backtesting {len(file_paths)} files in parallel")

    valid_paths = [fp for fp in file_paths if _validate_file_path(fp)]

    if not valid_paths:
        log_error("No valid file paths provided")
        return pd.DataFrame()

    if len(valid_paths) < len(file_paths):
        log_warn(f"Filtered {len(file_paths) - len(valid_paths)} invalid paths")

    try:
        ddf = dd.read_csv(
            valid_paths,
            blocksize=chunksize,
            dtype={symbol_column: "string", price_column: "float64"},
        )

        return _execute_dask_backtest(ddf, atc_config, symbol_column, price_column)
    except Exception as e:
        log_error(f"Failed to read or process files: {e}")
        return pd.DataFrame()
