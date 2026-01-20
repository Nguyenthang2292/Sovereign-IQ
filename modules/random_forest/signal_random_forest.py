"""
Test script for Random Forest signal generation across all Binance symbols.

This script fetches data for all Binance futures symbols across multiple timeframes,
computes Random Forest signals for each, and generates a detailed report.

Features:
- Batch processing to avoid RAM overflow
- Multi-threading for I/O-bound data fetching
- Multi-processing for CPU-intensive RF computation
- Resource monitoring and limiting (70% RAM and CPU)

Usage:
    python -m modules.random_forest.signal_random_forest_test
        (automatically shows interactive menu)
    python -m modules.random_forest.signal_random_forest_test --menu
        (explicitly shows interactive menu)
    python -m modules.random_forest.signal_random_forest_test --execution-mode hybrid --batch-size 50
        (use command-line arguments directly)
    python -m modules.random_forest.signal_random_forest_test --max-symbols 100 --max-workers-thread 16
        (use command-line arguments directly)

Output:
    Creates a CSV report: random_forest_signals_report_{timestamp}.csv
"""

import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from colorama import Fore, Style

from config.random_forest import MODELS_DIR, RANDOM_FOREST_MODEL_FILENAME
from core.signal_calculators import get_random_forest_signal
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.utils import (
    color_text,
    log_error,
    log_info,
    log_progress,
    log_success,
    log_warn,
    prompt_user_input,
)


class ResourceMonitor:
    """Monitor and limit system resources (RAM and CPU)."""

    def __init__(self, max_memory_pct: float = 70.0, max_cpu_pct: float = 70.0):
        """Initialize resource monitor.

        Args:
            max_memory_pct: Maximum memory usage percentage (default: 70%)
            max_cpu_pct: Maximum CPU usage percentage (default: 70%)
        """
        self.max_memory_pct = max_memory_pct
        self.max_cpu_pct = max_cpu_pct
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None

    def get_memory_usage(self) -> Tuple[float, float, float]:
        """Get current memory usage.

        Returns:
            Tuple of (process_memory_mb, system_memory_pct, available_memory_mb)
        """
        if not PSUTIL_AVAILABLE:
            return 0.0, 0.0, 0.0

        try:
            # Process memory
            process_memory = self.process.memory_info().rss / (1024**2)  # MB

            # System memory
            system_memory = psutil.virtual_memory()
            system_memory_pct = system_memory.percent
            available_memory_mb = system_memory.available / (1024**2)  # MB

            return process_memory, system_memory_pct, available_memory_mb
        except Exception:
            return 0.0, 0.0, 0.0

    def get_cpu_usage(self) -> Tuple[float, float]:
        """Get current CPU usage.

        Returns:
            Tuple of (process_cpu_pct, system_cpu_pct)
        """
        if not PSUTIL_AVAILABLE:
            return 0.0, 0.0

        try:
            # Process CPU
            process_cpu = self.process.cpu_percent(interval=0.1)

            # System CPU (average across all cores)
            system_cpu = psutil.cpu_percent(interval=0.1)

            return process_cpu, system_cpu
        except Exception:
            return 0.0, 0.0

    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit.

        Returns:
            True if memory usage is below limit, False otherwise
        """
        if not PSUTIL_AVAILABLE:
            return True

        _, system_memory_pct, _ = self.get_memory_usage()
        return system_memory_pct < self.max_memory_pct

    def check_cpu_limit(self) -> bool:
        """Check if CPU usage exceeds limit.

        Returns:
            True if CPU usage is below limit, False otherwise
        """
        if not PSUTIL_AVAILABLE:
            return True

        _, system_cpu = self.get_cpu_usage()
        return system_cpu < self.max_cpu_pct

    def wait_if_over_limit(self, max_wait_seconds: float = 5.0) -> None:
        """Wait if resource usage exceeds limits.

        Args:
            max_wait_seconds: Maximum time to wait in seconds
        """
        if not PSUTIL_AVAILABLE:
            return

        wait_time = 0.0
        wait_interval = 0.5

        while wait_time < max_wait_seconds:
            if self.check_memory_limit() and self.check_cpu_limit():
                break

            time.sleep(wait_interval)
            wait_time += wait_interval

            if wait_time >= max_wait_seconds:
                log_warn(
                    f"Resource usage still high after {max_wait_seconds}s wait. "
                    f"Memory: {self.get_memory_usage()[1]:.1f}%, "
                    f"CPU: {self.get_cpu_usage()[1]:.1f}%"
                )

    def get_status_string(self) -> str:
        """Get formatted status string for logging.

        Returns:
            Formatted string with current resource usage
        """
        if not PSUTIL_AVAILABLE:
            return "Resource monitoring unavailable (psutil not installed)"

        process_mem, system_mem_pct, available_mem = self.get_memory_usage()
        process_cpu, system_cpu = self.get_cpu_usage()

        return (
            f"Memory: {process_mem:.1f}MB (process), {system_mem_pct:.1f}% (system), "
            f"{available_mem:.1f}MB available | "
            f"CPU: {process_cpu:.1f}% (process), {system_cpu:.1f}% (system)"
        )


def get_signal_direction(signal: int) -> str:
    """Get signal direction from signal value.

    Args:
        signal: Random Forest signal value (1, -1, or 0)

    Returns:
        "LONG", "SHORT", or "NEUTRAL"
    """
    if signal == 1:
        return "LONG"
    elif signal == -1:
        return "SHORT"
    else:
        return "NEUTRAL"


def calculate_confidence(confidence_value: float) -> float:
    """Calculate confidence level from confidence value.

    Args:
        confidence_value: Confidence value from Random Forest (0.0 to 1.0)

    Returns:
        Confidence level (0.0 to 1.0)
    """
    return max(0.0, min(1.0, confidence_value))


def fetch_data_for_symbol(
    symbol: str,
    data_fetcher: DataFetcher,
    timeframe: str,
    limit: int,
) -> Optional[Tuple[pd.DataFrame, Optional[str]]]:
    """Fetch OHLCV data for a symbol (I/O-bound task).

    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        data_fetcher: DataFetcher instance
        timeframe: Timeframe string (e.g., '15m', '30m', '1h')
        limit: Number of candles to fetch

    Returns:
        Tuple of (DataFrame, exchange_id) or None if failed
    """
    try:
        df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=limit,
            timeframe=timeframe,
            check_freshness=True,
        )

        if df is None or df.empty:
            return None

        return df, exchange_id
    except Exception as e:
        log_error(f"Error fetching data for {symbol} ({timeframe}): {type(e).__name__}: {e}")
        return None


def compute_random_forest_signals_for_data(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    exchange_id: Optional[str],
    data_fetcher: DataFetcher,
    limit: int,
    model_path: Optional[str] = None,
) -> Optional[Dict]:
    """Compute Random Forest signals for fetched data (CPU-bound task).

    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe string
        df: OHLCV DataFrame
        exchange_id: Exchange ID
        data_fetcher: DataFetcher instance
        limit: Number of candles to fetch (for fallback)
        model_path: Optional path to model file (default: uses default from config)

    Returns:
        Dictionary with Random Forest results, or None if failed
    """
    try:
        # Get current price
        if "close" not in df.columns:
            return None

        current_price = df["close"].iloc[-1]

        # Calculate Random Forest signal
        rf_result = get_random_forest_signal(
            data_fetcher=data_fetcher,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            model_path=model_path,
            df=df,
        )

        if rf_result is None:
            return None

        signal, confidence = rf_result

        # Build result dictionary
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": current_price,
            "signal": signal,
            "signal_direction": get_signal_direction(signal),
            "confidence": calculate_confidence(confidence),
            "exchange": exchange_id or "UNKNOWN",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return result
    except Exception as e:
        log_error(f"Error computing Random Forest for {symbol} ({timeframe}): {type(e).__name__}: {e}")
        return None


def test_random_forest_for_symbol(
    symbol: str,
    data_fetcher: DataFetcher,
    timeframe: str,
    limit: int,
    model_path: Optional[str] = None,
) -> Optional[Dict]:
    """Test Random Forest for a single symbol on a specific timeframe (sequential mode).

    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        data_fetcher: DataFetcher instance
        timeframe: Timeframe string (e.g., '15m', '30m', '1h')
        limit: Number of candles to fetch
        model_path: Optional path to model file (default: uses default from config)

    Returns:
        Dictionary with Random Forest results, or None if failed
    """
    # Fetch data
    data_result = fetch_data_for_symbol(symbol, data_fetcher, timeframe, limit)
    if data_result is None:
        return None

    df, exchange_id = data_result

    # Compute Random Forest signals
    return compute_random_forest_signals_for_data(symbol, timeframe, df, exchange_id, data_fetcher, limit, model_path)


def create_batches(items: List[str], batch_size: int) -> List[List[str]]:
    """Create batches from a list of items.

    Args:
        items: List of items to batch
        batch_size: Size of each batch

    Returns:
        List of batches, each containing up to batch_size items
    """
    if not items or batch_size <= 0:
        return []

    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i : i + batch_size])

    return batches


def auto_train_model(
    data_fetcher: DataFetcher, symbols: List[str], timeframe: str, limit: int = 1000
) -> Tuple[bool, Optional[str]]:
    """Auto train a new Random Forest model using data from specified symbols.

    Args:
        data_fetcher: DataFetcher instance
        symbols: List of symbols to use for training
        timeframe: Timeframe for data fetching
        limit: Number of candles to fetch per symbol

    Returns:
        Tuple of (success, model_path). If success is True, model_path contains the path to saved model.
        If False, model_path contains error message.
    """
    try:

        import pandas as pd

        from modules.random_forest.core.model import train_random_forest_model

        log_info(f"Auto training model using {len(symbols)} symbols: {', '.join(symbols)}")

        # Get model path before training
        from config.random_forest import MODELS_DIR, RANDOM_FOREST_MODEL_FILENAME

        old_model_path = MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME
        old_model_exists = old_model_path.exists()

        if old_model_exists:
            log_info(f"Old model found at: {old_model_path}. Will delete after successful training.")

        # Fetch data for each symbol
        all_data = []
        for symbol in symbols:
            try:
                df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                    symbol, limit=limit, timeframe=timeframe, check_freshness=False
                )
                if df is not None and not df.empty:
                    # Add symbol column for tracking
                    df = df.copy()
                    df["symbol"] = symbol
                    all_data.append(df)
                    log_progress(f"Fetched {len(df)} rows for {symbol}")
                else:
                    log_warn(f"No data for {symbol}")
            except Exception as e:
                log_warn(f"Error fetching data for {symbol}: {e}")

        if not all_data:
            return False, "No data fetched for any symbol"

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        log_info(f"Combined data shape: {combined_df.shape}")

        # Train model
        model = train_random_forest_model(combined_df, save_model=True)

        if model is None:
            return False, "Model training failed"

        # Find the latest saved model file
        import glob
        import os

        model_files = glob.glob(str(MODELS_DIR / "*.joblib"))
        if not model_files:
            return False, "Model file not found after training"

        latest_model = max(model_files, key=os.path.getmtime)
        model_path = latest_model

        # Copy to default location for validation
        default_model_path = MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME
        import shutil

        shutil.copy2(latest_model, default_model_path)
        log_info(f"Copied model to default location: {default_model_path}")

        log_success(f"Auto training completed successfully. Model saved to: {model_path}")

        # Delete old model if it existed and is different from new model
        # (In case of versioning, old_model_path might be different from model_path)
        if old_model_exists and old_model_path.exists():
            try:
                # Only delete if it's actually the old model (not the newly created one)
                # Check by comparing modification times or just delete if paths are same
                old_model_path.unlink()
                log_info(f"Deleted old model: {old_model_path}")
            except Exception as e:
                log_warn(f"Could not delete old model: {e}")

        return True, model_path

    except Exception as e:
        error_msg = f"Auto training failed: {type(e).__name__}: {e}"
        log_error(error_msg)
        return False, error_msg


def get_top_volume_symbols(data_fetcher: DataFetcher, top_n: int = 10) -> List[str]:
    """Fetch top N symbols by 24h volume from market data.

    Args:
        data_fetcher: DataFetcher instance
        top_n: Number of top volume symbols to return

    Returns:
        List of top volume symbols (e.g., ['BTC/USDT', 'ETH/USDT', ...])
    """
    try:
        # Use existing robust method from DataFetcher to get top volume symbols
        top_symbols = data_fetcher.list_binance_futures_symbols(
            max_candidates=top_n, progress_label="Fetching Top Symbols"
        )
        return top_symbols
    except Exception as e:
        log_error(f"Error fetching top volume symbols: {type(e).__name__}: {e}")
        return []


def validate_model(
    model_path: Optional[str] = None,
    data_fetcher: Optional[DataFetcher] = None,
    symbols: Optional[List[str]] = None,
    timeframe: str = "1h",
    limit: int = 100,
) -> Tuple[bool, Optional[str]]:
    """Validate Random Forest model before processing.

    Args:
        model_path: Optional path to model file (default: uses default from config)
        data_fetcher: DataFetcher instance (required if symbols provided)
        symbols: Optional list of symbols to test compatibility with
        timeframe: Timeframe for testing (default: 1h)
        limit: Number of candles for testing (default: 100)

    Returns:
        Tuple of (is_valid, error_message). If is_valid is True, model is compatible.
        If False, error_message contains the reason.
    """
    try:
        from pathlib import Path

        from modules.random_forest.core.model import load_random_forest_model

        # Determine model path
        if model_path:
            validated_path = Path(model_path)
        else:
            validated_path = MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME

        if not validated_path.exists():
            return False, f"Model file not found: {validated_path}"

        # Load model
        model = load_random_forest_model(validated_path)
        if model is None:
            return False, "Failed to load model file"

        # Check if model uses deprecated raw OHLCV features
        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
            raw_ohlcv_features = ["open", "high", "low", "close", "volume"]
            model_raw_ohlcv = [f for f in model.feature_names_in_ if f in raw_ohlcv_features]
            if model_raw_ohlcv:
                error_msg = (
                    f"Model uses deprecated raw OHLCV features: {model_raw_ohlcv}. "
                    f"Raw OHLCV features are no longer supported. "
                    f"Please retrain model with derived features "
                    f"(returns_1, returns_5, log_volume, "
                    f"high_low_range, close_open_diff)."
                )
                return False, error_msg

        # If symbols provided, test model compatibility with their data
        if symbols and data_fetcher:
            log_info(f"Testing model compatibility with {len(symbols)} symbols...")
            incompatible_symbols = []

            for symbol in symbols:
                try:
                    # Fetch sample data
                    df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                        symbol, limit=limit, timeframe=timeframe, check_freshness=False
                    )
                    if df is None or df.empty:
                        continue

                    # Try to get signal (this will test feature compatibility)
                    rf_result = get_random_forest_signal(
                        data_fetcher=data_fetcher,
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=limit,
                        model_path=str(validated_path),
                        df=df,
                    )

                    if rf_result is None:
                        log_warn(f"Compatibility test: get_random_forest_signal returned None for {symbol}")
                        incompatible_symbols.append(symbol)

                except Exception as e:
                    log_warn(f"Compatibility test: Exception for {symbol}: {type(e).__name__}: {e}")
                    incompatible_symbols.append(symbol)

            if incompatible_symbols:
                symbol_list = ", ".join(incompatible_symbols[:5])
                if len(incompatible_symbols) > 5:
                    symbol_list += "..."
                return False, f"Model incompatible with data from symbols: {symbol_list}"

        return True, None
    except Exception as e:
        return False, f"Error validating model: {type(e).__name__}: {e}"


def calculate_optimal_batch_size(
    num_symbols: int,
    num_timeframes: int,
    max_memory_pct: float = 70.0,
    default_batch_size: int = 50,
) -> int:
    """Calculate optimal batch size based on available memory.

    Args:
        num_symbols: Total number of symbols to process
        num_timeframes: Number of timeframes to process
        max_memory_pct: Maximum memory usage percentage (default: 70%)
        default_batch_size: Default batch size if calculation fails (default: 50)

    Returns:
        Optimal batch size
    """
    if not PSUTIL_AVAILABLE:
        return default_batch_size

    try:
        # Get available memory
        memory = psutil.virtual_memory()
        available_memory_mb = memory.available / (1024**2)  # MB

        # Calculate usable memory (70% of available)
        usable_memory_mb = available_memory_mb * (max_memory_pct / 100.0)

        # Estimate memory per symbol per timeframe
        # Random Forest processing is memory-intensive
        # Estimate: ~0.5 MB per symbol per timeframe (includes model, features, predictions)
        memory_per_symbol_mb = 0.5

        # Calculate how many symbols we can process at once
        total_memory_needed = memory_per_symbol_mb * num_timeframes
        if total_memory_needed > 0:
            max_symbols_per_batch = int(usable_memory_mb / total_memory_needed)
        else:
            max_symbols_per_batch = default_batch_size

        # Limit batch size between 10 and 100
        batch_size = max(10, min(100, max_symbols_per_batch))

        return batch_size
    except Exception:
        return default_batch_size


def process_symbol_batch_sequential(
    batch: List[str],
    data_fetcher: DataFetcher,
    timeframe: str,
    limit: int,
    model_path: Optional[str] = None,
) -> List[Dict]:
    """Process a batch of symbols sequentially.

    Args:
        batch: List of symbols to process
        data_fetcher: DataFetcher instance
        timeframe: Timeframe string
        limit: Number of candles to fetch
        model_path: Optional path to model file

    Returns:
        List of result dictionaries
    """
    results = []

    for symbol in batch:
        try:
            result = test_random_forest_for_symbol(symbol, data_fetcher, timeframe, limit, model_path)
            if result:
                results.append(result)
        except Exception as e:
            log_error(f"Error processing {symbol}: {type(e).__name__}: {e}")

    return results


def process_symbol_batch_threading(
    batch: List[str],
    data_fetcher: DataFetcher,
    timeframe: str,
    limit: int,
    max_workers: int,
    model_path: Optional[str] = None,
) -> List[Dict]:
    """Process a batch of symbols using threading.

    Args:
        batch: List of symbols to process
        data_fetcher: DataFetcher instance
        timeframe: Timeframe string
        limit: Number of candles to fetch
        max_workers: Maximum number of worker threads
        model_path: Optional path to model file

    Returns:
        List of result dictionaries
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(test_random_forest_for_symbol, symbol, data_fetcher, timeframe, limit, model_path): symbol
            for symbol in batch
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                log_error(f"Error processing {symbol} in thread: {type(e).__name__}: {e}")

    return results


def process_symbol_batch_hybrid(
    batch: List[str],
    data_fetcher: DataFetcher,
    timeframe: str,
    limit: int,
    max_workers_thread: int,
    max_workers_process: int,
    model_path: Optional[str] = None,
) -> List[Dict]:
    """Process a batch of symbols using hybrid approach (threading for fetch, threading for compute).

    Args:
        batch: List of symbols to process
        data_fetcher: DataFetcher instance
        timeframe: Timeframe string
        limit: Number of candles to fetch
        max_workers_thread: Maximum number of worker threads for data fetching
        max_workers_process: Maximum number of worker threads for RF computation
        model_path: Optional path to model file

    Returns:
        List of result dictionaries
    """
    results = []

    # Step 1: Fetch data using threading (I/O-bound)
    fetched_data = {}
    with ThreadPoolExecutor(max_workers=max_workers_thread) as executor:
        future_to_symbol = {
            executor.submit(fetch_data_for_symbol, symbol, data_fetcher, timeframe, limit): symbol for symbol in batch
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data_result = future.result()
                if data_result is not None:
                    fetched_data[symbol] = data_result
            except Exception as e:
                log_error(f"Error fetching data for {symbol} in thread: {type(e).__name__}: {e}")

    # Step 2: Compute Random Forest signals using threading (CPU-bound)
    if not fetched_data:
        return results

    with ThreadPoolExecutor(max_workers=max_workers_process) as executor:
        future_to_symbol = {
            executor.submit(
                compute_random_forest_signals_for_data, symbol, timeframe, df, ex_id, data_fetcher, limit, model_path
            ): symbol
            for symbol, (df, ex_id) in fetched_data.items()
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                log_error(f"Error computing RF for {symbol} in thread: {type(e).__name__}: {e}")

    return results


def interactive_menu() -> dict:
    """Interactive menu for configuring Random Forest test parameters.

    Returns:
        Dictionary with configuration parameters matching argparse namespace.
    """
    print("\n" + color_text("=" * 60, Fore.CYAN))
    print(color_text("RANDOM FOREST SIGNAL TEST - INTERACTIVE CONFIGURATION", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 60, Fore.CYAN))

    config = {}

    # 1. Max symbols
    print("\n" + color_text("1. Maximum Symbols", Fore.YELLOW))
    print("   Leave empty for all symbols")
    max_symbols_input = prompt_user_input("   Enter max symbols [all]: ", default="")
    if max_symbols_input.strip():
        try:
            config["max_symbols"] = int(max_symbols_input)
        except ValueError:
            log_warn("Invalid number, using default: all symbols")
            config["max_symbols"] = None
    else:
        config["max_symbols"] = None

    # 2. Limit (candles)
    print("\n" + color_text("2. Data Limit", Fore.YELLOW))
    limit_input = prompt_user_input("   Number of candles to fetch per symbol [1500]: ", default="1500")
    try:
        config["limit"] = int(limit_input) if limit_input.strip() else 1500
    except ValueError:
        log_warn("Invalid number, using default: 1500")
        config["limit"] = 1500

    # 3. Timeframes
    print("\n" + color_text("3. Timeframes", Fore.YELLOW))
    print("   Available: 15m, 30m, 1h, 4h, 1d")
    print("   Enter comma-separated list (e.g., 1h,4h,1d)")
    timeframes_input = prompt_user_input("   Timeframes [1h,4h,1d]: ", default="1h,4h,1d")
    config["timeframes"] = timeframes_input.strip() if timeframes_input.strip() else "1h,4h,1d"

    # 4. Model path
    print("\n" + color_text("4. Model Path", Fore.YELLOW))
    print(f"   Default: {MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME}")
    print("   Leave empty for default model")
    model_path_input = prompt_user_input("   Model file path [default]: ", default="")
    config["model_path"] = model_path_input.strip() if model_path_input.strip() else None

    # 5. Output file
    print("\n" + color_text("5. Output File", Fore.YELLOW))
    print("   Leave empty for auto-generated filename")
    output_input = prompt_user_input("   Output CSV file path [auto]: ", default="")
    config["output"] = output_input.strip() if output_input.strip() else None

    # 6. Batch size
    print("\n" + color_text("6. Batch Size", Fore.YELLOW))
    print("   Leave empty for auto-calculation based on RAM")
    batch_size_input = prompt_user_input("   Batch size [auto]: ", default="")
    if batch_size_input.strip():
        try:
            config["batch_size"] = int(batch_size_input)
        except ValueError:
            log_warn("Invalid number, using default: auto")
            config["batch_size"] = None
    else:
        config["batch_size"] = None

    # 7. Max workers (thread)
    print("\n" + color_text("7. Worker Threads", Fore.YELLOW))
    print("   Leave empty for auto-calculation")
    max_workers_thread_input = prompt_user_input("   Max worker threads [auto]: ", default="")
    if max_workers_thread_input.strip():
        try:
            config["max_workers_thread"] = int(max_workers_thread_input)
        except ValueError:
            log_warn("Invalid number, using default: auto")
            config["max_workers_thread"] = None
    else:
        config["max_workers_thread"] = None

    # 8. Max workers (process)
    print("\n" + color_text("8. Worker Processes", Fore.YELLOW))
    print("   Leave empty for CPU count")
    max_workers_process_input = prompt_user_input("   Max worker processes [auto]: ", default="")
    if max_workers_process_input.strip():
        try:
            config["max_workers_process"] = int(max_workers_process_input)
        except ValueError:
            log_warn("Invalid number, using default: auto")
            config["max_workers_process"] = None
    else:
        config["max_workers_process"] = None

    # 9. Execution mode
    print("\n" + color_text("9. Execution Mode", Fore.YELLOW))
    print("   1) Sequential - Process one symbol at a time")
    print("   2) Threading - Use threads for I/O-bound tasks")
    print("   4) Hybrid - Use threads for fetch, threads for compute (recommended)")
    mode_choice = prompt_user_input("   Select execution mode [1/2/4] [4]: ", default="4")
    mode_map = {"1": "sequential", "2": "threading", "4": "hybrid"}
    config["execution_mode"] = mode_map.get(mode_choice.strip(), "hybrid")

    # 10. Max memory percentage
    print("\n" + color_text("10. Memory Limit", Fore.YELLOW))
    max_memory_input = prompt_user_input("   Maximum memory usage percentage [70.0]: ", default="70.0")
    try:
        config["max_memory_pct"] = float(max_memory_input) if max_memory_input.strip() else 70.0
    except ValueError:
        log_warn("Invalid number, using default: 70.0")
        config["max_memory_pct"] = 70.0

    # 11. Max CPU percentage
    print("\n" + color_text("11. CPU Limit", Fore.YELLOW))
    max_cpu_input = prompt_user_input("   Maximum CPU usage percentage [70.0]: ", default="70.0")
    try:
        config["max_cpu_pct"] = float(max_cpu_input) if max_cpu_input.strip() else 70.0
    except ValueError:
        log_warn("Invalid number, using default: 70.0")
        config["max_cpu_pct"] = 70.0

    # 12. Auto train timeframe
    print("\n" + color_text("12. Auto Train Timeframe", Fore.YELLOW))
    print("   This TF will be used if model is incompatible and auto training is enabled")
    print("   Available: 15m, 30m, 1h, 4h, 1d")
    train_tf_input = prompt_user_input("   Auto train TF [1h]: ", default="1h")
    valid_train_tfs = ["15m", "30m", "1h", "4h", "1d"]
    if train_tf_input.strip() in valid_train_tfs:
        config["auto_train_tf"] = train_tf_input.strip()
    else:
        log_warn("Invalid TF, using default: 1h")
        config["auto_train_tf"] = "1h"

    # Summary
    print("\n" + color_text("=" * 60, Fore.CYAN))
    print(color_text("CONFIGURATION SUMMARY", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 60, Fore.CYAN))
    print(f"  Max Symbols: {config['max_symbols'] or 'All'}")
    print(f"  Limit: {config['limit']} candles")
    print(f"  Timeframes: {config['timeframes']}")
    print(f"  Model Path: {config['model_path'] or 'Default'}")
    print(f"  Output: {config['output'] or 'Auto-generated'}")
    print(f"  Batch Size: {config['batch_size'] or 'Auto'}")
    print(f"  Worker Threads: {config['max_workers_thread'] or 'Auto'}")
    print(f"  Worker Processes: {config['max_workers_process'] or 'Auto'}")
    print(f"  Execution Mode: {config['execution_mode']}")
    print(f"  Memory Limit: {config['max_memory_pct']}%")
    print(f"  CPU Limit: {config['max_cpu_pct']}%")
    print(f"  Auto Train TF: {config['auto_train_tf']}%")

    confirm = prompt_user_input("\nProceed with this configuration? [Y/n]: ", default="Y")
    if confirm.strip().lower() not in ("", "y", "yes"):
        log_warn("Configuration cancelled.")
        return None

    return config


def main():
    """Main function to run Random Forest test across all symbols and timeframes."""
    import argparse

    # Check if any arguments were provided (excluding script name)
    # If no arguments, automatically show menu
    show_menu_automatically = len(sys.argv) == 1

    parser = argparse.ArgumentParser(description="Test Random Forest signals across all Binance symbols")
    parser.add_argument(
        "--menu",
        action="store_true",
        help="Launch interactive menu for configuration (default: shown if no args provided)",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Maximum number of symbols to process (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1500,
        help="Number of candles to fetch per symbol (default: 1500)",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="1h,4h,1d",
        help="Comma-separated list of timeframes (default: 1h,4h,1d)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to Random Forest model file (default: uses default from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: random_forest_signals_report_{timestamp}.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for processing symbols (default: auto-calculate from RAM)",
    )
    parser.add_argument(
        "--max-workers-thread",
        type=int,
        default=None,
        help="Maximum number of worker threads for data fetching (default: min(32, num_symbols))",
    )
    parser.add_argument(
        "--max-workers-process",
        type=int,
        default=None,
        help="Maximum number of worker threads for RF computation (default: CPU count)",
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="hybrid",
        choices=["sequential", "threading", "hybrid"],
        help="Execution mode: sequential, threading, or hybrid (default: hybrid)",
    )
    parser.add_argument(
        "--max-memory-pct",
        type=float,
        default=70.0,
        help="Maximum memory usage percentage (default: 70.0)",
    )
    parser.add_argument(
        "--max-cpu-pct",
        type=float,
        default=70.0,
        help="Maximum CPU usage percentage (default: 70.0)",
    )
    parser.add_argument(
        "--auto-train-tf",
        type=str,
        default="1h",
        choices=["15m", "30m", "1h", "4h", "1d"],
        help="Timeframe for auto training if model incompatible (default: 1h)",
    )
    args = parser.parse_args()

    # Import config
    from config.random_forest import MODELS_DIR, RANDOM_FOREST_MODEL_FILENAME

    # Interactive menu mode (show menu if --menu flag is set OR if no arguments provided)
    if args.menu or show_menu_automatically:
        menu_config = interactive_menu()
        if menu_config is None:
            log_warn("Exiting...")
            return 0

        # Override args with menu configuration
        for key, value in menu_config.items():
            setattr(args, key, value)

    log_info("=" * 60)
    log_info("RANDOM FOREST SIGNAL TEST ACROSS ALL BINANCE SYMBOLS")
    log_info("=" * 60)

    # Initialize Resource Monitor
    resource_monitor = ResourceMonitor(max_memory_pct=args.max_memory_pct, max_cpu_pct=args.max_cpu_pct)
    log_info(f"Resource limits: Memory {args.max_memory_pct}%, CPU {args.max_cpu_pct}%")
    if PSUTIL_AVAILABLE:
        log_info(f"Initial status: {resource_monitor.get_status_string()}")

    # Initialize Exchange Manager and Data Fetcher (needed for model validation and auto training)
    try:
        exchange_manager = ExchangeManager(testnet=False)
        data_fetcher = DataFetcher(exchange_manager)
    except Exception as e:
        log_error(f"Failed to initialize ExchangeManager: {e}")
        return 1

    # Validate model path
    if args.model_path:
        model_path_obj = Path(args.model_path)
        if not model_path_obj.exists():
            log_error(f"Model file not found: {args.model_path}")
            return 1
        model_path = args.model_path
    else:
        # Use default model path
        default_model_path = MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME
        if not default_model_path.exists():
            log_error(f"Default model file not found: {default_model_path}")
            log_error("Please train a model first or specify --model-path")
            return 1
        model_path = str(default_model_path)
        log_info(f"Using default model: {model_path}")

    # Validate model compatibility (check for deprecated features)
    log_info("Validating model compatibility...")
    is_valid, error_msg = validate_model(model_path)
    if not is_valid:
        log_error("=" * 60)
        log_error("MODEL VALIDATION FAILED")
        log_error("=" * 60)
        log_error(error_msg)
        log_error("")

        # Get top volume symbols for auto training
        log_info("Fetching top volume symbols for auto training...")
        top_symbols = get_top_volume_symbols(data_fetcher, top_n=10)
        if not top_symbols:
            log_error("Cannot fetch top volume symbols for auto training.")
            log_error("Please retrain the model manually or use a compatible model.")
            return 1

        # Ask user if they want to auto train
        if args.menu or show_menu_automatically:
            symbol_preview = ", ".join(top_symbols[:3]) + "..." if len(top_symbols) > 3 else ", ".join(top_symbols)
            auto_train = prompt_user_input(
                f"Model is incompatible. Auto train new model using top {len(top_symbols)} volume symbols "
                f"({symbol_preview}) with TF '{args.auto_train_tf}'? [Y/n]: ",
                default="Y",
            )
            auto_train = auto_train.strip().lower() in ("", "y", "yes")
        else:
            # Command line mode - assume yes for auto training
            auto_train = True
            log_info(
                f"Auto training enabled. Will train model using top {len(top_symbols)} "
                f"volume symbols with TF '{args.auto_train_tf}'"
            )

        if auto_train:
            log_info("Starting auto training...")
            success, result_msg = auto_train_model(data_fetcher, top_symbols, args.auto_train_tf, limit=1000)
            if success:
                log_success("Auto training completed. Retrying model validation...")
                # Model has been copied to default location, use that for validation
                from config.random_forest import MODELS_DIR, RANDOM_FOREST_MODEL_FILENAME

                model_path = str(MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME)
                # Re-validate
                is_valid, error_msg = validate_model(model_path, data_fetcher, top_symbols, args.auto_train_tf)
                if not is_valid:
                    log_error("Auto trained model still incompatible. Please check training data or parameters.")
                    return 1
                log_success("Auto trained model validation passed!")
            else:
                log_error(f"Auto training failed: {result_msg}")
                log_error("Please retrain the model manually or use a compatible model.")
                return 1
        else:
            log_error("Auto training declined. Please retrain the model manually or use a compatible model.")
            return 1

    log_success("Model validation passed - model is compatible")

    # Parse timeframes
    timeframes = [tf.strip() for tf in args.timeframes.split(",")]
    valid_timeframes = ["15m", "30m", "1h", "4h", "1d"]
    timeframes = [tf for tf in timeframes if tf in valid_timeframes]

    if not timeframes:
        log_error(f"No valid timeframes provided. Valid options: {valid_timeframes}")
        return 1

    log_info(f"Timeframes: {', '.join(timeframes)}")
    log_info(f"Data limit: {args.limit} candles")
    log_info(f"Execution mode: {args.execution_mode}")

    # Fetch all Binance futures symbols
    log_info("Fetching Binance futures symbols...")
    try:
        all_symbols = data_fetcher.list_binance_futures_symbols(
            max_candidates=None,
            progress_label="Symbol Discovery",
        )
    except Exception as e:
        log_error(f"Failed to fetch symbols: {e}")
        return 1

    if not all_symbols:
        log_error("No symbols found")
        return 1

    # Limit symbols if specified
    if args.max_symbols and args.max_symbols > 0:
        symbols = all_symbols[: args.max_symbols]
        log_info(f"Processing first {len(symbols)} symbols (limit: {args.max_symbols})")
    else:
        symbols = all_symbols
        log_info(f"Processing all {len(symbols)} symbols")

    # Calculate optimal batch size
    if args.batch_size:
        batch_size = args.batch_size
        log_info(f"Using specified batch size: {batch_size}")
    else:
        batch_size = calculate_optimal_batch_size(len(symbols), len(timeframes), args.max_memory_pct)
        log_info(f"Using calculated batch size: {batch_size}")

    # Create batches
    batches = create_batches(symbols, batch_size)
    log_info(f"Created {len(batches)} batches (batch size: {batch_size})")

    # Set default worker counts
    if args.max_workers_thread is None:
        max_workers_thread = min(32, len(symbols) + 4)
    else:
        max_workers_thread = args.max_workers_thread

    if args.max_workers_process is None:
        max_workers_process = cpu_count()
    else:
        max_workers_process = args.max_workers_process

    log_info(f"Worker counts: Thread={max_workers_thread}, Process={max_workers_process}")

    # Collect results
    all_results: List[Dict] = []

    # Process each timeframe
    total_combinations = len(symbols) * len(timeframes)
    processed_count = 0

    for timeframe in timeframes:
        log_info("")
        log_info("=" * 40)
        log_info(f"PROCESSING TIMEFRAME: {timeframe}")
        log_info("=" * 40)

        # Process batches
        for batch_idx, batch in enumerate(batches, 1):
            try:
                log_info(f"Processing batch {batch_idx}/{len(batches)} ({len(batch)} symbols)...")

                # Check resources before processing
                resource_monitor.wait_if_over_limit()

                # Process batch based on execution mode
                if args.execution_mode == "sequential":
                    batch_results = process_symbol_batch_sequential(
                        batch, data_fetcher, timeframe, args.limit, model_path
                    )
                elif args.execution_mode == "threading":
                    batch_results = process_symbol_batch_threading(
                        batch, data_fetcher, timeframe, args.limit, max_workers_thread, model_path
                    )
                else:  # hybrid
                    batch_results = process_symbol_batch_hybrid(
                        batch,
                        data_fetcher,
                        timeframe,
                        args.limit,
                        max_workers_thread,
                        max_workers_process,
                        model_path,
                    )

                # Add results
                all_results.extend(batch_results)
                processed_count += len(batch_results)

                # Progress update
                pct = (processed_count / total_combinations) * 100
                log_progress(
                    f"Progress: {processed_count}/{total_combinations} ({pct:.1f}%) | "
                    f"Batch {batch_idx}/{len(batches)} complete | "
                    f"{resource_monitor.get_status_string()}"
                )

                # Memory cleanup after batch
                gc.collect()

            except KeyboardInterrupt:
                log_warn("Interrupted by user")
                break
            except Exception as e:
                log_error(f"Error processing batch {batch_idx}: {type(e).__name__}: {e}")
                # Graceful degradation: try sequential mode for this batch
                if args.execution_mode != "sequential":
                    log_warn(f"Falling back to sequential mode for batch {batch_idx}")
                    try:
                        batch_results = process_symbol_batch_sequential(
                            batch, data_fetcher, timeframe, args.limit, model_path
                        )
                        all_results.extend(batch_results)
                        processed_count += len(batch_results)
                    except Exception as e2:
                        log_error(f"Sequential fallback also failed: {type(e2).__name__}: {e2}")
                continue

    log_info("")
    log_success(f"Completed! Total results: {len(all_results)}")

    # Generate report
    if all_results:
        df = pd.DataFrame(all_results)

        # Sort by confidence (descending)
        df = df.sort_values(by="confidence", ascending=False)

        # Add rank column
        df.insert(0, "rank", range(1, len(df) + 1))

        # Round numeric columns for better readability
        numeric_cols = ["current_price", "confidence"]
        df[numeric_cols] = df[numeric_cols].round(4)

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = args.output or f"random_forest_signals_report_{timestamp}.csv"

        # Save to CSV
        df.to_csv(output_file, index=False)
        log_success(f"Report saved to: {output_file}")

        # Print summary
        log_info("")
        log_info("=" * 60)
        log_info("SUMMARY")
        log_info("=" * 60)

        # Count signals by direction
        long_count = len(df[df["signal_direction"] == "LONG"])
        short_count = len(df[df["signal_direction"] == "SHORT"])
        neutral_count = len(df[df["signal_direction"] == "NEUTRAL"])

        log_info(f"Total signals: {len(df)}")
        log_info(f"  LONG: {long_count}")
        log_info(f"  SHORT: {short_count}")
        log_info(f"  NEUTRAL: {neutral_count}")

        # Top 10 signals by confidence
        log_info("")
        log_info("TOP 10 SIGNALS BY CONFIDENCE:")
        print(
            df[["rank", "symbol", "timeframe", "signal_direction", "confidence", "current_price"]]
            .head(10)
            .to_string(index=False)
        )

        # Top 5 per timeframe
        log_info("")
        for tf in timeframes:
            tf_data = df[df["timeframe"] == tf].head(5)
            if not tf_data.empty:
                log_info(f"TOP 5 - {tf}:")
                print(tf_data[["rank", "symbol", "signal_direction", "confidence"]].to_string(index=False))

    else:
        log_warn("No results generated")

    log_info("")
    log_success("Done!")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log_warn("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
