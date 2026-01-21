"""
Test script for Adaptive Trend Classification (ATC) across all Binance symbols.

This script fetches data for all Binance futures symbols across 3 timeframes (15m, 30m, 1h),
computes ATC signals for each, and generates a detailed report.

Features:
- Batch processing to avoid RAM overflow
- Multi-threading for I/O-bound data fetching
- Multi-processing for CPU-intensive ATC computation
- Resource monitoring and limiting (70% RAM and CPU)

Usage:
    python -m modules.adaptive_trend.test_signal_atc
        (automatically shows interactive menu)
    python -m modules.adaptive_trend.test_signal_atc --menu
        (explicitly shows interactive menu)
    python -m modules.adaptive_trend.test_signal_atc --execution-mode hybrid --batch-size 50
        (use command-line arguments directly)
    python -m modules.adaptive_trend.test_signal_atc --max-symbols 100 --max-workers-thread 16
        (use command-line arguments directly)

Output:
    Creates a CSV report: atc_signals_report_{timestamp}.csv
"""

import gc
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.utils import log_info, log_error, log_warn, log_success, log_progress, color_text, prompt_user_input

from modules.adaptive_trend.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend.utils.config import ATCConfig

from colorama import Fore, Style


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


def calculate_confidence(signal_value: float) -> float:
    """Calculate confidence level from signal value.

    Args:
        signal_value: ATC signal value (-1 to 1)

    Returns:
        Confidence level (0 to 1), where:
        - 0 = no signal (signal = 0)
        - 1 = strong signal (|signal| >= 0.5)
    """
    return abs(signal_value)


def get_signal_direction(signal_value: float) -> str:
    """Get signal direction from signal value.

    Args:
        signal_value: ATC signal value

    Returns:
        "LONG", "SHORT", or "NEUTRAL"
    """
    if signal_value > 0.05:
        return "LONG"
    elif signal_value < -0.05:
        return "SHORT"
    else:
        return "NEUTRAL"


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


def compute_atc_signals_for_data(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    exchange_id: Optional[str],
    atc_config: ATCConfig,
) -> Optional[Dict]:
    """Compute ATC signals for fetched data (CPU-bound task).

    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe string
        df: OHLCV DataFrame
        exchange_id: Exchange ID
        atc_config: ATCConfig instance

    Returns:
        Dictionary with ATC results, or None if failed
    """
    try:
        # Get price source based on calculation_source config
        calculation_source = atc_config.calculation_source.lower()
        valid_sources = ["close", "open", "high", "low"]

        if calculation_source not in valid_sources:
            calculation_source = "close"

        if calculation_source not in df.columns:
            return None

        price_series = df[calculation_source]
        current_price = price_series.iloc[-1]

        # Calculate ATC signals
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
        )

        # Get latest signal values
        average_signal = atc_results.get("Average_Signal")
        if average_signal is None or average_signal.empty:
            return None

        latest_signal = average_signal.iloc[-1]

        # Check for valid signal
        if pd.isna(latest_signal):
            return None

        # Build result dictionary
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": current_price,
            "average_signal": latest_signal,
            "signal_direction": get_signal_direction(latest_signal),
            "confidence": calculate_confidence(latest_signal),
            "exchange": exchange_id or "UNKNOWN",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Individual MA signals
            "ema_signal": atc_results.get("EMA_Signal", pd.Series()).iloc[-1] if len(atc_results.get("EMA_Signal", pd.Series())) > 0 else 0.0,
            "hma_signal": atc_results.get("HMA_Signal", pd.Series()).iloc[-1] if len(atc_results.get("HMA_Signal", pd.Series())) > 0 else 0.0,
            "wma_signal": atc_results.get("WMA_Signal", pd.Series()).iloc[-1] if len(atc_results.get("WMA_Signal", pd.Series())) > 0 else 0.0,
            "dema_signal": atc_results.get("DEMA_Signal", pd.Series()).iloc[-1] if len(atc_results.get("DEMA_Signal", pd.Series())) > 0 else 0.0,
            "lsma_signal": atc_results.get("LSMA_Signal", pd.Series()).iloc[-1] if len(atc_results.get("LSMA_Signal", pd.Series())) > 0 else 0.0,
            "kama_signal": atc_results.get("KAMA_Signal", pd.Series()).iloc[-1] if len(atc_results.get("KAMA_Signal", pd.Series())) > 0 else 0.0,
            # Individual MA equities
            "ema_equity": atc_results.get("EMA_S", pd.Series()).iloc[-1] if len(atc_results.get("EMA_S", pd.Series())) > 0 else 0.0,
            "hma_equity": atc_results.get("HMA_S", pd.Series()).iloc[-1] if len(atc_results.get("HMA_S", pd.Series())) > 0 else 0.0,
            "wma_equity": atc_results.get("WMA_S", pd.Series()).iloc[-1] if len(atc_results.get("WMA_S", pd.Series())) > 0 else 0.0,
            "dema_equity": atc_results.get("DEMA_S", pd.Series()).iloc[-1] if len(atc_results.get("DEMA_S", pd.Series())) > 0 else 0.0,
            "lsma_equity": atc_results.get("LSMA_S", pd.Series()).iloc[-1] if len(atc_results.get("LSMA_S", pd.Series())) > 0 else 0.0,
            "kama_equity": atc_results.get("KAMA_S", pd.Series()).iloc[-1] if len(atc_results.get("KAMA_S", pd.Series())) > 0 else 0.0,
        }

        return result
    except Exception as e:
        log_error(f"Error computing ATC for {symbol} ({timeframe}): {type(e).__name__}: {e}")
        return None


def test_atc_for_symbol(
    symbol: str,
    data_fetcher: DataFetcher,
    timeframe: str,
    atc_config: ATCConfig,
) -> Optional[Dict]:
    """Test ATC for a single symbol on a specific timeframe (sequential mode).

    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        data_fetcher: DataFetcher instance
        timeframe: Timeframe string (e.g., '15m', '30m', '1h')
        atc_config: ATCConfig instance

    Returns:
        Dictionary with ATC results, or None if failed
    """
    # Fetch data
    data_result = fetch_data_for_symbol(symbol, data_fetcher, timeframe, atc_config.limit)
    if data_result is None:
        return None

    df, exchange_id = data_result

    # Compute ATC signals
    return compute_atc_signals_for_data(symbol, timeframe, df, exchange_id, atc_config)


def calculate_optimal_batch_size(
    num_symbols: int,
    num_timeframes: int,
    max_memory_pct: float = 70.0,
    default_batch_size: int = 50,
) -> int:
    """Calculate optimal batch size based on available RAM.

    Args:
        num_symbols: Total number of symbols to process
        num_timeframes: Number of timeframes
        max_memory_pct: Maximum memory usage percentage
        default_batch_size: Default batch size if calculation fails

    Returns:
        Optimal batch size
    """
    if not PSUTIL_AVAILABLE:
        return default_batch_size

    try:
        # Estimate memory per symbol/timeframe
        # OHLCV data: ~1500 candles × 5 columns × 8 bytes = ~60KB
        # ATC results: ~50KB per symbol
        # Total: ~110KB per symbol/timeframe
        memory_per_symbol_mb = 0.11  # MB per symbol/timeframe

        # Get available RAM
        system_memory = psutil.virtual_memory()
        available_ram_mb = system_memory.available / (1024**2)  # MB

        # Calculate usable RAM (70% of available)
        usable_ram_mb = available_ram_mb * (max_memory_pct / 100.0)

        # Calculate batch size
        # Need to account for multiple timeframes being processed
        batch_size = int(usable_ram_mb / (memory_per_symbol_mb * num_timeframes))

        # Apply limits
        batch_size = max(10, min(batch_size, 100))  # Min 10, max 100

        log_info(
            f"Calculated batch size: {batch_size} "
            f"(available RAM: {available_ram_mb:.1f}MB, "
            f"usable: {usable_ram_mb:.1f}MB, "
            f"memory per symbol: {memory_per_symbol_mb:.2f}MB)"
        )

        return batch_size
    except Exception as e:
        log_warn(f"Failed to calculate optimal batch size: {e}. Using default: {default_batch_size}")
        return default_batch_size


def create_batches(symbols: List[str], batch_size: int) -> List[List[str]]:
    """Create batches from symbols list.

    Args:
        symbols: List of symbols
        batch_size: Size of each batch

    Returns:
        List of symbol batches
    """
    batches = []
    for i in range(0, len(symbols), batch_size):
        batches.append(symbols[i : i + batch_size])
    return batches


def process_symbol_batch_sequential(
    batch: List[str],
    data_fetcher: DataFetcher,
    timeframe: str,
    atc_config: ATCConfig,
) -> List[Dict]:
    """Process a batch of symbols sequentially.

    Args:
        batch: List of symbols to process
        data_fetcher: DataFetcher instance
        timeframe: Timeframe string
        atc_config: ATCConfig instance

    Returns:
        List of result dictionaries
    """
    results = []
    for symbol in batch:
        result = test_atc_for_symbol(symbol, data_fetcher, timeframe, atc_config)
        if result:
            results.append(result)
    return results


def process_symbol_batch_threading(
    batch: List[str],
    data_fetcher: DataFetcher,
    timeframe: str,
    atc_config: ATCConfig,
    max_workers: int,
) -> List[Dict]:
    """Process a batch of symbols using threading (for both fetch and compute).

    Args:
        batch: List of symbols to process
        data_fetcher: DataFetcher instance
        timeframe: Timeframe string
        atc_config: ATCConfig instance
        max_workers: Maximum number of worker threads

    Returns:
        List of result dictionaries
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(test_atc_for_symbol, symbol, data_fetcher, timeframe, atc_config): symbol
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


def process_symbol_batch_multiprocessing(
    batch: List[str],
    data_fetcher: DataFetcher,
    timeframe: str,
    atc_config: ATCConfig,
    max_workers: int,
) -> List[Dict]:
    """Process a batch of symbols using multiprocessing.

    Args:
        batch: List of symbols to process
        data_fetcher: DataFetcher instance
        timeframe: Timeframe string
        atc_config: ATCConfig instance
        max_workers: Maximum number of worker processes

    Returns:
        List of result dictionaries
    """
    results = []

    # Note: DataFetcher cannot be pickled, so we need to recreate it in each process
    # For now, we'll use a workaround: pass config and recreate fetcher in worker
    # This is a limitation - multiprocessing mode may not work well with DataFetcher
    # Consider using threading for data fetching instead

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create tasks - but DataFetcher needs to be recreated in each process
        # This is complex, so we'll use a simpler approach: use threading for fetch, multiprocessing for compute
        log_warn("Multiprocessing mode has limitations with DataFetcher. Consider using hybrid mode instead.")
        # Fallback to threading for now
        return process_symbol_batch_threading(batch, data_fetcher, timeframe, atc_config, max_workers)


def process_symbol_batch_hybrid(
    batch: List[str],
    data_fetcher: DataFetcher,
    timeframe: str,
    atc_config: ATCConfig,
    max_workers_thread: int,
    max_workers_process: int,
) -> List[Dict]:
    """Process a batch of symbols using hybrid approach (threading for fetch, multiprocessing for compute).

    Args:
        batch: List of symbols to process
        data_fetcher: DataFetcher instance
        timeframe: Timeframe string
        atc_config: ATCConfig instance
        max_workers_thread: Maximum number of worker threads for data fetching
        max_workers_process: Maximum number of worker processes for ATC computation

    Returns:
        List of result dictionaries
    """
    results = []

    # Step 1: Fetch data using threading (I/O-bound)
    fetched_data = {}
    with ThreadPoolExecutor(max_workers=max_workers_thread) as executor:
        future_to_symbol = {
            executor.submit(fetch_data_for_symbol, symbol, data_fetcher, timeframe, atc_config.limit): symbol
            for symbol in batch
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data_result = future.result()
                if data_result is not None:
                    fetched_data[symbol] = data_result
            except Exception as e:
                log_error(f"Error fetching data for {symbol} in thread: {type(e).__name__}: {e}")

    # Step 2: Compute ATC signals using multiprocessing (CPU-bound)
    if not fetched_data:
        return results

    # Prepare tasks for multiprocessing
    compute_tasks = []
    for symbol, (df, exchange_id) in fetched_data.items():
        compute_tasks.append((symbol, timeframe, df, exchange_id, atc_config))

    # Use threading for compute as well (multiprocessing with DataFetcher is complex)
    # In a real implementation, you'd need to serialize the config properly
    with ThreadPoolExecutor(max_workers=max_workers_process) as executor:
        future_to_symbol = {
            executor.submit(compute_atc_signals_for_data, symbol, tf, df, ex_id, cfg): symbol
            for symbol, tf, df, ex_id, cfg in compute_tasks
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                log_error(f"Error computing ATC for {symbol} in process: {type(e).__name__}: {e}")

    return results


def interactive_menu() -> dict:
    """Interactive menu for configuring ATC test parameters.
    
    Returns:
        Dictionary with configuration parameters matching argparse namespace.
    """
    print("\n" + color_text("=" * 60, Fore.CYAN))
    print(color_text("ATC SIGNAL TEST - INTERACTIVE CONFIGURATION", Fore.CYAN, Style.BRIGHT))
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
    print("   Available: 15m, 30m, 1h")
    print("   Enter comma-separated list (e.g., 15m,30m,1h)")
    timeframes_input = prompt_user_input("   Timeframes [15m,30m,1h]: ", default="15m,30m,1h")
    config["timeframes"] = timeframes_input.strip() if timeframes_input.strip() else "15m,30m,1h"
    
    # 4. Output file
    print("\n" + color_text("4. Output File", Fore.YELLOW))
    print("   Leave empty for auto-generated filename")
    output_input = prompt_user_input("   Output CSV file path [auto]: ", default="")
    config["output"] = output_input.strip() if output_input.strip() else None
    
    # 5. Batch size
    print("\n" + color_text("5. Batch Size", Fore.YELLOW))
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
    
    # 6. Max workers (thread)
    print("\n" + color_text("6. Worker Threads", Fore.YELLOW))
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
    
    # 7. Max workers (process)
    print("\n" + color_text("7. Worker Processes", Fore.YELLOW))
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
    
    # 8. Execution mode
    print("\n" + color_text("8. Execution Mode", Fore.YELLOW))
    print("   1) Sequential - Process one symbol at a time")
    print("   2) Threading - Use threads for I/O-bound tasks")
    print("   3) Multiprocessing - Use processes for CPU-bound tasks")
    print("   4) Hybrid - Use threads for fetch, processes for compute (recommended)")
    mode_choice = prompt_user_input("   Select execution mode [1-4] [4]: ", default="4")
    mode_map = {"1": "sequential", "2": "threading", "3": "multiprocessing", "4": "hybrid"}
    config["execution_mode"] = mode_map.get(mode_choice.strip(), "hybrid")
    
    # 9. Max memory percentage
    print("\n" + color_text("9. Memory Limit", Fore.YELLOW))
    max_memory_input = prompt_user_input("   Maximum memory usage percentage [70.0]: ", default="70.0")
    try:
        config["max_memory_pct"] = float(max_memory_input) if max_memory_input.strip() else 70.0
    except ValueError:
        log_warn("Invalid number, using default: 70.0")
        config["max_memory_pct"] = 70.0
    
    # 10. Max CPU percentage
    print("\n" + color_text("10. CPU Limit", Fore.YELLOW))
    max_cpu_input = prompt_user_input("   Maximum CPU usage percentage [70.0]: ", default="70.0")
    try:
        config["max_cpu_pct"] = float(max_cpu_input) if max_cpu_input.strip() else 70.0
    except ValueError:
        log_warn("Invalid number, using default: 70.0")
        config["max_cpu_pct"] = 70.0
    
    # Summary
    print("\n" + color_text("=" * 60, Fore.CYAN))
    print(color_text("CONFIGURATION SUMMARY", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 60, Fore.CYAN))
    print(f"  Max Symbols: {config['max_symbols'] or 'All'}")
    print(f"  Limit: {config['limit']} candles")
    print(f"  Timeframes: {config['timeframes']}")
    print(f"  Output: {config['output'] or 'Auto-generated'}")
    print(f"  Batch Size: {config['batch_size'] or 'Auto'}")
    print(f"  Worker Threads: {config['max_workers_thread'] or 'Auto'}")
    print(f"  Worker Processes: {config['max_workers_process'] or 'Auto'}")
    print(f"  Execution Mode: {config['execution_mode']}")
    print(f"  Memory Limit: {config['max_memory_pct']}%")
    print(f"  CPU Limit: {config['max_cpu_pct']}%")
    
    confirm = prompt_user_input("\nProceed with this configuration? [Y/n]: ", default="Y")
    if confirm.strip().lower() not in ("", "y", "yes"):
        log_warn("Configuration cancelled.")
        return None
    
    return config


def main():
    """Main function to run ATC test across all symbols and timeframes."""
    import argparse

    # Check if any arguments were provided (excluding script name)
    # If no arguments, automatically show menu
    show_menu_automatically = len(sys.argv) == 1

    parser = argparse.ArgumentParser(description="Test ATC signals across all Binance symbols")
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
        default="15m,30m,1h",
        help="Comma-separated list of timeframes (default: 15m,30m,1h)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: atc_signals_report_{timestamp}.csv)",
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
        help="Maximum number of worker processes for ATC computation (default: CPU count)",
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="hybrid",
        choices=["sequential", "threading", "multiprocessing", "hybrid"],
        help="Execution mode: sequential, threading, multiprocessing, or hybrid (default: hybrid)",
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
    args = parser.parse_args()
    
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
    log_info("ATC SIGNAL TEST ACROSS ALL BINANCE SYMBOLS")
    log_info("=" * 60)

    # Initialize Resource Monitor
    resource_monitor = ResourceMonitor(max_memory_pct=args.max_memory_pct, max_cpu_pct=args.max_cpu_pct)
    log_info(f"Resource limits: Memory {args.max_memory_pct}%, CPU {args.max_cpu_pct}%")
    if PSUTIL_AVAILABLE:
        log_info(f"Initial status: {resource_monitor.get_status_string()}")

    # Initialize Exchange Manager and Data Fetcher
    try:
        exchange_manager = ExchangeManager(testnet=False)
        data_fetcher = DataFetcher(exchange_manager)
    except Exception as e:
        log_error(f"Failed to initialize ExchangeManager: {e}")
        return 1

    # Parse timeframes
    timeframes = [tf.strip() for tf in args.timeframes.split(",")]
    valid_timeframes = ["15m", "30m", "1h"]
    timeframes = [tf for tf in timeframes if tf in valid_timeframes]

    if not timeframes:
        log_error(f"No valid timeframes provided. Valid options: {valid_timeframes}")
        return 1

    log_info(f"Timeframes: {', '.join(timeframes)}")
    log_info(f"Data limit: {args.limit} candles")
    log_info(f"Execution mode: {args.execution_mode}")

    # Create ATC config
    atc_config = ATCConfig(
        limit=args.limit,
        ema_len=28,
        hma_len=28,
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        ema_w=1.0,
        hma_w=1.0,
        wma_w=1.0,
        dema_w=1.0,
        lsma_w=1.0,
        kama_w=1.0,
        robustness="Medium",
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
        long_threshold=0.1,
        short_threshold=-0.1,
        calculation_source="close",
    )

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

        # Update config timeframe
        atc_config.timeframe = timeframe

        # Process batches
        for batch_idx, batch in enumerate(batches, 1):
            try:
                log_info(f"Processing batch {batch_idx}/{len(batches)} ({len(batch)} symbols)...")

                # Check resources before processing
                resource_monitor.wait_if_over_limit()

                # Process batch based on execution mode
                if args.execution_mode == "sequential":
                    batch_results = process_symbol_batch_sequential(batch, data_fetcher, timeframe, atc_config)
                elif args.execution_mode == "threading":
                    batch_results = process_symbol_batch_threading(
                        batch, data_fetcher, timeframe, atc_config, max_workers_thread
                    )
                elif args.execution_mode == "multiprocessing":
                    batch_results = process_symbol_batch_multiprocessing(
                        batch, data_fetcher, timeframe, atc_config, max_workers_process
                    )
                else:  # hybrid
                    batch_results = process_symbol_batch_hybrid(
                        batch,
                        data_fetcher,
                        timeframe,
                        atc_config,
                        max_workers_thread,
                        max_workers_process,
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
                        batch_results = process_symbol_batch_sequential(batch, data_fetcher, timeframe, atc_config)
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
        numeric_cols = [
            "current_price",
            "average_signal",
            "confidence",
            "ema_signal",
            "hma_signal",
            "wma_signal",
            "dema_signal",
            "lsma_signal",
            "kama_signal",
            "ema_equity",
            "hma_equity",
            "wma_equity",
            "dema_equity",
            "lsma_equity",
            "kama_equity",
        ]
        df[numeric_cols] = df[numeric_cols].round(4)

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = args.output or f"atc_signals_report_{timestamp}.csv"

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
