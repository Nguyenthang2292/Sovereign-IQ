"""
Position Sizing Calculator - Entry Point.

Calculate optimal position sizes using Bayesian Kelly Criterion and Regime Switching
based on results from main_hybrid.py or main_voting.py.

Example:
    Run from command line:
        $ python main_position_sizing.py --source hybrid --account-balance 10000
"""

import warnings
import sys
import os
from pathlib import Path
import pandas as pd
from typing import Optional, List, Dict, Tuple
import logging
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.common.utils import configure_windows_stdio

# Fix encoding issues on Windows for interactive CLI runs only
configure_windows_stdio()

from colorama import Fore, Style, init as colorama_init

from modules.common.utils import (
    color_text,
    log_error,
    log_progress,
    log_success,
    log_warn,
)
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.core.data_fetcher import DataFetcher
from modules.common.utils import initialize_components
from modules.position_sizing.core.position_sizer import PositionSizer
from modules.position_sizing.utils.data_loader import (
    load_symbols_from_results,
    load_symbols_from_file,
    validate_symbols,
)
from modules.position_sizing.cli.argument_parser import (
    parse_args,
    interactive_config_menu,
    parse_symbols_string,
)
from modules.position_sizing.cli.display import (
    display_position_sizing_results,
    display_configuration,
)
from config.position_sizing import (
    ENABLE_PARALLEL_PROCESSING,
    USE_GPU,
    ENABLE_MULTITHREADING,
)
from modules.common.utils import days_to_candles

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)

# Constants
PARALLEL_PROCESSING_THRESHOLD = 100  # Threshold for deciding between parallel and sequential processing


# Custom exception for data-related errors
class InsufficientDataError(Exception):
    """Raised when there is insufficient data to perform calculations."""
    pass


class DataError(Exception):
    """Raised when there is an error with data processing or validation."""
    pass


def _try_timeframes_auto(
    symbols: List[Dict],
    account_balance: float,
    lookback_days: int,
    data_fetcher: DataFetcher,
    max_position_size: float,
    signal_mode: str,
    signal_calculation_mode: str,
) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Tự động thử các timeframe song song để tìm timeframe có kết quả position sizing.
    
    Chạy song song các timeframe: 15m, 30m, 1h
    Dừng ngay khi tìm thấy timeframe có kết quả hợp lệ đầu tiên (early stopping).
    Tăng tốc độ đáng kể so với chạy tuần tự.
    
    Args:
        symbols: List of symbol dictionaries
        account_balance: Account balance in USDT
        lookback_days: Number of days to look back
        data_fetcher: DataFetcher instance (assumed thread-safe or each thread uses separate calls)
        max_position_size: Maximum position size as fraction
        signal_mode: Signal calculation mode ('majority_vote' or 'single_signal')
        signal_calculation_mode: Signal calculation approach ('precomputed' or 'incremental')
        
    Returns:
        (timeframe, results_df) nếu tìm thấy, (None, None) nếu không tìm thấy
    """
    timeframes = ['15m', '30m', '1h']
    
    # Tạo cancellation event để các tasks có thể kiểm tra và dừng sớm
    cancel_event = threading.Event()
    
    def try_timeframe(timeframe: str) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
        """Try a single timeframe and return result."""
        try:
            # Kiểm tra cancellation trước khi bắt đầu
            if cancel_event.is_set():
                log_progress(f"Skipping timeframe {timeframe} (cancelled)")
                return (None, None)
            
            log_progress(f"Trying timeframe: {timeframe}...")
            
            # Kiểm tra cancellation trước khi tạo PositionSizer
            if cancel_event.is_set():
                log_progress(f"Cancelling timeframe {timeframe} before PositionSizer creation")
                return (None, None)
            
            # Tạo PositionSizer với timeframe này
            position_sizer = PositionSizer(
                data_fetcher=data_fetcher,
                timeframe=timeframe,
                lookback_days=lookback_days,
                max_position_size=max_position_size,
                signal_mode=signal_mode,
                signal_calculation_mode=signal_calculation_mode,
            )
            
            # Kiểm tra cancellation trước khi tính toán
            if cancel_event.is_set():
                log_progress(f"Cancelling timeframe {timeframe} before calculation")
                return (None, None)
            
            # Tính position sizing
            results_df = position_sizer.calculate_portfolio_allocation(
                symbols=symbols,
                account_balance=account_balance,
                timeframe=timeframe,
                lookback=lookback_days,
            )
            
            # Kiểm tra cancellation sau khi tính toán
            if cancel_event.is_set():
                log_progress(f"Cancelling timeframe {timeframe} after calculation")
                return (None, None)
            
            # Kiểm tra kết quả
            if not results_df.empty and 'position_size_usdt' in results_df.columns:
                total_position_size = results_df['position_size_usdt'].sum()
                total_exposure_pct = results_df['position_size_pct'].sum() if 'position_size_pct' in results_df.columns else 0.0
                
                if total_position_size > 0 or total_exposure_pct > 0:
                    log_success(f"✓ Found valid position sizing at timeframe: {timeframe}")
                    log_progress(f"  Total Position Size: {total_position_size:.2f} USDT")
                    log_progress(f"  Total Exposure: {total_exposure_pct:.2f}%")
                    return (timeframe, results_df)
            
            log_warn(f"No position sizing result at {timeframe}")
            return (None, None)
            
        except (ValueError, KeyError, IndexError, pd.errors.EmptyDataError, InsufficientDataError, DataError) as e:
            # Recoverable data errors - log with context
            log_warn(f"Data error trying timeframe {timeframe}: {type(e).__name__}: {e}")
            logging.exception(f"Full traceback for data error at timeframe {timeframe}")
            return (None, None)
        except (ConnectionError, TimeoutError, OSError) as e:
            # Unrecoverable network/system errors - log and return None (let other threads continue)
            log_error(f"Network/system error trying timeframe {timeframe}: {type(e).__name__}: {e}")
            logging.exception(f"Full traceback for network/system error at timeframe {timeframe}")
            # Don't re-raise in parallel execution - let other threads continue
            return (None, None)
        except Exception as e:
            # Other unexpected errors - log with full traceback
            log_error(f"Unexpected error trying timeframe {timeframe}: {type(e).__name__}: {e}")
            logging.exception(f"Full traceback for unexpected error at timeframe {timeframe}")
            return (None, None)
    
    # Chạy song song với ThreadPoolExecutor
    log_progress("\nRunning timeframes in parallel for faster execution...")
    executor = ThreadPoolExecutor(max_workers=len(timeframes))
    early_exit = False  # Flag để đánh dấu early stopping
    try:
        # Submit tất cả tasks
        future_to_timeframe = {
            executor.submit(try_timeframe, tf): tf 
            for tf in timeframes
        }
        
        # Chờ kết quả, return ngay khi tìm thấy kết quả hợp lệ đầu tiên
        for future in as_completed(future_to_timeframe):
            timeframe = future_to_timeframe[future]
            try:
                result_timeframe, result_df = future.result()
                
                # Nếu tìm thấy kết quả hợp lệ, cancel các futures còn lại và return
                if result_timeframe is not None and result_df is not None:
                    log_progress(f"Early stopping: Found result at {result_timeframe}, cancelling other timeframes...")
                    
                    # Set cancellation event để các tasks đang chạy có thể dừng sớm
                    cancel_event.set()
                    
                    # Cancel các futures còn lại
                    remaining_futures = [f for f in future_to_timeframe.keys() if not f.done()]
                    cancelled_count = 0
                    for remaining_future in remaining_futures:
                        if remaining_future.cancel():
                            cancelled_count += 1
                    
                    if cancelled_count > 0:
                        log_progress(f"Cancelled {cancelled_count} pending timeframe(s)")
                    
                    # Đánh dấu early exit để finally block biết không cần chờ
                    early_exit = True
                    
                    return (result_timeframe, result_df)
            except Exception as e:
                log_warn(f"Exception getting result from timeframe {timeframe}: {e}")
                continue
        
        log_warn("✗ No timeframe found with valid position sizing results")
        return (None, None)
    finally:
        # Đảm bảo executor được shutdown ngay cả khi có exception để cleanup executor
        # Sử dụng early_exit flag để quyết định có chờ các tasks hoàn thành hay không
        # Nếu exception xảy ra trước khi tìm thấy kết quả, early_exit vẫn là False nên
        # shutdown(wait=True) sẽ block cho đến khi các tasks hoàn thành (làm chậm việc
        # propagate exception) nhưng đảm bảo shutdown sạch sẽ
        try:
            if sys.version_info >= (3, 9):
                executor.shutdown(wait=not early_exit, cancel_futures=True)
            else:
                # Với Python < 3.9, cancel các futures còn lại thủ công
                # future_to_timeframe có thể chưa được định nghĩa nếu exception xảy ra sớm
                if 'future_to_timeframe' in locals():
                    for future in future_to_timeframe.keys():
                        if not future.done():
                            future.cancel()
                executor.shutdown(wait=not early_exit)
        except Exception:
            # Ignore exceptions trong finally block để không che giấu exception gốc
            pass


def _prompt_for_balance() -> float:
    """
    Prompt user for account balance input and validate it.
    
    Returns:
        float: The validated account balance in USDT
        
    Exits:
        sys.exit(1) if input cannot be converted to float
    """
    account_balance_str = input("\nEnter account balance (USDT): ").strip()
    try:
        return float(account_balance_str)
    except ValueError:
        log_error("Invalid account balance. Exiting.")
        sys.exit(1)


def main() -> None:
    """
    Main entry point for position sizing CLI.
    
    Workflow:
    1. Parse arguments or show interactive menu
    2. Load symbols from source
    3. Initialize components (ExchangeManager, DataFetcher, PositionSizer)
    4. Calculate position sizes for all symbols
    5. Display results
    """
    try:
        # Parse arguments
        args = parse_args()
        
        # Interactive menu if not skipped
        config = {}
        if not args.no_menu:
            config = interactive_config_menu()
            # Merge config with args - always override with menu values if menu was shown
            for key, value in config.items():
                # Always override with menu values when menu is shown (user explicitly chose these values)
                setattr(args, key, value)
        
        # Initialize components first (needed for fetching balance)
        log_progress("\nInitializing components...")
        exchange_manager, data_fetcher = initialize_components()
        
        # Get account balance
        account_balance = args.account_balance
        
        # Try to fetch from Binance if requested
        if getattr(args, 'fetch_balance', False):
            log_progress("Fetching account balance from Binance...")
            try:
                fetched_balance = data_fetcher.fetch_binance_account_balance()
                if fetched_balance is not None and fetched_balance > 0:
                    account_balance = fetched_balance
                    log_success(f"Fetched account balance from Binance: {account_balance:.2f} USDT")
                else:
                    log_warn("Could not fetch balance from Binance. Please enter manually.")
            except Exception as e:
                log_warn(f"Error fetching balance from Binance: {e}. Please enter manually.")
        
        # Prompt for balance if still not set
        if account_balance is None:
            log_progress("\nAccount balance not provided. Options:")
            print("  1. Enter balance manually")
            print(color_text("  2. Fetch from Binance (requires API credentials) [default]", Fore.MAGENTA, Style.BRIGHT))
            default_choice_text = color_text("default=2", Fore.MAGENTA)
            choice = input(f"\nSelect option (1/2, {default_choice_text}): ").strip()
            
            # Default to option 2 if empty input
            if not choice:
                choice = "2"
            
            if choice == "2":
                try:
                    fetched_balance = data_fetcher.fetch_binance_account_balance()
                    if fetched_balance is not None and fetched_balance > 0:
                        account_balance = fetched_balance
                        log_success(f"Fetched account balance from Binance: {account_balance:.2f} USDT")
                    else:
                        log_warn("Could not fetch balance from Binance. Please enter manually.")
                        account_balance = _prompt_for_balance()
                except Exception as e:
                    log_warn(f"Error fetching balance from Binance: {e}. Please enter manually.")
                    account_balance = _prompt_for_balance()
            else:
                account_balance = _prompt_for_balance()
        
        if account_balance <= 0:
            log_error("Account balance must be positive. Exiting.")
            sys.exit(1)
        
        # Load symbols
        log_progress("\nLoading symbols...")
        symbols = []
        
        if args.symbols_file:
            symbols = load_symbols_from_file(args.symbols_file)
        elif args.symbols:
            # Parse comma-separated symbols and normalize (auto-add /USDT if needed)
            symbol_list = parse_symbols_string(args.symbols)
            symbols = [{'symbol': s, 'signal': 1} for s in symbol_list]
        elif args.source:
            symbols = load_symbols_from_results(args.source)
        else:
            log_error("No symbol source specified. Please provide --symbols-file, --symbols, or --source")
            sys.exit(1)
        
        # Validate symbols
        symbols = validate_symbols(symbols)
        
        if not symbols:
            log_error("No valid symbols found. Exiting.")
            sys.exit(1)
        
        log_success(f"Loaded {len(symbols)} symbols")
        
        # Get signal mode and calculation mode from args
        signal_mode = getattr(args, 'signal_mode', 'single_signal')
        signal_calculation_mode = getattr(args, 'signal_calculation_mode', 'precomputed')
        
        # Check auto timeframe mode (from config menu or command line)
        auto_timeframe = getattr(args, 'auto_timeframe', False)
        
        # Initialize Position Sizer (only if not using auto timeframe)
        if not auto_timeframe:
            position_sizer = PositionSizer(
                data_fetcher=data_fetcher,
                timeframe=args.timeframe,
                lookback_days=args.lookback_days,
                max_position_size=args.max_position_size,
                signal_mode=signal_mode,
                signal_calculation_mode=signal_calculation_mode,
            )
        
        # Display configuration
        print(color_text("\n" + "=" * 100, Fore.CYAN, Style.BRIGHT))
        print(color_text("POSITION SIZING CONFIGURATION", Fore.CYAN, Style.BRIGHT))
        print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))
        
        print(f"\n{color_text('Configuration:', Fore.WHITE)}")
        print(f"  Account Balance: {color_text(f'{account_balance:.2f} USDT', Fore.YELLOW)}")
        if auto_timeframe:
            print(f"  Timeframe: {color_text('Auto (15m -> 30m -> 1h)', Fore.YELLOW)}")
        else:
            print(f"  Timeframe: {color_text(args.timeframe, Fore.YELLOW)}")
        print(f"  Lookback: {color_text(f'{args.lookback_days} days', Fore.YELLOW)}")
        print(f"  Max Position Size: {color_text(f'{args.max_position_size*100:.1f}%', Fore.YELLOW)}")
        print(f"  Signal Mode: {color_text(signal_mode, Fore.YELLOW)}")
        if signal_mode == 'single_signal':
            print(f"    └─ Using single signal (highest confidence) approach - Lấy bất kỳ signal nào có confidence cao nhất")
        else:
            print(f"    └─ Using majority vote approach - Cần ít nhất {color_text('3 indicators', Fore.CYAN)} đồng ý")
        print(f"  Signal Calculation Mode: {color_text(signal_calculation_mode, Fore.YELLOW)}")
        if signal_calculation_mode == 'incremental':
            print(f"    └─ Incremental mode (Position-Aware Skipping): Pre-compute indicators once, extract signals per period")
            print(f"       • Pre-computes all indicators once for entire DataFrame (10-15x faster)")
            print(f"       • Skips signal calculation when position open (saves 30-50% time)")
            print(f"       • Combines signal calculation and trade simulation in single loop")
        else:
            print(f"    └─ Precomputed mode: Calculate all signals first, then simulate trades (default)")
            print(f"       • Better for analysis and debugging")
        
        if args.source:
            print(f"  Source: {color_text(args.source, Fore.YELLOW)}")
        elif args.symbols_file:
            print(f"  Symbols File: {color_text(args.symbols_file, Fore.YELLOW)}")
        elif args.symbols:
            print(f"  Symbols: {color_text(args.symbols, Fore.YELLOW)}")
        
        # Display optimization features status
        print(f"\n{color_text('Optimization Features (Architecture 5 - Hybrid Approach):', Fore.WHITE, Style.BRIGHT)}")
        print(color_text("-" * 100, Fore.CYAN))
        
        # 1. Vectorized Indicator Pre-computation (Always enabled in new implementation)
        vectorized_status = color_text("✅ ENABLED", Fore.GREEN)
        print(f"  1. {color_text('Vectorized Indicator Pre-computation:', Fore.WHITE)} {vectorized_status}")
        print(f"     └─ Pre-compute all indicators once for entire DataFrame (10-15x faster)")
        
        # 2. Incremental Signal Calculation (Always enabled in new implementation)
        incremental_status = color_text("✅ ENABLED", Fore.GREEN)
        print(f"  2. {color_text('Incremental Signal Calculation:', Fore.WHITE)} {incremental_status}")
        if signal_calculation_mode == 'incremental':
            print(f"     └─ Position-Aware Skipping: Skip signal calculation when position open")
            print(f"        • Pre-computes all indicators once, then extracts signals incrementally")
            print(f"        • Combines signal calculation and trade simulation in single loop")
            print(f"        • Saves 30-50% computation time for long-held positions")
            print(f"        • 10-15x faster when no position (uses precomputed indicators)")
        else:
            print(f"     └─ Calculate signals from pre-computed data using DataFrame views")
        
        # 3. Shared Memory for Parallel Processing
        try:
            from modules.backtester.core.shared_memory_utils import SHARED_MEMORY_AVAILABLE
            shared_memory_available = SHARED_MEMORY_AVAILABLE
        except ImportError:
            shared_memory_available = False
        
        # Calculate lookback_candles - may be approximate if auto_timeframe is enabled
        lookback_candles = days_to_candles(args.lookback_days, args.timeframe)
        if ENABLE_PARALLEL_PROCESSING:
            parallel_status = color_text("✅ ENABLED", Fore.GREEN)
            shared_mem_status = color_text("✅ AVAILABLE", Fore.GREEN) if shared_memory_available else color_text("❌ NOT AVAILABLE", Fore.YELLOW)
            print(f"  3. {color_text('Parallel Processing:', Fore.WHITE)} {parallel_status}")
            print(f"     └─ Shared Memory: {shared_mem_status}")
            if shared_memory_available:
                print(f"        • Using shared memory for efficient inter-process data sharing")
                print(f"        • Reduces memory overhead by 50-70% compared to pickle")
            else:
                print(f"        • Falling back to pickle serialization")
            if auto_timeframe:
                # When auto_timeframe is enabled, the timeframe is not yet determined
                # so we skip the lookback_candles display here (will be shown after timeframe is resolved)
                print(f"        • {color_text('Parallel processing decision', Fore.CYAN)} will be made after timeframe selection")
            else:
                if lookback_candles > PARALLEL_PROCESSING_THRESHOLD:
                    print(f"        • {color_text('Will use parallel processing', Fore.CYAN)} for {lookback_candles} periods")
                else:
                    print(f"        • {color_text('Using sequential processing', Fore.CYAN)} (dataset size <= {PARALLEL_PROCESSING_THRESHOLD})")
        else:
            parallel_status = color_text("❌ DISABLED", Fore.YELLOW)
            print(f"  3. {color_text('Parallel Processing:', Fore.WHITE)} {parallel_status}")
            print(f"     └─ Using optimized sequential vectorized processing")
        
        # 4. Multithreading
        multithreading_status = color_text("✅ ENABLED", Fore.GREEN) if ENABLE_MULTITHREADING else color_text("❌ DISABLED", Fore.YELLOW)
        print(f"  4. {color_text('Multithreading:', Fore.WHITE)} {multithreading_status}")
        if ENABLE_MULTITHREADING:
            print(f"     └─ Parallel indicator calculation using ThreadPoolExecutor")
        
        # 5. GPU Acceleration
        gpu_status = color_text("✅ ENABLED", Fore.GREEN) if USE_GPU else color_text("❌ DISABLED", Fore.YELLOW)
        print(f"  5. {color_text('GPU Acceleration:', Fore.WHITE)} {gpu_status}")
        if USE_GPU:
            print(f"     └─ GPU acceleration for ML models (XGBoost) if available")
        
        print(color_text("-" * 100, Fore.CYAN))
        print(f"  {color_text('Expected Performance:', Fore.CYAN)} 10-15x faster for large datasets (>1000 periods)")
        print(f"  {color_text('Memory Usage:', Fore.CYAN)} Reduced by 50-70% with shared memory")
        print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))
        
        # Calculate position sizes
        log_progress(f"\nCalculating position sizes for {len(symbols)} symbols...")
        log_progress("This may take a few minutes...\n")
        
        # Check if auto timeframe mode is enabled
        found_timeframe = None
        if auto_timeframe:
            log_progress("\nAuto timeframe testing enabled. Trying timeframes: 15m -> 30m -> 1h")
            log_progress("Will stop at first timeframe with valid position sizing results.\n")
            
            found_timeframe, results_df = _try_timeframes_auto(
                symbols=symbols,
                account_balance=account_balance,
                lookback_days=args.lookback_days,
                data_fetcher=data_fetcher,
                max_position_size=args.max_position_size,
                signal_mode=signal_mode,
                signal_calculation_mode=signal_calculation_mode,
            )
            
            if found_timeframe:
                log_success(f"\n✓ Found valid position sizing at timeframe: {found_timeframe}")
                # Cập nhật args.timeframe để hiển thị đúng
                args.timeframe = found_timeframe
            else:
                log_warn("\n✗ No timeframe found with valid position sizing results")
                log_warn("All timeframes (15m, 30m, 1h) were tried but none produced valid results")
                log_warn("This may indicate:")
                log_warn("  - Insufficient trades for Kelly calculation")
                log_warn("  - Win rate too low (< 40%)")
                log_warn("  - Invalid avg_win or avg_loss values")
                sys.exit(1)
        else:
            # Logic cũ: dùng timeframe từ args
            results_df = position_sizer.calculate_portfolio_allocation(
                symbols=symbols,
                account_balance=account_balance,
                timeframe=args.timeframe,
                lookback=args.lookback_days,  # This will be converted to candles internally
            )
        
        # Display final configuration summary (after finding timeframe if auto mode)
        if auto_timeframe and found_timeframe:
            # Calculate exact lookback_candles with the resolved timeframe
            final_lookback_candles = days_to_candles(args.lookback_days, found_timeframe)
            print(color_text("\n" + "=" * 100, Fore.CYAN, Style.BRIGHT))
            print(color_text("FINAL CONFIGURATION", Fore.CYAN, Style.BRIGHT))
            print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))
            print(f"  Selected Timeframe: {color_text(found_timeframe, Fore.GREEN, Style.BRIGHT)}")
            print(f"  Lookback Candles: {color_text(f'{final_lookback_candles}', Fore.GREEN, Style.BRIGHT)} periods")
            if ENABLE_PARALLEL_PROCESSING:
                if final_lookback_candles > PARALLEL_PROCESSING_THRESHOLD:
                    print(f"  Parallel Processing: {color_text('Was used', Fore.GREEN)} ({final_lookback_candles} periods > {PARALLEL_PROCESSING_THRESHOLD} — processed in parallel)")
                else:
                    print(f"  Parallel Processing: {color_text('Processed sequentially', Fore.YELLOW)} (dataset size <= {PARALLEL_PROCESSING_THRESHOLD} — processed sequentially)")
            print(color_text("=" * 100, Fore.CYAN, Style.BRIGHT))
        
        # Check if results_df is empty before displaying or saving
        if results_df.empty:
            log_warn("\nNo position sizing results found")
            log_warn("The calculation completed but produced no results.")
            log_warn("This may indicate:")
            log_warn("  - No valid signals were generated for the symbols")
            log_warn("  - All position sizes were filtered out")
            log_warn("  - Insufficient data for calculations")
            return
        
        # Display results
        display_position_sizing_results(results_df)
        
        # Save results if output file specified
        if args.output:
            results_df.to_csv(args.output, index=False)
            log_success(f"\nResults saved to {args.output}")
        
        log_success("\nPosition sizing calculation completed!")
        
    except KeyboardInterrupt:
        print(color_text("\n\nExiting by user request.", Fore.YELLOW))
        sys.exit(0)
    except Exception as e:
        log_error(f"Error: {type(e).__name__}: {e}")
        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
