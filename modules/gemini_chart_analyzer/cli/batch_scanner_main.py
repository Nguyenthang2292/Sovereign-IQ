"""
CLI Main Program for Market Batch Scanner.

Interactive menu for batch scanning entire market with Gemini.
"""

import sys
from pathlib import Path
import traceback

# Add project root to sys.path
if '__file__' in globals():
    project_root = Path(__file__).parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

# Ensure stdin is available on Windows before configuring stdio
# This is critical when running the file directly (not via wrapper)
if sys.platform == 'win32':
    try:
        if sys.stdin is None or (hasattr(sys.stdin, 'closed') and sys.stdin.closed):
            sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
    except (OSError, IOError, AttributeError):
        pass  # Continue if we can't fix stdin

# Fix encoding issues on Windows
from modules.common.utils import configure_windows_stdio
configure_windows_stdio()

from colorama import Fore, init as colorama_init
from modules.common.utils import color_text, log_error, log_warn
from modules.common.utils import normalize_timeframe
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner
from modules.common.core.exchange_manager import PublicExchangeManager
from modules.common.core.data_fetcher import DataFetcher
from modules.common.ui.logging import log_info, log_success
from core.voting_analyzer import VotingAnalyzer
import argparse
import time
from typing import List, Optional
from config import (
    DECISION_MATRIX_VOTING_THRESHOLD,
    DECISION_MATRIX_MIN_VOTES,
    SPC_LOOKBACK,
    SPC_P_LOW,
    SPC_P_HIGH,
    SPC_STRATEGY_PARAMETERS,
    RANGE_OSCILLATOR_LENGTH,
    RANGE_OSCILLATOR_MULTIPLIER,
    HMM_WINDOW_SIZE_DEFAULT,
    HMM_WINDOW_KAMA_DEFAULT,
    HMM_FAST_KAMA_DEFAULT,
    HMM_SLOW_KAMA_DEFAULT,
    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
)

import warnings
# Suppress specific noisy warnings if needed
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PIL.Image")
colorama_init(autoreset=True)


def safe_input(prompt: str, default: str = '') -> str:
    """
    Safely read input from stdin, handling I/O errors gracefully.
    
    Args:
        prompt: Prompt string to display
        default: Default value to return if input fails
        
    Returns:
        User input string or default if input fails
    """
    try:
        # Check if stdin is available and not closed
        if sys.stdin is None:
            log_warn("stdin is None, using default")
            return default
        
        # Check if stdin is closed (Windows-specific check)
        if hasattr(sys.stdin, 'closed') and sys.stdin.closed:
            log_warn("stdin is closed, attempting to reopen...")
            if sys.platform == 'win32':
                try:
                    # Try to reopen stdin on Windows
                    sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
                except (OSError, IOError) as reopen_err:
                    log_warn(f"Could not reopen stdin: {reopen_err}, using default")
                    return default
            else:
                log_warn("stdin is closed, using default")
                return default
        
        # Check if stdin is a TTY
        if not sys.stdin.isatty():
            # If stdin is not a TTY, return default
            return default
        
        # Double-check stdin is still available right before calling input()
        # This is important on Windows where stdin can be closed during execution
        if sys.platform == 'win32':
            try:
                # Check stdin status
                stdin_ok = (
                    sys.stdin is not None and 
                    not (hasattr(sys.stdin, 'closed') and sys.stdin.closed) and
                    sys.stdin.isatty()
                )
                
                if not stdin_ok:
                    # Try to reopen stdin one more time right before use
                    try:
                        sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
                        # Verify it's now open
                        if hasattr(sys.stdin, 'closed') and sys.stdin.closed:
                            # Still closed, return default
                            log_warn("Could not reopen stdin, using default")
                            return default
                    except (OSError, IOError) as reopen_err:
                        log_warn(f"Could not reopen stdin: {reopen_err}, using default")
                        return default
            except (AttributeError, ValueError, OSError, IOError):
                pass  # Continue and let the exception handler deal with it
        
        # Try to read input
        try:
            result = input(prompt).strip()
            return result
        except (ValueError, OSError, IOError) as input_err:
            # If input() itself fails, try to reopen stdin and return default
            if sys.platform == 'win32' and "closed file" in str(input_err).lower():
                try:
                    sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
                except (OSError, IOError):
                    pass
            raise  # Re-raise to be caught by outer exception handler
        
    except (ValueError, OSError, IOError) as e:
        # Handle I/O errors gracefully (e.g., "I/O operation on closed file")
        log_warn(f"Error reading input: {e}, using default: '{default}'")
        # Try to reopen stdin on Windows if it was closed
        if sys.platform == 'win32' and "closed file" in str(e):
            try:
                sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
                log_warn("Reopened stdin, but using default for this input")
            except (OSError, IOError):
                pass
        return default
    except EOFError:
        # Handle EOF (Ctrl+D or similar)
        log_warn(f"EOF reached, using default: '{default}'")
        return default
    except KeyboardInterrupt:
        # Handle Ctrl+C
        log_warn("Input interrupted by user")
        raise
    except AttributeError:
        # Handle case where stdin doesn't have expected attributes
        log_warn("stdin has unexpected attributes, using default")
        return default


class SymbolFetchError(Exception):
    """Custom exception for symbol fetching errors."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None, is_retryable: bool = False):
        super().__init__(message)
        self.original_exception = original_exception
        self.is_retryable = is_retryable


def get_all_symbols_from_exchange(
    exchange_name: str = 'binance',
    quote_currency: str = 'USDT',
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> List[str]:
    """
    Get all trading symbols from exchange with retry logic for transient errors.
    
    Args:
        exchange_name: Exchange name to connect to (default: 'binance')
        quote_currency: Quote currency to filter symbols (default: 'USDT')
        max_retries: Maximum number of retry attempts for transient errors (default: 3)
        retry_delay: Initial delay in seconds for exponential backoff (default: 1.0)
    
    Returns:
        List of symbol strings (e.g., ['BTC/USDT', 'ETH/USDT', ...])
        Empty list if no symbols found (but no error occurred)
    
    Raises:
        SymbolFetchError: If symbol fetching fails after all retries or encounters
                       a non-retryable error. The exception includes information
                       about whether the error is retryable and the original exception.
    """
    # Protect stdin before creating PublicExchangeManager (may close stdin on Windows)
    saved_stdin = None
    if sys.platform == 'win32' and sys.stdin is not None:
        try:
            saved_stdin = sys.stdin
            if hasattr(sys.stdin, 'closed') and sys.stdin.closed:
                try:
                    sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
                    saved_stdin = sys.stdin
                except (OSError, IOError):
                    pass
        except (AttributeError, ValueError, OSError, IOError):
            pass
    
    try:
        last_exception = None
        public_exchange_manager = PublicExchangeManager()
        
        for attempt in range(max_retries):
            try:
                # Use public exchange manager (no credentials needed for load_markets)
                exchange = public_exchange_manager.connect_to_exchange_with_no_credentials(exchange_name)
                
                # Load markets
                markets = exchange.load_markets()
                
                # Filter by quote currency and active status
                symbols = []
                for symbol, market in markets.items():
                    if (market.get('quote') == quote_currency and 
                        market.get('active', True) and
                        market.get('type') == 'spot'):  # Only spot markets
                        symbols.append(symbol)
                
                # Sort alphabetically
                symbols.sort()
                
                # Success - return symbols (empty list is valid if no symbols match criteria)
                return symbols
                
            except Exception as e:
                last_exception = e
                error_message = str(e)
                
                # Determine if error is retryable (network errors, rate limits, temporary unavailability)
                error_code = None
                if hasattr(e, 'status_code'):
                    error_code = e.status_code
                elif hasattr(e, 'code'):
                    error_code = e.code
                elif '503' in error_message or 'UNAVAILABLE' in error_message.upper():
                    error_code = 503
                elif '429' in error_message or 'RATE_LIMIT' in error_message.upper():
                    error_code = 429
                
                is_retryable = (
                    error_code in [503, 429] or
                    'overloaded' in error_message.lower() or
                    'rate limit' in error_message.lower() or
                    'unavailable' in error_message.lower() or
                    'timeout' in error_message.lower() or
                    'connection' in error_message.lower() or
                    'network' in error_message.lower()
                )
                
                # Log the error
                if attempt < max_retries - 1 and is_retryable:
                    wait_time = retry_delay * (2 ** attempt)
                    log_warn(
                        f"Retryable error getting symbols (attempt {attempt + 1}/{max_retries}): {error_message}. "
                        f"Waiting {wait_time}s before retrying..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error or final attempt failed
                    log_error(f"Error getting symbols: {error_message}")
                    if is_retryable:
                        # Final retry failed
                        raise SymbolFetchError(
                            f"Failed to fetch symbols after {max_retries} attempts: {error_message}",
                            original_exception=e,
                            is_retryable=True
                        ) from e
                    else:
                        # Non-retryable error
                        raise SymbolFetchError(
                            f"Failed to fetch symbols (non-retryable error): {error_message}",
                            original_exception=e,
                            is_retryable=False
                        ) from e
        
        # This should never be reached, but just in case
        if last_exception:
            raise SymbolFetchError(
                f"Failed to fetch symbols after {max_retries} attempts",
                original_exception=last_exception,
                is_retryable=True
            ) from last_exception
    finally:
        # Always restore stdin after exchange operations
        if sys.platform == 'win32' and saved_stdin is not None:
            try:
                if sys.stdin is None or (hasattr(sys.stdin, 'closed') and sys.stdin.closed):
                    if saved_stdin is not None and not (hasattr(saved_stdin, 'closed') and saved_stdin.closed):
                        sys.stdin = saved_stdin
                    else:
                        try:
                            sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
                        except (OSError, IOError):
                            pass
            except (AttributeError, ValueError, OSError, IOError):
                pass


def pre_filter_symbols_with_voting(
    all_symbols: List[str],
    percentage: float,
    timeframe: str,
    limit: int
) -> List[str]:
    """
    Pre-filter symbols using VotingAnalyzer to select top % symbols with highest confidence.
    
    Runs VotingAnalyzer in-process to avoid stdin issues.
    
    Args:
        all_symbols: List of all symbols to filter
        percentage: Percentage of symbols to select (0-100)
        timeframe: Timeframe string for analysis
        limit: Number of candles to fetch per symbol
        
    Returns:
        List of filtered symbols (top % by weighted_score), sorted by weighted_score descending.
        Returns all_symbols if filtering fails or no signals found.
    """
    if not all_symbols or percentage <= 0.0:
        return all_symbols
    
    total_symbols = len(all_symbols)
    target_count = int(total_symbols * percentage / 100.0)
    
    # Validate target count
    if target_count <= 0:
        log_warn(f"Pre-filter percentage {percentage}% results in 0 symbols, skipping pre-filter")
        return all_symbols
    
    log_info(f"Pre-filtering symbols using VotingAnalyzer ({percentage}% of {total_symbols} symbols = {target_count} symbols)...")
    
    # Protect stdin before initializing VotingAnalyzer (which may close it on Windows)
    saved_stdin = None
    if sys.platform == 'win32' and sys.stdin is not None:
        try:
            saved_stdin = sys.stdin
            if hasattr(sys.stdin, 'closed') and sys.stdin.closed:
                try:
                    sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
                    saved_stdin = sys.stdin
                except (OSError, IOError):
                    pass
        except (AttributeError, ValueError, OSError, IOError):
            pass
    
    try:
        # Create args namespace for VotingAnalyzer with default values
        args = argparse.Namespace()
        args.timeframe = timeframe
        args.no_menu = True
        args.limit = limit
        args.max_workers = 10
        args.osc_length = RANGE_OSCILLATOR_LENGTH
        args.osc_mult = RANGE_OSCILLATOR_MULTIPLIER
        args.osc_strategies = None
        args.enable_spc = True
        args.spc_k = 2
        args.spc_lookback = SPC_LOOKBACK
        args.spc_p_low = SPC_P_LOW
        args.spc_p_high = SPC_P_HIGH
        args.spc_min_signal_strength = SPC_STRATEGY_PARAMETERS['cluster_transition']['min_signal_strength']
        args.spc_min_rel_pos_change = SPC_STRATEGY_PARAMETERS['cluster_transition']['min_rel_pos_change']
        args.spc_min_regime_strength = SPC_STRATEGY_PARAMETERS['regime_following']['min_regime_strength']
        args.spc_min_cluster_duration = SPC_STRATEGY_PARAMETERS['regime_following']['min_cluster_duration']
        args.spc_extreme_threshold = SPC_STRATEGY_PARAMETERS['mean_reversion']['extreme_threshold']
        args.spc_min_extreme_duration = SPC_STRATEGY_PARAMETERS['mean_reversion']['min_extreme_duration']
        args.spc_strategy = "all"
        args.enable_xgboost = True
        args.enable_hmm = True
        args.hmm_window_size = HMM_WINDOW_SIZE_DEFAULT
        args.hmm_window_kama = HMM_WINDOW_KAMA_DEFAULT
        args.hmm_fast_kama = HMM_FAST_KAMA_DEFAULT
        args.hmm_slow_kama = HMM_SLOW_KAMA_DEFAULT
        args.hmm_orders_argrelextrema = HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
        args.hmm_strict_mode = HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
        args.enable_random_forest = True
        args.random_forest_model_path = None
        args.use_decision_matrix = True
        args.voting_threshold = DECISION_MATRIX_VOTING_THRESHOLD
        args.min_votes = DECISION_MATRIX_MIN_VOTES
        args.ema_len = 28
        args.hma_len = 28
        args.wma_len = 28
        args.dema_len = 28
        args.lsma_len = 28
        args.kama_len = 28
        args.robustness = "Medium"
        args.lambda_param = 0.5
        args.decay = 0.1
        args.cutout = 5
        args.min_signal = 0.01
        args.max_symbols = None
        
        # Create ExchangeManager and DataFetcher
        from modules.common.core.exchange_manager import ExchangeManager
        exchange_manager = ExchangeManager()
        data_fetcher = DataFetcher(exchange_manager)
        
        # Create VotingAnalyzer instance
        analyzer = VotingAnalyzer(args, data_fetcher)
        analyzer.selected_timeframe = timeframe
        analyzer.atc_analyzer.selected_timeframe = timeframe
        
        # Run ATC scan first to get initial signals
        log_info("Running ATC scan for pre-filtering...")
        if not analyzer.run_atc_scan():
            log_warn("No ATC signals found from VotingAnalyzer, cannot pre-filter")
            return all_symbols
        
        # Calculate signals for all indicators and apply voting system
        log_info("Calculating signals from all indicators...")
        analyzer.calculate_and_vote()
        
        # Extract symbols from long_signals_final and short_signals_final
        # Combine LONG and SHORT, but only include symbols that are in all_symbols
        # Sort by weighted_score descending
        all_symbols_set = set(all_symbols)  # For fast lookup
        all_signals_list = []
        
        if not analyzer.long_signals_final.empty:
            for _, row in analyzer.long_signals_final.iterrows():
                symbol = row.get('symbol', '')
                weighted_score = row.get('weighted_score', 0.0)
                if symbol and symbol in all_symbols_set:
                    all_signals_list.append((symbol, weighted_score, 'LONG'))
        
        if not analyzer.short_signals_final.empty:
            for _, row in analyzer.short_signals_final.iterrows():
                symbol = row.get('symbol', '')
                weighted_score = row.get('weighted_score', 0.0)
                if symbol and symbol in all_symbols_set:
                    all_signals_list.append((symbol, weighted_score, 'SHORT'))
        
        if not all_signals_list:
            log_warn("No signals found from VotingAnalyzer for the specified symbols, scanning all symbols instead")
            return all_symbols
        
        # Sort by weighted_score descending
        all_signals_list.sort(key=lambda x: x[1], reverse=True)
        
        # Get top % symbols
        filtered_symbols = [symbol for symbol, _, _ in all_signals_list[:target_count]]
        
        log_success(f"Pre-filter complete: {len(filtered_symbols)}/{total_symbols} symbols selected (top {percentage}%)")
        
        if len(filtered_symbols) < target_count:
            log_warn(f"Only {len(filtered_symbols)} symbols have signals (less than target {target_count})")
        
        return filtered_symbols
        
    except Exception as e:
        log_error(f"Error during pre-filtering: {e}")
        log_warn("Falling back to scanning all symbols")
        import traceback
        traceback.print_exc()
        return all_symbols
    finally:
        # Always restore stdin after pre-filter
        if sys.platform == 'win32' and saved_stdin is not None:
            try:
                if sys.stdin is None or (hasattr(sys.stdin, 'closed') and sys.stdin.closed):
                    if saved_stdin is not None and not (hasattr(saved_stdin, 'closed') and saved_stdin.closed):
                        sys.stdin = saved_stdin
                    else:
                        try:
                            sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
                        except (OSError, IOError):
                            pass
            except (AttributeError, ValueError, OSError, IOError):
                pass


def interactive_batch_scan():
    """Interactive menu for batch scanning."""
    print()
    print(color_text("=" * 60, Fore.CYAN))
    print(color_text("MARKET BATCH SCANNER", Fore.CYAN))
    print(color_text("=" * 60, Fore.CYAN))
    print()
    
    # Get timeframe(s) - single or multi
    print("\nAnalysis mode:")
    print("  1. Single timeframe")
    print("  2. Multi-timeframe (recommended)")
    mode = safe_input(color_text("Select mode (1/2) [2]: ", Fore.YELLOW), default='2').strip()
    if not mode:
        mode = '2'
    
    timeframe = None
    timeframes = None
    
    if mode == '2':
        # Multi-timeframe mode
        from modules.gemini_chart_analyzer.core.utils import DEFAULT_TIMEFRAMES, normalize_timeframes
        
        print(f"\nDefault timeframes: {', '.join(DEFAULT_TIMEFRAMES)}")
        print("Timeframes: 15m, 30m, 1h, 4h, 1d, 1w (comma-separated)")
        timeframes_input = safe_input(color_text(f"Enter timeframes (comma-separated) [{', '.join(DEFAULT_TIMEFRAMES)}]: ", Fore.YELLOW), default=','.join(DEFAULT_TIMEFRAMES)).strip()
        if not timeframes_input:
            timeframes = DEFAULT_TIMEFRAMES
        else:
            try:
                timeframes_list = [tf.strip() for tf in timeframes_input.split(',') if tf.strip()]
                timeframes = normalize_timeframes(timeframes_list)
                if not timeframes:
                    log_warn("No valid timeframes, using default")
                    timeframes = DEFAULT_TIMEFRAMES
            except Exception as e:
                log_warn(f"Error parsing timeframes: {e}, using default")
                timeframes = DEFAULT_TIMEFRAMES
    else:
        # Single timeframe mode
        print("\nTimeframes: 15m, 30m, 1h, 4h, 1d, 1w")
        timeframe = safe_input(color_text("Enter timeframe [1h]: ", Fore.YELLOW), default='1h').strip()
        if not timeframe:
            timeframe = '1h'
        
        # Normalize timeframe with exception handling
        try:
            timeframe = normalize_timeframe(timeframe)
        except Exception as e:
            log_warn(f"Error parsing timeframe: {e}, defaulting to '1h'")
            timeframe = '1h'
            try:
                timeframe = normalize_timeframe(timeframe)
            except Exception as e2:
                log_error(f"Critical error normalizing default timeframe: {e2}")
                raise
    
    # Get max symbols (optional)
    max_symbols_input = safe_input(color_text("Max symbols to scan (press Enter for all): ", Fore.YELLOW), default='').strip()
    max_symbols = None
    if max_symbols_input:
        try:
            max_symbols = int(max_symbols_input)
            # Validate: must be positive integer (>=1)
            if max_symbols < 1:
                log_warn(f"max_symbols ({max_symbols}) must be >= 1, resetting to default (all symbols)")
                max_symbols = None
        except ValueError:
            log_warn("Invalid input, scanning all symbols")
    
    # Get cooldown
    cooldown_input = safe_input(color_text("Cooldown between batches in seconds [2.5]: ", Fore.YELLOW), default='2.5').strip()
    cooldown = 2.5
    if cooldown_input:
        try:
            cooldown = float(cooldown_input)
            # Validate: must be non-negative (>=0.0)
            if cooldown < 0.0:
                log_warn(f"cooldown ({cooldown}) must be >= 0.0, clamping to 0.0")
                cooldown = 0.0
        except ValueError:
            log_warn("Invalid input, using default 2.5s")
    
    # Get candles limit
    limit_input = safe_input(color_text("Number of candles per symbol [500]: ", Fore.YELLOW), default='500').strip()
    limit = 500
    if limit_input:
        try:
            limit = int(limit_input)
            # Validate: must be positive integer (>=1)
            if limit < 1:
                log_warn(f"limit ({limit}) must be >= 1, clamping to 1")
                limit = 1
        except ValueError:
            log_warn("Invalid input, using default 500")
    
    # Get pre-filter percentage (optional)
    print("\nPre-filter option:")
    print("  Select top % symbols with high confidence using VotingAnalyzer before Gemini scan")
    pre_filter_input = safe_input(color_text("Pre-filter percentage (0-100, press Enter to skip) [0]: ", Fore.YELLOW), default='0').strip()
    pre_filter_percentage = 0.0
    if pre_filter_input:
        try:
            pre_filter_percentage = float(pre_filter_input)
            # Validate: must be between 0-100
            if pre_filter_percentage < 0.0 or pre_filter_percentage > 100.0:
                log_warn(f"Pre-filter percentage ({pre_filter_percentage}) must be between 0-100, resetting to 0 (no pre-filter)")
                pre_filter_percentage = 0.0
        except ValueError:
            log_warn("Invalid input, using default 0 (no pre-filter)")
            pre_filter_percentage = 0.0
    
    # Confirm BEFORE running pre-filter (to avoid stdin issues)
    # This ensures all user input is collected before any operations that might affect stdin
    print()
    print(color_text("=" * 60, Fore.CYAN))
    print(color_text("CONFIGURATION", Fore.CYAN))
    print(color_text("=" * 60, Fore.CYAN))
    if timeframes:
        print(f"Timeframes: {', '.join(timeframes)} (Multi-timeframe mode)")
    else:
        print(f"Timeframe: {timeframe} (Single timeframe mode)")
    print(f"Max symbols: {max_symbols or 'All'}")
    print(f"Cooldown: {cooldown}s")
    print(f"Candles per symbol: {limit}")
    if pre_filter_percentage > 0:
        print(f"Pre-filter: Top {pre_filter_percentage}% symbols using VotingAnalyzer")
    else:
        print(f"Pre-filter: Disabled")
    print()
    
    # # Read confirmation BEFORE initializing scanner and running pre-filter
    # # This ensures stdin is still available and not affected by VotingAnalyzer
    # # Flush stdout/stderr before reading input to ensure clean state
    # try:
    #     sys.stdout.flush()
    #     sys.stderr.flush()
    # except (OSError, IOError, AttributeError):
    #     pass  # Continue if flush fails
    
    confirm = safe_input(color_text("Start batch scan? (y/n) [y]: ", Fore.YELLOW), default='y').lower()
    
    if confirm and confirm not in ['y', 'yes']:
        log_warn("Cancelled by user")
        return
    
    # Pre-filter symbols BEFORE initializing scanner (if enabled)
    pre_filtered_symbols = None
    if pre_filter_percentage > 0:
        try:
            log_info("=" * 60)
            log_info("PRE-FILTERING SYMBOLS")
            log_info("=" * 60)
            
            # Get all symbols from exchange
            log_info("Step 1: Getting all symbols from exchange...")
            all_symbols = get_all_symbols_from_exchange(
                exchange_name='binance',
                quote_currency='USDT'
            )
            
            if not all_symbols:
                log_warn("No symbols found from exchange, skipping pre-filter")
                pre_filtered_symbols = None
            else:
                # Determine primary timeframe for pre-filtering
                if timeframes:
                    primary_timeframe = timeframes[0]  # Use first timeframe for pre-filtering
                elif timeframe:
                    primary_timeframe = timeframe
                else:
                    primary_timeframe = '1h'  # Default fallback
                
                # Run pre-filter
                pre_filtered_symbols = pre_filter_symbols_with_voting(
                    all_symbols=all_symbols,
                    percentage=pre_filter_percentage,
                    timeframe=primary_timeframe,
                    limit=limit
                )
                
                if pre_filtered_symbols and len(pre_filtered_symbols) < len(all_symbols):
                    log_info(f"Pre-filtered: {len(all_symbols)} → {len(pre_filtered_symbols)} symbols ({pre_filter_percentage}%)")
                else:
                    log_info(f"Pre-filtered: All {len(all_symbols)} symbols retained (no filtering applied)")
                    
        except SymbolFetchError as e:
            log_error(f"Failed to fetch symbols from exchange: {e}")
            if e.is_retryable:
                log_error("This was a retryable error (network/rate limit). Please check your connection and try again.")
            else:
                log_error("This was a non-retryable error. Please check exchange configuration and API access.")
            log_warn("Continuing without pre-filter...")
            pre_filtered_symbols = None
        except Exception as e:
            log_error(f"Error during pre-filtering: {e}")
            log_warn("Continuing without pre-filter...")
            pre_filtered_symbols = None
    
    # Initialize scanner AFTER pre-filter (if enabled)
    try:
        scanner = MarketBatchScanner(cooldown_seconds=cooldown)
        
        # Run scan with pre-filtered symbols (if available)
        results = scanner.scan_market(
            timeframe=timeframe,
            timeframes=timeframes,
            max_symbols=max_symbols,
            limit=limit,
            initial_symbols=pre_filtered_symbols
        )
        
        # Display results
        print()
        print(color_text("=" * 60, Fore.GREEN))
        print(color_text("SCAN RESULTS", Fore.GREEN))
        print(color_text("=" * 60, Fore.GREEN))
        print()
        
        # Display LONG signals with confidence (sorted by confidence)
        long_symbols = results.get('long_symbols', [])
        print(color_text(f"LONG Signals ({len(long_symbols)}):", Fore.GREEN))
        if results.get('long_symbols_with_confidence'):
            print("  (Sorted by confidence: High → Low)")
            for symbol, confidence in results['long_symbols_with_confidence']:
                clamped_confidence = min(max(confidence, 0.0), 1.0)
                length = int(clamped_confidence * 10)
                confidence_bar = "█" * length  # Visual bar
                print(f"  {symbol:15s} | Confidence: {confidence:.2f} {confidence_bar}")
        elif long_symbols:
            # Fallback if confidence data not available
            for i in range(0, len(long_symbols), 5):
                row = long_symbols[i:i+5]
                print("  " + "  ".join(f"{s:12s}" for s in row))
        else:
            print("  None")
        
        print()
        # Display SHORT signals with confidence (sorted by confidence)
        short_symbols = results.get('short_symbols', [])
        print(color_text(f"SHORT Signals ({len(short_symbols)}):", Fore.RED))
        if results.get('short_symbols_with_confidence'):
            print("  (Sorted by confidence: High → Low)")
            for symbol, confidence in results['short_symbols_with_confidence']:
                clamped_confidence = min(max(confidence, 0.0), 1.0)
                length = int(clamped_confidence * 10)
                confidence_bar = "█" * length  # Visual bar
                print(f"  {symbol:15s} | Confidence: {confidence:.2f} {confidence_bar}")
        elif short_symbols:
            # Fallback if confidence data not available
            for i in range(0, len(short_symbols), 5):
                row = short_symbols[i:i+5]
                print("  " + "  ".join(f"{s:12s}" for s in row))
        else:
            print("  None")
        
        # Display summary with average confidence
        if results.get('summary'):
            summary = results['summary']
            if summary.get('avg_long_confidence', 0) > 0:
                print()
                print(color_text("Summary:", Fore.CYAN))
                print(f"  Average LONG confidence: {summary['avg_long_confidence']:.2f}")
            if summary.get('avg_short_confidence', 0) > 0:
                print(f"  Average SHORT confidence: {summary['avg_short_confidence']:.2f}")
        
        print()
        print(color_text("=" * 60, Fore.GREEN))
        results_file = results.get('results_file', 'N/A')
        print(color_text(f"Results saved to: {results_file}", Fore.GREEN))
        print(color_text("=" * 60, Fore.GREEN))        
    except Exception as e:
        log_error(f"Error during batch scan: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    # Ensure stdin is available on Windows (critical when running file directly)
    if sys.platform == 'win32':
        try:
            if sys.stdin is None or (hasattr(sys.stdin, 'closed') and sys.stdin.closed):
                sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
        except (OSError, IOError, AttributeError):
            pass  # Continue if we can't fix stdin
    
    try:
        interactive_batch_scan()
    except KeyboardInterrupt:
        log_warn("\nExiting...")
        sys.exit(0)
    except Exception as e:
        log_error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


