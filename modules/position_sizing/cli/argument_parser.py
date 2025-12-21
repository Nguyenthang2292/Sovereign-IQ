"""
Argument parser for position sizing CLI.

This module provides functions for parsing command-line arguments
and interactive configuration for position sizing calculation.
"""

import argparse
from typing import Optional, List
from colorama import Fore, Style

from modules.common.utils import (
    color_text,
    prompt_user_input,
)
from modules.common.ui.formatting import prompt_user_input_with_backspace
from config.position_sizing import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TIMEFRAME,
    DEFAULT_MAX_POSITION_SIZE,
    SIGNAL_CALCULATION_MODE,
)


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol input to standard format (BASE/USDT).
    
    Examples:
        "btc" -> "BTC/USDT"
        "BTC" -> "BTC/USDT"
        "BTC/USDT" -> "BTC/USDT"
        "btcusdt" -> "BTCUSDT/USDT"
    
    Args:
        symbol: Raw symbol input from user
        
    Returns:
        Normalized symbol in BASE/USDT format
    """
    if not symbol:
        return ""
    
    cleaned = symbol.strip().upper()
    
    # If already has "/", just ensure quote is USDT
    if "/" in cleaned:
        base, quote = cleaned.split("/", 1)
        base = base.strip()
        quote = quote.strip() or "USDT"
        return f"{base}/{quote}"
    
    # If ends with USDT (like "BTCUSDT"), extract base
    if cleaned.endswith("USDT") and len(cleaned) > 4:
        base = cleaned[:-4]
        return f"{base}/USDT"
    
    # Otherwise, assume it's just the base currency
    return f"{cleaned}/USDT"


def parse_symbols_string(symbols_str: str) -> List[str]:
    """
    Parse comma-separated symbols string and normalize each symbol.
    
    Args:
        symbols_str: Comma-separated symbols (e.g., "btc,eth" or "BTC/USDT,ETH/USDT")
        
    Returns:
        List of normalized symbols
    """
    if not symbols_str:
        return []
    
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
    return [normalize_symbol(s) for s in symbols]


def parse_args():
    """
    Parse command-line arguments for position sizing.
    
    Returns:
        argparse.Namespace object with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Position Sizing Calculator with Bayesian Kelly Criterion and Regime Switching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input options
    parser.add_argument(
        '--symbols-file',
        type=str,
        help='Path to CSV/JSON file with symbols (from hybrid/voting results)',
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (e.g., "btc,eth" or "BTC/USDT,ETH/USDT")',
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['hybrid', 'voting'],
        help='Source of symbols: hybrid or voting analyzer results',
    )
    
    # Account settings
    parser.add_argument(
        '--account-balance',
        type=float,
        help='Account balance in USDT (if not provided, will try to fetch from Binance or prompt)',
    )
    parser.add_argument(
        '--fetch-balance',
        action='store_true',
        help='Automatically fetch account balance from Binance (requires API credentials)',
    )
    
    # Backtest settings
    parser.add_argument(
        '--timeframe',
        type=str,
        default=DEFAULT_TIMEFRAME,
        help=f'Timeframe for backtesting (default: {DEFAULT_TIMEFRAME})',
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f'Number of days to look back for backtesting (default: {DEFAULT_LOOKBACK_DAYS})',
    )
    parser.add_argument(
        '--auto-timeframe',
        action='store_true',
        help='Enable auto timeframe testing (tries 15m, 30m, 1h until finding valid results)',
    )
    
    # Position size constraints
    parser.add_argument(
        '--max-position-size',
        type=float,
        default=DEFAULT_MAX_POSITION_SIZE,
        help=f'Maximum position size as fraction of account (default: {DEFAULT_MAX_POSITION_SIZE} = {DEFAULT_MAX_POSITION_SIZE*100:.0f}% of account balance)',
    )
    
    # Signal calculation settings
    parser.add_argument(
        '--signal-mode',
        type=str,
        choices=['majority_vote', 'single_signal'],
        default='single_signal',
        help='Signal calculation mode: single_signal (default, highest confidence) or majority_vote',
    )
    parser.add_argument(
        '--signal-calculation-mode',
        type=str,
        choices=['precomputed', 'incremental'],
        default=SIGNAL_CALCULATION_MODE,
        help='Signal calculation approach: precomputed (default, calculate all signals first) or incremental (skip when position open)',
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for results (CSV format)',
    )
    parser.add_argument(
        '--no-menu',
        action='store_true',
        help='Skip interactive menu, use command-line arguments only',
    )
    
    return parser.parse_args()


def _format_current_value_ps(value) -> str:
    """Format current value for display in menu."""
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    if isinstance(value, str):
        return value if value else "not set"
    return str(value)


def _display_main_menu_ps(config):
    """Display main menu with current configuration values."""
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("POSITION SIZING CONFIGURATION", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print("\n" + color_text("MAIN MENU", Fore.YELLOW, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    
    # Format current values
    source_val = "not set"
    if 'source' in config:
        source_val = config['source']
    elif 'symbols_file' in config:
        source_val = f"file: {config['symbols_file']}"
    elif 'symbols' in config and config['symbols']:
        source_val = f"manual: {config['symbols']}"
    
    balance_val = "not set"
    if config.get('fetch_balance', False):
        balance_val = "fetch from Binance"
    elif 'account_balance' in config and config['account_balance'] is not None:
        balance_val = f"{config['account_balance']:.2f} USDT"
    
    timeframe_val = _format_current_value_ps(config.get('timeframe', DEFAULT_TIMEFRAME))
    auto_tf_val = "enabled" if config.get('auto_timeframe', False) else "disabled"
    lookback_val = _format_current_value_ps(config.get('lookback_days', DEFAULT_LOOKBACK_DAYS))
    max_size_val = _format_current_value_ps(config.get('max_position_size', DEFAULT_MAX_POSITION_SIZE))
    
    print(f"  1. Symbol Source [{color_text(source_val, Fore.GREEN)}]")
    print(f"  2. Account Balance [{color_text(balance_val, Fore.GREEN)}]")
    print(f"  3. Backtest Settings [{color_text(f'timeframe={timeframe_val}, auto={auto_tf_val}, lookback={lookback_val}', Fore.GREEN)}]")
    print(f"  4. Position Size Constraints [{color_text(f'max={max_size_val}', Fore.GREEN)}]")
    print(f"  5. Review and Confirm")
    print(f"  6. Exit")
    print(color_text("-" * 80, Fore.CYAN))


def _prompt_with_back_ps(prompt: str, default: str = None, allow_back: bool = True) -> tuple:
    """
    Prompt user with backspace key for back navigation.
    
    Returns:
        (value, action) where action is 'main' or 'continue'
    """
    if allow_back:
        back_prompt = f"{prompt} (press Backspace to go back): "
    else:
        back_prompt = prompt
    
    if allow_back:
        user_input, is_back = prompt_user_input_with_backspace(back_prompt, default=default)
        
        if is_back:
            return (None, 'main')
        
        # Convert to lowercase for consistency
        if user_input:
            user_input = user_input.strip().lower()
        
        return (user_input, 'continue')
    else:
        user_input = prompt_user_input(back_prompt, default=default).strip().lower()
        return (user_input, 'continue')


def _configure_symbol_source(config):
    """Configure symbol source with back option."""
    while True:
        print("\n" + color_text("1. SYMBOL SOURCE", Fore.WHITE, Style.BRIGHT))
        print("   a) Load from hybrid analyzer results")
        print("   b) Load from voting analyzer results")
        print("   c) Load from file (CSV/JSON)")
        print(color_text("   d) Manual input (comma-separated symbols) [default]", Fore.MAGENTA, Style.BRIGHT))
        print(color_text("   e) Back to main menu", Fore.CYAN))
        print()
        
        # Determine default based on current config
        current_source = None
        if 'source' in config:
            if config['source'] == 'hybrid':
                current_source = 'a'
            elif config['source'] == 'voting':
                current_source = 'b'
        elif 'symbols_file' in config:
            current_source = 'c'
        elif 'symbols' in config:
            current_source = 'd'
        else:
            current_source = 'd'
        
        default_source_text = color_text("default=d", Fore.MAGENTA)
        source_choice, action = _prompt_with_back_ps(f"Select source (a/b/c/d/e, {default_source_text}): ", default=current_source)
        
        if action == 'main' or source_choice == 'e':
            return 'main'
        
        if not source_choice:
            source_choice = 'd'
        
        # Clear previous source settings
        config.pop('source', None)
        config.pop('symbols_file', None)
        config.pop('symbols', None)
        
        if source_choice == 'a':
            config['source'] = 'hybrid'
            return 'main'
        elif source_choice == 'b':
            config['source'] = 'voting'
            return 'main'
        elif source_choice == 'c':
            filepath, action = _prompt_with_back_ps("Enter file path: ")
            if action == 'main':
                return 'main'
            if filepath:
                config['symbols_file'] = filepath
            return 'main'
        elif source_choice == 'd':
            while True:
                symbols_str, action = _prompt_with_back_ps("Enter symbols (comma-separated, e.g., btc,eth or BTC/USDT,ETH/USDT): ")
                if action == 'main':
                    return 'main'
                
                if not symbols_str:
                    print(color_text("Error: At least one symbol is required. Please enter symbols or press Backspace to go back.", Fore.RED))
                    continue
                
                normalized_symbols = parse_symbols_string(symbols_str)
                
                # Validate: normalized_symbols must be non-empty and contain only valid (non-empty) symbols
                if not normalized_symbols:
                    print(color_text("Error: No valid symbols found. Please enter at least one symbol or press Backspace to go back.", Fore.RED))
                    continue
                
                # Filter out any empty strings that might have been created during normalization
                valid_symbols = [s for s in normalized_symbols if s]
                
                if not valid_symbols:
                    print(color_text("Error: No valid symbols found after normalization. Please enter valid symbols or press Backspace to go back.", Fore.RED))
                    continue
                
                # Validation passed, set config and break out of loop
                config['symbols'] = ",".join(valid_symbols)
                print(color_text(f"Normalized symbols: {config['symbols']}", Fore.GREEN))
                break
            
            return 'main'
        else:
            print(color_text("Invalid choice. Please select a/b/c/d/e.", Fore.RED))


def _configure_account_balance(config):
    """Configure account balance with back option."""
    while True:
        print("\n" + color_text("2. ACCOUNT BALANCE", Fore.WHITE, Style.BRIGHT))
        print(color_text("   a) Fetch from Binance (requires API credentials) [default]", Fore.MAGENTA, Style.BRIGHT))
        print("   b) Manual input")
        print(color_text("   c) Back to main menu", Fore.CYAN))
        print()
        
        # Determine default based on current config
        current_choice = 'a' if config.get('fetch_balance', False) else 'b'
        
        default_balance_text = color_text("default=a", Fore.MAGENTA)
        balance_choice, action = _prompt_with_back_ps(f"Select option (a/b/c, {default_balance_text}): ", default=current_choice)
        
        if action == 'main' or balance_choice == 'c':
            return 'main'
        
        if not balance_choice:
            balance_choice = 'a'
        
        if balance_choice == 'a':
            config['fetch_balance'] = True
            config['account_balance'] = None
            return 'main'
        elif balance_choice == 'b':
            balance_str, action = _prompt_with_back_ps("Enter account balance (USDT): ")
            
            if action == 'main':
                return 'main'
            try:
                config['account_balance'] = float(balance_str) if balance_str else 10000.0
                if config['account_balance'] <= 0:
                    print(color_text(f"Invalid balance. Using default: 10000 USDT", Fore.YELLOW))
                    config['account_balance'] = 10000.0
                    config['fetch_balance'] = False
                else:
                    config['fetch_balance'] = False
            except ValueError:
                print(color_text(f"Invalid balance. Using default: 10000 USDT", Fore.YELLOW))
                config['account_balance'] = 10000.0
                config['fetch_balance'] = False
            return 'main'
        else:
            print(color_text("Invalid choice. Please select a/b/c.", Fore.RED))


def _configure_backtest_settings(config):
    """Configure backtest settings with back option."""
    while True:
        print("\n" + color_text("3. BACKTEST SETTINGS", Fore.WHITE, Style.BRIGHT))
        print(color_text("   Auto timeframe testing: Try multiple timeframes automatically", Fore.CYAN))
        print("   a) Enable auto timeframe testing (tries 15m -> 30m -> 1h)")
        print(color_text("   b) Use specified timeframe only [default]", Fore.MAGENTA, Style.BRIGHT))
        print(color_text("   c) Back to main menu", Fore.CYAN))
        print()
        
        current_auto_tf = config.get('auto_timeframe', False)
        auto_tf_choice, action = _prompt_with_back_ps(f"Select option (a/b/c, default=b): ", default='a' if current_auto_tf else 'b')
        
        if action == 'main' or auto_tf_choice == 'c':
            return 'main'
        
        if not auto_tf_choice:
            auto_tf_choice = 'b'
        
        config['auto_timeframe'] = (auto_tf_choice == 'a')
        
        if not config['auto_timeframe']:
            current_tf = config.get('timeframe', DEFAULT_TIMEFRAME)
            timeframe, action = _prompt_with_back_ps(f"Timeframe ({color_text(f'default: {DEFAULT_TIMEFRAME}', Fore.MAGENTA)}): ", default=current_tf)
            if action == 'main':
                return 'main'
            config['timeframe'] = timeframe if timeframe else DEFAULT_TIMEFRAME
        else:
            config['timeframe'] = DEFAULT_TIMEFRAME
        
        current_lookback = str(config.get('lookback_days', DEFAULT_LOOKBACK_DAYS))
        lookback_str, action = _prompt_with_back_ps(f"Lookback days ({color_text(f'default: {DEFAULT_LOOKBACK_DAYS}', Fore.MAGENTA)}): ", default=current_lookback)
        if action == 'main':
            return 'main'
        try:
            config['lookback_days'] = int(lookback_str) if lookback_str else DEFAULT_LOOKBACK_DAYS
            if config['lookback_days'] <= 0:
                print(color_text(f"Invalid lookback days (must be >= 1). Using default: {DEFAULT_LOOKBACK_DAYS}", Fore.YELLOW))
                config['lookback_days'] = DEFAULT_LOOKBACK_DAYS
        except ValueError:
            print(color_text(f"Invalid lookback days. Using default: {DEFAULT_LOOKBACK_DAYS}", Fore.YELLOW))
            config['lookback_days'] = DEFAULT_LOOKBACK_DAYS
        
        return 'main'


def _configure_position_size(config):
    """Configure position size constraints with back option."""
    while True:
        print("\n" + color_text("4. POSITION SIZE CONSTRAINTS", Fore.WHITE, Style.BRIGHT))
        print(f"   Note: Max position size is a fraction of account balance")
        print(f"   Example: 0.1 = 10% of account, 0.2 = 20% of account")
        print(color_text("   b) Back to main menu", Fore.CYAN))
        print()
        
        current_max_size = str(config.get('max_position_size', DEFAULT_MAX_POSITION_SIZE))
        default_max_size_text = color_text(f"default: {DEFAULT_MAX_POSITION_SIZE} = {DEFAULT_MAX_POSITION_SIZE*100:.0f}%", Fore.MAGENTA)
        max_size_str, action = _prompt_with_back_ps(f"Max position size (fraction, {default_max_size_text}): ", default=current_max_size)
        
        if action == 'main':
            return 'main'
        
        # Handle empty input as default
        if not max_size_str:
            config['max_position_size'] = DEFAULT_MAX_POSITION_SIZE
            return 'main'
        
        # Convert to float and validate range
        try:
            max_position_size = float(max_size_str)
            # Validate: must be within 0 < max_position_size <= 1
            if max_position_size <= 0 or max_position_size > 1:
                print(color_text(
                    f"Warning: Invalid max position size ({max_size_str}). "
                    f"Must be greater than 0 and less than or equal to 1 (0 < value <= 1). "
                    f"Using default: {DEFAULT_MAX_POSITION_SIZE} ({DEFAULT_MAX_POSITION_SIZE*100:.0f}%)",
                    Fore.YELLOW
                ))
                config['max_position_size'] = DEFAULT_MAX_POSITION_SIZE
            else:
                config['max_position_size'] = max_position_size
        except ValueError:
            # Non-numeric input - fall back to default with warning
            print(color_text(
                f"Warning: Invalid input '{max_size_str}'. "
                f"Must be a numeric value greater than 0 and less than or equal to 1 (0 < value <= 1). "
                f"Using default: {DEFAULT_MAX_POSITION_SIZE} ({DEFAULT_MAX_POSITION_SIZE*100:.0f}%)",
                Fore.YELLOW
            ))
            config['max_position_size'] = DEFAULT_MAX_POSITION_SIZE
        
        return 'main'


def _review_and_confirm_ps(config):
    """Review configuration and confirm."""
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("CONFIGURATION REVIEW", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    
    # Source
    source_display = "not set"
    if 'source' in config:
        source_display = f"from {config['source']} analyzer"
    elif 'symbols_file' in config:
        source_display = f"from file: {config['symbols_file']}"
    elif 'symbols' in config and config['symbols']:
        source_display = f"manual: {config['symbols']}"
    print(f"\nSymbol Source: {source_display}")
    
    # Balance
    balance_display = "not set"
    if config.get('fetch_balance', False):
        balance_display = "fetch from Binance"
    elif 'account_balance' in config and config['account_balance'] is not None:
        balance_display = f"{config['account_balance']:.2f} USDT"
    print(f"Account Balance: {balance_display}")
    
    # Backtest settings
    timeframe_display = config.get('timeframe', DEFAULT_TIMEFRAME)
    auto_tf_display = "enabled" if config.get('auto_timeframe', False) else "disabled"
    lookback_display = config.get('lookback_days', DEFAULT_LOOKBACK_DAYS)
    print(f"Backtest Settings: timeframe={timeframe_display}, auto={auto_tf_display}, lookback={lookback_display} days")
    
    # Position size
    max_size_display = config.get('max_position_size', DEFAULT_MAX_POSITION_SIZE)
    print(f"Max Position Size: {max_size_display} ({max_size_display*100:.0f}%)")
    
    print("\n" + color_text("-" * 80, Fore.CYAN))
    confirm = prompt_user_input("Confirm this configuration? (y/n) [y]: ", default="y").strip().lower()
    
    if confirm in ['y', 'yes', '']:
        return 'done'
    else:
        return 'main'


class ConfigurationCanceledError(Exception):
    """Exception raised when user cancels configuration."""
    pass


def interactive_config_menu() -> Optional[dict]:
    """
    Interactive menu for configuring position sizing.
    
    Returns:
        Dictionary with configuration values, or None if user cancels (chooses Exit)
        
    Note:
        Returns None when user selects option 6 (Exit) to indicate cancellation.
        Callers should handle None return value for cleanup and graceful termination.
    """
    config = {}
    
    # Main menu loop
    while True:
        _display_main_menu_ps(config)
        
        choice = prompt_user_input("\nSelect option [1-6]: ").strip()
        
        if choice == '1':
            _configure_symbol_source(config)
        elif choice == '2':
            _configure_account_balance(config)
        elif choice == '3':
            _configure_backtest_settings(config)
        elif choice == '4':
            _configure_position_size(config)
        elif choice == '5':
            result = _review_and_confirm_ps(config)
            if result == 'done':
                break
        elif choice == '6':
            print(color_text("\nExiting configuration menu.", Fore.YELLOW))
            return None
        else:
            print(color_text("Invalid choice. Please select 1-6.", Fore.RED))
    
    return config

