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


def interactive_config_menu() -> dict:
    """
    Interactive menu for configuring position sizing.
    
    Returns:
        Dictionary with configuration values
    """
    print("\n" + color_text("=" * 80, Fore.CYAN))
    print(color_text("POSITION SIZING CONFIGURATION", Fore.CYAN))
    print(color_text("=" * 80, Fore.CYAN))
    
    config = {}
    
    # Source selection
    print("\n" + color_text("1. SYMBOL SOURCE", Fore.WHITE))
    print("   a) Load from hybrid analyzer results")
    print("   b) Load from voting analyzer results")
    print("   c) Load from file (CSV/JSON)")
    print(color_text("   d) Manual input (comma-separated symbols) [default]", Fore.MAGENTA, Style.BRIGHT))
    
    default_source_text = color_text("default=d", Fore.MAGENTA)
    source_choice = prompt_user_input(f"Select source (a/b/c/d, {default_source_text}): ", default="d").strip().lower()
    
    # Default to 'd' if empty input
    if not source_choice:
        source_choice = 'd'
    
    if source_choice == 'a':
        config['source'] = 'hybrid'
    elif source_choice == 'b':
        config['source'] = 'voting'
    elif source_choice == 'c':
        filepath = prompt_user_input("Enter file path: ").strip()
        config['symbols_file'] = filepath
    elif source_choice == 'd':
        symbols_str = prompt_user_input("Enter symbols (comma-separated, e.g., btc,eth or BTC/USDT,ETH/USDT): ").strip()
        if symbols_str:
            # Normalize symbols (auto-add /USDT if needed)
            normalized_symbols = parse_symbols_string(symbols_str)
            config['symbols'] = ",".join(normalized_symbols)
            print(color_text(f"Normalized symbols: {config['symbols']}", Fore.GREEN))
        else:
            config['symbols'] = ""
    else:
        print(color_text("Invalid choice. Using default: manual input (d)", Fore.YELLOW))
        source_choice = 'd'
        symbols_str = prompt_user_input("Enter symbols (comma-separated, e.g., btc,eth or BTC/USDT,ETH/USDT): ").strip()
        if symbols_str:
            normalized_symbols = parse_symbols_string(symbols_str)
            config['symbols'] = ",".join(normalized_symbols)
            print(color_text(f"Normalized symbols: {config['symbols']}", Fore.GREEN))
        else:
            config['symbols'] = ""
    
    # Account balance
    print("\n" + color_text("2. ACCOUNT BALANCE", Fore.WHITE))
    print(color_text("   a) Fetch from Binance (requires API credentials) [default]", Fore.MAGENTA, Style.BRIGHT))
    print("   b) Manual input")
    
    default_balance_text = color_text("default=a", Fore.MAGENTA)
    balance_choice = prompt_user_input(f"Select option (a/b, {default_balance_text}): ", default="a").strip().lower()
    
    # Default to 'a' if empty input
    if not balance_choice:
        balance_choice = 'a'
    
    if balance_choice == 'a':
        config['fetch_balance'] = True
        config['account_balance'] = None  # Will be fetched later
    else:
        balance_str = prompt_user_input("Enter account balance (USDT): ").strip()
        try:
            config['account_balance'] = float(balance_str)
        except ValueError:
            print(color_text(f"Invalid balance. Using default: 10000 USDT", Fore.YELLOW))
            config['account_balance'] = 10000.0
    
    # Backtest settings
    print("\n" + color_text("3. BACKTEST SETTINGS", Fore.WHITE))
    
    # Auto timeframe testing option
    print(color_text("   Auto timeframe testing: Try multiple timeframes automatically", Fore.CYAN))
    print("   a) Enable auto timeframe testing (tries 15m -> 30m -> 1h)")
    print(color_text("   b) Use specified timeframe only [default]", Fore.MAGENTA, Style.BRIGHT))
    auto_tf_choice = prompt_user_input(f"Select option (a/b, default=b): ", default="b").strip().lower()
    
    # Default to 'b' if empty input
    if not auto_tf_choice:
        auto_tf_choice = 'b'
    
    config['auto_timeframe'] = (auto_tf_choice == 'a')
    
    # Only ask for timeframe if auto timeframe is disabled
    if not config['auto_timeframe']:
        timeframe = prompt_user_input(f"Timeframe ({color_text(f'default: {DEFAULT_TIMEFRAME}', Fore.MAGENTA)}): ").strip()
        config['timeframe'] = timeframe if timeframe else DEFAULT_TIMEFRAME
    else:
        # Set a default timeframe (will be overridden by auto testing)
        config['timeframe'] = DEFAULT_TIMEFRAME
    
    lookback_str = prompt_user_input(f"Lookback days ({color_text(f'default: {DEFAULT_LOOKBACK_DAYS}', Fore.MAGENTA)}): ").strip()
    try:
        config['lookback_days'] = int(lookback_str) if lookback_str else DEFAULT_LOOKBACK_DAYS
    except ValueError:
        config['lookback_days'] = DEFAULT_LOOKBACK_DAYS
    
    # Position size constraints
    print("\n" + color_text("4. POSITION SIZE CONSTRAINTS", Fore.WHITE))
    print(f"   Note: Max position size is a fraction of account balance")
    print(f"   Example: 0.1 = 10% of account, 0.2 = 20% of account")
    default_max_size_text = color_text(f"default: {DEFAULT_MAX_POSITION_SIZE} = {DEFAULT_MAX_POSITION_SIZE*100:.0f}%", Fore.MAGENTA)
    max_size_str = prompt_user_input(f"Max position size (fraction, {default_max_size_text}): ").strip()
    try:
        config['max_position_size'] = float(max_size_str) if max_size_str else DEFAULT_MAX_POSITION_SIZE
    except ValueError:
        config['max_position_size'] = DEFAULT_MAX_POSITION_SIZE
    
    return config

