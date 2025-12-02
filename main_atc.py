"""
Adaptive Trend Classification (ATC) Main Program

Analyzes futures pairs on Binance using Adaptive Trend Classification:
- Fetches OHLCV data from Binance futures
- Calculates ATC signals using multiple moving averages
- Displays trend signals and analysis
"""

import warnings
import sys
import pandas as pd
import argparse
from typing import Optional

from modules.common.utils import configure_windows_stdio

# Fix encoding issues on Windows for interactive CLI runs only
configure_windows_stdio()

from colorama import Fore, Style, init as colorama_init

from modules.config import (
    DEFAULT_SYMBOL,
    DEFAULT_QUOTE,
    DEFAULT_TIMEFRAME,
    DEFAULT_LIMIT,
)
from modules.common.utils import (
    color_text,
    format_price,
    normalize_symbol,
    log_warn,
    log_error,
    log_info,
    log_analysis,
    log_data,
    log_success,
    log_progress,
)
from modules.common.ExchangeManager import ExchangeManager
from modules.common.DataFetcher import DataFetcher
from modules.adaptive_trend.atc import compute_atc_signals
from modules.adaptive_trend.layer1 import trend_sign, cut_signal

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


def parse_args():
    """Parse command-line arguments for ATC analysis."""
    parser = argparse.ArgumentParser(
        description="Adaptive Trend Classification (ATC) Analysis for Binance Futures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help=f"Symbol pair to analyze (default: {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--quote",
        type=str,
        default=DEFAULT_QUOTE,
        help=f"Quote currency (default: {DEFAULT_QUOTE})",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=DEFAULT_TIMEFRAME,
        help=f"Timeframe for analysis (default: {DEFAULT_TIMEFRAME})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of candles to fetch (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--ema-len",
        type=int,
        default=28,
        help="EMA length (default: 28)",
    )
    parser.add_argument(
        "--hma-len",
        type=int,
        default=28,
        help="HMA length (default: 28)",
    )
    parser.add_argument(
        "--wma-len",
        type=int,
        default=28,
        help="WMA length (default: 28)",
    )
    parser.add_argument(
        "--dema-len",
        type=int,
        default=28,
        help="DEMA length (default: 28)",
    )
    parser.add_argument(
        "--lsma-len",
        type=int,
        default=28,
        help="LSMA length (default: 28)",
    )
    parser.add_argument(
        "--kama-len",
        type=int,
        default=28,
        help="KAMA length (default: 28)",
    )
    parser.add_argument(
        "--robustness",
        type=str,
        choices=["Narrow", "Medium", "Wide"],
        default="Medium",
        help="Robustness setting (default: Medium)",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.02,
        dest="lambda_param",
        help="Lambda parameter for exponential growth (default: 0.02)",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.03,
        help="Decay rate (default: 0.03)",
    )
    parser.add_argument(
        "--cutout",
        type=int,
        default=0,
        help="Number of bars to skip at start (default: 0)",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive prompts",
    )
    parser.add_argument(
        "--no-menu",
        action="store_true",
        help="Disable interactive menu",
    )
    parser.add_argument(
        "--list-symbols",
        action="store_true",
        help="List available futures symbols and exit",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Maximum number of symbols to scan in auto mode",
    )
    parser.add_argument(
        "--min-signal",
        type=float,
        default=0.01,
        help="Minimum signal strength to display (default: 0.01)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Force auto mode (scan all symbols)",
    )

    return parser.parse_args()


def display_atc_signals(
    symbol: str,
    df: pd.DataFrame,
    atc_results: dict,
    current_price: float,
    exchange_label: str,
):
    """
    Display ATC signals and analysis results.

    Args:
        symbol: Symbol being analyzed
        df: DataFrame with OHLCV data
        atc_results: Dictionary with ATC signal results
        current_price: Current price
        exchange_label: Exchange name label
    """
    average_signal = atc_results.get("Average_Signal")
    if average_signal is None or len(average_signal) == 0:
        log_error("No ATC signals available")
        return

    # Get latest signal values
    latest_signal = average_signal.iloc[-1] if not average_signal.empty else 0.0
    latest_trend = trend_sign(average_signal)
    latest_trend_value = latest_trend.iloc[-1] if not latest_trend.empty else 0

    # Get individual MA signals
    ema_signal = atc_results.get("EMA_Signal", pd.Series())
    hma_signal = atc_results.get("HMA_Signal", pd.Series())
    wma_signal = atc_results.get("WMA_Signal", pd.Series())
    dema_signal = atc_results.get("DEMA_Signal", pd.Series())
    lsma_signal = atc_results.get("LSMA_Signal", pd.Series())
    kama_signal = atc_results.get("KAMA_Signal", pd.Series())

    # Get equity weights
    ema_s = atc_results.get("EMA_S", pd.Series())
    hma_s = atc_results.get("HMA_S", pd.Series())
    wma_s = atc_results.get("WMA_S", pd.Series())
    dema_s = atc_results.get("DEMA_S", pd.Series())
    lsma_s = atc_results.get("LSMA_S", pd.Series())
    kama_s = atc_results.get("KAMA_S", pd.Series())

    # Determine trend direction
    if latest_trend_value > 0:
        trend_direction = "BULLISH"
        trend_color = Fore.GREEN
    elif latest_trend_value < 0:
        trend_direction = "BEARISH"
        trend_color = Fore.RED
    else:
        trend_direction = "NEUTRAL"
        trend_color = Fore.YELLOW

    # Display header
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(
        color_text(
            f"ADAPTIVE TREND CLASSIFICATION (ATC) - {symbol} | {exchange_label}",
            Fore.CYAN,
            Style.BRIGHT,
        )
    )
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

    # Current price
    print(color_text(f"Current Price: {format_price(current_price)}", Fore.WHITE))
    print(color_text("-" * 80, Fore.CYAN))

    # Average Signal
    print(
        color_text(
            f"Average Signal: {latest_signal:.4f}",
            trend_color,
            Style.BRIGHT,
        )
    )
    print(
        color_text(
            f"Trend Direction: {trend_direction}",
            trend_color,
            Style.BRIGHT,
        )
    )
    print(color_text("-" * 80, Fore.CYAN))

    # Individual MA Signals
    print(color_text("Individual MA Signals:", Fore.MAGENTA, Style.BRIGHT))
    ma_signals = [
        ("EMA", ema_signal),
        ("HMA", hma_signal),
        ("WMA", wma_signal),
        ("DEMA", dema_signal),
        ("LSMA", lsma_signal),
        ("KAMA", kama_signal),
    ]

    for ma_name, ma_sig in ma_signals:
        if not ma_sig.empty:
            latest_ma_sig = ma_sig.iloc[-1]
            ma_trend = trend_sign(ma_sig)
            ma_trend_value = ma_trend.iloc[-1] if not ma_trend.empty else 0

            if ma_trend_value > 0:
                ma_color = Fore.GREEN
                ma_dir = "^"
            elif ma_trend_value < 0:
                ma_color = Fore.RED
                ma_dir = "v"
            else:
                ma_color = Fore.YELLOW
                ma_dir = "-"

            print(
                color_text(
                    f"  {ma_name:6s}: {latest_ma_sig:8.4f} {ma_dir}",
                    ma_color,
                )
            )

    print(color_text("-" * 80, Fore.CYAN))

    # Equity Weights (Layer 2)
    print(color_text("Equity Weights (Layer 2):", Fore.MAGENTA, Style.BRIGHT))
    ma_weights = [
        ("EMA", ema_s),
        ("HMA", hma_s),
        ("WMA", wma_s),
        ("DEMA", dema_s),
        ("LSMA", lsma_s),
        ("KAMA", kama_s),
    ]

    for ma_name, ma_weight in ma_weights:
        if not ma_weight.empty:
            latest_weight = ma_weight.iloc[-1]
            if pd.notna(latest_weight):
                print(
                    color_text(
                        f"  {ma_name:6s}: {latest_weight:8.4f}",
                        Fore.WHITE,
                    )
                )

    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))


def analyze_symbol(
    symbol: str,
    data_fetcher: DataFetcher,
    timeframe: str,
    limit: int,
    ema_len: int,
    hma_len: int,
    wma_len: int,
    dema_len: int,
    lsma_len: int,
    kama_len: int,
    robustness: str,
    lambda_param: float,
    decay: float,
    cutout: int,
):
    """
    Analyze a single symbol using ATC.

    Args:
        symbol: Symbol to analyze
        data_fetcher: DataFetcher instance
        timeframe: Timeframe for data
        limit: Number of candles
        ema_len: EMA length
        hma_len: HMA length
        wma_len: WMA length
        dema_len: DEMA length
        lsma_len: LSMA length
        kama_len: KAMA length
        robustness: Robustness setting
        lambda_param: Lambda parameter
        decay: Decay rate
        cutout: Cutout period

    Returns:
        bool: True if analysis succeeded, False otherwise
    """
    try:
        # Fetch OHLCV data
        df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=limit,
            timeframe=timeframe,
            check_freshness=True,
        )

        if df is None or df.empty:
            log_error(f"No data available for {symbol}")
            return False

        exchange_label = exchange_id.upper() if exchange_id else "UNKNOWN"

        # Get close prices
        if "close" not in df.columns:
            log_error(f"No 'close' column in data for {symbol}")
            return False

        close_prices = df["close"]
        current_price = close_prices.iloc[-1]

        # Calculate ATC signals
        log_progress(f"Calculating ATC signals for {symbol}...")

        atc_results = compute_atc_signals(
            prices=close_prices,
            src=None,  # Use close prices as source
            ema_len=ema_len,
            hull_len=hma_len,
            wma_len=wma_len,
            dema_len=dema_len,
            lsma_len=lsma_len,
            kama_len=kama_len,
            ema_w=1.0,
            hma_w=1.0,
            wma_w=1.0,
            dema_w=1.0,
            lsma_w=1.0,
            kama_w=1.0,
            robustness=robustness,
            La=lambda_param,
            De=decay,
            cutout=cutout,
        )

        # Display results
        display_atc_signals(
            symbol=symbol,
            df=df,
            atc_results=atc_results,
            current_price=current_price,
            exchange_label=exchange_label,
        )

        return True

    except Exception as e:
        log_error(f"Error analyzing {symbol}: {type(e).__name__}: {e}")
        import traceback

        log_error(f"Traceback: {traceback.format_exc()}")
        return False


def prompt_interactive_mode() -> dict:
    """
    Interactive menu for selecting analysis mode.
    
    Returns:
        dict with 'mode' key ('auto' or 'manual')
    """
    log_data("=" * 60)
    log_info("Adaptive Trend Classification (ATC) - Interactive Launcher")
    log_data("=" * 60)
    print(
        color_text(
            "1) Auto mode  - scan entire market for LONG/SHORT signals",
            Fore.MAGENTA,
            Style.BRIGHT,
        )
    )
    print("2) Manual mode - analyze specific symbol")
    print("3) Exit")

    while True:
        choice = input(color_text("\nSelect option [1-3] (default 1): ", Fore.YELLOW)).strip() or "1"
        if choice in {"1", "2", "3"}:
            break
        log_error("Invalid choice. Please enter 1, 2, or 3.")

    if choice == "3":
        log_warn("Exiting by user request.")
        sys.exit(0)

    return {"mode": "auto" if choice == "1" else "manual"}


def scan_all_symbols(
    data_fetcher: DataFetcher,
    timeframe: str,
    limit: int,
    ema_len: int,
    hma_len: int,
    wma_len: int,
    dema_len: int,
    lsma_len: int,
    kama_len: int,
    robustness: str,
    lambda_param: float,
    decay: float,
    cutout: int,
    max_symbols: Optional[int] = None,
    min_signal: float = 0.01,
):
    """
    Scan all futures symbols and filter those with LONG/SHORT signals.
    
    Args:
        data_fetcher: DataFetcher instance
        timeframe: Timeframe for data
        limit: Number of candles
        ema_len: EMA length
        hma_len: HMA length
        wma_len: WMA length
        dema_len: DEMA length
        lsma_len: LSMA length
        kama_len: KAMA length
        robustness: Robustness setting
        lambda_param: Lambda parameter
        decay: Decay rate
        cutout: Cutout period
        max_symbols: Maximum number of symbols to scan
        min_signal: Minimum signal strength to display
        
    Returns:
        tuple: (long_signals_df, short_signals_df) DataFrames with symbol, signal, price
    """
    try:
        log_progress("Fetching futures symbols from Binance...")
        all_symbols = data_fetcher.list_binance_futures_symbols(
            max_candidates=None,  # Get all symbols first
            progress_label="Symbol Discovery",
        )

        if not all_symbols:
            log_error("No symbols found")
            return pd.DataFrame(), pd.DataFrame()

        # Limit symbols if max_symbols specified
        if max_symbols and max_symbols > 0:
            symbols = all_symbols[:max_symbols]
            log_success(f"Found {len(all_symbols)} futures symbols, scanning first {len(symbols)} symbols")
        else:
            symbols = all_symbols
            log_success(f"Found {len(symbols)} futures symbols")
        
        log_progress(f"Scanning {len(symbols)} symbols for ATC signals...")

        results = []
        total = len(symbols)
        
        for idx, symbol in enumerate(symbols, 1):
            try:
                # Fetch OHLCV data
                df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                    symbol,
                    limit=limit,
                    timeframe=timeframe,
                    check_freshness=True,
                )

                if df is None or df.empty or "close" not in df.columns:
                    continue

                close_prices = df["close"]
                current_price = close_prices.iloc[-1]

                # Calculate ATC signals
                atc_results = compute_atc_signals(
                    prices=close_prices,
                    src=None,
                    ema_len=ema_len,
                    hull_len=hma_len,
                    wma_len=wma_len,
                    dema_len=dema_len,
                    lsma_len=lsma_len,
                    kama_len=kama_len,
                    ema_w=1.0,
                    hma_w=1.0,
                    wma_w=1.0,
                    dema_w=1.0,
                    lsma_w=1.0,
                    kama_w=1.0,
                    robustness=robustness,
                    La=lambda_param,
                    De=decay,
                    cutout=cutout,
                )

                average_signal = atc_results.get("Average_Signal")
                if average_signal is None or average_signal.empty:
                    continue

                latest_signal = average_signal.iloc[-1]
                latest_trend = trend_sign(average_signal)
                latest_trend_value = latest_trend.iloc[-1] if not latest_trend.empty else 0

                # Only include signals above threshold
                if abs(latest_signal) < min_signal:
                    continue

                results.append({
                    "symbol": symbol,
                    "signal": latest_signal,
                    "trend": latest_trend_value,
                    "price": current_price,
                    "exchange": exchange_id or "UNKNOWN",
                })

                # Progress update every 10 symbols
                if idx % 10 == 0 or idx == total:
                    log_progress(f"Scanned {idx}/{total} symbols... Found {len(results)} signals")

            except KeyboardInterrupt:
                log_warn("Scan interrupted by user")
                break
            except Exception as e:
                # Skip symbols with errors, continue scanning
                continue

        if not results:
            log_warn("No signals found above threshold")
            return pd.DataFrame(), pd.DataFrame()

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Filter LONG and SHORT signals
        long_signals = results_df[results_df["trend"] > 0].copy()
        short_signals = results_df[results_df["trend"] < 0].copy()

        # Sort by signal strength (absolute value)
        long_signals = long_signals.sort_values("signal", ascending=False).reset_index(drop=True)
        short_signals = short_signals.sort_values("signal", ascending=True).reset_index(drop=True)

        return long_signals, short_signals

    except Exception as e:
        log_error(f"Error scanning symbols: {type(e).__name__}: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), pd.DataFrame()


def display_scan_results(long_signals: pd.DataFrame, short_signals: pd.DataFrame, min_signal: float):
    """
    Display scan results for LONG and SHORT signals.
    
    Args:
        long_signals: DataFrame with LONG signals
        short_signals: DataFrame with SHORT signals
        min_signal: Minimum signal threshold used
    """
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("ATC SIGNAL SCAN RESULTS", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

    # LONG Signals
    print("\n" + color_text("LONG SIGNALS (BULLISH)", Fore.GREEN, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    
    if long_signals.empty:
        print(color_text("  No LONG signals found", Fore.YELLOW))
    else:
        print(color_text(f"  Found {len(long_signals)} symbols with LONG signals (min: {min_signal:.4f})", Fore.WHITE))
        print()
        print(color_text(f"{'Symbol':<15} {'Signal':>12} {'Price':>15} {'Exchange':<10}", Fore.MAGENTA))
        print(color_text("-" * 80, Fore.CYAN))
        
        for _, row in long_signals.iterrows():
            signal_str = f"{row['signal']:+.6f}"
            price_str = format_price(row['price'])
            print(
                color_text(
                    f"{row['symbol']:<15} {signal_str:>12} {price_str:>15} {row['exchange']:<10}",
                    Fore.GREEN,
                )
            )

    # SHORT Signals
    print("\n" + color_text("SHORT SIGNALS (BEARISH)", Fore.RED, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    
    if short_signals.empty:
        print(color_text("  No SHORT signals found", Fore.YELLOW))
    else:
        print(color_text(f"  Found {len(short_signals)} symbols with SHORT signals (min: {min_signal:.4f})", Fore.WHITE))
        print()
        print(color_text(f"{'Symbol':<15} {'Signal':>12} {'Price':>15} {'Exchange':<10}", Fore.MAGENTA))
        print(color_text("-" * 80, Fore.CYAN))
        
        for _, row in short_signals.iterrows():
            signal_str = f"{row['signal']:+.6f}"
            price_str = format_price(row['price'])
            print(
                color_text(
                    f"{row['symbol']:<15} {signal_str:>12} {price_str:>15} {row['exchange']:<10}",
                    Fore.RED,
                )
            )

    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text(f"Total: {len(long_signals)} LONG + {len(short_signals)} SHORT = {len(long_signals) + len(short_signals)} signals", Fore.WHITE))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))


def list_futures_symbols(data_fetcher: DataFetcher, max_symbols: Optional[int] = None):
    """
    List available futures symbols from Binance.

    Args:
        data_fetcher: DataFetcher instance
        max_symbols: Maximum number of symbols to display
    """
    try:
        log_progress("Fetching futures symbols from Binance...")
        symbols = data_fetcher.list_binance_futures_symbols(
            max_candidates=max_symbols,
            progress_label="Symbol Discovery",
        )

        if not symbols:
            log_error("No symbols found")
            return

        log_success(f"Found {len(symbols)} futures symbols")

        print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
        print(color_text("AVAILABLE FUTURES SYMBOLS", Fore.CYAN, Style.BRIGHT))
        print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

        # Display symbols in columns
        cols = 4
        for i in range(0, len(symbols), cols):
            row_symbols = symbols[i : i + cols]
            row_text = "  ".join(f"{sym:15s}" for sym in row_symbols)
            print(color_text(row_text, Fore.WHITE))

        print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

    except Exception as e:
        log_error(f"Error listing symbols: {type(e).__name__}: {e}")


def main() -> None:
    """
    Main function for ATC analysis.

    Orchestrates the complete ATC analysis workflow:
    1. Parse command-line arguments
    2. Initialize components (ExchangeManager, DataFetcher)
    3. Fetch OHLCV data from Binance futures
    4. Calculate ATC signals
    5. Display results
    """
    args = parse_args()

    # List symbols if requested
    if args.list_symbols:
        exchange_manager = ExchangeManager()
        data_fetcher = DataFetcher(exchange_manager)
        list_futures_symbols(data_fetcher)
        return

    # Initialize components
    log_progress("Initializing components...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)

    # Interactive menu
    mode = "manual"
    if args.auto:
        mode = "auto"
    elif not args.no_menu:
        menu_result = prompt_interactive_mode()
        mode = menu_result["mode"]

    # Auto mode: scan all symbols
    if mode == "auto":
        # Display configuration
        if log_analysis:
            log_analysis("=" * 80)
            log_analysis("ADAPTIVE TREND CLASSIFICATION (ATC) - AUTO SCAN MODE")
            log_analysis("=" * 80)
            log_analysis("Configuration:")
        if log_data:
            log_data(f"  Mode: AUTO (scan all symbols)")
            log_data(f"  Timeframe: {args.timeframe}")
            log_data(f"  Limit: {args.limit} candles")
            log_data(f"  Robustness: {args.robustness}")
            log_data(f"  MA Lengths: EMA={args.ema_len}, HMA={args.hma_len}, WMA={args.wma_len}, DEMA={args.dema_len}, LSMA={args.lsma_len}, KAMA={args.kama_len}")
            log_data(f"  Lambda: {args.lambda_param}, Decay: {args.decay}, Cutout: {args.cutout}")
            log_data(f"  Min Signal: {args.min_signal}")
            if args.max_symbols:
                log_data(f"  Max Symbols: {args.max_symbols}")

        # Scan all symbols
        long_signals, short_signals = scan_all_symbols(
            data_fetcher=data_fetcher,
            timeframe=args.timeframe,
            limit=args.limit,
            ema_len=args.ema_len,
            hma_len=args.hma_len,
            wma_len=args.wma_len,
            dema_len=args.dema_len,
            lsma_len=args.lsma_len,
            kama_len=args.kama_len,
            robustness=args.robustness,
            lambda_param=args.lambda_param,
            decay=args.decay,
            cutout=args.cutout,
            max_symbols=args.max_symbols,
            min_signal=args.min_signal,
        )

        # Display results
        display_scan_results(long_signals, short_signals, args.min_signal)
        return

    # Manual mode: analyze specific symbol
    # Get symbol input
    quote = args.quote.upper() if args.quote else DEFAULT_QUOTE
    symbol_input = args.symbol

    if not symbol_input and not args.no_prompt:
        symbol_input = input(
            color_text(
                f"Enter symbol pair (default: {DEFAULT_SYMBOL}): ",
                Fore.YELLOW,
            )
        ).strip()
        if not symbol_input:
            symbol_input = DEFAULT_SYMBOL

    if not symbol_input:
        symbol_input = DEFAULT_SYMBOL

    symbol = normalize_symbol(symbol_input, quote)

    # Display configuration
    if log_analysis:
        log_analysis("=" * 80)
        log_analysis("ADAPTIVE TREND CLASSIFICATION (ATC) ANALYSIS")
        log_analysis("=" * 80)
        log_analysis("Configuration:")
    if log_data:
        log_data(f"  Symbol: {symbol}")
        log_data(f"  Timeframe: {args.timeframe}")
        log_data(f"  Limit: {args.limit} candles")
        log_data(f"  Robustness: {args.robustness}")
        log_data(f"  MA Lengths: EMA={args.ema_len}, HMA={args.hma_len}, WMA={args.wma_len}, DEMA={args.dema_len}, LSMA={args.lsma_len}, KAMA={args.kama_len}")
        log_data(f"  Lambda: {args.lambda_param}, Decay: {args.decay}, Cutout: {args.cutout}")

    # Analyze symbol
    success = analyze_symbol(
        symbol=symbol,
        data_fetcher=data_fetcher,
        timeframe=args.timeframe,
        limit=args.limit,
        ema_len=args.ema_len,
        hma_len=args.hma_len,
        wma_len=args.wma_len,
        dema_len=args.dema_len,
        lsma_len=args.lsma_len,
        kama_len=args.kama_len,
        robustness=args.robustness,
        lambda_param=args.lambda_param,
        decay=args.decay,
        cutout=args.cutout,
    )

    if not success:
        log_error("Analysis failed")
        return

    # Interactive loop if prompts enabled
    if not args.no_prompt:
        try:
            while True:
                print(
                    color_text(
                        "\nPress Ctrl+C to exit. Provide a new symbol to continue.",
                        Fore.YELLOW,
                    )
                )
                symbol_input = input(
                    color_text(
                        f"Enter symbol pair (default: {symbol}): ",
                        Fore.YELLOW,
                    )
                ).strip()

                if not symbol_input:
                    symbol_input = symbol

                symbol = normalize_symbol(symbol_input, quote)

                analyze_symbol(
                    symbol=symbol,
                    data_fetcher=data_fetcher,
                    timeframe=args.timeframe,
                    limit=args.limit,
                    ema_len=args.ema_len,
                    hma_len=args.hma_len,
                    wma_len=args.wma_len,
                    dema_len=args.dema_len,
                    lsma_len=args.lsma_len,
                    kama_len=args.kama_len,
                    robustness=args.robustness,
                    lambda_param=args.lambda_param,
                    decay=args.decay,
                    cutout=args.cutout,
                )
        except KeyboardInterrupt:
            print(color_text("\nExiting program by user request.", Fore.YELLOW))


if __name__ == "__main__":
    main()

