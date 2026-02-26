"""
Interactive menu for Gemini Gann Square CLI.

Displayed when no --symbol / --timeframe args are provided.
"""

from __future__ import annotations

from .runner import run_analysis

_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"]
_DEFAULT_SYMBOL = "BTC/USDT"
_DEFAULT_TIMEFRAME = "4h"
_DEFAULT_LIMIT = 200
_DEFAULT_LOOKBACK = 5


def run_interactive_menu() -> None:
    """Run the interactive CLI menu."""
    _print_banner()

    symbol = _DEFAULT_SYMBOL
    timeframe = _DEFAULT_TIMEFRAME
    limit = _DEFAULT_LIMIT
    lookback = _DEFAULT_LOOKBACK
    output_dir = "charts"

    while True:
        _print_main_menu(symbol, timeframe, limit, lookback)
        choice = input("  Select option: ").strip()

        if choice == "1":
            _run_now(symbol, timeframe, limit, lookback, output_dir)

        elif choice == "2":
            symbol = _ask_symbol(symbol)
            timeframe = _ask_timeframe(timeframe)
            limit = _ask_int("  Number of candles", limit, min_val=50, max_val=2000)
            lookback = _ask_int("  Zigzag lookback window (candles each side)", lookback, min_val=2, max_val=20)

        elif choice == "3":
            print("\n  Goodbye! 👋\n")
            break

        else:
            print("  ⚠  Invalid option. Please enter 1-3.\n")


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────


def _print_banner() -> None:
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║     GEMINI GANN SQUARE ANALYZER      ║")
    print("  ║  Gann Theory + Google Gemini AI      ║")
    print("  ╚══════════════════════════════════════╝")
    print()


def _print_main_menu(symbol: str, timeframe: str, limit: int, lookback: int) -> None:
    print("  ┌──────────────────────────────────────┐")
    print(f"  │  Symbol    : {symbol:<24}│")
    print(f"  │  Timeframe : {timeframe:<24}│")
    print(f"  │  Candles   : {limit:<24}│")
    print(f"  │  Lookback  : {lookback:<24}│")
    print("  ├──────────────────────────────────────┤")
    print("  │  1. ▶  Run Analysis                  │")
    print("  │  2.    Change Settings               │")
    print("  │  3.    Exit                          │")
    print("  └──────────────────────────────────────┘")


def _run_now(symbol: str, timeframe: str, limit: int, lookback: int, output_dir: str) -> None:
    print()
    try:
        run_analysis(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            lookback=lookback,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"\n  ❌ Error: {e}\n")
    print()


def _ask_symbol(current: str) -> str:
    val = input(f"  Symbol (e.g. BTC/USDT, ETHUSDT) [{current}]: ").strip()
    return val if val else current


def _ask_timeframe(current: str) -> str:
    print(f"  Available timeframes: {', '.join(_TIMEFRAMES)}")
    val = input(f"  Timeframe [{current}]: ").strip()
    if val and val in _TIMEFRAMES:
        return val
    if val:
        print(f"  ⚠  '{val}' not in list, keeping '{current}'.")
    return current


def _ask_int(prompt: str, current: int, min_val: int = 1, max_val: int = 9999) -> int:
    val_str = input(f"  {prompt} [{current}]: ").strip()
    if not val_str:
        return current
    try:
        val = int(val_str)
        if min_val <= val <= max_val:
            return val
        print(f"  ⚠  Value must be {min_val}–{max_val}. Keeping {current}.")
    except ValueError:
        print(f"  ⚠  Invalid number. Keeping {current}.")
    return current
