"""
CLI entry point for Gemini Gann Square module.

Supports:
  - Command-line args mode: python -m modules.gemini_gann_square --symbol BTCUSDT --timeframe 4h
  - Interactive menu fallback: python -m modules.gemini_gann_square
"""

from __future__ import annotations

from .argument_parser import parse_args
from .interactive_menu import run_interactive_menu
from .runner import run_analysis


def main() -> None:
    """Entry point: parse args or fall back to interactive menu."""
    args = parse_args()

    # If symbol and timeframe provided via args → run directly
    if args.symbol and args.timeframe:
        run_analysis(
            symbol=args.symbol,
            timeframe=args.timeframe,
            limit=args.limit,
            lookback=args.lookback,
            output_dir=args.output_dir,
        )
    else:
        # Fall back to interactive menu
        run_interactive_menu()


if __name__ == "__main__":
    main()
