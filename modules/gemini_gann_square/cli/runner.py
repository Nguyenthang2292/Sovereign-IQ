"""
Runner — fetch OHLCV data and run the full Gann analysis pipeline.
"""

from __future__ import annotations

from typing import Optional

from modules.common.ui.logging import log_error, log_info

from ..core.gann_signal_engine import GannSignalEngine


def run_analysis(
    symbol: str,
    timeframe: str,
    limit: int = 200,
    lookback: int = 5,
    output_dir: str = "charts",
    gemini_api_key: Optional[str] = None,
) -> None:
    """
    Run the full Gann + Gemini analysis.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT' or 'BTCUSDT').
        timeframe: Candle timeframe (e.g., '4h').
        limit: Number of candles to fetch.
        lookback: Zigzag pivot lookback window.
        output_dir: Directory to save chart PNGs.
        gemini_api_key: Optional Gemini API key override.
    """
    log_info(f"Starting Gann analysis: {symbol} {timeframe}")

    try:
        # Run Gann + Gemini analysis
        engine = GannSignalEngine(
            lookback=lookback,
            gemini_api_key=gemini_api_key,
            chart_output_dir=output_dir,
        )
        result = engine.analyze(symbol=symbol, timeframe=timeframe, limit=limit)

        # Print result to terminal
        print(result.display())

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
    except Exception as e:
        log_error(f"Analysis failed: {e}")
        raise
