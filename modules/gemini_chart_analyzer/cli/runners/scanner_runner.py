"""Scanner runner for batch scanner."""

from typing import Any, List, Optional

from colorama import Fore

from modules.common.utils import color_text
from modules.gemini_chart_analyzer.cli.config.display import DisplayConfig
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner
from modules.gemini_chart_analyzer.core.scanner_types import BatchScanResult


def run_scanner(
    timeframe: Optional[str],
    timeframes: Optional[List[str]],
    max_symbols: Optional[int],
    limit: int,
    cooldown: float,
    pre_filtered_symbols: Optional[List[str]],
) -> BatchScanResult:
    """
    Run market batch scanner.

    Args:
        timeframe: Single timeframe (or None if multi-timeframe)
        timeframes: List of timeframes (or None if single timeframe)
        max_symbols: Maximum symbols to scan (or None for all)
        limit: Number of candles per symbol
        cooldown: Cooldown between batches in seconds
        pre_filtered_symbols: Optional list of pre-filtered symbols

    Returns:
        Dictionary containing scan results
    """
    scanner = MarketBatchScanner(cooldown_seconds=cooldown)

    results = scanner.scan_market(
        timeframe=timeframe,
        timeframes=timeframes,
        max_symbols=max_symbols,
        limit=limit,
        initial_symbols=pre_filtered_symbols,
    )

    return results


def _display_symbols_group(
    title: str,
    color: str,
    symbols: List[str],
    symbols_with_confidence: List[Any],
    config: DisplayConfig,
) -> None:
    """Helper to display a group of symbols with confidence bars."""
    print(color_text(f"{title} ({len(symbols)}):", color))
    if symbols_with_confidence:
        print("  (Sorted by confidence: High → Low)")
        for symbol, confidence in symbols_with_confidence:
            clamped_confidence = min(max(confidence, config.confidence_min), config.confidence_max)
            length = int(clamped_confidence * config.confidence_bar_length)
            confidence_bar = "█" * length
            print(f"  {symbol:{config.symbol_column_width}s} | Confidence: {confidence:.2f} {confidence_bar}")
    elif symbols:
        # Fallback if confidence data not available
        for i in range(0, len(symbols), config.symbols_per_row_fallback):
            row = symbols[i : i + config.symbols_per_row_fallback]
            print("  " + "  ".join(f"{s:{config.fallback_column_width}s}" for s in row))
    else:
        print("  None")


def display_scan_results(results: BatchScanResult) -> None:
    """
    Display scan results in formatted output.

    Args:
        results: BatchScanResult containing scan results
    """
    config = DisplayConfig()
    print()
    print(color_text("=" * config.divider_length, Fore.GREEN))
    print(color_text("SCAN RESULTS", Fore.GREEN))
    print(color_text("=" * config.divider_length, Fore.GREEN))
    print()

    # Display LONG signals
    _display_symbols_group(
        "LONG Signals",
        Fore.GREEN,
        results.long_symbols,
        results.long_symbols_with_confidence,
        config,
    )

    print()
    # Display SHORT signals
    _display_symbols_group(
        "SHORT Signals",
        Fore.RED,
        results.short_symbols,
        results.short_symbols_with_confidence,
        config,
    )

    # Display summary with average confidence
    if results.summary:
        summary = results.summary
        if summary.get("avg_long_confidence", 0) > 0:
            print()
            print(color_text("Summary:", Fore.CYAN))
            print(f"  Average LONG confidence: {summary['avg_long_confidence']:.2f}")
        if summary.get("avg_short_confidence", 0) > 0:
            print(f"  Average SHORT confidence: {summary['avg_short_confidence']:.2f}")

    print()
    print(color_text("=" * config.divider_length, Fore.GREEN))
    results_file = results.results_file or "N/A"
    print(color_text(f"Results saved to: {results_file}", Fore.GREEN))
    print(color_text("=" * config.divider_length, Fore.GREEN))
