"""Diagnostic script to investigate why all 20 symbols are generating signals.

This script analyzes:
1. Final signal values (are they actually non-zero?)
2. Signal persistence (how long since last crossover?)
3. Threshold sensitivity (how close are signals to thresholds?)
4. Distribution of signals across symbols
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from tabulate import tabulate  # type: ignore

from modules.adaptive_trend_LTS.benchmarks.benchmark_comparison.data import fetch_symbols_data
from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals
from modules.common.utils import log_error, log_info, log_success, log_warn


def analyze_signal_characteristics(symbol: str, result: dict, prices: pd.Series) -> dict:
    """Analyze characteristics of a signal to diagnose overfitting.

    Args:
        symbol: Symbol name
        result: ATC result dictionary
        prices: Price series

    Returns:
        Dictionary with diagnostic metrics
    """
    avg_signal = result.get("Average_Signal")

    if avg_signal is None or len(avg_signal) == 0:
        return {
            "symbol": symbol,
            "has_signal": False,
            "final_value": None,
            "signal_type": "NONE",
            "bars_since_change": None,
            "signal_strength": None,
            "non_zero_bars": 0,
            "zero_bars": 0,
        }

    # Get final signal value
    final_value = avg_signal.iloc[-1] if len(avg_signal) > 0 else None

    # Determine signal type based on final value
    if pd.isna(final_value):
        signal_type = "NaN"
    elif abs(final_value) < 0.01:  # Near zero
        signal_type = "NEUTRAL"
    elif final_value > 0.1:
        signal_type = "LONG"
    elif final_value < -0.1:
        signal_type = "SHORT"
    else:
        signal_type = "WEAK"

    # Find when signal last changed
    signal_values = avg_signal.values
    bars_since_change = 0

    if len(signal_values) > 1:
        for i in range(len(signal_values) - 1, 0, -1):
            if not np.isnan(signal_values[i]) and not np.isnan(signal_values[i - 1]):
                # Check if signal direction changed
                if abs(signal_values[i] - signal_values[i - 1]) > 0.01:
                    bars_since_change = len(signal_values) - 1 - i
                    break
        else:
            bars_since_change = len(signal_values) - 1  # No change found

    # Count non-zero vs zero bars
    valid_signals = signal_values[~np.isnan(signal_values)]
    non_zero_bars = np.sum(np.abs(valid_signals) > 0.01)
    zero_bars = np.sum(np.abs(valid_signals) <= 0.01)

    # Signal strength (distance from neutral)
    signal_strength = abs(final_value) if not pd.isna(final_value) else 0

    return {
        "symbol": symbol,
        "has_signal": signal_type != "NEUTRAL" and signal_type != "NaN",
        "final_value": final_value,
        "signal_type": signal_type,
        "bars_since_change": bars_since_change,
        "signal_strength": signal_strength,
        "non_zero_bars": int(non_zero_bars),
        "zero_bars": int(zero_bars),
    }


def main():
    """Run diagnostic analysis on benchmark signals."""
    log_info("=" * 80)
    log_info("SIGNAL OVERFITTING DIAGNOSTIC")
    log_info("=" * 80)

    # Fetch same data as benchmark
    log_info("Fetching 20 symbols with 500 bars (1h timeframe)...")
    prices_data = fetch_symbols_data(num_symbols=20, bars=500, timeframe="1h")

    if len(prices_data) == 0:
        log_error("No data fetched, exiting")
        return

    log_success(f"Fetched {len(prices_data)} symbols")

    # Common config (same as benchmark)
    config = {
        "ema_len": 28,
        "hma_len": 28,
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "ema_w": 1.0,
        "hma_w": 1.0,
        "wma_w": 1.0,
        "dema_w": 1.0,
        "lsma_w": 1.0,
        "kama_w": 1.0,
        "robustness": "Medium",
        "La": 0.02,
        "De": 0.03,
        "cutout": 0,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
        "strategy_mode": False,
    }

    log_info("Computing ATC signals for all symbols...")

    diagnostics = []

    for idx, (symbol, prices) in enumerate(prices_data.items(), 1):
        try:
            result = compute_atc_signals(prices=prices, **config)
            diag = analyze_signal_characteristics(symbol, result, prices)
            diagnostics.append(diag)

            if idx % 5 == 0:
                log_info(f"Processed {idx}/{len(prices_data)} symbols")

        except Exception as e:
            log_error(f"Error processing {symbol}: {e}")
            diagnostics.append(
                {
                    "symbol": symbol,
                    "has_signal": False,
                    "final_value": None,
                    "signal_type": "ERROR",
                    "bars_since_change": None,
                    "signal_strength": None,
                    "non_zero_bars": 0,
                    "zero_bars": 0,
                }
            )

    # Create summary table
    log_info("\n" + "=" * 80)
    log_info("DIAGNOSTIC RESULTS")
    log_info("=" * 80)

    # Overall statistics
    total_symbols = len(diagnostics)
    symbols_with_signal = sum(1 for d in diagnostics if d["has_signal"])

    signal_types = {}
    for d in diagnostics:
        signal_type = d["signal_type"]
        signal_types[signal_type] = signal_types.get(signal_type, 0) + 1

    print("\n### OVERALL STATISTICS ###")
    print(f"Total symbols: {total_symbols}")
    print(
        f"Symbols with signal (non-neutral): {symbols_with_signal} ({symbols_with_signal / total_symbols * 100:.1f}%)"
    )
    print("\nSignal Type Distribution:")
    for sig_type, count in sorted(signal_types.items()):
        print(f"  {sig_type}: {count} ({count / total_symbols * 100:.1f}%)")

    # Detailed table
    print("\n### DETAILED SIGNAL ANALYSIS ###")
    table_data = []
    for d in diagnostics:
        table_data.append(
            [
                d["symbol"][:15],  # Truncate long symbols
                d["signal_type"],
                f"{d['final_value']:.4f}" if d["final_value"] is not None else "N/A",
                f"{d['signal_strength']:.4f}" if d["signal_strength"] is not None else "N/A",
                d["bars_since_change"] if d["bars_since_change"] is not None else "N/A",
                f"{d['non_zero_bars']}/{d['zero_bars']}",
            ]
        )

    headers = ["Symbol", "Type", "Final Value", "Strength", "Bars Since Change", "NonZero/Zero"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Analyze stale signals
    stale_threshold = 50  # Consider signal stale if no change in 50+ bars
    stale_signals = [d for d in diagnostics if d["bars_since_change"] and d["bars_since_change"] > stale_threshold]

    print("\n### STALE SIGNAL ANALYSIS ###")
    print(f"Signals with no change in {stale_threshold}+ bars: {len(stale_signals)}")
    if stale_signals:
        print("\nStale signals may indicate overfitting (old crossovers persisting):")
        for d in stale_signals[:5]:  # Show first 5
            print(f"  - {d['symbol']}: {d['bars_since_change']} bars since change ({d['signal_type']})")

    # Threshold sensitivity analysis
    print("\n### THRESHOLD SENSITIVITY ###")
    print("Current thresholds: LONG > 0.1, SHORT < -0.1")

    weak_signals = [d for d in diagnostics if d["signal_strength"] and 0.05 < d["signal_strength"] < 0.15]
    print(f"Signals near threshold (0.05-0.15): {len(weak_signals)}")
    print("These signals are sensitive to threshold changes and may be unreliable.")

    # Recommendations
    print("\n### RECOMMENDATIONS ###")

    if symbols_with_signal > total_symbols * 0.8:
        log_warn("⚠️  HIGH SIGNAL RATE DETECTED (>80%)")
        print("This suggests potential overfitting. Consider:")
        print("  1. Increase thresholds (e.g., ±0.2 or ±0.3)")
        print("  2. Add recency filter (ignore signals older than N bars)")
        print("  3. Add signal strength filter (ignore weak signals)")

    if stale_signals and len(stale_signals) > total_symbols * 0.5:
        log_warn("⚠️  MANY STALE SIGNALS DETECTED (>50%)")
        print("Signal persistence is causing old crossovers to be counted as active signals.")
        print("Consider adding a recency requirement in the benchmark comparison.")

    if weak_signals and len(weak_signals) > total_symbols * 0.3:
        log_warn("⚠️  MANY WEAK SIGNALS DETECTED (>30%)")
        print("Thresholds may be too permissive.")
        print("Consider tightening long_threshold and short_threshold.")

    log_success("\nDiagnostic analysis complete!")


if __name__ == "__main__":
    main()
