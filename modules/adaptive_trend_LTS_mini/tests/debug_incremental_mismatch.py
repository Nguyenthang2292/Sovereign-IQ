"""
Debug tool to investigate incremental vs full calculation mismatch.

This script traces calculations step-by-step to identify where the 17% discrepancy occurs.
"""

import numpy as np
import pandas as pd
from modules.adaptive_trend_LTS.core.compute_atc_signals import IncrementalATC, compute_atc_signals


def compare_calculations_detailed():
    """Compare incremental and full calculations with detailed tracing."""

    # Use same data as failing test
    np.random.seed(42)
    base_price = 100.0
    n = 200
    returns = np.random.normal(0.001, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)
    sample_prices = pd.Series(prices, index=range(n))

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
        "La": 0.02,
        "De": 0.03,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
        "cutout": 0,
        "strategy_mode": False,
    }

    print("=" * 80)
    print("INCREMENTAL VS FULL CALCULATION DEBUG")
    print("=" * 80)

    # Initialize incremental with first N-10 prices
    init_prices = sample_prices[:-10]
    atc = IncrementalATC(config)
    init_results = atc.initialize(init_prices)

    print(f"\n📊 INITIALIZATION")
    print(f"Init prices: {len(init_prices)} bars")
    print(f"Last init signal: {init_results['Average_Signal'].iloc[-1]:.6f}")

    # Update incrementally with remaining 10 prices
    print(f"\n📈 INCREMENTAL UPDATES (last 10 bars)")
    incremental_signals = []

    for i, price in enumerate(sample_prices[-10:]):
        # Capture state before update
        before_state = {
            "bar_index": atc.state.get("bar_index"),
            "ma_values": {k: v.copy() if isinstance(v, list) else v for k, v in atc.state["ma_values"].items()},
            "equity": atc.state["equity"].copy() if atc.state["equity"] else None,
            "layer1_signals": atc.state.get("layer1_signals", {}).copy(),
            "average_signal": atc.state.get("average_signal"),
        }

        signal = atc.update(price)
        incremental_signals.append(signal)

        # Capture state after update
        after_state = {
            "bar_index": atc.state.get("bar_index"),
            "ma_values": {k: v.copy() if isinstance(v, list) else v for k, v in atc.state["ma_values"].items()},
            "equity": atc.state["equity"].copy() if atc.state["equity"] else None,
            "layer1_signals": atc.state.get("layer1_signals", {}).copy(),
            "average_signal": atc.state.get("average_signal"),
        }

        print(f"\n  Bar {len(init_prices) + i}: price={price:.4f}, signal={signal:.6f}")
        print(f"    Bar index: {before_state['bar_index']} → {after_state['bar_index']}")
        print(f"    Avg signal: {before_state['average_signal']:.6f} → {after_state['average_signal']:.6f}")

        # Print Layer 1 signals
        print(f"    Layer1 signals:")
        for ma_type in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
            before_l1 = before_state["layer1_signals"].get(ma_type, 0.0)
            after_l1 = after_state["layer1_signals"].get(ma_type, 0.0)
            print(f"      {ma_type}: {before_l1:+.6f} → {after_l1:+.6f}")

    # Full recalculation on all prices
    print(f"\n🔄 FULL CALCULATION")
    full_results = compute_atc_signals(sample_prices, **config)
    full_signals = full_results["Average_Signal"].iloc[-10:].values

    print(f"Last 10 signals from full calculation:")
    for i, sig in enumerate(full_signals):
        print(f"  Bar {len(sample_prices) - 10 + i}: {sig:.6f}")

    # Compare
    print(f"\n⚖️  COMPARISON")
    print(f"{'Bar':<6} {'Incremental':<15} {'Full':<15} {'Diff':<15} {'% Error':<10}")
    print("-" * 70)

    max_diff = 0
    max_diff_idx = -1

    for i, (inc, full) in enumerate(zip(incremental_signals, full_signals)):
        diff = inc - full
        pct_error = abs(diff / full * 100) if full != 0 else 0

        if abs(diff) > max_diff:
            max_diff = abs(diff)
            max_diff_idx = i

        marker = "❌" if abs(diff) > 0.001 else "✅"
        print(f"{len(init_prices) + i:<6} {inc:<15.6f} {full:<15.6f} {diff:<15.6f} {pct_error:<10.2f}% {marker}")

    print(f"\n📍 MAXIMUM DIFFERENCE:")
    print(f"  Bar: {len(init_prices) + max_diff_idx}")
    print(f"  Incremental: {incremental_signals[max_diff_idx]:.6f}")
    print(f"  Full: {full_signals[max_diff_idx]:.6f}")
    print(f"  Difference: {max_diff:.6f} ({abs(max_diff / full_signals[max_diff_idx] * 100):.2f}%)")

    # Detailed analysis at the problematic bar
    print(f"\n🔍 DETAILED ANALYSIS AT BAR {len(init_prices) + max_diff_idx}")

    # Check full calculation state at this bar
    print(f"\n  Full calculation Layer 2 equities:")
    for ma_type in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
        equity_key = f"{ma_type}_S"
        if equity_key in full_results:
            equity_val = full_results[equity_key].iloc[len(init_prices) + max_diff_idx]
            print(f"    {ma_type}_S: {equity_val:.6f}")

    print(f"\n  Incremental Layer 2 equities (final state):")
    if atc.state["equity"]:
        for ma_type, equity_val in atc.state["equity"].items():
            print(f"    {ma_type}: {equity_val:.6f}")

    print(f"\n  Full calculation Layer 1 signals:")
    for ma_type in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
        signal_key = f"{ma_type}_Signal"
        if signal_key in full_results:
            signal_val = full_results[signal_key].iloc[len(init_prices) + max_diff_idx]
            print(f"    {ma_type}_Signal: {signal_val:+.6f}")

    print(f"\n  Incremental Layer 1 signals (final state):")
    for ma_type, signal_val in atc.state.get("layer1_signals", {}).items():
        print(f"    {ma_type}: {signal_val:+.6f}")

    # Analyze cut_signal thresholds
    print(f"\n  Threshold analysis:")
    print(f"    long_threshold: {config['long_threshold']}")
    print(f"    short_threshold: {config['short_threshold']}")

    # Check if issue is with discretization (cut_signal)
    for ma_type in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
        inc_l1 = atc.state.get("layer1_signals", {}).get(ma_type, 0.0)

        # Apply same discretization as full calculation
        if inc_l1 > config['long_threshold']:
            c_inc = 1.0
        elif inc_l1 < config['short_threshold']:
            c_inc = -1.0
        else:
            c_inc = 0.0

        print(f"    {ma_type}: L1={inc_l1:+.6f} → C={c_inc:+.1f}")

    return incremental_signals, full_signals


def analyze_ma_values():
    """Analyze MA values to see if they match."""

    np.random.seed(42)
    base_price = 100.0
    n = 200
    returns = np.random.normal(0.001, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)
    sample_prices = pd.Series(prices, index=range(n))

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
        "La": 0.02,
        "De": 0.03,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
        "cutout": 0,
        "strategy_mode": False,
    }

    print("\n" + "=" * 80)
    print("MA VALUES COMPARISON")
    print("=" * 80)

    # Get full calculation MAs
    from modules.adaptive_trend_LTS.core.compute_moving_averages import set_of_moving_averages

    full_mas = {}
    for ma_type in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
        length = config[f"{ma_type.lower()}_len"]
        ma_tuple = set_of_moving_averages(
            length=length,
            source=sample_prices,
            ma_type=ma_type,
            robustness="Medium",
            use_cache=False,
            use_rust=config.get("use_rust_backend", True),
            use_cuda=False,
        )
        full_mas[ma_type] = ma_tuple

    # Initialize incremental
    init_prices = sample_prices[:-1]
    atc = IncrementalATC(config)
    atc.initialize(init_prices)

    # Update with last price
    last_price = sample_prices.iloc[-1]
    atc.update(last_price)

    # Compare MA values
    print(f"\nComparing MA values at last bar (bar {len(sample_prices)-1}):")

    for ma_type in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
        print(f"\n  {ma_type}:")

        full_ma_tuple = full_mas[ma_type]
        inc_ma_list = atc.state["ma_values"].get(ma_type.lower())

        if full_ma_tuple and inc_ma_list:
            # Compare all 9 variations
            for i in range(9):
                full_val = full_ma_tuple[i].iloc[-1] if i < len(full_ma_tuple) else None
                inc_val = inc_ma_list[i] if i < len(inc_ma_list) else None

                if full_val is not None and inc_val is not None:
                    diff = abs(full_val - inc_val)
                    pct_diff = abs(diff / full_val * 100) if full_val != 0 else 0

                    status = "✅" if diff < 1e-6 else "❌"
                    print(f"    Var {i}: Full={full_val:.6f}, Inc={inc_val:.6f}, Diff={diff:.6e} ({pct_diff:.4f}%) {status}")


if __name__ == "__main__":
    # Run detailed comparison
    incremental_signals, full_signals = compare_calculations_detailed()

    # Analyze MA values
    analyze_ma_values()

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
