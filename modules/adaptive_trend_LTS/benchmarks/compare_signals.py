"""Compare full ATC signals between Original and Rust at different pipeline stages."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

# Import both implementations
from modules.adaptive_trend.core import compute_atc_signals as compute_original
from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals as compute_rust
from modules.common.core import DataFetcher, ExchangeManager


def main():
    print("Fetching data...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=1500, timeframe="1h")
    if df is None or len(df) < 100:
        print("ERROR: Could not fetch data")
        return

    prices = DataFetcher.dataframe_to_close_series(df)
    print(f"Got {len(prices)} bars")

    # Common config
    config = {
        "ema_len": 28,
        "hull_len": 28,
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

    print("\nRunning Original implementation...")
    try:
        orig_result = compute_original(prices=prices, **config)
        print(f"  Original: {len(orig_result)} keys")
    except Exception as e:
        print(f"  Original FAILED: {e}")
        orig_result = None

    print("\nRunning Rust implementation...")
    try:
        rust_result = compute_rust(prices=prices, **config)
        print(f"  Rust: {len(rust_result)} keys")
    except Exception as e:
        print(f"  Rust FAILED: {e}")
        rust_result = None

    if orig_result is None or rust_result is None:
        print("Cannot compare - one failed")
        return

    print("\n" + "=" * 60)
    print("SIGNAL COMPARISON")
    print("=" * 60)

    # Compare each signal key
    common_keys = set(orig_result.keys()) & set(rust_result.keys())
    print(f"\nCommon keys: {len(common_keys)}")

    for key in sorted(common_keys):
        orig_val = orig_result[key]
        rust_val = rust_result[key]

        if not isinstance(orig_val, pd.Series) or not isinstance(rust_val, pd.Series):
            continue

        valid_mask = (~orig_val.isna()) & (~rust_val.isna())
        if valid_mask.sum() == 0:
            print(f"{key}: NO VALID DATA")
            continue

        orig_valid = orig_val[valid_mask].values
        rust_valid = rust_val[valid_mask].values
        diff = np.abs(orig_valid - rust_valid)
        max_diff = np.max(diff)

        status = "MATCH" if max_diff < 1e-10 else f"DIFF (max={max_diff:.2e})"
        print(f"{key}: {status}")

        # Show details for differences
        if max_diff >= 1e-6:
            first_diff_idx = np.argmax(diff > 1e-10)
            print(f"  First diff at idx={first_diff_idx}")
            print(f"    Original: {orig_valid[first_diff_idx]:.10f}")
            print(f"    Rust:     {rust_valid[first_diff_idx]:.10f}")

    # Final Average_Signal comparison
    print("\n" + "-" * 60)
    if "Average_Signal" in orig_result and "Average_Signal" in rust_result:
        orig_avg = orig_result["Average_Signal"]
        rust_avg = rust_result["Average_Signal"]

        valid_mask = (~orig_avg.isna()) & (~rust_avg.isna())
        if valid_mask.sum() > 0:
            orig_valid = orig_avg[valid_mask].values
            rust_valid = rust_avg[valid_mask].values
            diff = np.abs(orig_valid - rust_valid)

            # Count matches (within tolerance)
            tolerance = 1e-6
            matches = np.sum(diff < tolerance)
            match_pct = (matches / len(diff)) * 100

            print(f"\nAverage_Signal Match Rate: {match_pct:.2f}%")
            print(f"Max difference: {np.max(diff):.6e}")
            print(f"Avg difference: {np.mean(diff):.6e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
