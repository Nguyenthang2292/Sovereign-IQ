"""Debug CUDA batch processing issue."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import process_symbols_batch_cuda
from modules.common.core import DataFetcher, ExchangeManager

print("Fetching test data...")
em = ExchangeManager()
df_obj = DataFetcher(em)
symbols = df_obj.list_binance_futures_symbols(max_candidates=3)

prices_data = {}
for s in symbols[:3]:
    try:
        df, _ = df_obj.fetch_ohlcv_with_fallback_exchange(s, limit=500, timeframe="1h")
        if df is not None and len(df) >= 500:
            prices_data[s] = df_obj.dataframe_to_close_series(df)
            print(f"✓ Loaded {s}: {len(prices_data[s])} bars")
            if len(prices_data) >= 2:
                break
    except Exception as e:
        print(f"✗ Failed {s}: {e}")

if len(prices_data) < 2:
    print("Not enough data, exiting")
    sys.exit(1)

config = {
    "ema_len": 28,
    "hull_len": 28,
    "wma_len": 28,
    "dema_len": 28,
    "lsma_len": 28,
    "kama_len": 28,
    "robustness": "Medium",
    "La": 0.02,
    "De": 0.03,
    "cutout": 0,
    "long_threshold": 0.1,
    "short_threshold": -0.1,
    "use_cuda": True,
}

print("\n" + "=" * 60)
print("Testing CUDA Batch Processing")
print("=" * 60)

try:
    cuda_results = process_symbols_batch_cuda(prices_data, config)
    print(f"\n✓ CUDA batch completed")
    print(f"  Symbols processed: {len(cuda_results)}")

    for sym in list(cuda_results.keys())[:1]:
        print(f"\n  Sample symbol: {sym}")
        print(f"  Result type: {type(cuda_results[sym])}")
        print(f"  Result keys: {list(cuda_results[sym].keys()) if isinstance(cuda_results[sym], dict) else 'N/A'}")

        if isinstance(cuda_results[sym], dict) and "Average_Signal" in cuda_results[sym]:
            avg_sig = cuda_results[sym]["Average_Signal"]
            print(f"  Average_Signal type: {type(avg_sig)}")
            print(f"  Average_Signal length: {len(avg_sig)}")
            print(
                f"  Average_Signal sample: {avg_sig.iloc[:5].values if isinstance(avg_sig, pd.Series) else avg_sig[:5]}"
            )
            print(
                f"  Average_Signal stats: min={np.nanmin(avg_sig):.3f}, max={np.nanmax(avg_sig):.3f}, mean={np.nanmean(avg_sig):.3f}"
            )

except Exception as e:
    print(f"\n✗ CUDA batch failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing Regular (Rust) Processing")
print("=" * 60)

try:
    sym = list(prices_data.keys())[0]
    rust_result = compute_atc_signals(prices=prices_data[sym], **config)
    print(f"\n✓ Rust processing completed for {sym}")
    print(f"  Result type: {type(rust_result)}")
    print(f"  Result keys: {list(rust_result.keys())}")

    if "Average_Signal" in rust_result:
        avg_sig = rust_result["Average_Signal"]
        print(f"  Average_Signal type: {type(avg_sig)}")
        print(f"  Average_Signal length: {len(avg_sig)}")
        print(f"  Average_Signal sample: {avg_sig.iloc[:5].values}")
        print(f"  Average_Signal stats: min={avg_sig.min():.3f}, max={avg_sig.max():.3f}, mean={avg_sig.mean():.3f}")

except Exception as e:
    print(f"\n✗ Rust processing failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)

if len(cuda_results) > 0 and sym in cuda_results:
    cuda_sig = cuda_results[sym]["Average_Signal"]
    rust_sig = rust_result["Average_Signal"]

    # Align indices
    common_idx = (
        cuda_sig.index.intersection(rust_sig.index)
        if isinstance(cuda_sig, pd.Series)
        else range(min(len(cuda_sig), len(rust_sig)))
    )

    if isinstance(cuda_sig, pd.Series):
        cuda_vals = cuda_sig.loc[common_idx].values
        rust_vals = rust_sig.loc[common_idx].values
    else:
        cuda_vals = np.array(cuda_sig[: len(common_idx)])
        rust_vals = rust_sig.iloc[: len(common_idx)].values

    diff = np.abs(cuda_vals - rust_vals)
    print(f"  Common length: {len(common_idx)}")
    print(f"  Max difference: {np.nanmax(diff):.6f}")
    print(f"  Mean difference: {np.nanmean(diff):.6f}")
    print(f"  Match (< 1e-6): {np.sum(diff < 1e-6)} / {len(diff)} ({np.sum(diff < 1e-6) / len(diff) * 100:.1f}%)")
