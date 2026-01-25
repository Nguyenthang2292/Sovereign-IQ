"""Trace mismatch with parallel_l2=False."""

import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from modules.adaptive_trend.core import compute_atc_signals as compute_orig
from modules.adaptive_trend_LTS.core.compute_atc_signals import compute_atc_signals as compute_rust
from modules.common.core import DataFetcher, ExchangeManager


def main():
    em = ExchangeManager()
    df = DataFetcher(em)
    data, _ = df.fetch_ohlcv_with_fallback_exchange("ONT/USDT", limit=1500, timeframe="1h")
    prices = df.dataframe_to_close_series(data)

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

    print("Running Original...")
    res_orig = compute_orig(prices=prices, **config)

    print("Running Rust (Sequential L2)...")
    rust_config = config.copy()
    rust_config.update(
        {
            "parallel_l1": False,
            "parallel_l2": False,
            "precision": "float64",
            "use_rust_backend": True,
            "use_cache": False,
            "fast_mode": True,
        }
    )
    res_rust = compute_rust(prices=prices, **rust_config)

    o = res_orig["Average_Signal"].values
    r = res_rust["Average_Signal"].values
    diff = np.abs(o - r)
    max_d = np.nanmax(diff)
    print(f"Average_Signal: max_diff={max_d:.2e}")
    if max_d > 1e-6:
        print("STILL MISMATCHED with Sequential L2")
    else:
        print("MATCHED with Sequential L2! Lỗi nằm ở parallel_l2")


if __name__ == "__main__":
    main()
