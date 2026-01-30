"""
Phase 3 Benchmark: Caching & Persistence for XGBoost Module

Target: 50-100x speedup for repeated runs (instant loading).
"""

import time
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from modules.xgboost_LTS.utils.cache_manager import CacheManager
from modules.xgboost_LTS.core.model import train_and_predict
from modules.xgboost_LTS.core.labeling import apply_directional_labels
from config import MODEL_FEATURES


def benchmark_caching():
    print("=" * 60)
    print("PHASE 3 BENCHMARK - Caching & Persistence")
    print("=" * 60)

    rows = 10000
    columns = list(set(MODEL_FEATURES + ["close", "high", "low", "open", "volume"]))
    df = pd.DataFrame(np.random.randn(rows, len(columns)), columns=columns)

    df["close"] = np.cumsum(np.random.randn(rows)) + 100
    df["high"] = df["close"] + 1
    df["low"] = df["close"] - 1
    df["open"] = df["close"]
    df["volume"] = np.random.randint(100, 1000, rows)

    print(f"Dataset: {rows} rows, {len(columns)} columns")

    print("\n--- Labeling Caching ---")
    start = time.time()
    labeled_1 = apply_directional_labels(df.copy(), use_cache=True)
    t1 = time.time() - start
    print(f"Run 1 (Uncached): {t1:.4f}s")

    start = time.time()
    labeled_2 = apply_directional_labels(df.copy(), use_cache=True)
    t2 = time.time() - start
    print(f"Run 2 (Cached):   {t2:.4f}s")
    if t2 > 0:
        print(f"Speedup: {t1 / t2:.2f}x")
    else:
        print("Speedup: Instant (cache hit)")

    print("\n--- Model Caching ---")
    training_df = labeled_1.dropna(subset=["Target"])
    if len(training_df) < 100:
        print("Not enough valid data for training benchmark.")
        return

    print(f"Training on {len(training_df)} rows")

    start = time.time()
    model_1 = train_and_predict(training_df.copy(), use_cache=True)
    t1 = time.time() - start
    print(f"Train 1 (Uncached): {t1:.4f}s")

    start = time.time()
    model_2 = train_and_predict(training_df.copy(), use_cache=True)
    t2 = time.time() - start
    print(f"Train 2 (Cached):   {t2:.4f}s")
    if t2 > 0:
        print(f"Speedup: {t1 / t2:.2f}x")
    else:
        print("Speedup: Instant (cache hit)")

    cache = CacheManager()
    print(f"\nCache location: {cache.cache_dir}")
    print(f"Cache size: {sum(f.stat().st_size for f in cache.cache_dir.rglob('*') if f.is_file()) / 1024:.1f} KB")

    return {
        "labeling_uncached": t1,
        "labeling_cached": t2,
        "training_uncached": t1 if "model_1" in locals() else None,
        "training_cached": t2 if "model_2" in locals() else None,
    }


if __name__ == "__main__":
    benchmark_caching()
