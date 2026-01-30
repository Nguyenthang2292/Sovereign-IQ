"""
Phase 2 Benchmark: Memory & Vectorization for XGBoost Module

Target: 3-5x labeling speedup and 30-50% memory reduction with Numba JIT and float32.
"""

import time
import pandas as pd
import numpy as np
import gc
import sys
import os
import psutil
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from modules.xgboost_LTS.core.labeling import apply_directional_labels
from modules.xgboost_LTS.core.model import train_and_predict
from modules.xgboost_LTS.utils.cache_manager import CacheManager
from config import MODEL_FEATURES


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def benchmark_phase2():
    """Benchmark Phase 2 optimizations: Numba JIT Labeling and Float32."""
    print("=" * 60)
    print("PHASE 2 BENCHMARK - Memory & Vectorization")
    print("=" * 60)

    rows = 100000
    df = pd.DataFrame(np.random.randn(rows, len(MODEL_FEATURES)), columns=MODEL_FEATURES)
    df["close"] = np.cumsum(np.random.randn(rows)) + 100
    df["high"] = np.cumsum(np.random.randn(rows)) + 101
    df["low"] = np.cumsum(np.random.randn(rows)) + 99
    df["open"] = np.cumsum(np.random.randn(rows)) + 100
    df["volume"] = np.random.randint(100, 1000, rows)

    print(f"Dataset: {rows} rows, {len(MODEL_FEATURES)} features")

    mem_before = get_memory_mb()
    print(f"Memory before processing: {mem_before:.2f} MB")

    start = time.time()
    labeled_df = apply_directional_labels(df.copy())
    label_time = time.time() - start
    print(f"Labeling Time: {label_time:.4f}s ({label_time * 1000 / rows:.4f}s per 1K rows)")

    mem_after_label = get_memory_mb()
    print(f"Memory after labeling: {mem_after_label:.2f} MB")

    labeled_df = labeled_df.dropna(subset=["Target"])
    print(f"Valid samples for training: {len(labeled_df)}")

    mem_before_train = get_memory_mb()
    print(f"Memory before training: {mem_before_train:.2f} MB")

    start = time.time()
    train_and_predict(labeled_df.copy(), use_cache=False)
    train_time = time.time() - start

    mem_after_train = get_memory_mb()
    print(f"Training Time: {train_time:.4f}s")
    print(f"Memory after training: {mem_after_train:.2f} MB")

    peak_memory = max(mem_before, mem_after_label, mem_before_train, mem_after_train)
    print(f"Peak Memory: {peak_memory:.2f} MB")

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    return {
        "rows": rows,
        "features": len(MODEL_FEATURES),
        "labeling_time": label_time,
        "training_time": train_time,
        "peak_memory_mb": peak_memory,
    }


def benchmark_float32_impact():
    """Benchmark the impact of float32 precision on memory usage."""
    print("\n" + "=" * 60)
    print("FLOAT32 PRECISION IMPACT")
    print("=" * 60)

    n_samples = 10000
    n_features = len(MODEL_FEATURES)

    X_float64 = pd.DataFrame(np.random.randn(n_samples, n_features), columns=MODEL_FEATURES).astype(np.float64)

    X_float32 = X_float64.astype(np.float32)

    mem_float64 = X_float64.memory_usage(deep=True).sum()
    mem_float32 = X_float32.memory_usage(deep=True).sum()

    print(f"Float64 Memory: {mem_float64 / 1024**2:.2f} MB")
    print(f"Float32 Memory: {mem_float32 / 1024**2:.2f} MB")
    print(f"Memory Reduction: {(1 - mem_float32 / mem_float64) * 100:.1f}%")

    precision_loss = np.abs(X_float64.values - X_float32.values).max()
    print(f"Max Precision Loss: {precision_loss:.2e}")

    return {
        "float64_mb": mem_float64 / 1024**2,
        "float32_mb": mem_float32 / 1024**2,
        "reduction_percent": (1 - mem_float32 / mem_float64) * 100,
        "max_precision_loss": precision_loss,
    }


if __name__ == "__main__":
    results = benchmark_phase2()
    float32_results = benchmark_float32_impact()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Labeling Speed: {results['labeling_time']:.4f}s for {results['rows']} rows")
    print(f"Peak Memory: {results['peak_memory_mb']:.2f} MB")
    print(f"Float32 Reduction: {float32_results['reduction_percent']:.1f}%")
