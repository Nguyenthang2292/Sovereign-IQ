"""
Phase 1 Benchmark: Core Optimizations for XGBoost Module

Target: 2-8x speedup with GPU detection caching, parallel CV, and parallel Optuna trials.
"""

import time
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config import MODEL_FEATURES
from modules.xgboost_LTS.core.model import train_and_predict
from modules.xgboost_LTS.core.labeling import apply_directional_labels
from modules.xgboost_LTS.utils.gpu_utils import detect_cuda_available


def benchmark_baseline():
    """Establish baseline performance before optimizations."""
    print("=" * 60)
    print("PHASE 1 BENCHMARK - BASELINE")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 2000

    df = pd.DataFrame(
        {
            **{f: np.random.randn(n_samples) for f in MODEL_FEATURES},
            "close": np.cumsum(np.random.randn(n_samples)) + 100,
            "high": np.cumsum(np.random.randn(n_samples)) + 101,
            "low": np.cumsum(np.random.randn(n_samples)) + 99,
            "open": np.cumsum(np.random.randn(n_samples)) + 100,
            "volume": np.random.randint(100, 1000, n_samples),
            "Target": np.random.randint(0, 3, n_samples),
        }
    )

    print(f"Dataset: {n_samples} rows, {len(MODEL_FEATURES)} features")

    times = []
    for _ in range(5):
        test_df = df.copy()
        start = time.perf_counter()
        apply_directional_labels(test_df)
        times.append(time.perf_counter() - start)
    avg_label_time = sum(times) / len(times)
    print(f"Labeling Time: {avg_label_time:.4f}s (mean of 5)")

    labeled_df = apply_directional_labels(df.copy())
    labeled_df = labeled_df.dropna(subset=["Target"])

    times = []
    for _ in range(3):
        test_df = labeled_df.copy()
        start = time.perf_counter()
        train_and_predict(test_df, use_cache=False)
        times.append(time.perf_counter() - start)
    avg_train_time = sum(times) / len(times)
    print(f"Training Time: {avg_train_time:.4f}s (mean of 3)")

    gpu_available = detect_cuda_available()
    print(f"GPU Available: {gpu_available}")

    return {
        "labeling_time": avg_label_time,
        "training_time": avg_train_time,
        "gpu_available": gpu_available,
    }


def benchmark_optimized():
    """Benchmark with optimizations enabled."""
    print("=" * 60)
    print("PHASE 1 BENCHMARK - OPTIMIZED")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 2000

    df = pd.DataFrame(
        {
            **{f: np.random.randn(n_samples) for f in MODEL_FEATURES},
            "close": np.cumsum(np.random.randn(n_samples)) + 100,
            "high": np.cumsum(np.random.randn(n_samples)) + 101,
            "low": np.cumsum(np.random.randn(n_samples)) + 99,
            "open": np.cumsum(np.random.randn(n_samples)) + 100,
            "volume": np.random.randint(100, 1000, n_samples),
            "Target": np.random.randint(0, 3, n_samples),
        }
    )

    print(f"Dataset: {n_samples} rows, {len(MODEL_FEATURES)} features")

    times = []
    for _ in range(5):
        test_df = df.copy()
        start = time.perf_counter()
        apply_directional_labels(test_df)
        times.append(time.perf_counter() - start)
    avg_label_time = sum(times) / len(times)
    print(f"Labeling Time: {avg_label_time:.4f}s (mean of 5)")

    labeled_df = apply_directional_labels(df.copy())
    labeled_df = labeled_df.dropna(subset=["Target"])

    times = []
    for _ in range(3):
        test_df = labeled_df.copy()
        start = time.perf_counter()
        train_and_predict(test_df, use_cache=False)
        times.append(time.perf_counter() - start)
    avg_train_time = sum(times) / len(times)
    print(f"Training Time: {avg_train_time:.4f}s (mean of 3)")

    gpu_available = detect_cuda_available()
    print(f"GPU Available: {gpu_available}")

    return {
        "labeling_time": avg_label_time,
        "training_time": avg_train_time,
        "gpu_available": gpu_available,
    }


if __name__ == "__main__":
    baseline = benchmark_baseline()
    print("\n")
    optimized = benchmark_optimized()

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Labeling Speedup: {baseline['labeling_time'] / optimized['labeling_time']:.2f}x")
    print(f"Training Speedup: {baseline['training_time'] / optimized['training_time']:.2f}x")
