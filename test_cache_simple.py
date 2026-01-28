"""Simple cache test to verify cache hits."""

import pandas as pd
import numpy as np
from modules.adaptive_trend_LTS.utils.cache_manager import get_cache_manager, reset_cache_manager
from modules.adaptive_trend_LTS.core.compute_atc_signals.compute_atc_signals import compute_atc_signals

# Reset and get clean cache
reset_cache_manager()
cache_mgr = get_cache_manager()
cache_mgr.clear()

# Generate test data
np.random.seed(42)
prices = pd.Series(100 + np.cumsum(np.random.randn(1000) * 0.5))
print(f"Initial stats: {cache_mgr.get_stats()}")

# First run (cache miss expected)
print("\n=== First run (should be cache miss) ===")
result1 = compute_atc_signals(prices, ema_len=28, use_rust_backend=False)
print(f"Stats after first run: {cache_mgr.get_stats()}")
print(f"Has EMA_Signal: {'EMA_Signal' in result1}")

# Second run with same params (cache hit expected)
print("\n=== Second run with same params (should be cache hit) ===")
result2 = compute_atc_signals(prices, ema_len=28, use_rust_backend=False)
print(f"Stats after second run: {cache_mgr.get_stats()}")
print(f"Results match: {result1.get('EMA_Signal').equals(result2.get('EMA_Signal'))}")
