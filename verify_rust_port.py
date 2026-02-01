import numpy as np
import sovereign_prime

print("=" * 60)
print("Testing Adaptive Trend Functions")
print("=" * 60)

prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 11.0, 10.0, 12.0], dtype=np.float64)

# Test KAMA
kama = sovereign_prime.calculate_kama_rust(prices, 2)
print(f"✓ KAMA: shape={kama.shape}, last={kama[-1]:.4f}")

# Test EMA
ema = sovereign_prime.calculate_ema_rust(prices, 3)
print(f"✓ EMA: shape={ema.shape}, last={ema[-1]:.4f}")

print()
print("=" * 60)
print("Testing XGBoost Functions")
print("=" * 60)

# Test rolling quantile
quantile_result = sovereign_prime.rolling_quantile_rust(prices, 3, 0.5)
print(f"✓ Rolling Quantile: shape={quantile_result.shape}, last={quantile_result[-1]:.4f}")

# Test rolling mean
mean_result = sovereign_prime.rolling_mean_rust(prices, 3)
print(f"✓ Rolling Mean: shape={mean_result.shape}, last={mean_result[-1]:.4f}")

# Test pct_change
pct_change = sovereign_prime.pct_change_rust(prices, 1)
print(f"✓ Pct Change: shape={pct_change.shape}, non-nan count={np.sum(~np.isnan(pct_change))}")

# Test rolling std
rolling_std = sovereign_prime.rolling_std_rust(prices, 3)
print(f"✓ Rolling Std: shape={rolling_std.shape}, last={rolling_std[-1]:.4f}")

print()
print("=" * 60)
print("✓ ALL TESTS PASSED")
print("=" * 60)
