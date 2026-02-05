"""Unit tests for O(1) incremental moving averages.

This module verifies correctness of O(1) MA implementations by:
1. Comparing O(1) MAs against reference implementations
2. Testing edge cases (warmup, constant series, step changes)
3. Testing with randomized seeded sequences
4. Testing window parameter validation and buffer rollover
"""

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental_mas_o1 import (
    TrueO1WMA,
    TrueO1HMA,
    TrueO1LSMA,
    TrueO1KAMA,
)


class TestTrueO1WMA:
    """Test O(1) WMA implementation."""

    def test_initialization(self):
        """Test WMA initialization."""
        wma = TrueO1WMA(10)
        assert wma.length == 10
        assert wma.denominator == 55.0  # 10 * 11 / 2
        assert not wma.is_initialized

    def test_constant_series(self):
        """Test WMA with constant price series."""
        wma = TrueO1WMA(5)
        prices = [100.0] * 20

        for price in prices:
            wma.update(price)

        # All prices are 100, so WMA should be 100
        assert abs(wma.current_value - 100.0) < 1e-10

    def test_linear_series(self):
        """Test WMA with linearly increasing series."""
        wma = TrueO1WMA(5)
        prices = [100.0 + i for i in range(20)]

        for price in prices:
            wma.update(price)

        # For window [104, 105, 106, 107, 108], WMA should be:
        # (104*1 + 105*2 + 106*3 + 107*4 + 108*5) / 15 = 106.67
        expected = (104 * 1 + 105 * 2 + 106 * 3 + 107 * 4 + 108 * 5) / 15.0
        assert abs(wma.current_value - expected) < 1e-6

    def test_vs_reference_wma(self):
        """Compare O(1) WMA against reference implementation."""
        length = 10
        wma_o1 = TrueO1WMA(length)

        # Generate test data
        np.random.seed(42)
        prices = np.random.randn(100) * 10 + 100

        # Update O(1) WMA
        wma_values = []
        for price in prices:
            wma_values.append(wma_o1.update(price))

        # Compute reference WMA
        def reference_wma(prices_arr, n):
            result = []
            for i in range(len(prices_arr)):
                if i < n - 1:
                    result.append(prices_arr[i])
                else:
                    window = prices_arr[i - n + 1 : i + 1]
                    weights = np.arange(1, n + 1)
                    result.append(np.dot(window, weights) / weights.sum())
            return result

        ref_values = reference_wma(prices, length)

        # Compare after warmup
        for i in range(length, len(prices)):
            assert abs(wma_values[i] - ref_values[i]) < 1e-6, (
                f"Mismatch at index {i}: {wma_values[i]} vs {ref_values[i]}"
            )

    def test_reset_functionality(self):
        """Test reset clears state."""
        wma = TrueO1WMA(5)
        for price in [100.0, 101.0, 102.0, 103.0, 104.0]:
            wma.update(price)

        initial_value = wma.current_value
        wma.reset()

        assert not wma.is_initialized
        assert len(wma.price_window) == 0

        # Re-initialize with different prices
        for price in [200.0, 201.0, 202.0, 203.0, 204.0]:
            wma.update(price)

        # Value should be different from initial
        assert abs(wma.current_value - initial_value) > 10

    def test_window_parameter_validation(self):
        """Test window parameter validation."""
        with pytest.raises(ValueError, match="length must be > 0"):
            TrueO1WMA(0)

        with pytest.raises(ValueError, match="length must be > 0"):
            TrueO1WMA(-5)


class TestTrueO1HMA:
    """Test O(1) HMA implementation."""

    def test_initialization(self):
        """Test HMA initialization."""
        hma = TrueO1HMA(28)
        assert hma.length == 28
        assert hma.half_len == 14
        assert hma.sqrt_len == 5

    def test_constant_series(self):
        """Test HMA with constant price series."""
        hma = TrueO1HMA(10)
        prices = [100.0] * 30

        for price in prices:
            hma.update(price)

        # All prices are 100, so HMA should be 100
        assert abs(hma.current_value - 100.0) < 1e-10

    def test_vs_reference_hma(self):
        """Compare O(1) HMA against reference implementation."""
        length = 20
        hma_o1 = TrueO1HMA(length)

        # Generate test data
        np.random.seed(42)
        prices = np.random.randn(150) * 10 + 100

        # Update O(1) HMA
        hma_values = []
        for price in prices:
            hma_values.append(hma_o1.update(price))

        # Compute reference HMA
        def reference_wma(prices_arr, n):
            result = []
            for i in range(len(prices_arr)):
                if i < n - 1:
                    result.append(prices_arr[i])
                else:
                    window = prices_arr[i - n + 1 : i + 1]
                    weights = np.arange(1, n + 1)
                    result.append(np.dot(window, weights) / weights.sum())
            return result

        def reference_hma(prices_arr, n):
            half_len = max(1, n // 2)
            sqrt_len = max(1, int(np.sqrt(n)))

            wma_half = reference_wma(prices_arr, half_len)
            wma_full = reference_wma(prices_arr, n)

            intermediate = [2.0 * h - f for h, f in zip(wma_half, wma_full)]
            hma_final = reference_wma(intermediate, sqrt_len)

            return hma_final

        ref_values = reference_hma(prices, length)

        # Compare after warmup (need enough data for all nested WMAs)
        warmup = length + int(np.sqrt(length))
        for i in range(warmup, len(prices)):
            assert abs(hma_values[i] - ref_values[i]) < 1e-5, (
                f"Mismatch at index {i}: {hma_values[i]} vs {ref_values[i]}"
            )

    def test_reset_functionality(self):
        """Test reset clears state."""
        hma = TrueO1HMA(10)
        for price in [100.0 + i for i in range(30)]:
            hma.update(price)

        initial_value = hma.current_value
        hma.reset()

        assert not hma.is_initialized
        assert len(hma.intermediate_series) == 0


class TestTrueO1LSMA:
    """Test O(1) LSMA implementation."""

    def test_initialization(self):
        """Test LSMA initialization."""
        lsma = TrueO1LSMA(10)
        assert lsma.length == 10
        assert lsma.denom != 0
        assert not lsma.is_initialized

    def test_constant_series(self):
        """Test LSMA with constant price series."""
        lsma = TrueO1LSMA(5)
        prices = [100.0] * 20

        for price in prices:
            lsma.update(price)

        # All prices are 100, so LSMA should be 100
        assert abs(lsma.current_value - 100.0) < 1e-10

    def test_linear_series(self):
        """Test LSMA with linearly increasing series."""
        lsma = TrueO1LSMA(5)
        prices = [100.0 + i for i in range(20)]

        for price in prices:
            lsma.update(price)

        # For linear series, LSMA should match the last price exactly
        assert abs(lsma.current_value - prices[-1]) < 1e-10

    def test_vs_reference_lsma(self):
        """Compare O(1) LSMA against reference implementation."""
        length = 10
        lsma_o1 = TrueO1LSMA(length)

        # Generate test data
        np.random.seed(42)
        prices = np.random.randn(100) * 10 + 100

        # Update O(1) LSMA
        lsma_values = []
        for price in prices:
            lsma_values.append(lsma_o1.update(price))

        # Compute reference LSMA
        def reference_lsma(prices_arr, n):
            result = []
            for i in range(len(prices_arr)):
                if i < n - 1:
                    result.append(prices_arr[i])
                else:
                    window = prices_arr[i - n + 1 : i + 1]
                    x = np.arange(n)
                    y = np.array(window)

                    sum_x = n * (n - 1) / 2
                    sum_x2 = n * (n - 1) * (2 * n - 1) / 6
                    sum_y = np.sum(y)
                    sum_xy = np.dot(x, y)

                    denom = n * sum_x2 - sum_x**2
                    if denom == 0:
                        result.append(window[-1])
                    else:
                        slope = (n * sum_xy - sum_x * sum_y) / denom
                        intercept = (sum_y - slope * sum_x) / n
                        lsma = intercept + slope * (n - 1)
                        result.append(lsma)
            return result

        ref_values = reference_lsma(prices, length)

        # Compare after warmup
        for i in range(length, len(prices)):
            assert abs(lsma_values[i] - ref_values[i]) < 1e-6, (
                f"Mismatch at index {i}: {lsma_values[i]} vs {ref_values[i]}"
            )

    def test_reset_functionality(self):
        """Test reset clears state."""
        lsma = TrueO1LSMA(5)
        for price in [100.0 + i for i in range(20)]:
            lsma.update(price)

        initial_value = lsma.current_value
        lsma.reset()

        assert not lsma.is_initialized
        assert len(lsma.price_window) == 0


class TestTrueO1KAMA:
    """Test O(1) KAMA implementation."""

    def test_initialization(self):
        """Test KAMA initialization."""
        kama = TrueO1KAMA(28)
        assert kama.length == 28
        assert not kama.is_initialized

    def test_constant_series(self):
        """Test KAMA with constant price series."""
        kama = TrueO1KAMA(10)
        prices = [100.0] * 40

        for price in prices:
            kama.update(price)

        # All prices are 100, so KAMA should be 100
        assert abs(kama.current_value - 100.0) < 1e-10

    def test_vs_reference_kama(self):
        """Compare O(1) KAMA against reference implementation."""
        length = 20
        kama_o1 = TrueO1KAMA(length)

        # Generate test data
        np.random.seed(42)
        prices = np.random.randn(100) * 10 + 100

        # Update O(1) KAMA
        kama_values = []
        for price in prices:
            kama_values.append(kama_o1.update(price))

        # Compute reference KAMA
        def reference_kama(prices_arr, n):
            result = []
            kama = prices_arr[0]
            for i in range(len(prices_arr)):
                if i == 0:
                    result.append(kama)
                    continue

                if i < n:
                    result.append(kama)
                    continue

                noise = 0.0
                for j in range(i - n + 1, i + 1):
                    if j <= 0:
                        continue
                    noise += abs(prices_arr[j] - prices_arr[j - 1])

                signal = abs(prices_arr[i] - prices_arr[i - n])
                ratio = 0.0 if noise == 0 else signal / noise

                fast_sc = 2.0 / 3.0  # 2/(2+1)
                slow_sc = 2.0 / 31.0  # 2/(30+1)
                sc = (ratio * (fast_sc - slow_sc) + slow_sc) ** 2

                kama = kama + sc * (prices_arr[i] - kama)
                result.append(kama)
            return result

        ref_values = reference_kama(prices, length)

        # Compare after warmup
        for i in range(length + 1, len(prices)):
            assert abs(kama_values[i] - ref_values[i]) < 1e-6, (
                f"Mismatch at index {i}: {kama_values[i]} vs {ref_values[i]}"
            )

    def test_reset_functionality(self):
        """Test reset clears state."""
        kama = TrueO1KAMA(10)
        for price in [100.0 + np.random.randn() for _ in range(40)]:
            kama.update(price)

        initial_value = kama.current_value
        kama.reset()

        assert not kama.is_initialized
        assert len(kama.price_window) == 0


class TestIncrementalATCWithO1:
    """Test IncrementalATC integration with O(1) MAs."""

    def test_o1_vs_legacy_atc(self):
        """Test that O(1) and legacy implementations produce identical results."""
        from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import IncrementalATC

        config = {
            "ema_len": 20,
            "hma_len": 20,
            "wma_len": 20,
            "dema_len": 20,
            "lsma_len": 20,
            "kama_len": 20,
            "La": 0.02,
            "De": 0.03,
            "long_threshold": 0.1,
            "short_threshold": -0.1,
        }

        # Generate test data
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(200) * 10)

        # Initialize with O(1) MAs
        config_o1 = config.copy()
        config_o1["use_o1_mas"] = True
        atc_o1 = IncrementalATC(config_o1)
        atc_o1.initialize(prices[:-10])

        # Initialize with legacy MAs
        config_legacy = config.copy()
        config_legacy["use_o1_mas"] = False
        atc_legacy = IncrementalATC(config_legacy)
        atc_legacy.initialize(prices[:-10])

        # Compare incremental updates
        for price in prices[-10:]:
            signal_o1 = atc_o1.update(price)
            signal_legacy = atc_legacy.update(price)

            # Should match very closely
            assert abs(signal_o1 - signal_legacy) < 1e-3, f"O(1): {signal_o1}, Legacy: {signal_legacy}"

    def test_buffer_rollover(self):
        """Test that O(1) MAs handle buffer rollover correctly."""
        from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import IncrementalATC

        config = {
            "ema_len": 10,
            "hma_len": 10,
            "wma_len": 10,
            "dema_len": 10,
            "lsma_len": 10,
            "kama_len": 10,
            "use_o1_mas": True,
            "La": 0.02,
            "De": 0.03,
        }

        atc = IncrementalATC(config)

        # Generate more data than the buffer size
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(500) * 10)

        # Initialize with first half
        atc.initialize(prices[:250])

        # Update with remaining prices
        signals = []
        for price in prices[250:]:
            signal = atc.update(price)
            signals.append(signal)
            assert np.isfinite(signal), f"Signal became NaN/inf: {signal}"

        # All signals should be valid
        assert len(signals) == 250
        assert all(np.isfinite(s) for s in signals)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
