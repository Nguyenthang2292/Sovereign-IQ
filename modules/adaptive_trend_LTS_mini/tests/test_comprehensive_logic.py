"""
Comprehensive Test Suite for Adaptive Trend LTS Module
Tests for logic errors, edge cases, and potential bugs
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig, create_atc_config_from_dict
from modules.adaptive_trend_LTS_mini.utils.diflen import diflen
from modules.adaptive_trend_LTS_mini.utils.exp_growth import exp_growth
from modules.adaptive_trend_LTS_mini.utils.rate_of_change import rate_of_change
from modules.adaptive_trend_LTS_mini.core.process_layer1.weighted_signal import weighted_signal
from modules.adaptive_trend_LTS_mini.core.signal_detection.crossover import crossover
from modules.adaptive_trend_LTS_mini.core.signal_detection.crossunder import crossunder
from modules.adaptive_trend_LTS_mini.core.signal_detection.generate_signal import generate_signal_from_ma


class TestParameterScaling:
    """Test parameter scaling logic - Giả thuyết 1"""

    def test_lambda_scaling_in_config(self):
        """Test lambda scaling trong ATCConfig"""
        config = ATCConfig(lambda_param=0.02)
        assert config.lambda_param == 0.02  # Unscaled
        assert config.lambda_scaled == 0.00002  # Scaled (divided by 1000)

    def test_decay_scaling_in_config(self):
        """Test decay scaling trong ATCConfig"""
        config = ATCConfig(decay=0.03)
        assert config.decay == 0.03  # Unscaled
        assert config.decay_scaled == 0.0003  # Scaled (divided by 100)

    def test_no_double_scaling_risk(self):
        """Kiểm tra rủi ro double scaling khi dùng config với compute_atc_signals"""
        # Nếu dùng config.lambda_scaled và truyền vào compute_atc_signals sẽ bị double-scale
        config = ATCConfig(lambda_param=0.02, decay=0.03)

        # Giả lập compute_atc_signals scaling
        la_from_config_scaled = config.lambda_scaled  # 0.00002
        de_from_config_scaled = config.decay_scaled  # 0.0003

        # Nếu truyền vào compute_atc_signals, nó scale lại:
        la_double_scaled = la_from_config_scaled / 1000  # 0.00000002 - SAI!
        de_double_scaled = de_from_config_scaled / 100  # 0.000003 - SAI!

        # Correct usage là truyền unscaled values:
        la_correct = config.lambda_param / 1000  # 0.00002
        de_correct = config.decay / 100  # 0.0003

        assert la_double_scaled != la_correct, "Double scaling detected!"
        assert de_double_scaled != de_correct, "Double scaling detected!"


class TestDiflenFunction:
    """Test diflen logic và edge cases"""

    def test_diflen_narrow(self):
        """Test Narrow robustness"""
        L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen(10, "Narrow")
        assert L1 == 11 and L_1 == 9
        assert L2 == 12 and L_2 == 8
        assert L3 == 13 and L_3 == 7
        assert L4 == 14 and L_4 == 6
        assert all(l > 0 for l in [L1, L2, L3, L4, L_1, L_2, L_3, L_4])

    def test_diflen_medium(self):
        """Test Medium robustness"""
        L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen(10, "Medium")
        assert L1 == 11 and L_1 == 9
        assert L2 == 12 and L_2 == 8
        assert L3 == 14 and L_3 == 6
        assert L4 == 16 and L_4 == 4
        assert all(l > 0 for l in [L1, L2, L3, L4, L_1, L_2, L_3, L_4])

    def test_diflen_wide(self):
        """Test Wide robustness"""
        L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen(10, "Wide")
        assert L1 == 11 and L_1 == 9
        assert L2 == 13 and L_2 == 7
        assert L3 == 15 and L_3 == 5
        assert L4 == 17 and L_4 == 3
        assert all(l > 0 for l in [L1, L2, L3, L4, L_1, L_2, L_3, L_4])

    def test_diflen_invalid_length(self):
        """Test với length quá nhỏ"""
        with pytest.raises(ValueError):
            diflen(3, "Wide")  # Wide cần min 8

    def test_diflen_invalid_robustness(self):
        """Test với robustness không hợp lệ"""
        # Nên fallback về Medium
        result = diflen(10, "Invalid")
        # Should use Medium offsets
        L1, L2, L3, L4, L_1, L_2, L_3, L_4 = result
        assert L4 == 16 and L_4 == 4  # Medium pattern


class TestExpGrowth:
    """Test exponential growth calculation"""

    def test_exp_growth_basic(self):
        """Test basic exp_growth calculation"""
        index = pd.RangeIndex(0, 10)
        result = exp_growth(L=0.001, index=index, cutout=0)

        assert len(result) == 10
        # Bar 0 in Pine Script: bars = 1 (special case when bar_index == 0)
        # So bar 0 = e^(L * 1) = e^L when cutout=0
        # Bar 1: bars = 1, so also e^L
        # Growth only increases from bar 2 onwards (bar_index >= 2)
        assert result.iloc[0] > 1.0  # e^(0.001 * 1) ≈ 1.001
        assert result.iloc[1] == result.iloc[0]  # Both bar 0 and 1 use bars=1
        assert result.iloc[2] > result.iloc[1]  # Growth increases from bar 2

    def test_exp_growth_with_cutout(self):
        """Test exp_growth với cutout"""
        index = pd.RangeIndex(0, 10)
        result = exp_growth(L=0.001, index=index, cutout=3)

        # First 3 bars should be 1.0
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 1.0
        assert result.iloc[2] == 1.0
        # After cutout, growth starts
        assert result.iloc[3] == 1.0  # e^(0.001 * 0) = 1
        assert result.iloc[4] > 1.0  # e^(0.001 * 1) > 1

    def test_exp_growth_overflow_protection(self):
        """Test overflow detection - uses log_warn, not warnings module"""
        index = pd.RangeIndex(0, 1000)
        # L too large should trigger log_warn (not pytest warnings)
        result = exp_growth(L=1.0, index=index, cutout=0)

        # Should not contain inf (replaced with max float)
        assert not np.isinf(result).any()
        # Check that overflow was handled (some values should be max float)
        assert (result == np.finfo(np.float64).max).any() or result.max() > 1e200

    def test_exp_growth_invalid_L(self):
        """Test với L không hợp lệ"""
        index = pd.RangeIndex(0, 10)
        with pytest.raises(ValueError):
            exp_growth(L=np.nan, index=index)
        with pytest.raises(ValueError):
            exp_growth(L=np.inf, index=index)


class TestCrossoverCrossunder:
    """Test crossover và crossunder detection"""

    def test_crossover_detection(self):
        """Test crossover detection"""
        price = pd.Series([1, 2, 3, 2, 3, 4])
        ma = pd.Series([2, 2, 2, 2, 2, 2])

        result = crossover(price, ma)

        # Crossover at index 1: price=2 > ma=2 (false, <= prev), index 4: price=3 > ma=2 and prev=2 <= 2
        assert result.dtype == bool
        assert not result.iloc[0]  # First is NaN -> False

    def test_crossunder_detection(self):
        """Test crossunder detection"""
        price = pd.Series([4, 3, 2, 3, 2, 1])
        ma = pd.Series([2, 2, 2, 2, 2, 2])

        result = crossunder(price, ma)

        assert result.dtype == bool
        assert not result.iloc[0]  # First is NaN -> False

    def test_crossover_empty_series(self):
        """Test với empty series"""
        price = pd.Series([], dtype=float)
        ma = pd.Series([], dtype=float)

        result = crossover(price, ma)
        assert len(result) == 0

    def test_crossover_different_indices(self):
        """Test với different indices - nên align"""
        price = pd.Series([1, 2, 3], index=[0, 1, 2])
        ma = pd.Series([2, 2, 2], index=[1, 2, 3])

        result = crossover(price, ma)
        # Should align to common indices [1, 2]
        assert len(result) == 2


class TestWeightedSignal:
    """Test weighted signal calculation - Giả thuyết 4"""

    def test_weighted_signal_basic(self):
        """Test basic weighted signal"""
        signals = [
            pd.Series([1, 1, -1]),
            pd.Series([1, -1, 1]),
        ]
        weights = [
            pd.Series([1.0, 1.0, 1.0]),
            pd.Series([1.0, 1.0, 1.0]),
        ]

        result = weighted_signal(signals, weights)

        # (1*1 + 1*1) / (1+1) = 1, (1*1 + -1*1) / 2 = 0, (-1*1 + 1*1) / 2 = 0
        assert len(result) == 3
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 0.0
        assert result.iloc[2] == 0.0

    def test_weighted_signal_zero_weights(self):
        """Test với tất cả weights = 0 - ĐÃ FIX"""
        signals = [
            pd.Series([1, 1, -1]),
            pd.Series([1, -1, 1]),
        ]
        weights = [
            pd.Series([0.0, 0.0, 0.0]),
            pd.Series([0.0, 0.0, 0.0]),
        ]

        result = weighted_signal(signals, weights)

        # FIXED: Now returns neutral signal (0.0) when all weights are zero
        print(f"Zero weights result: {result.values}")
        assert (result == 0.0).all(), "Should return 0.0 for zero weights"

    def test_weighted_signal_mismatched_lengths(self):
        """Test với mismatched signal và weight lengths - Auto-align behavior"""
        signals = [pd.Series([1, 2, 3])]
        weights = [pd.Series([1.0, 2.0])]

        # Code auto-aligns instead of raising ValueError
        result = weighted_signal(signals, weights)
        # Should align và tạo NaN cho giá trị thiếu
        assert len(result) == 3

    def test_weighted_signal_different_indices(self):
        """Test alignment với different indices"""
        signals = [
            pd.Series([1, 2], index=[0, 1]),
            pd.Series([3, 4], index=[1, 2]),
        ]
        weights = [
            pd.Series([1.0, 1.0], index=[0, 1]),
            pd.Series([1.0, 1.0], index=[1, 2]),
        ]

        result = weighted_signal(signals, weights)
        # Should align to first index [0, 1]
        assert len(result) == 2


class TestSignalGeneration:
    """Test signal generation from MA"""

    def test_generate_signal_crossover(self):
        """Test signal generation với crossover"""
        price = pd.Series([1, 2, 3, 4, 5, 4, 3, 4, 5])
        ma = pd.Series([2, 2, 2, 2, 2, 2, 2, 2, 2])

        result = generate_signal_from_ma(price, ma)

        assert result.dtype == np.int8
        assert len(result) == len(price)
        # Signal persists after crossover
        assert result.iloc[0] == 0  # No signal initially
        # After price crosses above MA, signal should be 1

    def test_generate_signal_persistence(self):
        """Test signal persistence (var behavior)"""
        # Create scenario: crossover up, hold, crossunder down
        price = pd.Series([1, 3, 3, 3, 1])  # Cross up at 1, cross down at 4
        ma = pd.Series([2, 2, 2, 2, 2])

        result = generate_signal_from_ma(price, ma)

        # Signal should persist at 1 after crossover up
        assert result.iloc[2] == 1  # Still bullish
        assert result.iloc[3] == 1  # Still bullish
        # After crossunder, should be -1
        assert result.iloc[4] == -1  # Bearish


class TestEdgeCases:
    """Test các edge cases quan trọng"""

    def test_single_bar_data(self):
        """Test với single bar data"""
        price = pd.Series([100.0])
        ma = pd.Series([100.0])

        result = generate_signal_from_ma(price, ma)
        # Single bar không thể có crossover
        assert len(result) == 1
        assert result.iloc[0] == 0  # No signal

    def test_all_nan_input(self):
        """Test với all NaN input"""
        price = pd.Series([np.nan, np.nan, np.nan])
        ma = pd.Series([np.nan, np.nan, np.nan])

        result = generate_signal_from_ma(price, ma)
        # Should handle gracefully
        assert len(result) == 3

    def test_constant_values(self):
        """Test với constant values (no crossovers)"""
        price = pd.Series([5, 5, 5, 5, 5])
        ma = pd.Series([5, 5, 5, 5, 5])

        result = generate_signal_from_ma(price, ma)
        # No crossovers possible
        assert (result == 0).all()

    def test_extreme_price_values(self):
        """Test với extreme price values"""
        price = pd.Series([1e10, 1e-10, 1e10, 1e-10])
        ma = pd.Series([1.0, 1.0, 1.0, 1.0])

        result = generate_signal_from_ma(price, ma)
        assert len(result) == 4
        assert not np.isinf(result).any()


class TestConfigFromDict:
    """Test config creation from dict"""

    def test_create_from_dict_basic(self):
        """Test basic dict conversion"""
        params = {
            "lambda_param": 0.05,
            "decay": 0.04,
            "ema_len": 30,
        }

        config = create_atc_config_from_dict(params, timeframe="1h")

        assert config.lambda_param == 0.05
        assert config.decay == 0.04
        assert config.ema_len == 30
        assert config.timeframe == "1h"

    def test_create_from_dict_defaults(self):
        """Test dict với missing values"""
        params = {}

        config = create_atc_config_from_dict(params)

        # Should use defaults
        assert config.lambda_param == 0.02
        assert config.decay == 0.03
        assert config.ema_len == 28

    def test_backward_compat_prefer_gpu(self):
        """Test backward compatibility với prefer_gpu"""
        params = {
            "prefer_gpu": True,
        }

        config = create_atc_config_from_dict(params)

        # prefer_gpu should map to use_rust_backend
        assert config.use_rust_backend == True


class TestRateOfChange:
    """Test rate of change calculation"""

    def test_rate_of_change_basic(self):
        """Test basic RoC calculation"""
        prices = pd.Series([100, 110, 99, 108])

        result = rate_of_change(prices)

        # (110 - 100) / 100 = 0.1
        # (99 - 110) / 110 = -0.1
        # (108 - 99) / 99 = 0.0909...
        assert len(result) == len(prices)
        assert np.isnan(result.iloc[0])  # First is NaN
        assert abs(result.iloc[1] - 0.1) < 0.001

    def test_rate_of_change_with_zeros(self):
        """Test RoC với zero prices"""
        prices = pd.Series([0, 100, 0, 100])

        result = rate_of_change(prices)

        # Division by zero should produce inf or nan
        assert len(result) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
