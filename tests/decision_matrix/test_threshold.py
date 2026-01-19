"""
Unit tests for Threshold Calculator.

Tests dynamic threshold calculation for different feature types.
"""

import pytest

from modules.decision_matrix.utils.threshold import ThresholdCalculator


class TestThresholdCalculator:
    """Test ThresholdCalculator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = ThresholdCalculator(volume_std_length=14)

    def test_calculate_threshold_volume(self):
        """Test threshold calculation for Volume feature."""
        historical_values = [100, 110, 105, 115, 95, 120, 90, 130, 85, 125]

        threshold = self.calculator.calculate_threshold("Volume", historical_values)

        # Should return standard deviation of historical values
        mean = sum(historical_values) / len(historical_values)
        variance = sum((x - mean) ** 2 for x in historical_values) / len(historical_values)
        expected_std = variance**0.5

        assert abs(threshold - expected_std) < 0.001

    def test_calculate_threshold_volume_no_historical_data(self):
        """Test that Volume threshold requires historical data."""
        with pytest.raises(ValueError, match="Historical values required"):
            self.calculator.calculate_threshold("Volume", None)

        with pytest.raises(ValueError, match="Historical values required"):
            self.calculator.calculate_threshold("Volume", [])

    def test_calculate_threshold_z_score(self):
        """Test threshold calculation for Z-Score feature."""
        threshold = self.calculator.calculate_threshold("Z-Score", None)

        # Should return fixed 0.05
        assert threshold == 0.05

    def test_calculate_threshold_stochastic(self):
        """Test threshold calculation for Stochastic feature."""
        threshold = self.calculator.calculate_threshold("Stochastic", None)

        # Should return fixed 0.5
        assert threshold == 0.5

    def test_calculate_threshold_rsi(self):
        """Test threshold calculation for RSI feature."""
        threshold = self.calculator.calculate_threshold("RSI", None)

        # Should return fixed 0.5
        assert threshold == 0.5

    def test_calculate_threshold_mfi(self):
        """Test threshold calculation for MFI feature."""
        threshold = self.calculator.calculate_threshold("MFI", None)

        # Should return fixed 0.5
        assert threshold == 0.5

    def test_calculate_threshold_ema(self):
        """Test threshold calculation for EMA feature."""
        threshold = self.calculator.calculate_threshold("EMA", None)

        # Should return fixed 0.5
        assert threshold == 0.5

    def test_calculate_threshold_sma(self):
        """Test threshold calculation for SMA feature."""
        threshold = self.calculator.calculate_threshold("SMA", None)

        # Should return fixed 0.5
        assert threshold == 0.5

    def test_calculate_threshold_case_insensitive(self):
        """Test that feature type matching is case-insensitive."""
        # Test different cases
        assert self.calculator.calculate_threshold("volume", [100, 110]) > 0
        assert self.calculator.calculate_threshold("VOLUME", [100, 110]) > 0
        assert self.calculator.calculate_threshold("Volume", [100, 110]) > 0

        assert self.calculator.calculate_threshold("z-score", None) == 0.05
        assert self.calculator.calculate_threshold("Z-SCORE", None) == 0.05
        assert self.calculator.calculate_threshold("Z-Score", None) == 0.05

    def test_calculate_stdev_basic(self):
        """Test standard deviation calculation."""
        values = [2, 4, 4, 4, 5, 5, 7, 9]

        std = self.calculator._calculate_stdev(values)

        # Calculate expected
        mean = 5.0
        variance = ((2 - 5) ** 2 + (4 - 5) ** 2 * 3 + (5 - 5) ** 2 * 2 + (7 - 5) ** 2 + (9 - 5) ** 2) / 8
        expected = variance**0.5

        assert abs(std - expected) < 0.001

    def test_calculate_stdev_empty_list(self):
        """Test standard deviation with empty list."""
        std = self.calculator._calculate_stdev([])

        assert std == 0.0

    def test_calculate_stdev_single_value(self):
        """Test standard deviation with single value."""
        std = self.calculator._calculate_stdev([100])

        # Variance = 0 for single value, should return epsilon
        assert std == 1e-8

    def test_calculate_stdev_identical_values(self):
        """Test standard deviation with identical values."""
        values = [100, 100, 100, 100, 100]

        std = self.calculator._calculate_stdev(values)

        # Variance = 0 for identical values, should return epsilon
        assert std == 1e-8
