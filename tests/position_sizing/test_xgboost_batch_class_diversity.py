
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from modules.position_sizing.core.hybrid_signal_calculator import HybridSignalCalculator
from modules.xgboost.model import ClassDiversityError
from modules.position_sizing.core.hybrid_signal_calculator import HybridSignalCalculator
from modules.xgboost.model import ClassDiversityError

"""
Tests for XGBoost batch calculation with class diversity validation.

Tests cover:
- Class diversity check logic (at least 2 classes)
- Handling ClassDiversityError exceptions from train_and_predict
- Handling general exceptions for other errors
"""





@pytest.fixture
def mock_data_fetcher():
    """Create a mock data fetcher for testing."""
    return SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=Mock(return_value=(None, None)),
    )


@pytest.fixture
def sample_dataframe_with_features():
    """Create a sample DataFrame with features for XGBoost testing."""
    # Set deterministic seed for reproducible test data
    rng = np.random.default_rng(42)

    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    prices = 100 + np.cumsum(rng.standard_normal(100) * 0.5)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": rng.uniform(1000, 10000, 100),
        },
        index=dates,
    )

    # Add some mock feature columns that might be added by indicator engine
    df["sma_20"] = df["close"].rolling(20).mean()
    df["rsi"] = 50 + rng.standard_normal(100) * 10

    return df


class TestXGBoostBatchClassDiversity:
    """Test class diversity validation in XGBoost batch calculation."""

    def test_skips_when_less_than_2_classes(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test that periods with less than 2 classes are skipped."""
        calculator = HybridSignalCalculator(mock_data_fetcher, enabled_indicators=["xgboost"])

        # Mock indicator engine to return DataFrame with only 1 class in Target column
        def mock_compute_features(df):
            df_copy = df.copy()
            # Add Target column with only one class (e.g., all 1s)
            df_copy["Target"] = 1
            return df_copy

        with patch("modules.position_sizing.core.hybrid_signal_calculator.IndicatorEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine.compute_features = Mock(side_effect=mock_compute_features)
            mock_engine_class.return_value = mock_engine

            # Mock apply_directional_labels to preserve Target column
            with patch(
                "modules.position_sizing.core.hybrid_signal_calculator.apply_directional_labels"
            ) as mock_apply_labels:

                def mock_apply(df):
                    # Return DataFrame with only one class
                    df_copy = df.copy()
                    if "Target" not in df_copy.columns:
                        df_copy["Target"] = 1  # Only class 1
                    return df_copy

                mock_apply_labels.side_effect = mock_apply

                # Mock train_and_predict to not be called (should be skipped)
                with patch("modules.position_sizing.core.hybrid_signal_calculator.train_and_predict") as mock_train:
                    result_df = calculator._calc_xgboost_batch(
                        df=sample_dataframe_with_features, symbol="BTC/USDT", timeframe="1h"
                    )

                    # Should return DataFrame with zeros (no predictions made)
                    assert isinstance(result_df, pd.DataFrame)
                    assert "signal" in result_df.columns
                    assert "confidence" in result_df.columns
                    # train_and_predict should not be called because class diversity check fails
                    # The check at line 1119 (len(unique_classes) < 2) happens before train_and_predict
                    mock_train.assert_not_called()

    def test_handles_class_diversity_error(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test that ClassDiversityError exceptions are handled correctly."""
        calculator = HybridSignalCalculator(mock_data_fetcher, enabled_indicators=["xgboost"])

        # Mock indicator engine
        def mock_compute_features(df):
            df_copy = df.copy()
            df_copy["Target"] = [0, 1] * (len(df) // 2)  # Two classes
            return df_copy

        with patch("modules.position_sizing.core.hybrid_signal_calculator.IndicatorEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine.compute_features = Mock(side_effect=mock_compute_features)
            mock_engine_class.return_value = mock_engine

            # Mock apply_directional_labels
            with patch(
                "modules.position_sizing.core.hybrid_signal_calculator.apply_directional_labels"
            ) as mock_apply_labels:

                def mock_apply(df):
                    df_copy = df.copy()
                    if "Target" not in df_copy.columns:
                        df_copy["Target"] = [0, 1] * (len(df) // 2)
                    return df_copy

                mock_apply_labels.side_effect = mock_apply

                # Mock train_and_predict to raise ClassDiversityError
                with patch("modules.position_sizing.core.hybrid_signal_calculator.train_and_predict") as mock_train:
                    mock_train.side_effect = ClassDiversityError(
                        "Invalid classes inferred from unique values of `y`. Expected: [0], got [1]"
                    )

                    result_df = calculator._calc_xgboost_batch(
                        df=sample_dataframe_with_features, symbol="BTC/USDT", timeframe="1h"
                    )

                    # Should handle the error gracefully and skip the period
                    assert isinstance(result_df, pd.DataFrame)
                    # Verify expected columns exist
                    expected_columns = ["signal", "confidence"]
                    assert set(result_df.columns) == set(expected_columns), (
                        f"Expected columns {expected_columns}, got {result_df.columns.tolist()}"
                    )
                    # Verify DataFrame has at least one row
                    assert len(result_df) > 0, "Result DataFrame should have at least one row"
                    # Verify column dtypes
                    assert pd.api.types.is_integer_dtype(result_df["signal"]) or pd.api.types.is_numeric_dtype(
                        result_df["signal"]
                    ), f"Expected 'signal' column to be numeric, got {result_df['signal'].dtype}"
                    assert pd.api.types.is_float_dtype(result_df["confidence"]) or pd.api.types.is_numeric_dtype(
                        result_df["confidence"]
                    ), f"Expected 'confidence' column to be float/numeric, got {result_df['confidence'].dtype}"
                    # Verify key columns are not all null (at least some values exist)
                    assert not result_df["signal"].isna().all(), "Signal column should not be all null"
                    assert not result_df["confidence"].isna().all(), "Confidence column should not be all null"

    def test_handles_missing_class_error(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test that ClassDiversityError for missing class 0 is handled correctly."""
        calculator = HybridSignalCalculator(mock_data_fetcher, enabled_indicators=["xgboost"])

        # Mock indicator engine
        def mock_compute_features(df):
            df_copy = df.copy()
            df_copy["Target"] = [1, 2] * (len(df) // 2)  # Classes 1 and 2, missing 0
            return df_copy

        with patch("modules.position_sizing.core.hybrid_signal_calculator.IndicatorEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine.compute_features = Mock(side_effect=mock_compute_features)
            mock_engine_class.return_value = mock_engine

            # Mock apply_directional_labels
            with patch(
                "modules.position_sizing.core.hybrid_signal_calculator.apply_directional_labels"
            ) as mock_apply_labels:

                def mock_apply(df):
                    df_copy = df.copy()
                    if "Target" not in df_copy.columns:
                        df_copy["Target"] = [1, 2] * (len(df) // 2)
                    return df_copy

                mock_apply_labels.side_effect = mock_apply

                # Mock train_and_predict to raise ClassDiversityError
                with patch("modules.position_sizing.core.hybrid_signal_calculator.train_and_predict") as mock_train:
                    mock_train.side_effect = ClassDiversityError(
                        "Training set missing class 0 (DOWN). Found classes: [1, 2]."
                    )

                    result_df = calculator._calc_xgboost_batch(
                        df=sample_dataframe_with_features, symbol="BTC/USDT", timeframe="1h"
                    )

                    # Should handle the error gracefully
                    assert isinstance(result_df, pd.DataFrame)
                    assert "signal" in result_df.columns
                    assert "confidence" in result_df.columns

    def test_handles_invalid_classes_error(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test that ClassDiversityError for invalid classes is handled correctly."""
        calculator = HybridSignalCalculator(mock_data_fetcher, enabled_indicators=["xgboost"])

        # Mock indicator engine
        def mock_compute_features(df):
            df_copy = df.copy()
            df_copy["Target"] = [0, 1] * (len(df) // 2)
            return df_copy

        with patch("modules.position_sizing.core.hybrid_signal_calculator.IndicatorEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine.compute_features = Mock(side_effect=mock_compute_features)
            mock_engine_class.return_value = mock_engine

            # Mock apply_directional_labels
            with patch(
                "modules.position_sizing.core.hybrid_signal_calculator.apply_directional_labels"
            ) as mock_apply_labels:

                def mock_apply(df):
                    df_copy = df.copy()
                    if "Target" not in df_copy.columns:
                        df_copy["Target"] = [0, 1] * (len(df) // 2)
                    return df_copy

                mock_apply_labels.side_effect = mock_apply

                # Mock train_and_predict to raise ClassDiversityError
                with patch("modules.position_sizing.core.hybrid_signal_calculator.train_and_predict") as mock_train:
                    mock_train.side_effect = ClassDiversityError(
                        "Number of classes, 1, does not match size of target_names, 3. Invalid classes inferred."
                    )

                    result_df = calculator._calc_xgboost_batch(
                        df=sample_dataframe_with_features, symbol="BTC/USDT", timeframe="1h"
                    )

                    # Should handle the error gracefully
                    assert isinstance(result_df, pd.DataFrame)
                    # Verify expected columns exist
                    expected_columns = ["signal", "confidence"]
                    assert set(result_df.columns) == set(expected_columns), (
                        f"Expected columns {expected_columns}, got {result_df.columns.tolist()}"
                    )
                    # Verify DataFrame has at least one row
                    assert len(result_df) > 0, "Result DataFrame should have at least one row"
                    # Verify column dtypes
                    assert pd.api.types.is_integer_dtype(result_df["signal"]) or pd.api.types.is_numeric_dtype(
                        result_df["signal"]
                    ), f"Expected 'signal' column to be numeric, got {result_df['signal'].dtype}"
                    assert pd.api.types.is_float_dtype(result_df["confidence"]) or pd.api.types.is_numeric_dtype(
                        result_df["confidence"]
                    ), f"Expected 'confidence' column to be float/numeric, got {result_df['confidence'].dtype}"
                    # Verify key columns are not all null (at least some values exist)
                    assert not result_df["signal"].isna().all(), "Signal column should not be all null"
                    assert not result_df["confidence"].isna().all(), "Confidence column should not be all null"

    def test_reraises_non_class_related_errors(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test that non-class-related errors are re-raised."""
        calculator = HybridSignalCalculator(mock_data_fetcher, enabled_indicators=["xgboost"])

        # Mock indicator engine
        def mock_compute_features(df):
            df_copy = df.copy()
            df_copy["Target"] = [0, 1, 2] * (len(df) // 3 + 1)
            return df_copy[: len(df)]

        with patch("modules.position_sizing.core.hybrid_signal_calculator.IndicatorEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine.compute_features = Mock(side_effect=mock_compute_features)
            mock_engine_class.return_value = mock_engine

            # Mock apply_directional_labels
            with patch(
                "modules.position_sizing.core.hybrid_signal_calculator.apply_directional_labels"
            ) as mock_apply_labels:

                def mock_apply(df):
                    df_copy = df.copy()
                    if "Target" not in df_copy.columns:
                        df_copy["Target"] = [0, 1, 2] * (len(df) // 3 + 1)
                    return df_copy[: len(df)]

                mock_apply_labels.side_effect = mock_apply

                # Mock train_and_predict to raise a different error (not class-related)
                with patch("modules.position_sizing.core.hybrid_signal_calculator.train_and_predict") as mock_train:
                    mock_train.side_effect = ValueError("Some other error that is not related to classes")

                    # Should re-raise the error (it will be caught by outer exception handler)
                    with pytest.raises(ValueError, match="Some other error"):
                        calculator._calc_xgboost_batch(
                            df=sample_dataframe_with_features, symbol="BTC/USDT", timeframe="1h"
                        )

    def test_handles_class_diversity_error_from_xgboost(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test that ClassDiversityError raised by train_and_predict is handled correctly."""
        calculator = HybridSignalCalculator(mock_data_fetcher, enabled_indicators=["xgboost"])

        # Mock indicator engine
        def mock_compute_features(df):
            df_copy = df.copy()
            df_copy["Target"] = [0, 1] * (len(df) // 2)
            return df_copy

        with patch("modules.position_sizing.core.hybrid_signal_calculator.IndicatorEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine.compute_features = Mock(side_effect=mock_compute_features)
            mock_engine_class.return_value = mock_engine

            # Mock apply_directional_labels
            with patch(
                "modules.position_sizing.core.hybrid_signal_calculator.apply_directional_labels"
            ) as mock_apply_labels:

                def mock_apply(df):
                    df_copy = df.copy()
                    if "Target" not in df_copy.columns:
                        df_copy["Target"] = [0, 1] * (len(df) // 2)
                    return df_copy

                mock_apply_labels.side_effect = mock_apply

                # Mock train_and_predict to raise ClassDiversityError
                with patch("modules.position_sizing.core.hybrid_signal_calculator.train_and_predict") as mock_train:
                    mock_train.side_effect = ClassDiversityError(
                        "XGBoost class mismatch: Invalid classes in training data"
                    )

                    result_df = calculator._calc_xgboost_batch(
                        df=sample_dataframe_with_features, symbol="BTC/USDT", timeframe="1h"
                    )

                    # Should handle the error gracefully
                    assert isinstance(result_df, pd.DataFrame)
                    # Verify expected columns (order matters as it's consistent with implementation)
                    expected_columns = ["signal", "confidence"]
                    assert result_df.columns.tolist() == expected_columns, (
                        f"Expected columns {expected_columns}, got {result_df.columns.tolist()}"
                    )

    def test_skips_when_no_target_column(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test that periods without Target column are skipped."""
        calculator = HybridSignalCalculator(mock_data_fetcher, enabled_indicators=["xgboost"])

        # Mock indicator engine to return DataFrame without Target column
        def mock_compute_features(df):
            df_copy = df.copy()
            # Don't add Target column
            return df_copy

        with patch("modules.position_sizing.core.hybrid_signal_calculator.IndicatorEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine.compute_features = Mock(side_effect=mock_compute_features)
            mock_engine_class.return_value = mock_engine

            # Mock apply_directional_labels to not add Target column
            with patch(
                "modules.position_sizing.core.hybrid_signal_calculator.apply_directional_labels"
            ) as mock_apply_labels:

                def mock_apply(df):
                    # Return DataFrame without Target column
                    return df.copy()

                mock_apply_labels.side_effect = mock_apply

                result_df = calculator._calc_xgboost_batch(
                    df=sample_dataframe_with_features, symbol="BTC/USDT", timeframe="1h"
                )

                # Should return DataFrame with zeros (no predictions made)
                assert isinstance(result_df, pd.DataFrame)
                assert "signal" in result_df.columns
                assert "confidence" in result_df.columns

    def test_handles_insufficient_periods(self, mock_data_fetcher):
        """Test that DataFrame with less than min_periods (50) returns empty results."""
        calculator = HybridSignalCalculator(mock_data_fetcher, enabled_indicators=["xgboost"])

        # Create DataFrame with less than 50 periods
        # Set deterministic seed for reproducible test data
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="h")
        df = pd.DataFrame(
            {
                "open": rng.standard_normal(30) + 100,
                "high": rng.standard_normal(30) + 101,
                "low": rng.standard_normal(30) + 99,
                "close": rng.standard_normal(30) + 100,
            },
            index=dates,
        )

        result_df = calculator._calc_xgboost_batch(df=df, symbol="BTC/USDT", timeframe="1h")

        # Should return DataFrame with zeros
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(df)
        assert (result_df["signal"] == 0).all()
        assert (result_df["confidence"] == 0.0).all()

    def test_handles_missing_required_columns(self, mock_data_fetcher):
        """Test that missing required columns (high, low, close) returns empty results."""
        calculator = HybridSignalCalculator(mock_data_fetcher, enabled_indicators=["xgboost"])

        # Create DataFrame without required columns
        # Set deterministic seed for reproducible test data
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        df = pd.DataFrame(
            {
                "open": rng.standard_normal(100) + 100,
                # Missing 'high', 'low', 'close'
            },
            index=dates,
        )

        result_df = calculator._calc_xgboost_batch(df=df, symbol="BTC/USDT", timeframe="1h")

        # Should return DataFrame with zeros
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(df)
        assert (result_df["signal"] == 0).all()
        assert (result_df["confidence"] == 0.0).all()
