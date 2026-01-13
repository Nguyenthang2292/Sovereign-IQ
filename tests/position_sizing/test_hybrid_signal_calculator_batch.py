from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from modules.position_sizing.core.hybrid_signal_calculator import HybridSignalCalculator

"""
Tests for Hybrid Signal Calculator batch processing and precomputed indicators.

Tests cover:
- precompute_all_indicators_vectorized with XGBoost
- Exception handling in batch processing
- XGBoost batch calculation error handling
"""


@pytest.fixture
def mock_data_fetcher():
    """Create a mock data fetcher for testing."""
    return SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=Mock(return_value=(None, None)),
    )


@pytest.fixture
def sample_dataframe_with_features():
    """Create a sample DataFrame with features for batch testing."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
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

    return df


class TestPrecomputeAllIndicatorsVectorized:
    """Test precompute_all_indicators_vectorized method."""

    def test_precompute_returns_dict_of_dataframes(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test that precompute_all_indicators_vectorized returns dict of DataFrames."""
        calculator = HybridSignalCalculator(mock_data_fetcher, enabled_indicators=["range_oscillator", "xgboost"])

        with (
            patch.object(calculator, "_calc_range_oscillator_vectorized") as mock_osc,
            patch.object(calculator, "_calc_xgboost_batch") as mock_xgb,
        ):
            # Mock return values
            mock_osc_df = pd.DataFrame(
                {
                    "signal": [1] * len(sample_dataframe_with_features),
                    "confidence": [0.8] * len(sample_dataframe_with_features),
                },
                index=sample_dataframe_with_features.index,
            )

            mock_xgb_df = pd.DataFrame(
                {
                    "signal": [1] * len(sample_dataframe_with_features),
                    "confidence": [0.7] * len(sample_dataframe_with_features),
                },
                index=sample_dataframe_with_features.index,
            )

            mock_osc.return_value = mock_osc_df
            mock_xgb.return_value = mock_xgb_df

            result = calculator.precompute_all_indicators_vectorized(
                df=sample_dataframe_with_features,
                symbol="BTC/USDT",
                timeframe="1h",
                osc_length=50,
                osc_mult=2.0,
                osc_strategies=None,
                spc_params=None,
            )

            # Should return dict
            assert isinstance(result, dict)
            assert "range_oscillator" in result
            assert "xgboost" in result
            assert isinstance(result["range_oscillator"], pd.DataFrame)
            assert isinstance(result["xgboost"], pd.DataFrame)

    def test_precompute_handles_xgboost_exception(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test that XGBoost batch calculation exceptions are handled gracefully."""
        calculator = HybridSignalCalculator(mock_data_fetcher, enabled_indicators=["xgboost"])

        with patch.object(calculator, "_calc_xgboost_batch") as mock_xgb:
            # Mock XGBoost to raise exception
            mock_xgb.side_effect = Exception("XGBoost batch calculation failed")

            result = calculator.precompute_all_indicators_vectorized(
                df=sample_dataframe_with_features,
                symbol="BTC/USDT",
                timeframe="1h",
                osc_length=50,
                osc_mult=2.0,
                osc_strategies=None,
                spc_params=None,
            )

            # Should handle exception and return DataFrame with zeros
            assert isinstance(result, dict)
            assert "xgboost" in result
            assert isinstance(result["xgboost"], pd.DataFrame)
            # Should have zero signals and confidence when error occurs
            assert (result["xgboost"]["signal"] == 0).all()
            assert (result["xgboost"]["confidence"] == 0.0).all()

    def test_precompute_handles_all_indicators(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test that all enabled indicators are precomputed."""
        calculator = HybridSignalCalculator(
            mock_data_fetcher, enabled_indicators=["range_oscillator", "spc", "xgboost", "hmm", "random_forest"]
        )

        # Mock all batch calculation methods
        with (
            patch.object(calculator, "_calc_range_oscillator_vectorized") as mock_osc,
            patch.object(calculator, "_calc_spc_vectorized") as mock_spc,
            patch.object(calculator, "_calc_xgboost_batch") as mock_xgb,
            patch.object(calculator, "_calc_hmm_batch") as mock_hmm,
            patch.object(calculator, "_calc_random_forest_batch") as mock_rf,
        ):
            # Create mock DataFrames - each mock gets its own distinct DataFrame instance
            # to prevent in-place mutations in one mock from affecting others
            base_mock_df = pd.DataFrame(
                {
                    "signal": [1] * len(sample_dataframe_with_features),
                    "confidence": [0.8] * len(sample_dataframe_with_features),
                },
                index=sample_dataframe_with_features.index,
            )

            mock_osc.return_value = base_mock_df.copy()
            mock_spc.return_value = base_mock_df.copy()
            mock_xgb.return_value = base_mock_df.copy()
            mock_hmm.return_value = base_mock_df.copy()
            mock_rf.return_value = base_mock_df.copy()

            result = calculator.precompute_all_indicators_vectorized(
                df=sample_dataframe_with_features,
                symbol="BTC/USDT",
                timeframe="1h",
                osc_length=50,
                osc_mult=2.0,
                osc_strategies=None,
                spc_params=None,
            )

            # Should have all indicators
            assert "range_oscillator" in result
            assert "spc_cluster_transition" in result
            assert "spc_regime_following" in result
            assert "spc_mean_reversion" in result
            assert "xgboost" in result
            assert "hmm" in result
            assert "random_forest" in result


class TestXGBoostBatchExceptionHandling:
    """Test exception handling in XGBoost batch calculation."""

    def test_batch_calculation_handles_general_exception(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test that general exceptions in batch calculation are caught."""
        calculator = HybridSignalCalculator(mock_data_fetcher, enabled_indicators=["xgboost"])

        # Mock indicator engine to raise exception
        with patch("modules.position_sizing.core.hybrid_signal_calculator.IndicatorEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine.compute_features = Mock(side_effect=Exception("Unexpected error"))
            mock_engine_class.return_value = mock_engine

            result_df = calculator._calc_xgboost_batch(
                df=sample_dataframe_with_features, symbol="BTC/USDT", timeframe="1h"
            )

            # Should return DataFrame with zeros when exception occurs
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == len(sample_dataframe_with_features)
            # All signals should be 0 when error occurs in loop
            assert (result_df["signal"] == 0).all()
            assert (result_df["confidence"] == 0.0).all()

    def test_batch_calculation_continues_after_period_error(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test that batch calculation continues processing after error in one period."""
        calculator = HybridSignalCalculator(mock_data_fetcher, enabled_indicators=["xgboost"])

        call_count = 0

        def mock_compute_features(df):
            nonlocal call_count
            call_count += 1
            df_copy = df.copy()
            # Add Target column with sufficient classes
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

                # Mock train_and_predict to raise error on first call, succeed on others
                with (
                    patch("modules.position_sizing.core.hybrid_signal_calculator.train_and_predict") as mock_train,
                    patch("modules.position_sizing.core.hybrid_signal_calculator.predict_next_move") as mock_predict,
                ):

                    def train_side_effect(df):
                        # Raise error only on first call
                        if mock_train.call_count == 1:
                            raise ValueError("First call error")
                        # Return mock model for subsequent calls
                        return Mock()

                    mock_train.side_effect = train_side_effect
                    mock_predict.return_value = np.array([0.3, 0.4, 0.3])  # Mock probabilities

                    result_df = calculator._calc_xgboost_batch(
                        df=sample_dataframe_with_features, symbol="BTC/USDT", timeframe="1h"
                    )

                    # Should continue processing despite first error
                    assert isinstance(result_df, pd.DataFrame)
                    assert len(result_df) == len(sample_dataframe_with_features)
                    assert "signal" in result_df.columns
                    assert "confidence" in result_df.columns

                    # Verify that multiple periods were processed
                    # compute_features should be called multiple times (once per period after min_periods)
                    assert call_count > 1, (
                        f"Expected compute_features to be called multiple times, but was called {call_count} times"
                    )

                    # train_and_predict should be called multiple times (proving processing continued after first error)
                    assert mock_train.call_count > 1, (
                        f"Expected train_and_predict to be called multiple times, but was called "
                        f"{mock_train.call_count} times"
                    )

                    # Verify that processing continued after the first error
                    # The first call should have failed, but subsequent calls should have succeeded
                    # Since mock_train raises error on first call and succeeds on others,
                    # we should have at least 2 calls (first fails, second+ succeed)
                    assert mock_train.call_count >= 2, (
                        "Expected at least 2 calls to train_and_predict (first fails, subsequent succeed)"
                    )

                    # Verify that result_df contains data for periods after the first error
                    # Since the first period (min_periods) fails but subsequent ones succeed,
                    # we should have some non-zero signals/confidence values
                    # Note: The first period will have 0 signal/confidence due to error,
                    # but later periods should have values from successful predictions
                    non_zero_signals = (result_df["signal"] != 0).sum()
                    non_zero_confidence = (result_df["confidence"] != 0.0).sum()
                    # At least some periods should have been successfully processed
                    assert non_zero_signals > 0 or non_zero_confidence > 0, (
                        "Expected some periods to have non-zero signals/confidence after first error, but all are zero"
                    )


class TestCalculateSignalFromPrecomputed:
    """Test calculate_signal_from_precomputed method."""

    def test_calculate_signal_from_precomputed_returns_tuple(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test that calculate_signal_from_precomputed returns tuple."""
        calculator = HybridSignalCalculator(mock_data_fetcher)

        # Create precomputed indicators
        precomputed = {
            "range_oscillator": pd.DataFrame(
                {
                    "signal": [1] * len(sample_dataframe_with_features),
                    "confidence": [0.8] * len(sample_dataframe_with_features),
                },
                index=sample_dataframe_with_features.index,
            ),
            "xgboost": pd.DataFrame(
                {
                    "signal": [1] * len(sample_dataframe_with_features),
                    "confidence": [0.7] * len(sample_dataframe_with_features),
                },
                index=sample_dataframe_with_features.index,
            ),
        }

        signal, confidence = calculator.calculate_signal_from_precomputed(
            precomputed_indicators=precomputed,
            period_index=50,
            signal_type="LONG",
        )

        assert isinstance(signal, int)
        assert signal in [-1, 0, 1]
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_calculate_single_signal_from_precomputed(self, mock_data_fetcher, sample_dataframe_with_features):
        """Test calculate_single_signal_from_precomputed method."""
        calculator = HybridSignalCalculator(mock_data_fetcher)

        # Create precomputed indicators
        precomputed = {
            "range_oscillator": pd.DataFrame(
                {
                    "signal": [1] * len(sample_dataframe_with_features),
                    "confidence": [0.8] * len(sample_dataframe_with_features),
                },
                index=sample_dataframe_with_features.index,
            ),
            "xgboost": pd.DataFrame(
                {
                    "signal": [-1] * len(sample_dataframe_with_features),
                    "confidence": [0.9] * len(sample_dataframe_with_features),  # Higher confidence
                },
                index=sample_dataframe_with_features.index,
            ),
        }

        signal, confidence = calculator.calculate_single_signal_from_precomputed(
            precomputed_indicators=precomputed,
            period_index=50,
        )

        # Should return signal with highest confidence (xgboost with -1 and 0.9)
        assert isinstance(signal, int)
        assert signal in [-1, 0, 1]
        assert isinstance(confidence, float)
        assert confidence == 0.9  # Should use xgboost confidence (higher)
        assert signal == -1  # Should use xgboost signal
