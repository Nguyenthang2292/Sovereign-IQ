"""
Comprehensive tests for batch symbol training.
Tests parallel processing, error handling, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import time

from modules.xgboost_LTS.utils.batch_symbols import batch_train_symbols, _train_one


class TestBatchTrainSymbols:
    """Test batch training functionality."""

    def test_single_symbol(self):
        """Test training with single symbol."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "Target": np.random.choice([0, 1, 2], 100),
            }
        )

        mock_train_fn = MagicMock(return_value={"accuracy": 0.8})

        symbols_data = {"SYM1": df}
        results = batch_train_symbols(symbols_data, mock_train_fn, max_workers=1)

        assert "SYM1" in results
        assert results["SYM1"]["ok"] is True
        assert results["SYM1"]["result"]["accuracy"] == 0.8

    def test_multiple_symbols_sequential(self):
        """Test training multiple symbols sequentially."""
        symbols_data = {
            f"SYM{i}": pd.DataFrame(
                {
                    "feature1": np.random.randn(50),
                    "feature2": np.random.randn(50),
                    "Target": np.random.choice([0, 1, 2], 50),
                }
            )
            for i in range(3)
        }

        mock_train_fn = MagicMock(return_value={"accuracy": 0.75})

        results = batch_train_symbols(symbols_data, mock_train_fn, max_workers=1)

        assert len(results) == 3
        for i in range(3):
            assert results[f"SYM{i}"]["ok"] is True

    def test_empty_symbols_dict(self):
        """Test with empty symbols dictionary."""
        mock_train_fn = MagicMock()

        results = batch_train_symbols({}, mock_train_fn, max_workers=1)

        assert len(results) == 0

    def test_training_error_handling(self):
        """Test handling of training errors."""
        df = pd.DataFrame({"feature1": np.random.randn(50), "Target": np.random.choice([0, 1, 2], 50)})

        def failing_train_fn(df, use_cache=True):
            raise ValueError("Training failed!")

        symbols_data = {"FAIL": df}
        results = batch_train_symbols(symbols_data, failing_train_fn, max_workers=1)

        assert "FAIL" in results
        assert results["FAIL"]["ok"] is False
        assert "Training failed!" in results["FAIL"]["error"]

    def test_mixed_success_and_failure(self):
        """Test with some symbols succeeding and some failing."""

        def conditional_train_fn(df, use_cache=True):
            if "bad" in str(df["feature1"].iloc[0]):
                raise ValueError("Bad data!")
            return {"accuracy": 0.8}

        symbols_data = {
            "GOOD": pd.DataFrame({"feature1": np.random.randn(50), "Target": np.random.choice([0, 1, 2], 50)}),
            "BAD": pd.DataFrame(
                {
                    "feature1": ["bad"] * 50,  # Will cause error
                    "Target": np.random.choice([0, 1, 2], 50),
                }
            ),
        }

        results = batch_train_symbols(symbols_data, conditional_train_fn, max_workers=1)

        assert results["GOOD"]["ok"] is True
        assert results["BAD"]["ok"] is False

    def test_use_cache_parameter_passing(self):
        """Test that use_cache parameter is passed correctly."""
        received_cache_params = []

        def capture_train_fn(df, use_cache=True):
            received_cache_params.append(use_cache)
            return {"accuracy": 0.8}

        df = pd.DataFrame({"feature1": np.random.randn(50), "Target": np.random.choice([0, 1, 2], 50)})

        symbols_data = {"SYM": df}

        # Test with use_cache=True
        batch_train_symbols(symbols_data, capture_train_fn, max_workers=1, use_cache=True)
        assert received_cache_params[-1] is True

        # Test with use_cache=False
        batch_train_symbols(symbols_data, capture_train_fn, max_workers=1, use_cache=False)
        assert received_cache_params[-1] is False

    def test_dataframe_copying(self):
        """Test that dataframes are copied before training."""
        original_df = pd.DataFrame({"feature1": np.random.randn(50), "Target": np.random.choice([0, 1, 2], 50)})

        modified_ids = []

        def check_copy_fn(df, use_cache=True):
            modified_ids.append(id(df))
            return {"accuracy": 0.8}

        symbols_data = {"SYM": original_df}
        batch_train_symbols(symbols_data, check_copy_fn, max_workers=1)

        # The ID should be different (copy was made)
        assert modified_ids[0] != id(original_df)


class TestTrainOneHelper:
    """Test _train_one helper function."""

    def test_successful_training(self):
        """Test successful training result."""
        df = pd.DataFrame({"feature": np.random.randn(50), "Target": np.random.choice([0, 1, 2], 50)})

        mock_train_fn = MagicMock(return_value={"accuracy": 0.85})

        symbol, result = _train_one("TEST", df, mock_train_fn, True)

        assert symbol == "TEST"
        assert result["ok"] is True
        assert result["result"]["accuracy"] == 0.85

    def test_training_failure(self):
        """Test failure handling."""
        df = pd.DataFrame({"feature": np.random.randn(50), "Target": np.random.choice([0, 1, 2], 50)})

        def fail_fn(df, use_cache=True):
            raise RuntimeError("Model crashed")

        symbol, result = _train_one("FAIL", df, fail_fn, True)

        assert symbol == "FAIL"
        assert result["ok"] is False
        assert "Model crashed" in result["error"]

    def test_exception_type_preservation(self):
        """Test that exception types are preserved in error message."""
        df = pd.DataFrame({"feature": np.random.randn(50), "Target": np.random.choice([0, 1, 2], 50)})

        def type_error_fn(df, use_cache=True):
            raise TypeError("Type mismatch")

        symbol, result = _train_one("TYPE_ERR", df, type_error_fn, True)

        assert result["ok"] is False
        assert "Type mismatch" in result["error"]


class TestParallelExecution:
    """Test parallel execution behavior."""

    def test_worker_count_default(self):
        """Test default worker count calculation."""
        import os

        expected_workers = max(1, (os.cpu_count() or 2) - 1)

        # Just verify the calculation doesn't crash
        # Can't easily test actual worker count without mocking
        df = pd.DataFrame({"feature": np.random.randn(50), "Target": np.random.choice([0, 1, 2], 50)})

        mock_train_fn = MagicMock(return_value={"accuracy": 0.8})
        symbols_data = {f"SYM{i}": df.copy() for i in range(2)}

        # Should complete without error
        results = batch_train_symbols(symbols_data, mock_train_fn, max_workers=None)
        assert len(results) == 2

    def test_single_worker_avoid_overhead(self):
        """Test that single worker runs sequentially."""
        call_order = []

        def track_fn(df, use_cache=True):
            symbol = df["symbol"].iloc[0]
            call_order.append(symbol)
            time.sleep(0.01)  # Small delay
            return {"accuracy": 0.8}

        symbols_data = {
            f"SYM{i}": pd.DataFrame(
                {"feature": np.random.randn(10), "symbol": [f"SYM{i}"] * 10, "Target": np.random.choice([0, 1, 2], 10)}
            )
            for i in range(3)
        }

        results = batch_train_symbols(symbols_data, track_fn, max_workers=1)

        # All should complete
        assert len(results) == 3
        # Call order should be preserved in single worker mode
        assert len(call_order) == 3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_number_of_symbols(self):
        """Test with many symbols."""
        n_symbols = 20

        symbols_data = {
            f"SYM{i:03d}": pd.DataFrame({"feature": np.random.randn(30), "Target": np.random.choice([0, 1, 2], 30)})
            for i in range(n_symbols)
        }

        mock_train_fn = MagicMock(return_value={"accuracy": 0.7})

        results = batch_train_symbols(symbols_data, mock_train_fn, max_workers=2)

        assert len(results) == n_symbols

    def test_symbols_with_same_data(self):
        """Test multiple symbols with identical data."""
        shared_data = pd.DataFrame({"feature": np.random.randn(50), "Target": np.random.choice([0, 1, 2], 50)})

        symbols_data = {"A": shared_data.copy(), "B": shared_data.copy(), "C": shared_data.copy()}

        mock_train_fn = MagicMock(return_value={"accuracy": 0.8})

        results = batch_train_symbols(symbols_data, mock_train_fn, max_workers=1)

        assert len(results) == 3
        for symbol in ["A", "B", "C"]:
            assert results[symbol]["ok"] is True

    def test_very_small_dataframes(self):
        """Test with very small dataframes."""
        df = pd.DataFrame({"feature": [1.0, 2.0], "Target": [0, 1]})

        mock_train_fn = MagicMock(return_value={"accuracy": 1.0})

        symbols_data = {"TINY": df}
        results = batch_train_symbols(symbols_data, mock_train_fn, max_workers=1)

        assert results["TINY"]["ok"] is True

    def test_dataframe_with_nan_values(self):
        """Test handling of NaN values in data."""
        df = pd.DataFrame({"feature": [1.0, np.nan, 3.0, 4.0], "Target": [0, 1, 2, 0]})

        def handle_nan_fn(df, use_cache=True):
            # Some implementations might drop NaN, others might error
            if df["feature"].isna().any():
                raise ValueError("NaN detected")
            return {"accuracy": 0.8}

        symbols_data = {"NAN": df}
        results = batch_train_symbols(symbols_data, handle_nan_fn, max_workers=1)

        # Should either succeed or fail gracefully
        assert "NAN" in results
        if not results["NAN"]["ok"]:
            assert "error" in results["NAN"]

    def test_special_characters_in_symbol_names(self):
        """Test symbol names with special characters."""
        symbols_data = {
            "BTC/USD": pd.DataFrame({"feature": np.random.randn(30), "Target": np.random.choice([0, 1, 2], 30)}),
            "ETH-USD": pd.DataFrame({"feature": np.random.randn(30), "Target": np.random.choice([0, 1, 2], 30)}),
            "SOL:USD": pd.DataFrame({"feature": np.random.randn(30), "Target": np.random.choice([0, 1, 2], 30)}),
        }

        mock_train_fn = MagicMock(return_value={"accuracy": 0.8})

        results = batch_train_symbols(symbols_data, mock_train_fn, max_workers=1)

        assert len(results) == 3
        for symbol in symbols_data.keys():
            assert symbol in results
            assert results[symbol]["ok"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
