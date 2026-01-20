"""
Unit tests for signal_random_forest.py

This module contains comprehensive tests for the Random Forest signal generation script,
including validation, data fetching, batch processing, and resource monitoring.
"""

import os

# Add parent directories to path for imports
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.random_forest.signal_random_forest import (
    ResourceMonitor,
    auto_train_model,
    calculate_confidence,
    create_batches,
    get_signal_direction,
    get_top_volume_symbols,
    validate_model,
)


class TestSignalFunctions(unittest.TestCase):
    """Test basic signal processing functions."""

    def test_get_signal_direction(self):
        """Test signal direction mapping."""
        self.assertEqual(get_signal_direction(1), "LONG")
        self.assertEqual(get_signal_direction(-1), "SHORT")
        self.assertEqual(get_signal_direction(0), "NEUTRAL")
        self.assertEqual(get_signal_direction(2), "NEUTRAL")  # Invalid signal

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        self.assertEqual(calculate_confidence(0.5), 0.5)
        self.assertEqual(calculate_confidence(1.5), 1.0)  # Clamped to 1.0
        self.assertEqual(calculate_confidence(-0.5), 0.0)  # Clamped to 0.0


class TestValidateModel(unittest.TestCase):
    """Test model validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")

        # Create a mock model with deprecated features
        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["open", "high", "low", "close", "volume"]

        # Save mock model
        import joblib

        joblib.dump(mock_model, self.model_path)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("modules.random_forest.signal_random_forest.load_random_forest_model")
    def test_validate_model_deprecated_features(self, mock_load):
        """Test validation of model with deprecated features."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["open", "high", "low", "close", "volume"]
        mock_load.return_value = mock_model

        is_valid, error_msg = validate_model(self.model_path)
        self.assertFalse(is_valid)
        self.assertIn("deprecated raw OHLCV features", error_msg)

    @patch("modules.random_forest.signal_random_forest.load_random_forest_model")
    def test_validate_model_valid(self, mock_load):
        """Test validation of valid model."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["returns_1", "log_volume", "high_low_range"]
        mock_load.return_value = mock_model

        is_valid, error_msg = validate_model(self.model_path)
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)

    @patch("modules.random_forest.signal_random_forest.load_random_forest_model")
    def test_validate_model_with_symbols(self, mock_load):
        """Test validation with specific symbols."""
        mock_model = MagicMock()
        mock_model.feature_names_in_ = ["returns_1", "log_volume", "high_low_range"]
        mock_load.return_value = mock_model

        # Mock data fetcher
        mock_data_fetcher = MagicMock()
        mock_df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )

        with patch("modules.random_forest.signal_random_forest.get_random_forest_signal", return_value=(1, 0.8)):
            mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

            is_valid, error_msg = validate_model(
                self.model_path, data_fetcher=mock_data_fetcher, symbols=["BTC/USDT"], timeframe="1h"
            )
            self.assertTrue(is_valid)
            self.assertIsNone(error_msg)


class TestCreateBatches(unittest.TestCase):
    """Test batch creation functionality."""

    def test_create_batches_normal(self):
        """Test normal batch creation."""
        items = ["a", "b", "c", "d", "e"]
        batches = create_batches(items, 2)
        expected = [["a", "b"], ["c", "d"], ["e"]]
        self.assertEqual(batches, expected)

    def test_create_batches_empty(self):
        """Test batch creation with empty list."""
        batches = create_batches([], 2)
        self.assertEqual(batches, [])

    def test_create_batches_large_batch(self):
        """Test batch creation with batch size larger than list."""
        items = ["a", "b"]
        batches = create_batches(items, 5)
        expected = [["a", "b"]]
        self.assertEqual(batches, expected)


class TestResourceMonitor(unittest.TestCase):
    """Test resource monitoring functionality."""

    @patch("modules.random_forest.signal_random_forest.PSUTIL_AVAILABLE", True)
    @patch("psutil.Process")
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    def test_resource_monitor_with_psutil(self, mock_cpu, mock_memory, mock_process):
        """Test resource monitor when psutil is available."""
        # Mock process
        mock_proc_instance = MagicMock()
        mock_proc_instance.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)  # 100MB
        mock_proc_instance.cpu_percent.return_value = 10.0
        mock_process.return_value = mock_proc_instance

        # Mock system memory
        mock_mem_instance = MagicMock()
        mock_mem_instance.percent = 50.0
        mock_mem_instance.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_memory.return_value = mock_mem_instance

        # Mock CPU
        mock_cpu.return_value = 25.0

        monitor = ResourceMonitor(max_memory_pct=70.0, max_cpu_pct=80.0)

        # Test memory usage
        mem_mb, mem_pct, avail_mb = monitor.get_memory_usage()
        self.assertEqual(mem_mb, 100.0)
        self.assertEqual(mem_pct, 50.0)
        self.assertEqual(avail_mb, 4 * 1024.0)

        # Test CPU usage
        proc_cpu, sys_cpu = monitor.get_cpu_usage()
        self.assertEqual(proc_cpu, 10.0)
        self.assertEqual(sys_cpu, 25.0)

        # Test limits
        self.assertTrue(monitor.check_memory_limit())
        self.assertTrue(monitor.check_cpu_limit())

    @patch("modules.random_forest.signal_random_forest.PSUTIL_AVAILABLE", False)
    def test_resource_monitor_without_psutil(self):
        """Test resource monitor when psutil is not available."""
        monitor = ResourceMonitor()

        mem_mb, mem_pct, avail_mb = monitor.get_memory_usage()
        self.assertEqual((mem_mb, mem_pct, avail_mb), (0.0, 0.0, 0.0))

        proc_cpu, sys_cpu = monitor.get_cpu_usage()
        self.assertEqual((proc_cpu, sys_cpu), (0.0, 0.0))

        self.assertTrue(monitor.check_memory_limit())
        self.assertTrue(monitor.check_cpu_limit())


class TestGetTopVolumeSymbols(unittest.TestCase):
    """Test fetching top volume symbols."""

    @patch("modules.random_forest.signal_random_forest.ExchangeManager")
    def test_get_top_volume_symbols_success(self, mock_exchange_manager):
        """Test successful fetching of top volume symbols."""
        # Mock exchange
        mock_exchange = MagicMock()
        mock_tickers = {
            "BTC/USDT": {"quoteVolume": 1000000},
            "ETH/USDT": {"quoteVolume": 800000},
            "ADA/USDT": {"quoteVolume": 600000},
        }
        mock_exchange.fetch_tickers.return_value = mock_tickers

        mock_exchange_manager_instance = MagicMock()
        mock_exchange_manager_instance.get_exchange.return_value = mock_exchange
        mock_exchange_manager.return_value = mock_exchange_manager_instance

        # Mock data fetcher
        mock_data_fetcher = MagicMock()

        symbols = get_top_volume_symbols(mock_data_fetcher, top_n=2)
        self.assertEqual(symbols, ["BTC/USDT", "ETH/USDT"])

    @patch("modules.random_forest.signal_random_forest.ExchangeManager")
    def test_get_top_volume_symbols_no_volume(self, mock_exchange_manager):
        """Test fetching symbols when no volume data available."""
        # Mock exchange with no volume data
        mock_exchange = MagicMock()
        mock_tickers = {
            "BTC/USDT": {},
            "ETH/USDT": {"quoteVolume": 0},
        }
        mock_exchange.fetch_tickers.return_value = mock_tickers

        mock_exchange_manager_instance = MagicMock()
        mock_exchange_manager_instance.get_exchange.return_value = mock_exchange
        mock_exchange_manager.return_value = mock_exchange_manager_instance

        mock_data_fetcher = MagicMock()

        symbols = get_top_volume_symbols(mock_data_fetcher, top_n=5)
        self.assertEqual(symbols, [])


class TestAutoTrainModel(unittest.TestCase):
    """Test auto training functionality."""

    @patch("modules.random_forest.signal_random_forest.train_random_forest_model")
    def test_auto_train_model_success(self, mock_train):
        """Test successful auto training."""
        mock_train.return_value = MagicMock()  # Mock trained model

        mock_data_fetcher = MagicMock()
        mock_df = pd.DataFrame(
            {"open": [100, 101], "high": [105, 106], "low": [95, 96], "close": [102, 103], "volume": [1000, 1100]}
        )
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

        symbols = ["BTC/USDT", "ETH/USDT"]

        success, model_path = auto_train_model(mock_data_fetcher, symbols, "1h")
        self.assertTrue(success)
        self.assertIsInstance(model_path, str)
        mock_train.assert_called_once()

    @patch("modules.random_forest.signal_random_forest.train_random_forest_model")
    def test_auto_train_model_no_data(self, mock_train):
        """Test auto training when no data is available."""
        mock_data_fetcher = MagicMock()
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (None, None)

        symbols = ["BTC/USDT"]

        success, error_msg = auto_train_model(mock_data_fetcher, symbols, "1h")
        self.assertFalse(success)
        self.assertEqual(error_msg, "No data fetched for any symbol")
        mock_train.assert_not_called()

    @patch("modules.random_forest.signal_random_forest.train_random_forest_model")
    def test_auto_train_model_training_fails(self, mock_train):
        """Test auto training when model training fails."""
        mock_train.return_value = None

        mock_data_fetcher = MagicMock()
        mock_df = pd.DataFrame(
            {"open": [100, 101], "high": [105, 106], "low": [95, 96], "close": [102, 103], "volume": [1000, 1100]}
        )
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, "binance")

        symbols = ["BTC/USDT"]

        success, error_msg = auto_train_model(mock_data_fetcher, symbols, "1h")
        self.assertFalse(success)
        self.assertEqual(error_msg, "Model training failed")


if __name__ == "__main__":
    unittest.main()
