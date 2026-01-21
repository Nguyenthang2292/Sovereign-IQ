"""
Command line interface for training, evaluating, and running LSTM-based models
for time series prediction, such as cryptocurrency price forecasting.

This module provides configurable options for model architectures (including
CNN and attention mechanisms) and data sources, and orchestrates
end-to-end workflows for data fetching, preprocessing, training, evaluation,
and real-time inference. Designed for extensibility and integration within a
modular forecasting system.

Typical usage:
    python main.py [OPTIONS] [COMMANDS]

Key Components:
    - ModelConfiguration: Class defining LSTM or hybrid model options.
    - Data loading utilities: Functions to fetch and validate time series data.
    - Trainer (LSTMTrainer): Handles model training and evaluation routines.
    - System tools: GPU management, logging, and error handling utilities.
"""

import inspect
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config import (
    DEFAULT_CRYPTO_SYMBOLS_FOR_TRAINING_DL,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    DEFAULT_TIMEFRAMES_FOR_TRAINING_DL,
)
from config.lstm import ENABLE_KALMAN_FILTER, KALMAN_OBSERVATION_VARIANCE, KALMAN_PROCESS_VARIANCE
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.system import PyTorchGPUManager
from modules.common.ui.logging import log_debug, log_error, log_info, log_warn
from modules.common.utils.data import fetch_ohlcv_data_dict, validate_ohlcv_input
from modules.lstm.models import LSTMTrainer
from modules.lstm.models.model_utils import get_latest_signal


class ModelConfiguration:
    """
    Configuration class for different model types

    Attributes:
        name (str): Model name identifier
        use_cnn (bool): Whether to use CNN layers
        use_attention (bool): Whether to use attention mechanism
        attention_heads (int): Number of attention heads if attention is used
        look_back (int): Number of time steps to look back for sequence models
        output_mode (str): Model output type ('classification' or 'regression')
    """

    def __init__(
        self,
        name: str,
        use_cnn: bool,
        use_attention: bool,
        attention_heads: int = 8,
        look_back: int = 50,
        output_mode: str = "classification",
    ):
        self.name = name
        self.use_cnn = use_cnn
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.look_back = look_back
        self.output_mode = output_mode


def safe_execute_with_gpu(function, *args, gpu_manager=None, **kwargs):
    """
    Execute function with proper GPU resource management

    Args:
        function: Function to execute
        gpu_manager: GPU resource manager instance (PyTorchGPUManager)
        *args, **kwargs: Arguments to pass to the function (kwargs will be forwarded including gpu_manager)

    Returns:
        Return value from the function

    Raises:
        Exception: If function execution fails
    """
    if gpu_manager is None:
        gpu_manager = PyTorchGPUManager()

    try:
        # Configure GPU memory if available
        if gpu_manager.is_available():
            gpu_manager.configure_memory()

        # Check if function accepts gpu_manager parameter
        sig = inspect.signature(function)
        accepts_gpu_manager = "gpu_manager" in sig.parameters

        # Forward gpu_manager only if function accepts it or if it's already present
        if accepts_gpu_manager and "gpu_manager" not in kwargs:
            kwargs["gpu_manager"] = gpu_manager

        result = function(*args, **kwargs)
        return result
    except Exception as e:
        log_error(f"Error executing {getattr(function, '__name__', str(function))}: {e}")
        log_error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        try:
            if gpu_manager.is_available():
                torch.cuda.empty_cache()
        except Exception as cleanup_err:
            log_warn(f"Failed to release GPU cache: {cleanup_err}")


def cleanup_resources(model=None):
    """
    Cleanup resources after model training, with error checking.

    Args:
        model: Model instance to clean up
    """
    try:
        if model is not None:
            del model
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as cuda_err:
                log_warn(f"Error while emptying CUDA cache: {cuda_err}")
        log_info("Model resources cleaned up")
    except Exception as e:
        log_error(f"Error cleaning up model resources: {e}")


def get_model_configurations() -> List[ModelConfiguration]:
    """
    Define all 4 model configurations

    Returns:
        List of ModelConfiguration objects
    """
    return [
        ModelConfiguration("LSTM", use_cnn=False, use_attention=False),
        ModelConfiguration("LSTM-Attention", use_cnn=False, use_attention=True, attention_heads=8),
        ModelConfiguration("CNN-LSTM", use_cnn=True, use_attention=False, look_back=50),
        ModelConfiguration("CNN-LSTM-Attention", use_cnn=True, use_attention=True, attention_heads=8, look_back=50),
    ]


def prepare_training_dataset(symbols: List[str], timeframes: List[str]) -> Optional[pd.DataFrame]:
    """
    Prepare combined training dataset by fetching and combining OHLCV data.

    Args:
        symbols: List of crypto symbols
        timeframes: List of timeframes to load

    Returns:
        Combined DataFrame or None if failed
    """
    try:
        log_info("=" * 80)
        log_info("STEP 1: LOADING ALL PAIRS DATA")
        log_info("=" * 80)

        log_info(f"Loading data for {len(symbols)} symbols across {len(timeframes)} timeframes...")
        log_debug(f"Symbols: {symbols}")
        log_debug(f"Timeframes: {timeframes}")

        # Initialize exchange manager and data fetcher
        exchange_manager = ExchangeManager()
        data_fetcher = DataFetcher(exchange_manager)

        all_symbols_data = fetch_ohlcv_data_dict(
            symbols=symbols, timeframes=timeframes, exchange_manager=exchange_manager, data_fetcher=data_fetcher
        )

        if not all_symbols_data:
            log_error("Failed to load any pairs data")
            return None

        total_pairs = len(all_symbols_data)
        successful_pairs = sum(1 for v in all_symbols_data.values() if v is not None)
        log_info(f"Data loading results: {successful_pairs}/{total_pairs} pairs loaded successfully")

        log_info("=" * 80)
        log_info("STEP 2: COMBINING ALL DATAFRAMES")
        log_info("=" * 80)

        combined_frames = []

        for symbol, symbol_data in all_symbols_data.items():
            if symbol_data is None:
                continue

            # Handle nested dict structure (multi-timeframe)
            if isinstance(symbol_data, dict):
                for timeframe, df in symbol_data.items():
                    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                        continue

                    # Create a copy to avoid modifying original
                    df_copy = df.copy()

                    # Add metadata columns if they don't exist
                    if "pair" not in df_copy.columns:
                        df_copy["pair"] = symbol
                    if "timeframe" not in df_copy.columns:
                        df_copy["timeframe"] = timeframe

                    combined_frames.append(df_copy)
            # Handle direct DataFrame structure
            elif isinstance(symbol_data, pd.DataFrame):
                if symbol_data.empty:
                    continue

                df_copy = symbol_data.copy()

                # Add metadata columns if they don't exist
                if "pair" not in df_copy.columns:
                    df_copy["pair"] = symbol

                combined_frames.append(df_copy)

        if not combined_frames:
            log_error("Combined DataFrame is empty")
            return None

        # Concatenate all DataFrames
        try:
            combined_df = pd.concat(combined_frames, ignore_index=True)
        except Exception as e:
            log_error(f"Error combining DataFrames: {e}")
            return None

        if combined_df.empty:
            log_error("Combined DataFrame is empty")
            return None

        log_info(f"Successfully combined data: {len(combined_df)} total rows")
        log_debug(f"Combined DataFrame shape: {combined_df.shape}")
        log_debug(f"Columns: {list(combined_df.columns)}")

        if "pair" in combined_df.columns:
            log_info(f"Unique pairs in combined data: {combined_df['pair'].nunique()}")

        if "timeframe" in combined_df.columns:
            log_info(f"Unique timeframes in combined data: {combined_df['timeframe'].nunique()}")

        return combined_df

    except Exception as e:
        log_error(f"Error in data loading and preparation: {e}")
        return None


def train_model_configuration(
    config: ModelConfiguration,
    combined_df: pd.DataFrame,
    gpu_manager=None,
    use_kalman_filter: Optional[bool] = None,
    kalman_params: Optional[dict] = None,
) -> Tuple[Optional[object], str]:
    """
    Train a specific model configuration with proper resource management

    Args:
        config: Model configuration
        combined_df: Combined training data
        gpu_manager: Optional GPU resource manager (PyTorchGPUManager)
        use_kalman_filter: Whether to enable Kalman Filter preprocessing. If None, uses config default.
        kalman_params: Kalman Filter parameters. If None, uses config defaults.

    Returns:
        Tuple of (model, model_path)
    """
    if gpu_manager is None:
        gpu_manager = PyTorchGPUManager()

    try:
        log_info("=" * 80)
        log_info(f"TRAINING {config.name.upper()} MODEL")
        log_info("=" * 80)

        log_info(f"Configuration: {config.name}")
        log_debug(f"  - Use CNN: {config.use_cnn}")
        log_debug(f"  - Use Attention: {config.use_attention}")
        if config.use_attention:
            log_debug(f"  - Attention Heads: {config.attention_heads}")
        if config.use_cnn:
            log_debug(f"  - Look Back: {config.look_back}")
            log_debug(f"  - Output Mode: {config.output_mode}")

        # Validate DataFrame structure using common utility
        try:
            validate_ohlcv_input(combined_df, ["timestamp", "open", "high", "low", "close", "volume"])
        except ValueError as e:
            log_error(str(e))
            log_error(f"Invalid input data for {config.name}: {e}")
            return None, ""

        if config.use_cnn and len(combined_df) < config.look_back + 100:
            error = f"Insufficient data for CNN model: {len(combined_df)} rows, need at least {config.look_back + 100}"
            log_error(f"Invalid input data for {config.name}: {error}")
            return None, ""

        if combined_df.isnull().values.any():
            error = "DataFrame contains NaN values"
            log_error(f"Invalid input data for {config.name}: {error}")
            return None, ""

        log_debug(f"Training data shape: {combined_df.shape}")
        log_debug(f"Training data columns: {list(combined_df.columns)}")

        start_time = time.time()

        # Configure GPU memory if available
        if gpu_manager.is_available():
            gpu_manager.configure_memory()

        # Use unified trainer for all model configurations
        # Determine Kalman Filter settings
        if use_kalman_filter is None:
            use_kalman_filter = ENABLE_KALMAN_FILTER

        if use_kalman_filter:
            if kalman_params is None:
                kalman_params = {
                    "process_variance": KALMAN_PROCESS_VARIANCE,
                    "observation_variance": KALMAN_OBSERVATION_VARIANCE,
                }
            log_info("Kalman Filter: Enabled")
            log_debug(f"  - Process Variance: {kalman_params.get('process_variance', KALMAN_PROCESS_VARIANCE)}")
            log_debug(
                f"  - Observation Variance: {kalman_params.get('observation_variance', KALMAN_OBSERVATION_VARIANCE)}"
            )
        else:
            kalman_params = None
            log_info("Kalman Filter: Disabled")

        trainer = LSTMTrainer(
            use_cnn=config.use_cnn,
            use_attention=config.use_attention,
            look_back=config.look_back,
            output_mode=config.output_mode,
            attention_heads=config.attention_heads,
            use_kalman_filter=use_kalman_filter,
            kalman_params=kalman_params,
        )

        model, _, model_path = safe_execute_with_gpu(
            trainer.train,
            df_input=combined_df,
            epochs=50,
            save_model=True,
            model_filename=None,
            gpu_manager=gpu_manager,
        )

        training_time = time.time() - start_time

        if model is not None:
            log_info(f"{config.name} model trained successfully in {training_time:.2f}s")
            log_info(f"Model saved to: {model_path}")
        else:
            log_error(f"{config.name} model training failed")

        return model, model_path

    except Exception as e:
        log_error(f"Error training {config.name} model: {e}")
        log_debug(f"Full traceback: {traceback.format_exc()}")
        return None, ""


def test_signal_generation(
    config: ModelConfiguration, model: object, test_symbol: str, test_timeframe: str, all_symbols_ohlcv_data: Dict
) -> str:
    """
    Test signal generation for a specific model

    Args:
        config: Model configuration
        model: Trained model
        test_symbol: Symbol to test (e.g., 'BTCUSDT')
        test_timeframe: Timeframe to test (e.g., '1h')
        all_pairs_data: All loaded pairs data

    Returns:
        Generated signal string
    """
    try:
        log_info(f"Testing {config.name} signal generation for {test_symbol} {test_timeframe}...")

        # Get test data
        test_df = None
        if (
            all_symbols_ohlcv_data
            and test_symbol in all_symbols_ohlcv_data
            and all_symbols_ohlcv_data[test_symbol]
            and isinstance(all_symbols_ohlcv_data[test_symbol], dict)
            and test_timeframe in all_symbols_ohlcv_data[test_symbol]
        ):
            test_df = all_symbols_ohlcv_data[test_symbol][test_timeframe].copy()
            log_debug(f"Using real data: {len(test_df)} rows")
            log_debug(f"Current columns: {list(test_df.columns)}")

            # Normalize column names to lowercase (case-insensitive matching)
            column_name_map = {}
            for col in test_df.columns:
                lower_col = col.lower()
                if lower_col in column_name_map:
                    log_warn(f"Multiple columns map to '{lower_col}': {column_name_map[lower_col]} and {col}")
                column_name_map[lower_col] = col

            for required_col in ["timestamp", "open", "high", "low", "close", "volume"]:
                if required_col not in test_df.columns:
                    # Try to find case-insensitive match
                    if required_col in column_name_map:
                        actual_col = column_name_map[required_col]
                        test_df[required_col] = test_df[actual_col]
                        log_debug(f"Mapped {actual_col} -> {required_col}")
                    else:
                        log_error(f"Cannot find column for {required_col}")

        else:
            # Create sample data
            log_warn("Real data not available, creating sample data")
            np.random.seed(42)

            base_time = pd.Timestamp.now()
            sample_data = []
            base_price = 45000
            for i in range(100):
                open_price = base_price + np.random.normal(0, 1000)
                high_price = open_price + np.random.uniform(0, 2000)
                low_price = open_price - np.random.uniform(0, 2000)
                close_price = np.random.uniform(low_price, high_price)
                volume = np.random.uniform(50, 500)

                sample_data.append(
                    {
                        "timestamp": base_time - pd.Timedelta(hours=100 - i),
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                    }
                )

            test_df = pd.DataFrame(sample_data)

        # Final verification of required columns
        missing_columns = [
            col for col in ["timestamp", "open", "high", "low", "close", "volume"] if col not in test_df.columns
        ]

        if missing_columns:
            log_error(f"Still missing required columns after mapping: {missing_columns}")
            log_debug(f"Available columns: {list(test_df.columns)}")
            return "ERROR - Missing columns"

        # Generate signal
        signal = get_latest_signal(test_df, model)
        log_info(f"{config.name} signal for {test_symbol} {test_timeframe}: {signal}")

        return signal

    except Exception as e:
        log_error(f"Error testing {config.name} signal generation: {e}")
        log_debug(f"Full traceback: {traceback.format_exc()}")
        return "ERROR"


def main():
    """
    Main function that loads data, trains all model configurations,
    and tests signal generation on specified symbol and timeframe.

    Returns:
        Dict containing training results, signal results, and metrics,
        or None if execution fails
    """
    total_start_time = time.time()

    log_info("=" * 80)
    log_info("COMPREHENSIVE MODEL TRAINING AND TESTING")
    log_info("=" * 80)
    log_info("Testing 4 model configurations:")
    log_info("  1. LSTM")
    log_info("  2. LSTM + Attention")
    log_info("  3. CNN + LSTM")
    log_info("  4. CNN + LSTM + Attention")
    log_info("=" * 80)

    gpu_manager = PyTorchGPUManager()

    try:
        # Get symbols and timeframes (limited subset for testing)
        symbols = DEFAULT_CRYPTO_SYMBOLS_FOR_TRAINING_DL[:5]
        timeframes = DEFAULT_TIMEFRAMES_FOR_TRAINING_DL[:3]

        log_debug(f"Selected symbols: {symbols}")
        log_debug(f"Selected timeframes: {timeframes}")

        # Load and prepare training data
        combined_df = prepare_training_dataset(symbols, timeframes)
        if combined_df is None:
            log_error("Failed to load and prepare data")
            return None

        # Load test data for signal generation
        log_info(f"Loading test data for {DEFAULT_SYMBOL} {DEFAULT_TIMEFRAME}...")
        exchange_manager = ExchangeManager()
        data_fetcher = DataFetcher(exchange_manager)
        test_data = fetch_ohlcv_data_dict(
            symbols=[DEFAULT_SYMBOL],
            timeframes=[DEFAULT_TIMEFRAME],
            exchange_manager=exchange_manager,
            data_fetcher=data_fetcher,
        )

        # Get model configurations and prepare result storage
        model_configs = get_model_configurations()
        training_results = {}
        signal_results = {}

        # Train and test each model configuration
        for i, config in enumerate(model_configs, 1):
            log_info(f"\n{'=' * 20} MODEL {i}/4: {config.name.upper()} {'=' * 20}")

            model, model_path = train_model_configuration(config, combined_df, gpu_manager)

            if model is not None:
                training_results[config.name] = {"model": model, "model_path": model_path, "success": True}

                signal = test_signal_generation(config, model, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, test_data)
                signal_results[config.name] = signal

                # Clean up model resources before next iteration
                cleanup_resources(model)
            else:
                training_results[config.name] = {"model": None, "model_path": "", "success": False}
                signal_results[config.name] = "FAILED"

        # Generate final results summary
        total_time = time.time() - total_start_time

        log_info("=" * 80)
        log_info("FINAL RESULTS SUMMARY")
        log_info("=" * 80)

        log_info(f"Total execution time: {total_time:.2f} seconds")
        log_info(f"Data processed: {len(combined_df)} rows from {len(symbols)} pairs")

        # Training results summary
        log_info("\nTRAINING RESULTS:")
        successful_models = 0
        for config_name, result in training_results.items():
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            log_info(f"  {config_name:20} : {status}")
            if result["success"]:
                successful_models += 1
                log_debug(f"    Model saved: {result['model_path']}")

        # Signal generation results
        log_info(f"\nSIGNAL GENERATION RESULTS FOR {DEFAULT_SYMBOL} {DEFAULT_TIMEFRAME}:")
        for config_name, signal in signal_results.items():
            if signal == "FAILED":
                log_error(f"  {config_name:20} : ❌ FAILED")
            elif signal == "ERROR":
                log_warn(f"  {config_name:20} : ⚠️  ERROR")
            else:
                log_info(f"  {config_name:20} : 📊 {signal}")

        # Performance summary
        log_info("\nPERFORMANCE SUMMARY:")
        log_info(f"  Models trained successfully: {successful_models}/4")
        log_info(f"  Success rate: {successful_models / 4 * 100:.1f}%")
        log_info(f"  Average time per model: {total_time / 4:.2f}s")

        # Detailed configuration comparison
        log_info("\nCONFIGURATION COMPARISON:")
        for config in model_configs:
            log_debug(f"  {config.name}:")
            log_debug(f"    - CNN: {config.use_cnn}")
            log_debug(f"    - Attention: {config.use_attention}")
            if config.use_attention:
                log_debug(f"    - Attention Heads: {config.attention_heads}")
            if config.use_cnn:
                log_debug(f"    - Look Back: {config.look_back}")
            log_info(f"    - Training: {'✅' if training_results[config.name]['success'] else '❌'}")
            log_info(f"    - Signal: {signal_results[config.name]}")
        log_info("Comprehensive model testing completed!")

        return {
            "training_results": training_results,
            "signal_results": signal_results,
            "total_time": total_time,
            "success_rate": successful_models / 4,
        }

    except Exception as e:
        log_error(f"Error in main execution: {e}")
        log_debug(f"Full traceback: {traceback.format_exc()}")
        return None

    finally:
        cleanup_resources()


if __name__ == "__main__":
    log_info("Starting comprehensive model training and testing...")
    log_info("This will test all 4 model configurations:")
    log_info("  1. LSTM")
    log_info("  2. LSTM + Attention")
    log_info("  3. CNN + LSTM")
    log_info("  4. CNN + LSTM + Attention")
    log_info("")

    result = main()

    if result:
        log_info("All tests completed successfully!")
        log_info(f"Overall success rate: {result['success_rate'] * 100:.1f}%")
    else:
        log_error("Tests failed!")
