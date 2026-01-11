
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import random
import sys

import pandas as pd
import pandas as pd

"""
Script để test tất cả models trong artifacts/models/lstm và so sánh signals.
Giúp phát hiện bias và phân tích performance của từng model.
"""



# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from colorama import Fore, Style, init

from config.common import DEFAULT_TIMEFRAMES_FOR_TRAINING_DL
from config.evaluation import CONFIDENCE_THRESHOLD
from config.lstm import MODELS_DIR
from modules.common.ui.logging import log_error, log_info, log_warn
from modules.common.utils import initialize_components
from modules.common.utils.data import fetch_ohlcv_data_dict
from modules.lstm.models.model_utils import get_latest_signal, load_model_and_scaler

init(autoreset=True)


def load_all_models(allow_missing_scaler: bool = False) -> List[Tuple[Path, object, object, int]]:
    """
    Load tất cả models từ thư mục MODELS_DIR.

    Args:
        allow_missing_scaler: Nếu True, vẫn load model ngay cả khi không có scaler (với warning)

    Returns:
        List of tuples: (model_path, model, scaler, look_back)
    """
    models = []

    if not MODELS_DIR.exists():
        log_error(f"Models directory not found: {MODELS_DIR}")
        return models

    model_files = sorted(MODELS_DIR.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)

    log_info(f"\nFound {len(model_files)} model(s) in {MODELS_DIR}")

    for model_path in model_files:
        try:
            log_info(f"\nLoading: {model_path.name}...")
            model, scaler, look_back = load_model_and_scaler(model_path)

            if model is None:
                log_warn("  ⚠️  Failed to load model")
                continue

            if scaler is None:
                if allow_missing_scaler:
                    log_warn(f"  ⚠️  Scaler not found in checkpoint for {model_path.name}")
                    log_warn("  ⚠️  Continuing without scaler (predictions may be unreliable)")
                else:
                    log_warn(f"  ⚠️  Scaler not found in checkpoint for {model_path.name}")
                    log_warn("  ⚠️  Skipping this model (use --allow-missing-scaler to test anyway)")
                    continue

            if look_back is None:
                log_warn("  ⚠️  Look_back not found in checkpoint, using default")
                from config.lstm import WINDOW_SIZE_LSTM

                look_back = WINDOW_SIZE_LSTM

            models.append((model_path, model, scaler, look_back))
            scaler_status = "✓" if scaler is not None else "⚠️ (no scaler)"
            log_info(f"  ✓ Loaded successfully (look_back: {look_back}, scaler: {scaler_status})")

        except Exception as e:
            log_error(f"  ❌ Error loading {model_path.name}: {e}")
            import traceback

            log_error(f"  Traceback: {traceback.format_exc()}")
            continue

    return models


def test_model_on_data(
    model: object, scaler: object, look_back: int, df_market_data: pd.DataFrame
) -> Tuple[str, float, int, Dict[str, float]]:
    """
    Test một model trên data và trả về signal, confidence, và chi tiết.

    Args:
        model: Trained model
        scaler: Pre-fitted scaler (có thể None nếu allow_missing_scaler=True)
        look_back: Sequence length
        df_market_data: Market data DataFrame

    Returns:
        Tuple: (signal, confidence, predicted_class, prediction_probs_dict)
    """
    try:
        # Note: get_latest_signal may fail if scaler is None, but we handle it below
        if scaler is None:
            # Try to generate signal without scaler (will use raw features)
            log_warn("Testing model without scaler - using raw features (results may be unreliable)")

        signal = get_latest_signal(df_market_data=df_market_data, model=model, scaler=scaler, look_back=look_back)

        # Get detailed prediction info
        # We need to replicate the logic to get confidence
        import numpy as np
        import torch

        from config.model_features import MODEL_FEATURES
        from modules.lstm.utils.indicator_features import generate_indicator_features
        from modules.lstm.utils.kalman_filter import apply_kalman_to_ohlc

        # Check if Kalman filter should be applied
        use_kalman_filter = getattr(model, "_use_kalman_filter", False)
        kalman_params = getattr(model, "_kalman_params", None)

        df_for_features = df_market_data.copy()
        if use_kalman_filter and kalman_params:
            try:
                df_for_features = apply_kalman_to_ohlc(df_market_data.copy(), **kalman_params)
            except Exception:
                pass

        df_features = generate_indicator_features(df_for_features)
        if df_features.empty or len(df_features) < look_back:
            return signal, 0.0, 0, {}

        available_features = [col for col in MODEL_FEATURES if col in df_features.columns]
        if not available_features:
            return signal, 0.0, 0, {}

        feature_array = df_features[available_features].values
        scaled_features = scaler.transform(feature_array)

        if not np.all(np.isfinite(scaled_features)):
            return signal, 0.0, 0, {}

        window_data = scaled_features[-look_back:]
        model_device = next(model.parameters()).device
        input_window = torch.tensor(window_data, dtype=torch.float32, device=model_device).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            model_output = model(input_window)

            # Handle model output shape (batch, classes) or (classes,)
            if len(model_output.shape) == 2:
                prediction_probs = model_output[0].cpu().numpy()
            else:
                prediction_probs = model_output.cpu().numpy()

            # Normalize probabilities if needed (apply softmax if not normalized)
            prob_sum = np.sum(prediction_probs)
            if abs(prob_sum - 1.0) > 0.01:
                # Apply softmax to normalize
                exp_probs = np.exp(prediction_probs - np.max(prediction_probs))
                prediction_probs = exp_probs / np.sum(exp_probs)

            # Ensure probabilities are valid
            if np.any(prediction_probs < 0):
                prediction_probs = np.maximum(prediction_probs, 0.0)
                prediction_probs = prediction_probs / np.sum(prediction_probs)

        predicted_class_idx = np.argmax(prediction_probs)
        predicted_class = predicted_class_idx - 1  # -1, 0, 1 for SELL, NEUTRAL, BUY
        confidence = float(np.max(prediction_probs))

        # Create probabilities dict - ensure we have 3 classes
        if len(prediction_probs) == 3:
            probs_dict = {
                "SELL": float(prediction_probs[0]),
                "NEUTRAL": float(prediction_probs[1]),
                "BUY": float(prediction_probs[2]),
            }
        else:
            # Handle edge case where num_classes != 3
            log_warn(f"Unexpected number of classes: {len(prediction_probs)}, expected 3")
            probs_dict = {
                "SELL": float(prediction_probs[0]) if len(prediction_probs) > 0 else 0.0,
                "NEUTRAL": float(prediction_probs[1])
                if len(prediction_probs) > 1
                else float(prediction_probs[0])
                if len(prediction_probs) > 0
                else 0.0,
                "BUY": float(prediction_probs[2]) if len(prediction_probs) > 2 else 0.0,
            }

        return signal, confidence, predicted_class, probs_dict

    except Exception as e:
        log_error(f"Error testing model: {e}")
        return "NEUTRAL", 0.0, 0, {}


def print_aggregated_results(all_results: List[Dict], results_by_model: Dict[str, List[Dict]], models: List) -> None:
    """In kết quả tổng hợp cho tất cả models và symbols."""
    print("\n" + "=" * 120)
    print(f"{'AGGREGATED RESULTS SUMMARY':^120}")
    print("=" * 120)

    # Overall statistics
    print(f"\n{Fore.CYAN}{'OVERALL STATISTICS':^120}{Style.RESET_ALL}")
    print("-" * 120)

    total_tests = len(all_results)
    signal_counts = Counter([r["signal"] for r in all_results if r["signal"] != "ERROR"])
    error_count = sum([1 for r in all_results if r["signal"] == "ERROR"])

    print(f"Total tests: {total_tests}")
    print(f"Successful tests: {total_tests - error_count}")
    print(f"Errors: {error_count}")

    if signal_counts:
        print("\nSignal distribution (across all models and symbols):")
        for signal, count in signal_counts.most_common():
            percentage = (count / (total_tests - error_count) * 100) if (total_tests - error_count) > 0 else 0
            signal_color = Fore.GREEN if signal == "BUY" else (Fore.RED if signal == "SELL" else Fore.YELLOW)
            print(f"  {signal_color}{signal:<10}{Style.RESET_ALL}: {count:>5} ({percentage:>5.1f}%)")

    # Average confidence
    valid_confidences = [r["confidence"] for r in all_results if r["confidence"] > 0]
    if valid_confidences:
        avg_confidence = sum(valid_confidences) / len(valid_confidences)
        print(f"\nAverage confidence: {avg_confidence:.3f}")
        low_confidence_count = sum([1 for c in valid_confidences if c < CONFIDENCE_THRESHOLD])
        print(
            f"Low confidence tests (< {CONFIDENCE_THRESHOLD}): {low_confidence_count}/{len(valid_confidences)} ({low_confidence_count / len(valid_confidences) * 100:.1f}%)"
        )

    # Per-model statistics
    print(f"\n{Fore.CYAN}{'PER-MODEL STATISTICS':^120}{Style.RESET_ALL}")
    print("-" * 120)
    print(f"{'Model Name':<60} {'Tests':<8} {'BUY':<8} {'NEUTRAL':<10} {'SELL':<8} {'Avg Conf':<10}")
    print("-" * 120)

    for model_name, model_results in sorted(results_by_model.items()):
        if not model_results:
            continue

        test_count = len(model_results)
        model_signal_counts = Counter([r["signal"] for r in model_results if r["signal"] != "ERROR"])
        model_valid_confidences = [r["confidence"] for r in model_results if r["confidence"] > 0]
        avg_conf = sum(model_valid_confidences) / len(model_valid_confidences) if model_valid_confidences else 0.0

        buy_count = model_signal_counts.get("BUY", 0)
        neutral_count = model_signal_counts.get("NEUTRAL", 0)
        sell_count = model_signal_counts.get("SELL", 0)

        # Truncate model name if too long
        display_name = model_name[:57] + "..." if len(model_name) > 60 else model_name

        print(
            f"{display_name:<60} {test_count:<8} {buy_count:<8} {neutral_count:<10} {sell_count:<8} {avg_conf:<10.3f}"
        )

    # Check for NEUTRAL bias
    if signal_counts:
        neutral_ratio = (
            signal_counts.get("NEUTRAL", 0) / (total_tests - error_count) if (total_tests - error_count) > 0 else 0
        )
        if neutral_ratio > 0.7:
            print(
                f"\n{Fore.YELLOW}⚠️  WARNING: Potential NEUTRAL bias detected! ({neutral_ratio * 100:.1f}% NEUTRAL){Style.RESET_ALL}"
            )
            print("   This might indicate:")
            print("   - Models are too conservative")
            print("   - Confidence threshold is too high")
            print("   - Models need retraining with better data/features")

    print("=" * 120)


def print_results_table(results: List[Dict]) -> None:
    """In bảng kết quả so sánh tất cả models."""
    print("\n" + "=" * 100)
    print(f"{'MODEL COMPARISON RESULTS':^100}")
    print("=" * 100)

    # Header
    header = f"{'Model Name':<50} {'Signal':<10} {'Confidence':<12} {'BUY':<10} {'NEUTRAL':<10} {'SELL':<10}"
    print(header)
    print("-" * 100)

    # Results
    for r in results:
        model_name = Path(r["model_path"]).name
        signal = r["signal"]
        confidence = r["confidence"]
        probs = r["probabilities"]

        # Color code signal
        if signal == "BUY":
            signal_str = f"{Fore.GREEN}{signal}{Style.RESET_ALL}"
        elif signal == "SELL":
            signal_str = f"{Fore.RED}{signal}{Style.RESET_ALL}"
        else:
            signal_str = f"{Fore.YELLOW}{signal}{Style.RESET_ALL}"

        # Truncate model name if too long
        if len(model_name) > 47:
            model_name = model_name[:44] + "..."

        confidence_str = f"{confidence:.3f}"
        if confidence < CONFIDENCE_THRESHOLD:
            confidence_str = f"{Fore.YELLOW}{confidence_str} (LOW){Style.RESET_ALL}"

        buy_prob = probs.get("BUY", 0.0)
        neutral_prob = probs.get("NEUTRAL", 0.0)
        sell_prob = probs.get("SELL", 0.0)

        row = f"{model_name:<50} {signal_str:<20} {confidence_str:<20} {buy_prob:.3f}   {neutral_prob:.3f}   {sell_prob:.3f}"
        print(row)

    print("=" * 100)

    # Statistics
    signal_counts = Counter([r["signal"] for r in results])
    print(f"\n{'STATISTICS':^100}")
    print("-" * 100)
    print(f"Total models tested: {len(results)}")
    print(f"BUY signals:    {signal_counts.get('BUY', 0)} ({signal_counts.get('BUY', 0) / len(results) * 100:.1f}%)")
    print(f"SELL signals:   {signal_counts.get('SELL', 0)} ({signal_counts.get('SELL', 0) / len(results) * 100:.1f}%)")
    print(
        f"NEUTRAL signals: {signal_counts.get('NEUTRAL', 0)} ({signal_counts.get('NEUTRAL', 0) / len(results) * 100:.1f}%)"
    )

    # Average confidence
    avg_confidence = sum([r["confidence"] for r in results]) / len(results) if results else 0
    print(f"Average confidence: {avg_confidence:.3f}")

    # Confidence threshold info
    print(f"\nConfidence threshold: {CONFIDENCE_THRESHOLD}")
    low_confidence_count = sum([1 for r in results if r["confidence"] < CONFIDENCE_THRESHOLD])
    print(f"Models with low confidence (< {CONFIDENCE_THRESHOLD}): {low_confidence_count}/{len(results)}")

    # Check for NEUTRAL bias
    neutral_ratio = signal_counts.get("NEUTRAL", 0) / len(results) if results else 0
    if neutral_ratio > 0.7:
        print(
            f"\n{Fore.YELLOW}⚠️  WARNING: Potential NEUTRAL bias detected! ({neutral_ratio * 100:.1f}% NEUTRAL){Style.RESET_ALL}"
        )
        print("   This might indicate:")
        print("   - Models are too conservative")
        print("   - Confidence threshold is too high")
        print("   - Models need retraining with better data/features")

    print("=" * 100)


def main():
    """Main function để test tất cả models."""
    print(f"\n{Fore.CYAN}{'=' * 100}")
    print(f"{'LSTM MODELS BATCH TESTING':^100}")
    print(f"{'=' * 100}{Style.RESET_ALL}\n")

    # Check command line arguments
    allow_missing_scaler = "--allow-missing-scaler" in sys.argv or "-a" in sys.argv

    if allow_missing_scaler:
        print(
            f"{Fore.YELLOW}⚠️  Mode: Allowing models without scaler (predictions may be unreliable){Style.RESET_ALL}\n"
        )

    # Ask user if they want to allow missing scaler
    if not allow_missing_scaler:
        response = input("Allow models without scaler? (y/n) [n]: ").strip().lower()
        if response in ["y", "yes"]:
            allow_missing_scaler = True
            log_info("Will load models even without scaler (with warnings)")

    # Load all models
    models = load_all_models(allow_missing_scaler=allow_missing_scaler)

    if not models:
        log_error("No models loaded. Exiting.")
        if not allow_missing_scaler:
            log_info("Tip: Use --allow-missing-scaler flag or answer 'y' to allow models without scaler")
        return

    # Get random symbols and timeframes
    print(f"\n{Fore.CYAN}Fetching available symbols from Binance...{Style.RESET_ALL}")

    try:
        exchange_manager, data_fetcher = initialize_components()

        # Get list of available symbols from Binance
        all_symbols = data_fetcher.list_binance_futures_symbols(
            exclude_symbols=None,
            max_candidates=500,  # Get more to have enough for random selection
            progress_label="Fetching Symbols",
        )

        if not all_symbols:
            log_error("Failed to fetch symbols from Binance")
            return

        # Random select 100 symbols
        num_symbols_to_test = min(100, len(all_symbols))
        selected_symbols = random.sample(all_symbols, num_symbols_to_test)

        # Available timeframes
        available_timeframes = (
            DEFAULT_TIMEFRAMES_FOR_TRAINING_DL if DEFAULT_TIMEFRAMES_FOR_TRAINING_DL else ["15m", "1h", "4h", "1d"]
        )

        # Random assign timeframe for each symbol
        symbol_timeframe_pairs = []
        for symbol in selected_symbols:
            random_tf = random.choice(available_timeframes)
            symbol_timeframe_pairs.append((symbol, random_tf))

        log_info(f"\n✓ Selected {num_symbols_to_test} symbols with random timeframes")
        log_info(f"  Timeframes used: {', '.join(set([tf for _, tf in symbol_timeframe_pairs]))}")

        # Fetch data for all symbol/timeframe pairs
        print(f"\n{Fore.CYAN}Fetching market data for {num_symbols_to_test} symbols...{Style.RESET_ALL}")

        symbols_to_fetch = [s.replace("/", "") for s, _ in symbol_timeframe_pairs]
        timeframes_to_fetch = list(set([tf for _, tf in symbol_timeframe_pairs]))

        all_data = fetch_ohlcv_data_dict(
            symbols=symbols_to_fetch,
            timeframes=timeframes_to_fetch,
            exchange_manager=exchange_manager,
            data_fetcher=data_fetcher,
            limit=1500,
        )

        if not all_data:
            log_error("Failed to fetch market data")
            return

        # Filter successful fetches
        valid_pairs = []
        for symbol, timeframe in symbol_timeframe_pairs:
            symbol_key = symbol.replace("/", "")
            if symbol_key in all_data and timeframe in all_data[symbol_key]:
                df = all_data[symbol_key][timeframe]
                if not df.empty and len(df) > 50:  # Ensure enough data
                    valid_pairs.append((symbol, timeframe, df))

        log_info(f"✓ Successfully fetched data for {len(valid_pairs)}/{num_symbols_to_test} symbols")

        if not valid_pairs:
            log_error("No valid data fetched. Exiting.")
            return

    except Exception as e:
        log_error(f"Error fetching data: {e}")
        import traceback

        log_error(f"Traceback: {traceback.format_exc()}")
        return

    # Test each model on each symbol/timeframe
    print(f"\n{Fore.CYAN}Testing all models on {len(valid_pairs)} symbol/timeframe combinations...{Style.RESET_ALL}\n")

    # Store results per model and per symbol
    all_results = []  # All results for aggregation
    results_by_model = defaultdict(list)  # Group by model

    total_tests = len(models) * len(valid_pairs)
    current_test = 0

    for model_path, model, scaler, look_back in models:
        model_name = Path(model_path).name
        log_info(f"\n{Fore.YELLOW}Testing model: {model_name}{Style.RESET_ALL}")

        for symbol, timeframe, df_market_data in valid_pairs:
            current_test += 1
            try:
                signal, confidence, predicted_class, probabilities = test_model_on_data(
                    model, scaler, look_back, df_market_data
                )

                result = {
                    "model_path": model_path,
                    "model_name": model_name,
                    "symbol": symbol.replace("/", ""),
                    "timeframe": timeframe,
                    "signal": signal,
                    "confidence": confidence,
                    "predicted_class": predicted_class,
                    "probabilities": probabilities,
                }

                all_results.append(result)
                results_by_model[model_name].append(result)

                # Progress indicator
                if current_test % 10 == 0:
                    log_info(f"  Progress: {current_test}/{total_tests} tests completed")

            except Exception as e:
                log_error(f"Error testing {model_name} on {symbol} {timeframe}: {e}")
                result = {
                    "model_path": model_path,
                    "model_name": model_name,
                    "symbol": symbol.replace("/", ""),
                    "timeframe": timeframe,
                    "signal": "ERROR",
                    "confidence": 0.0,
                    "predicted_class": 0,
                    "probabilities": {},
                }
                all_results.append(result)
                results_by_model[model_name].append(result)

    # Print aggregated results
    print_aggregated_results(all_results, results_by_model, models)

    print(
        f"\n{Fore.GREEN}Test completed! Tested {len(models)} models on {len(valid_pairs)} symbols.{Style.RESET_ALL}\n"
    )


if __name__ == "__main__":
    # Print usage help if requested
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("\nUsage:")
        print("  python test_all_models.py                          # Interactive mode")
        print("  python test_all_models.py --allow-missing-scaler   # Allow models without scaler")
        print("  python test_all_models.py -a                       # Short form")
        print("\nOptions:")
        print("  --allow-missing-scaler, -a    Load and test models even if scaler is missing")
        print("                                 (predictions may be unreliable)")
        print("  --help, -h                    Show this help message")
        sys.exit(0)

    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        log_error(f"Error: {e}")
        import traceback

        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
