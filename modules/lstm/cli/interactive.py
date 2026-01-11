
from typing import List, Tuple

from colorama import Fore, Style

from config.common import (

from config.common import (

"""
Interactive menu and prompt utilities for LSTM CLI.

This module provides interactive menus and prompts for:
- Main menu navigation
- Model component selection
- Symbol management
- Training workflow
"""



    DEFAULT_CRYPTO_SYMBOLS_FOR_TRAINING_DL,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    DEFAULT_TIMEFRAMES_FOR_TRAINING_DL,
)
from config.lstm import ENABLE_KALMAN_FILTER, KALMAN_OBSERVATION_VARIANCE, KALMAN_PROCESS_VARIANCE, WINDOW_SIZE_LSTM
from modules.common.ui.logging import log_error, log_info, log_warn
from modules.common.utils import color_text, prompt_user_input
from modules.common.utils.system import PyTorchGPUManager
from modules.lstm.cli.main import (
    ModelConfiguration,
    cleanup_resources,
    prepare_training_dataset,
    train_model_configuration,
)


def prompt_symbol() -> str:
    """Prompt user for symbol."""
    symbol = (
        prompt_user_input(f"Enter trading symbol (e.g., BTCUSDT) [{DEFAULT_SYMBOL}]: ", default=DEFAULT_SYMBOL)
        .strip()
        .upper()
    )
    return symbol


def prompt_timeframe() -> str:
    """Prompt user for timeframe."""
    timeframe = (
        prompt_user_input(f"Enter timeframe (e.g., 15m, 1h, 4h) [{DEFAULT_TIMEFRAME}]: ", default=DEFAULT_TIMEFRAME)
        .strip()
        .lower()
    )
    return timeframe


def display_main_menu() -> None:
    """Display main menu options."""
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("LSTM MODEL MANAGER", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("\nMAIN MENU", Fore.YELLOW, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    print("  1. Train new LSTM model")
    print("  2. Generate trading signal (using existing model)")
    print("  3. Exit")
    print(color_text("-" * 80, Fore.CYAN))


def prompt_menu_choice() -> str:
    """Prompt user for menu choice."""
    choice = prompt_user_input("\nSelect an option (1-3): ", default="2").strip()
    return choice


def select_model_components() -> ModelConfiguration:
    """
    Interactive menu to select model components (LSTM, CNN, Attention).

    Returns:
        ModelConfiguration object with selected components
    """
    print("\n" + color_text("MODEL COMPONENT SELECTION", Fore.YELLOW, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))

    # LSTM is always required
    log_info("✓ LSTM: Enabled (required)")

    # Ask for CNN
    cnn_choice = prompt_user_input("Enable CNN component? (y/n) [n]: ", default="n").strip().lower()
    use_cnn = cnn_choice in ["y", "yes"]

    if use_cnn:
        log_info("✓ CNN: Enabled")
    else:
        log_info("✗ CNN: Disabled")

    # Ask for Attention (only if LSTM is enabled)
    attention_choice = prompt_user_input("Enable Attention mechanism? (y/n) [n]: ", default="n").strip().lower()
    use_attention = attention_choice in ["y", "yes"]

    if use_attention:
        log_info("✓ Attention: Enabled")
    else:
        log_info("✗ Attention: Disabled")

    # Build model name
    name_parts = []
    if use_cnn:
        name_parts.append("CNN")
    name_parts.append("LSTM")
    if use_attention:
        name_parts.append("Attention")
    model_name = "-".join(name_parts)

    # Create configuration
    config_kwargs = {
        "name": model_name,
        "use_cnn": use_cnn,
        "use_attention": use_attention,
        "look_back": 50 if use_cnn else WINDOW_SIZE_LSTM,
        "output_mode": "classification",
    }

    # Only add attention_heads if attention is enabled
    if use_attention:
        config_kwargs["attention_heads"] = 8

    config = ModelConfiguration(**config_kwargs)

    log_info(f"\nSelected configuration: {model_name}")
    return config


def _normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol name: uppercase and add USDT suffix if not present.

    Args:
        symbol: Input symbol (e.g., "act", "BTC", "1000bonk")

    Returns:
        Normalized symbol (e.g., "ACTUSDT", "BTCUSDT", "1000BONKUSDT")
    """
    symbol = symbol.strip().upper()

    # Add USDT suffix if not already present
    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"

    return symbol


def manage_symbols_menu(initial_symbols: List[str]) -> List[str]:
    """
    Interactive menu to add/remove symbols from training list.

    Args:
        initial_symbols: Initial list of symbols

    Returns:
        Final list of selected symbols
    """
    symbols = list(initial_symbols)  # Copy list

    while True:
        print("\n" + color_text("SYMBOL MANAGEMENT", Fore.YELLOW, Style.BRIGHT))
        print(color_text("-" * 80, Fore.CYAN))
        print(f"\nCurrent symbols ({len(symbols)}):")
        if symbols:
            for i, sym in enumerate(symbols, 1):
                print(f"  {i}. {sym}")
        else:
            print("  (No symbols in the list)")

        print("\nOptions:")
        print("  1. Add symbol(s) (comma-separated, e.g., btc, eth, 1000bonk)")
        print("  2. Remove symbol")
        print("  3. Clear all symbols")
        print("  4. Done (use current list)")

        choice = prompt_user_input("\nSelect option (1-4) [4]: ", default="4").strip()

        if choice == "1":
            # Add symbol(s)
            print("\nEnter symbol(s) separated by commas.")
            print("Examples: btc, eth, sol  or  act, 1000bonk, doge")
            print("Program will automatically add USDT suffix if not present.")

            symbol_input = prompt_user_input("Symbols to add: ", default="").strip()

            if not symbol_input:
                log_warn("No symbols entered")
                continue

            # Parse comma-separated symbols
            symbol_list = [s.strip() for s in symbol_input.split(",") if s.strip()]

            if not symbol_list:
                log_warn("No valid symbols found in input")
                continue

            # Normalize and add symbols
            added_count = 0
            skipped_count = 0
            for symbol_raw in symbol_list:
                try:
                    normalized_symbol = _normalize_symbol(symbol_raw)

                    # Check if already exists
                    if normalized_symbol in symbols:
                        log_warn(f"  {normalized_symbol} is already in the list (skipped)")
                        skipped_count += 1
                    else:
                        symbols.append(normalized_symbol)
                        log_info(f"  ✓ Added {normalized_symbol}")
                        added_count += 1
                except Exception as e:
                    log_warn(f"  Failed to process '{symbol_raw}': {e} (skipped)")
                    skipped_count += 1

            if added_count > 0:
                log_info(f"\n✓ Successfully added {added_count} symbol(s)")
            if skipped_count > 0:
                log_warn(f"Skipped {skipped_count} symbol(s)")

        elif choice == "2":
            # Remove symbol
            if not symbols:
                log_warn("No symbols to remove")
                continue

            print("\nCurrent symbols:")
            for i, sym in enumerate(symbols, 1):
                print(f"  {i}. {sym}")

            sym_choice = prompt_user_input(f"Select symbol to remove (1-{len(symbols)}): ", default="").strip()

            try:
                sym_idx = int(sym_choice) - 1
                if 0 <= sym_idx < len(symbols):
                    symbol_to_remove = symbols.pop(sym_idx)
                    log_info(f"✓ Removed {symbol_to_remove}")
                else:
                    log_error(f"Invalid choice: {sym_choice}")
            except ValueError:
                log_error(f"Invalid input: {sym_choice}")

        elif choice == "3":
            # Clear all
            if symbols:
                confirm = (
                    prompt_user_input("Are you sure you want to clear all symbols? (y/n) [n]: ", default="n")
                    .strip()
                    .lower()
                )
                if confirm in ["y", "yes"]:
                    symbols.clear()
                    log_info("✓ Cleared all symbols")
            else:
                log_warn("List is already empty")

        elif choice == "4":
            # Done
            break

        else:
            log_warn(f"Invalid choice: {choice}")

    return symbols


def train_model_menu() -> Tuple[bool, str]:
    """
    Train LSTM model menu with component selection and symbol management.

    Returns:
        Tuple of (success: bool, model_path: str). model_path is empty string if failed.
    """
    log_info("\n" + "=" * 80)
    log_info("LSTM MODEL TRAINING")
    log_info("=" * 80)

    # Step 1: Select model components
    selected_config = select_model_components()

    # Step 2: Manage symbols
    print("\n" + color_text("=" * 80, Fore.CYAN))
    initial_symbols = DEFAULT_CRYPTO_SYMBOLS_FOR_TRAINING_DL[:5]  # Default: first 5
    symbols = manage_symbols_menu(initial_symbols)

    if not symbols:
        log_error("No symbols selected for training. Exiting.")
        return False, ""

    # Step 3: Select timeframes
    print("\n" + color_text("TIMEFRAME SELECTION", Fore.YELLOW, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    print("Available timeframes:")
    for i, tf in enumerate(DEFAULT_TIMEFRAMES_FOR_TRAINING_DL, 1):
        print(f"  {i}. {tf}")

    tf_choice = prompt_user_input(
        f"Number of timeframes to use (1-{len(DEFAULT_TIMEFRAMES_FOR_TRAINING_DL)}) [3]: ", default="3"
    ).strip()

    try:
        num_timeframes = int(tf_choice)
        if num_timeframes < 1 or num_timeframes > len(DEFAULT_TIMEFRAMES_FOR_TRAINING_DL):
            num_timeframes = 3
            log_warn(f"Invalid number, using default: {num_timeframes}")
    except ValueError:
        num_timeframes = 3
        log_warn(f"Invalid input, using default: {num_timeframes}")

    timeframes = DEFAULT_TIMEFRAMES_FOR_TRAINING_DL[:num_timeframes]

    # Step 4: Kalman Filter configuration
    print("\n" + color_text("KALMAN FILTER PREPROCESSING", Fore.YELLOW, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    print("Kalman Filter helps reduce noise in price data before generating indicators.")
    print("This can help reduce overfitting in LSTM models.")

    kalman_choice = (
        prompt_user_input(
            f"Enable Kalman Filter preprocessing? (y/n) [{'y' if ENABLE_KALMAN_FILTER else 'n'}]: ",
            default="y" if ENABLE_KALMAN_FILTER else "n",
        )
        .strip()
        .lower()
    )

    use_kalman_filter = kalman_choice in ["y", "yes"]
    kalman_params = None

    if use_kalman_filter:
        print("\nKalman Filter Parameters:")
        print(f"  - Process Variance (Q): Controls smoothing (default: {KALMAN_PROCESS_VARIANCE})")
        print(f"  - Observation Variance (R): Controls trust in observations (default: {KALMAN_OBSERVATION_VARIANCE})")

        custom_params = prompt_user_input("Use custom parameters? (y/n) [n]: ", default="n").strip().lower()

        if custom_params in ["y", "yes"]:
            try:
                process_var_input = prompt_user_input(
                    f"Process Variance (Q) [{KALMAN_PROCESS_VARIANCE}]: ", default=str(KALMAN_PROCESS_VARIANCE)
                ).strip()
                obs_var_input = prompt_user_input(
                    f"Observation Variance (R) [{KALMAN_OBSERVATION_VARIANCE}]: ",
                    default=str(KALMAN_OBSERVATION_VARIANCE),
                ).strip()

                kalman_params = {
                    "process_variance": float(process_var_input) if process_var_input else KALMAN_PROCESS_VARIANCE,
                    "observation_variance": float(obs_var_input) if obs_var_input else KALMAN_OBSERVATION_VARIANCE,
                }
            except (ValueError, TypeError):
                log_warn("Invalid parameters entered, using defaults")
                kalman_params = {
                    "process_variance": KALMAN_PROCESS_VARIANCE,
                    "observation_variance": KALMAN_OBSERVATION_VARIANCE,
                }
        else:
            kalman_params = {
                "process_variance": KALMAN_PROCESS_VARIANCE,
                "observation_variance": KALMAN_OBSERVATION_VARIANCE,
            }

    # Summary
    log_info("\n" + "=" * 80)
    log_info("TRAINING CONFIGURATION SUMMARY")
    log_info("=" * 80)
    log_info(f"Model: {selected_config.name}")
    log_info("  - LSTM: ✓ (required)")
    log_info(f"  - CNN: {'✓' if selected_config.use_cnn else '✗'}")
    log_info(f"  - Attention: {'✓' if selected_config.use_attention else '✗'}")
    log_info(f"Symbols ({len(symbols)}): {', '.join(symbols)}")
    log_info(f"Timeframes ({len(timeframes)}): {', '.join(timeframes)}")
    log_info(f"Kalman Filter: {'✓ Enabled' if use_kalman_filter else '✗ Disabled'}")
    if use_kalman_filter and kalman_params:
        log_info(f"  - Process Variance (Q): {kalman_params.get('process_variance', KALMAN_PROCESS_VARIANCE)}")
        log_info(
            f"  - Observation Variance (R): {kalman_params.get('observation_variance', KALMAN_OBSERVATION_VARIANCE)}"
        )
    log_info("=" * 80)

    confirm = prompt_user_input("\nProceed with training? (y/n) [y]: ", default="y").strip().lower()

    if confirm not in ["y", "yes"]:
        log_info("Training cancelled by user.")
        return False, ""

    # Initialize GPU manager
    gpu_manager = PyTorchGPUManager()

    try:
        # Prepare training dataset
        log_info("\nPreparing training dataset...")
        combined_df = prepare_training_dataset(symbols, timeframes)

        if combined_df is None:
            log_error("Failed to prepare training dataset")
            return False, ""

        # Train model
        log_info(f"\nTraining {selected_config.name} model...")
        model, model_path = train_model_configuration(
            selected_config, combined_df, gpu_manager, use_kalman_filter=use_kalman_filter, kalman_params=kalman_params
        )

        if model is not None and model_path:
            log_info("\n✅ Model trained successfully!")
            log_info(f"Model saved to: {model_path}")
            cleanup_resources(model)
            return True, str(model_path)
        else:
            log_error("\n❌ Model training failed")
            return False, ""

    except Exception as e:
        log_error(f"Error during training: {e}")
        import traceback

        log_error(f"Traceback: {traceback.format_exc()}")
        return False, ""
    finally:
        cleanup_resources()
