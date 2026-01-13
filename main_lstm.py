"""
LSTM Model Manager Entry Point.

Entry point for managing LSTM models: training and signal generation.
Supports all LSTM model variants (LSTM, LSTM-Attention, CNN-LSTM, CNN-LSTM-Attention).

Features:
    1. Train new LSTM models with different configurations
    2. Generate trading signals using trained models
    3. Interactive menu for easy navigation

Workflow:
    Training:
        1. Select model configuration
        2. Prepare training dataset from multiple symbols/timeframes
        3. Train model and save checkpoint

    Signal Generation:
        1. Load trained LSTM model from checkpoint
        2. Fetch OHLCV data for specified symbol and timeframe
        3. Generate technical indicators
        4. Generate trading signal using model inference
        5. Display signal result with confidence

Example:
    # Interactive menu mode
    $ python main_lstm.py

    # Direct signal generation (skip menu)
    $ python main_lstm.py --symbol BTCUSDT --timeframe 1h
    $ python main_lstm.py --symbol ETHUSDT --timeframe 15m --model-path \
    artifacts/models/lstm/cnn_lstm_attention_model.pth
"""

import sys
import warnings

from modules.common.utils import color_text, configure_windows_stdio, log_error, log_info, log_warn
from modules.lstm.cli import (
    display_main_menu,
    generate_signal_workflow,
    parse_args,
    prompt_menu_choice,
    prompt_symbol,
    prompt_timeframe,
    train_model_menu,
)

# Fix encoding issues on Windows for interactive CLI runs only
configure_windows_stdio()

from colorama import Fore
from colorama import init as colorama_init

colorama_init(autoreset=True)

# Suppress only benign deprecation and future warnings from third-party libraries
_EXTERNAL_MODULES = [
    "pandas",
    "numpy",
    "sklearn",
    "xgboost",
    "statsmodels",
    "pandas_ta",
    "hmmlearn",
    "pykalman",
    "numba",
    "matplotlib",
    "mplfinance",
    "torch",
]
for _module in _EXTERNAL_MODULES:
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=f"^{_module}(\\.|$)")
    warnings.filterwarnings("ignore", category=FutureWarning, module=f"^{_module}(\\.|$)")


def main() -> None:
    """
    Main entry point for LSTM model manager.

    Workflow steps:
        1. Display main menu
        2. User selects option (train or generate signal)
        3. Execute selected workflow
    """
    args = parse_args()

    # If command-line arguments provided, skip menu and go directly to signal generation
    if args.symbol is not None or args.timeframe is not None:
        generate_signal_workflow(
            model_path=args.model_path, symbol=args.symbol, timeframe=args.timeframe, limit=args.limit
        )
        return

    # Interactive menu mode
    while True:
        display_main_menu()
        choice = prompt_menu_choice()

        if choice == "1":
            # Train model
            success, model_path = train_model_menu()
            if success:
                log_info("\n✅ Training completed successfully!")

                # Ask if user wants to generate signal with the newly trained model
                from modules.common.utils import prompt_user_input

                generate_choice = (
                    prompt_user_input(
                        "\nWould you like to generate a trading signal with this model? (y/n) [y]: ", default="y"
                    )
                    .strip()
                    .lower()
                )

                if generate_choice in ["y", "yes"]:
                    log_info("\n" + "=" * 80)
                    log_info("GENERATE TRADING SIGNAL")
                    log_info("=" * 80)

                    # Prompt for symbol and timeframe
                    symbol = prompt_symbol()
                    timeframe = prompt_timeframe()

                    # Generate signal with the newly trained model
                    generate_signal_workflow(
                        model_path=model_path if model_path else None,
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=args.limit,
                    )
                else:
                    log_info("Skipping signal generation.")
            else:
                log_error("\n❌ Training failed!")

            # Ask if user wants to continue
            from modules.common.utils import prompt_user_input

            continue_choice = (
                prompt_user_input("\nPress Enter to return to main menu, or 'q' to quit: ", default="").strip().lower()
            )
            if continue_choice == "q":
                break

        elif choice == "2":
            # Generate signal
            generate_signal_workflow(model_path=args.model_path, limit=args.limit)

            # Ask if user wants to continue
            from modules.common.utils import prompt_user_input

            continue_choice = (
                prompt_user_input("\nPress Enter to return to main menu, or 'q' to quit: ", default="").strip().lower()
            )
            if continue_choice == "q":
                break

        elif choice == "3" or choice.lower() == "q":
            log_info("\nExiting...")
            break
        else:
            log_warn(f"Invalid choice: {choice}. Please select 1-3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\nExiting program by user request.", Fore.YELLOW))
        sys.exit(0)
    except Exception as e:
        log_error(f"Error: {type(e).__name__}: {e}")
        import traceback

        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
