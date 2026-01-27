"""Model training prompts for batch scanner."""

from typing import Any, Dict, List, Optional

from colorama import Fore

from modules.common.ui.formatting import color_text
from modules.common.ui.logging import log_info, log_warn
from modules.common.utils import normalize_timeframe, safe_input


def prompt_model_action(rf_model_status: Dict[str, Any]) -> str:
    """
    Prompt user for action based on Random Forest model status.
    """
    # Display model status
    print("\nRandom Forest Model Status:")
    if rf_model_status["exists"]:
        print(f"  Model file: {rf_model_status['model_path']}")
        if rf_model_status.get("modification_date"):
            print(f"  Last modified: {rf_model_status['modification_date']}")

        if rf_model_status["compatible"]:
            print("  Status: Compatible")
        else:
            print("  Status: Incompatible")
            if rf_model_status.get("uses_deprecated_features"):
                print("    (uses deprecated features)")

        if rf_model_status.get("error_message"):
            print(f"  Warning: {rf_model_status['error_message']}")
    else:
        print("  Model file not found")
        if rf_model_status and rf_model_status.get("error_message"):
            print(f"  Error: {rf_model_status['error_message']}")

    # Offer options if model is incompatible or doesn't exist
    if rf_model_status and (not rf_model_status["exists"] or not rf_model_status["compatible"]):
        print("\nRandom Forest Model Options:")
        print("  1. Retrain model now")
        print("  2. Skip (continue without model / use default if available)")
        rf_option_input = safe_input(color_text("Select option (1/2) [1]: ", Fore.YELLOW), default="1", allow_back=True)
        if not rf_option_input:
            rf_option_input = "1"

        if rf_option_input == "1":
            return "retrain"
        else:
            return "skip"
    else:
        # Model is compatible - optionally offer retrain
        print("\nRandom Forest Model Options:")
        print("  1. Continue with current model")
        print("  2. Retrain model now (optional)")
        print("  3. Skip")
        rf_option_input = safe_input(
            color_text("Select option (1/2/3) [1]: ", Fore.YELLOW), default="1", allow_back=True
        )
        if not rf_option_input:
            rf_option_input = "1"

        if rf_option_input == "2":
            return "retrain"
        elif rf_option_input == "3":
            return "skip"
        else:
            return "continue"


def prompt_training_symbols(default: Optional[List[str]] = None) -> List[str]:
    """
    Prompt user to enter training symbols.
    """
    print("\nTraining Symbols:")
    print("  Enter comma-separated list of symbols (e.g., BTC/USDT,ETH/USDT,BNB/USDT)")
    print("  Or leave empty to use top symbols by volume (default: top 10)")
    training_symbols_input = safe_input(
        color_text("Training symbols [auto - top 10]: ", Fore.YELLOW), default="", allow_back=True
    )

    training_symbols = []
    if training_symbols_input and training_symbols_input.strip():
        training_symbols = [s.strip() for s in training_symbols_input.split(",") if s.strip()]
    else:
        try:
            log_info("Fetching top symbols by volume for training...")
            from modules.common.core.data_fetcher import DataFetcher
            from modules.common.core.exchange_manager import ExchangeManager

            exchange_manager = ExchangeManager()
            data_fetcher = DataFetcher(exchange_manager)
            all_training_symbols = data_fetcher.get_spot_symbols(exchange_name="binance", quote_currency="USDT")
            training_symbols = all_training_symbols[:10] if all_training_symbols else []
            if training_symbols:
                log_info(f"Using top {len(training_symbols)} symbols: {', '.join(training_symbols)}")
        except Exception as e:
            log_warn(f"Failed to fetch symbols for training: {e}")
            training_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]  # Fallback

    if not training_symbols:
        log_warn("No symbols available for training. Using default symbols.")
        training_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

    return training_symbols


def prompt_training_timeframe(default: str = "1h") -> str:
    """
    Prompt user to enter training timeframe.
    """
    training_timeframe_input = safe_input(
        color_text("Training timeframe [1h]: ", Fore.YELLOW), default="1h", allow_back=True
    )
    training_timeframe = training_timeframe_input.strip() if training_timeframe_input.strip() else "1h"
    try:
        training_timeframe = normalize_timeframe(training_timeframe)
    except Exception:
        log_warn("Invalid timeframe, using default: 1h")
        training_timeframe = "1h"

    return training_timeframe


def prompt_training_limit(default: int = 1500) -> int:
    """
    Prompt user to enter training limit.
    """
    training_limit_input = safe_input(
        color_text("Number of candles per symbol for training [1500]: ", Fore.YELLOW), default="1500", allow_back=True
    )
    try:
        training_limit = int(training_limit_input) if training_limit_input.strip() else 1500
        if training_limit < 100:
            log_warn("Training limit too low, using minimum: 100")
            training_limit = 100
    except ValueError:
        log_warn("Invalid limit, using default: 1500")
        training_limit = 1500

    return training_limit


def confirm_training(symbols: List[str], timeframe: str, limit: int) -> bool:
    """
    Prompt user to confirm training configuration.
    """
    print("\nTraining Configuration:")
    print(f"  Symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''} ({len(symbols)} symbols)")
    print(f"  Timeframe: {timeframe}")
    print(f"  Candles per symbol: {limit}")
    confirm_training = safe_input(
        color_text("Start model training? (y/n) [y]: ", Fore.YELLOW), default="y", allow_back=True
    ).lower()
    if confirm_training in ["y", "yes", ""]:
        return True
    return False
