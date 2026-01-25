"""Random Forest model training prompts for batch scanner."""

from typing import Dict, Optional

from colorama import Fore

from modules.common.ui.logging import log_warn
from modules.common.utils import color_text, safe_input


def prompt_rf_training_mode(rf_model_status: Dict, loaded_config: Optional[Dict] = None) -> Dict:
    """
    Prompt user for RF model training configuration.

    Args:
        rf_model_status: Status dict from check_random_forest_model_status()
        loaded_config: Optional loaded configuration with defaults

    Returns:
        Dict with training configuration
    """
    config = {
        "auto_train": False,
        "training_symbols_mode": "auto",
        "training_symbols_count": 10,
        "manual_symbols": [],
        "training_timeframe": "1h",
        "training_limit": 1500,
    }

    # Load defaults from config if available
    force_retrain = False
    auto_train_if_invalid = True
    if loaded_config and "rf_training" in loaded_config:
        rf_cfg = loaded_config["rf_training"]
        config["training_symbols_mode"] = rf_cfg.get("training_symbols_mode", "auto")
        config["training_symbols_count"] = rf_cfg.get("training_symbols_count", 10)
        config["manual_symbols"] = rf_cfg.get("manual_symbols", [])
        config["training_timeframe"] = rf_cfg.get("training_timeframe", "1h")
        config["training_limit"] = rf_cfg.get("training_limit", 1500)
        force_retrain = rf_cfg.get("force_retrain", False)
        auto_train_if_invalid = rf_cfg.get("auto_train_if_invalid", True)

    # Check model status and handle training/retraining
    if rf_model_status["exists"] and rf_model_status["compatible"]:
        print(f"\n✓ Random Forest model is available: {rf_model_status.get('model_path', 'Unknown path')}")

        if force_retrain:
            print(color_text("🚀 force_retrain is ENABLED in config. Forcing model retraining...", Fore.CYAN))
            config["auto_train"] = True
            return prompt_rf_symbol_selection(config)

        retrain_input = safe_input(
            color_text("Would you like to RETRAIN the model anyway? (y/n) [n]: ", Fore.YELLOW),
            default="n",
            allow_back=True,
        ).lower()

        if retrain_input in ["y", "yes"]:
            config["auto_train"] = True
            return prompt_rf_symbol_selection(config)
        else:
            config["auto_train"] = False
            return config
    else:
        # Model missing or incompatible
        print("\n" + color_text("=" * 60, Fore.YELLOW))
        print(color_text("⚠️  Random Forest Model Not Available", Fore.YELLOW))
        print(color_text("=" * 60, Fore.YELLOW))
        print(f"Status: {rf_model_status.get('message', 'Model not found')}")

        if auto_train_if_invalid:
            print(color_text("\n🚀 auto_train_if_invalid is ENABLED. Preparing for training...", Fore.CYAN))
            config["auto_train"] = True
            return prompt_rf_symbol_selection(config)

        print("\nWould you like to train a new model?")

        auto_train_input = safe_input(
            color_text("Train RF model? (y/n) [y]: ", Fore.YELLOW), default="y", allow_back=True
        ).lower()

        if auto_train_input in ["y", "yes", ""]:
            config["auto_train"] = True
            return prompt_rf_symbol_selection(config)
        else:
            config["auto_train"] = False
            return config


def prompt_rf_symbol_selection(config: Dict) -> Dict:
    """
    Prompt for RF training symbol selection.

    Args:
        config: Current training config dict

    Returns:
        Updated config dict with symbol selection
    """
    print("\n" + color_text("RF Model Training - Symbol Selection", Fore.CYAN))
    print("  1. Auto (top N symbols by volume)")
    print("  2. Manual selection")

    mode_input = safe_input(color_text("Symbol selection mode (1/2) [1]: ", Fore.YELLOW), default="1", allow_back=True)

    if mode_input == "2":
        config["training_symbols_mode"] = "manual"
        print("\nEnter symbols for training (comma-separated):")
        print("Example: BTC/USDT,ETH/USDT,BNB/USDT")

        symbols_input = safe_input(color_text("Training symbols: ", Fore.YELLOW), default="", allow_back=True)

        if symbols_input:
            config["manual_symbols"] = [s.strip() for s in symbols_input.split(",") if s.strip()]
        else:
            log_warn("No symbols entered, falling back to auto mode")
            config["training_symbols_mode"] = "auto"
            config["manual_symbols"] = []
    else:
        config["training_symbols_mode"] = "auto"
        count_input = safe_input(
            color_text(f"Number of top symbols to use [{config['training_symbols_count']}]: ", Fore.YELLOW),
            default=str(config["training_symbols_count"]),
            allow_back=True,
        )

        try:
            config["training_symbols_count"] = int(count_input)
            if config["training_symbols_count"] < 1:
                log_warn("Count must be >= 1, using default: 10")
                config["training_symbols_count"] = 10
        except ValueError:
            log_warn("Invalid count, using default: 10")
            config["training_symbols_count"] = 10

    # Timeframe for training
    timeframe_input = safe_input(
        color_text(f"Training timeframe [{config['training_timeframe']}]: ", Fore.YELLOW),
        default=config["training_timeframe"],
        allow_back=True,
    )
    config["training_timeframe"] = timeframe_input or config["training_timeframe"]

    # Candle limit
    limit_input = safe_input(
        color_text(f"Candles per symbol [{config['training_limit']}]: ", Fore.YELLOW),
        default=str(config["training_limit"]),
        allow_back=True,
    )

    try:
        config["training_limit"] = int(limit_input)
        if config["training_limit"] < 100:
            log_warn("Limit too low, using minimum: 100")
            config["training_limit"] = 100
    except ValueError:
        log_warn(f"Invalid limit, using default: {config['training_limit']}")

    # Summary
    print("\n" + color_text("Training Configuration:", Fore.GREEN))
    if config["training_symbols_mode"] == "auto":
        print(f"  Symbols: Top {config['training_symbols_count']} by volume")
    else:
        manual_syms = config["manual_symbols"]
        syms_list = ", ".join(manual_syms[:5])
        dots = "..." if len(manual_syms) > 5 else ""
        print(f"  Symbols: {syms_list}{dots} ({len(manual_syms)} total)")
    print(f"  Timeframe: {config['training_timeframe']}")
    print(f"  Candles: {config['training_limit']}")

    return config


def prompt_atc_performance(loaded_config: Optional[Dict] = None) -> Dict:
    """
    Prompt for ATC high-performance parameters.

    Args:
        loaded_config: Optional loaded configuration with defaults

    Returns:
        Dict with ATC performance config
    """
    # Defaults from ATC settings guide (High-Performance preset)
    config = {
        "batch_processing": True,
        "use_cuda": False,
        "parallel_l1": True,
        "parallel_l2": True,
        "use_cache": True,
        "fast_mode": True,
    }

    # Load defaults from config if available
    if loaded_config and "atc_performance" in loaded_config:
        atc_cfg = loaded_config["atc_performance"]
        config.update(
            {
                "batch_processing": atc_cfg.get("batch_processing", True),
                "use_cuda": atc_cfg.get("use_cuda", False),
                "parallel_l1": atc_cfg.get("parallel_l1", True),
                "parallel_l2": atc_cfg.get("parallel_l2", True),
                "use_cache": atc_cfg.get("use_cache", True),
                "fast_mode": atc_cfg.get("fast_mode", True),
            }
        )

    print("\n" + color_text("ATC High-Performance Configuration", Fore.CYAN))
    print("  Configure Rust/CUDA acceleration and parallelization")
    print("\nPresets:")
    print("  1. High-Performance (Recommended) - Rust Rayon batch + all optimizations")
    print("  2. GPU Accelerated - CUDA (slower for < 500 symbols)")
    print("  3. Custom settings")
    print("  4. Use defaults")

    preset_input = safe_input(color_text("Select preset (1/2/3/4) [1]: ", Fore.YELLOW), default="1", allow_back=True)

    if preset_input == "2":
        # GPU preset
        config["batch_processing"] = False
        config["use_cuda"] = True
        print(color_text("\n✓ GPU (CUDA) preset applied", Fore.GREEN))
    elif preset_input == "3":
        # Custom settings
        print("\nCustom ATC Performance Settings:")

        batch_input = safe_input(
            color_text(
                f"Enable Rust Rayon batch processing? (y/n) [{'y' if config['batch_processing'] else 'n'}]: ",
                Fore.YELLOW,
            ),
            default="y" if config["batch_processing"] else "n",
            allow_back=True,
        ).lower()
        config["batch_processing"] = batch_input in ["y", "yes", ""]

        cuda_input = safe_input(
            color_text(f"Enable CUDA GPU acceleration? (y/n) [{'y' if config['use_cuda'] else 'n'}]: ", Fore.YELLOW),
            default="y" if config["use_cuda"] else "n",
            allow_back=True,
        ).lower()
        config["use_cuda"] = cuda_input in ["y", "yes"]

        parallel_l1_input = safe_input(
            color_text(f"Enable parallel Layer 1? (y/n) [{'y' if config['parallel_l1'] else 'n'}]: ", Fore.YELLOW),
            default="y" if config["parallel_l1"] else "n",
            allow_back=True,
        ).lower()
        config["parallel_l1"] = parallel_l1_input in ["y", "yes", ""]

        parallel_l2_input = safe_input(
            color_text(f"Enable parallel Layer 2? (y/n) [{'y' if config['parallel_l2'] else 'n'}]: ", Fore.YELLOW),
            default="y" if config["parallel_l2"] else "n",
            allow_back=True,
        ).lower()
        config["parallel_l2"] = parallel_l2_input in ["y", "yes", ""]

        cache_input = safe_input(
            color_text(f"Enable MA caching? (y/n) [{'y' if config['use_cache'] else 'n'}]: ", Fore.YELLOW),
            default="y" if config["use_cache"] else "n",
            allow_back=True,
        ).lower()
        config["use_cache"] = cache_input in ["y", "yes", ""]

        fast_mode_input = safe_input(
            color_text(f"Enable fast mode? (y/n) [{'y' if config['fast_mode'] else 'n'}]: ", Fore.YELLOW),
            default="y" if config["fast_mode"] else "n",
            allow_back=True,
        ).lower()
        config["fast_mode"] = fast_mode_input in ["y", "yes", ""]
    elif preset_input == "4":
        # Use defaults (already set)
        print(color_text("\n✓ Using default ATC performance settings", Fore.GREEN))
    else:
        # High-Performance preset (default)
        print(color_text("\n✓ High-Performance preset applied (Rust Rayon + all optimizations)", Fore.GREEN))

    return config
