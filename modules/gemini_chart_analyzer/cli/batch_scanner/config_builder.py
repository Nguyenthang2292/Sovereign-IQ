"""
Configuration builder for batch scanner.

Handles gathering configuration from interactive prompts or loading from files.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from colorama import Fore

from modules.common.utils import NavigationBack, color_text, safe_input
from modules.gemini_chart_analyzer.cli.config.display import display_loaded_configuration
from modules.gemini_chart_analyzer.cli.config.loader import (
    list_configuration_files,
    load_configuration_from_file,
)
from modules.gemini_chart_analyzer.cli.models.random_forest_manager import (
    check_random_forest_model_status,
)
from modules.gemini_chart_analyzer.cli.prompts.pre_filter import (
    prompt_enable_pre_filter,
    prompt_fast_mode,
    prompt_pre_filter_mode,
    prompt_pre_filter_percentage,
)
from modules.gemini_chart_analyzer.cli.prompts.rf_training import (
    prompt_atc_performance,
    prompt_rf_training_mode,
)
from modules.gemini_chart_analyzer.cli.prompts.spc import (
    prompt_spc_config_mode,
    prompt_spc_custom_config,
    prompt_spc_preset,
)
from modules.gemini_chart_analyzer.cli.prompts.timeframe import (
    prompt_analysis_mode,
    prompt_cooldown,
    prompt_limit,
    prompt_market_coverage,
)


def gather_scan_configuration() -> Dict[str, Any]:
    """Gather all scan configuration through prompts or file loading with Backspace support."""
    # Configuration state
    config = {
        "loaded_config": None,
        "analysis_mode": "multi-timeframe",
        "timeframe": "1h",
        "timeframes": None,
        "max_symbols": None,
        "cooldown": 2.5,
        "limit": 700,
        "enable_pre_filter": True,
        "pre_filter_mode": "voting",
        "pre_filter_percentage": None,
        "fast_mode": True,
        "stage0_sample_percentage": None,
        "spc_config_mode": "3",
        "spc_preset": None,
        "spc_config": {
            "volatility_adjustment": False,
            "use_correlation_weights": False,
            "time_decay_factor": None,
            "interpolation_mode": None,
            "min_flip_duration": None,
            "flip_confidence_threshold": None,
            "enable_mtf": False,
            "mtf_timeframes": None,
            "mtf_require_alignment": None,
        },
        "random_forest_model": {"status": None, "retrained": False, "model_path": None},
        "rf_training": {
            "auto_train": False,
            "training_symbols_mode": "auto",
            "training_symbols_count": 10,
            "manual_symbols": [],
            "training_timeframe": "1h",
            "training_limit": 1500,
            "force_retrain": False,
            "auto_train_if_invalid": True,
        },
        "atc_performance": {
            "batch_processing": True,
            "use_cuda": False,
            "parallel_l1": True,
            "parallel_l2": True,
            "use_cache": True,
            "fast_mode": True,
        },
        "use_loaded_config": False,
    }

    # Define steps
    steps = [
        "load_config_prompt",
        "analysis_mode",
        "market_coverage",
        "cooldown",
        "limit",
        "enable_pre_filter",
        "pre_filter_mode",
        "pre_filter_percentage",
        "fast_mode",
        "spc_config_mode",
        "spc_settings",
        "atc_performance",
        "rf_training",
    ]

    current_step = 0
    config_files = list_configuration_files()

    print(color_text("\n(Tip: Press Backspace or type 'b' at any prompt to go back)\n", Fore.CYAN))

    while current_step < len(steps):
        step_name = steps[current_step]
        try:
            if step_name == "load_config_prompt":
                if not config["use_loaded_config"]:
                    print("\nLoad Configuration:")
                    if config_files:
                        print(f"  {len(config_files)} configuration file(s) found in project root")
                    else:
                        print("  No configuration files found in project root")
                        print("  You can still enter a full path to a JSON or YAML file")

                    load_config_input = safe_input(
                        color_text("Load configuration from file? (y/n) [n]: ", Fore.YELLOW),
                        default="n",
                        allow_back=False,
                    ).lower()

                    if load_config_input in ["y", "yes"]:
                        if config_files:
                            print("\nAvailable configuration files:")
                            for idx, config_file in enumerate(config_files[:15], 1):
                                try:
                                    mtime = datetime.fromtimestamp(config_file.stat().st_mtime)
                                    mtime_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
                                    print(f"  {idx}. {config_file.name} ({mtime_str})")
                                except (OSError, ValueError):
                                    print(f"  {idx}. {config_file.name}")

                        file_choice = safe_input(
                            color_text("\nSelect file number or enter full path (press Enter to skip): ", Fore.YELLOW),
                            default="",
                            allow_back=True,
                        ).strip()

                        if file_choice:
                            config_path = None
                            try:
                                file_idx = int(file_choice)
                                if config_files and 1 <= file_idx <= len(config_files):
                                    config_path = config_files[file_idx - 1]
                            except ValueError:
                                config_path = Path(file_choice)

                            if config_path:
                                config["loaded_config"] = load_configuration_from_file(config_path)
                                if config["loaded_config"]:
                                    display_loaded_configuration(config["loaded_config"])
                                    print("\nConfiguration Options:")
                                    print("  1. Use loaded configuration as-is")
                                    print("  2. Use as defaults and adjust")
                                    print("  3. Start fresh (ignore loaded config)")
                                    use_choice = safe_input(
                                        color_text("Select option (1/2/3) [2]: ", Fore.YELLOW),
                                        default="2",
                                        allow_back=True,
                                    )
                                    if use_choice == "1":
                                        config["use_loaded_config"] = True
                                        loaded_data = config["loaded_config"]

                                        # Basic settings
                                        config["analysis_mode"] = loaded_data.get("analysis_mode", "multi-timeframe")
                                        config["timeframe"] = loaded_data.get("timeframe", "1h")
                                        config["timeframes"] = loaded_data.get("timeframes", [])
                                        config["max_symbols"] = loaded_data.get("max_symbols")
                                        config["cooldown"] = loaded_data.get("cooldown", 2.5)
                                        config["limit"] = loaded_data.get("limit", 700)

                                        # Pre-filter
                                        config["enable_pre_filter"] = loaded_data.get("enable_pre_filter", True)
                                        config["pre_filter_mode"] = loaded_data.get("pre_filter_mode", "voting")
                                        config["pre_filter_percentage"] = loaded_data.get("pre_filter_percentage")
                                        config["fast_mode"] = loaded_data.get("fast_mode", True)
                                        config["stage0_sample_percentage"] = loaded_data.get("stage0_sample_percentage")

                                        # SPC
                                        spc_cfg = loaded_data.get("spc_config", {})
                                        config["spc_preset"] = spc_cfg.get("preset", "3")
                                        config["spc_config"].update({k: v for k, v in spc_cfg.items() if k != "preset"})

                                        # RF & Performance
                                        if "rf_training" in loaded_data:
                                            rf_cfg = loaded_data["rf_training"]
                                            config["rf_training"].update(rf_cfg)

                                            # Determine if we should auto-train
                                            rf_status = check_random_forest_model_status()
                                            force_retrain = rf_cfg.get("force_retrain", False)
                                            auto_train_if_invalid = rf_cfg.get("auto_train_if_invalid", True)

                                            if force_retrain or (
                                                auto_train_if_invalid
                                                and (not rf_status["exists"] or not rf_status["compatible"])
                                            ):
                                                config["rf_training"]["auto_train"] = True

                                        if "atc_performance" in loaded_data:
                                            config["atc_performance"].update(loaded_data["atc_performance"])

                                        current_step = len(steps)
                                        continue
                                    elif use_choice == "3":
                                        config["loaded_config"] = None

                current_step += 1

            elif step_name == "analysis_mode":
                config["analysis_mode"], config["timeframe"], config["timeframes"] = prompt_analysis_mode(
                    default="2", loaded_config=config["loaded_config"]
                )
                current_step += 1

            elif step_name == "market_coverage":
                config["max_symbols"], config["stage0_sample_percentage"] = prompt_market_coverage(
                    loaded_config=config["loaded_config"]
                )
                current_step += 1

            elif step_name == "cooldown":
                config["cooldown"] = prompt_cooldown(default=2.5, loaded_config=config["loaded_config"])
                current_step += 1

            elif step_name == "limit":
                config["limit"] = prompt_limit(default=700, loaded_config=config["loaded_config"])
                current_step += 1

            elif step_name == "enable_pre_filter":
                config["enable_pre_filter"] = prompt_enable_pre_filter(
                    default=True, loaded_config=config["loaded_config"]
                )
                if not config["enable_pre_filter"]:
                    # Skip pre-filter settings and go to summary
                    current_step = len(steps)
                else:
                    current_step += 1

            elif step_name == "pre_filter_mode":
                config["pre_filter_mode"] = prompt_pre_filter_mode(
                    default="voting", loaded_config=config["loaded_config"]
                )
                current_step += 1

            elif step_name == "pre_filter_percentage":
                config["pre_filter_percentage"] = prompt_pre_filter_percentage(
                    default=None, loaded_config=config["loaded_config"]
                )
                current_step += 1

            elif step_name == "fast_mode":
                config["fast_mode"] = prompt_fast_mode(default=True, loaded_config=config["loaded_config"])
                current_step += 1

            elif step_name == "spc_config_mode":
                config["spc_config_mode"] = prompt_spc_config_mode(default="3", loaded_config=config["loaded_config"])
                current_step += 1

            elif step_name == "spc_settings":
                if config["spc_config_mode"] == "1":
                    config["spc_preset"] = prompt_spc_preset()
                elif config["spc_config_mode"] == "2":
                    config["spc_config"] = prompt_spc_custom_config(loaded_config=config["loaded_config"])
                current_step += 1

            elif step_name == "atc_performance":
                # Prompt for ATC high-performance parameters
                atc_perf_config = prompt_atc_performance(loaded_config=config["loaded_config"])
                config["atc_performance"].update(atc_perf_config)
                current_step += 1

            elif step_name == "rf_training":
                # Check RF model status and prompt for training if needed
                rf_status = check_random_forest_model_status()
                config["random_forest_model"]["status"] = rf_status

                # Prompt for RF training configuration
                rf_training_config = prompt_rf_training_mode(rf_status, loaded_config=config["loaded_config"])
                config["rf_training"].update(rf_training_config)

                # If auto-train is enabled, we'll handle training later in the batch scan service
                if rf_training_config["auto_train"]:
                    config["random_forest_model"]["retrained"] = True

                current_step += 1

        except NavigationBack:
            if current_step > 0:
                current_step -= 1
                # No more special jumps needed since export_config is gone
                # Handle other jumps if necessary
                print(color_text(f"\n[Go back to: {steps[current_step]}]", Fore.CYAN))
            else:
                print(color_text("\nAlready at first step.", Fore.YELLOW))

    # Construct final dictionary correctly
    rf_status = config["random_forest_model"].get("status")
    return {
        "analysis_mode": config["analysis_mode"],
        "timeframe": config["timeframe"],
        "timeframes": config["timeframes"],
        "max_symbols": config["max_symbols"],
        "cooldown": config["cooldown"],
        "limit": config["limit"],
        "enable_pre_filter": config["enable_pre_filter"],
        "pre_filter_mode": config["pre_filter_mode"],
        "pre_filter_percentage": config["pre_filter_percentage"],
        "fast_mode": config["fast_mode"],
        "stage0_sample_percentage": config["stage0_sample_percentage"],
        "spc_config": {**config["spc_config"], "preset": config["spc_preset"]},
        "random_forest_model": {
            "status": rf_status,
            "retrained": config["random_forest_model"]["retrained"],
            "model_path": rf_status.get("model_path") if rf_status else None,
        },
        "rf_training": config["rf_training"],
        "atc_performance": config["atc_performance"],
    }
