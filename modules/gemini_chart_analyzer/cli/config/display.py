"""Configuration display module for batch scanner."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

from colorama import Fore

from modules.common.utils import color_text


@dataclass
class DisplayConfig:
    """Configuration for CLI display formatting."""

    confidence_bar_length: int = 10
    confidence_min: float = 0.0
    confidence_max: float = 1.0
    divider_length: int = 60
    symbol_column_width: int = 15
    symbols_per_row_fallback: int = 5
    fallback_column_width: int = 12


def display_loaded_configuration(config_data: Dict[str, Any]) -> None:
    """Display loaded configuration in a readable format.

    Args:
        config_data: Dictionary containing configuration parameters
    """
    print()
    print(color_text("=" * 60, Fore.CYAN))
    print(color_text("LOADED CONFIGURATION", Fore.CYAN))
    print(color_text("=" * 60, Fore.CYAN))

    # Analysis mode
    analysis_mode = config_data.get("analysis_mode", "single-timeframe")
    if analysis_mode == "multi-timeframe":
        timeframes = config_data.get("timeframes", [])
        print(f"Timeframes: {', '.join(timeframes)} (Multi-timeframe mode)")
    else:
        timeframe = config_data.get("timeframe", "1h")
        print(f"Timeframe: {timeframe} (Single timeframe mode)")

    # Basic settings
    max_symbols = config_data.get("max_symbols")
    print(f"Max symbols: {max_symbols or 'All'}")
    print(f"Cooldown: {config_data.get('cooldown', 2.5)}s")
    print(f"Candles per symbol: {config_data.get('limit', 700)}")

    # Pre-filter settings
    enable_pre_filter = config_data.get("enable_pre_filter", False)
    if enable_pre_filter:
        pre_filter_mode = config_data.get("pre_filter_mode", "voting")
        fast_mode = config_data.get("fast_mode", True)
        mode_display = "Voting mode" if pre_filter_mode == "voting" else "Hybrid mode"
        speed_display = "Fast (3-stage: ATC → RangeOsc+SPC → ML models)" if fast_mode else "Full (All indicators)"
        print(f"Pre-filter: Enabled ({mode_display}, {speed_display})")

        # SPC configuration
        spc_config = config_data.get("spc_config")
        if spc_config:
            print("\nSPC Enhancements:")
            if spc_config.get("preset"):
                print(f"  Preset: {spc_config['preset'].capitalize()}")
            else:
                enhancements_list = []
                if spc_config.get("volatility_adjustment"):
                    enhancements_list.append("Volatility Adjustment")
                if spc_config.get("use_correlation_weights"):
                    enhancements_list.append("Correlation Weighting")
                time_decay = spc_config.get("time_decay_factor")
                if time_decay and time_decay != 1.0:
                    enhancements_list.append(f"Time Decay ({time_decay})")
                interp_mode = spc_config.get("interpolation_mode")
                if interp_mode and interp_mode != "linear":
                    enhancements_list.append(f"Interpolation ({interp_mode})")
                min_flip = spc_config.get("min_flip_duration")
                if min_flip:
                    enhancements_list.append(f"Min Flip Duration ({min_flip})")
                conf_thresh = spc_config.get("flip_confidence_threshold")
                if conf_thresh:
                    enhancements_list.append(f"Flip Confidence ({conf_thresh})")
                if spc_config.get("enable_mtf"):
                    mtf_tfs = spc_config.get("mtf_timeframes", [])
                    align = spc_config.get("mtf_require_alignment", False)
                    enhancements_list.append(f"Multi-Timeframe ({', '.join(mtf_tfs)}, align={align})")
                if enhancements_list:
                    print(f"  Custom: {', '.join(enhancements_list)}")
                else:
                    print("  Using defaults from config file")

        # Random Forest model
        rf_model = config_data.get("random_forest_model")
        if rf_model and isinstance(rf_model, dict):
            print("\nRandom Forest Model:")
            status = rf_model.get("status") or {}
            if status.get("exists"):
                compatible = status.get("compatible", False)
                status_display = "Compatible" if compatible else "Incompatible (uses deprecated features)"
                status_color = Fore.GREEN if compatible else Fore.RED
                print(f"  Status: {color_text(status_display, status_color)}")
                if rf_model.get("retrained"):
                    print(f"  {color_text('Model retrained', Fore.GREEN)}")
            else:
                print(f"  Status: {color_text('Not found', Fore.YELLOW)}")
    else:
        print("Pre-filter: Disabled")

    print()

    # Export timestamp if available
    export_timestamp = config_data.get("export_timestamp")
    if export_timestamp:
        try:
            dt = datetime.fromisoformat(export_timestamp)
            print(f"Configuration exported: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        except (ValueError, TypeError):
            pass
