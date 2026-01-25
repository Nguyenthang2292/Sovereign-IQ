"""SPC enhancement prompts for batch scanner."""

from colorama import Fore
from typing import Any, Dict, Optional

from modules.common.ui.formatting import color_text
from modules.common.ui.logging import log_warn
from modules.common.utils import safe_input


def prompt_spc_config_mode(default: str = "3", loaded_config: Optional[Dict] = None) -> str:
    """
    Prompt user to select SPC configuration mode.
    """
    if loaded_config:
        loaded_spc_config = loaded_config.get("spc_config", {})
        if loaded_spc_config:
            loaded_spc_config.get("config_mode", "3")

    print("\nSPC Enhancements Configuration:")
    print("  Configure Simplified Percentile Clustering enhancements")
    print("  1. Use preset (Conservative/Balanced/Aggressive)")
    print("  2. Custom configuration")
    print("  3. Skip (use defaults from config file)")
    spc_config_mode = safe_input(
        color_text("Select SPC config mode (1/2/3) [3]: ", Fore.YELLOW), default="3", allow_back=True
    )
    if not spc_config_mode:
        spc_config_mode = "3"

    return spc_config_mode


def prompt_spc_preset() -> Optional[str]:
    """
    Prompt user to select SPC preset.
    """
    print("\nSPC Preset:")
    print("  1. Conservative (Most stable - choppy markets)")
    print("  2. Balanced (Recommended - most crypto markets) ⭐")
    print("  3. Aggressive (Most responsive - trending markets)")
    preset_input = safe_input(color_text("Select preset (1/2/3) [2]: ", Fore.YELLOW), default="2", allow_back=True)
    if not preset_input:
        preset_input = "2"

    if preset_input == "1":
        return "conservative"
    elif preset_input == "2":
        return "balanced"
    elif preset_input == "3":
        return "aggressive"
    return None


def prompt_spc_custom_config(loaded_config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Prompt user for custom SPC configuration.
    """
    # Load defaults from loaded_config if available
    spc_volatility_adjustment = False
    spc_use_correlation_weights = False
    spc_time_decay_factor = None
    spc_interpolation_mode = None
    spc_min_flip_duration = None
    spc_flip_confidence_threshold = None
    spc_enable_mtf = False
    spc_mtf_timeframes = None
    spc_mtf_require_alignment = None

    if loaded_config:
        loaded_spc_config = loaded_config.get("spc_config", {})
        if loaded_spc_config:
            spc_volatility_adjustment = loaded_spc_config.get("volatility_adjustment", False)
            spc_use_correlation_weights = loaded_spc_config.get("use_correlation_weights", False)
            spc_time_decay_factor = loaded_spc_config.get("time_decay_factor", None)
            spc_interpolation_mode = loaded_spc_config.get("interpolation_mode", None)
            spc_min_flip_duration = loaded_spc_config.get("min_flip_duration", None)
            spc_flip_confidence_threshold = loaded_spc_config.get("flip_confidence_threshold", None)
            spc_enable_mtf = loaded_spc_config.get("enable_mtf", False)
            spc_mtf_timeframes = loaded_spc_config.get("mtf_timeframes", None)
            spc_mtf_require_alignment = loaded_spc_config.get("mtf_require_alignment", None)

    print("\nSPC Enhancement Options:")

    # Volatility adjustment
    vol_adj_input = safe_input(
        color_text("Enable volatility-adaptive percentiles? (y/n) [n]: ", Fore.YELLOW), default="n", allow_back=True
    ).lower()
    spc_volatility_adjustment = vol_adj_input in ["y", "yes"]

    # Correlation weights
    corr_weights_input = safe_input(
        color_text("Enable correlation-based feature weighting? (y/n) [n]: ", Fore.YELLOW), default="n", allow_back=True
    ).lower()
    spc_use_correlation_weights = corr_weights_input in ["y", "yes"]

    # Time decay factor
    time_decay_input = safe_input(
        color_text("Time decay factor (1.0=no decay, 0.99=light, 0.95=moderate) [1.0]: ", Fore.YELLOW),
        default="1.0",
        allow_back=True,
    )
    if time_decay_input:
        try:
            spc_time_decay_factor = float(time_decay_input)
            if not (0.5 <= spc_time_decay_factor <= 1.0):
                log_warn("Time decay factor must be in [0.5, 1.0], using default 1.0")
                spc_time_decay_factor = None
        except ValueError:
            log_warn("Invalid time decay factor, using default")
            spc_time_decay_factor = None

    # Interpolation mode
    print("\nInterpolation mode:")
    print("  1. Linear (default)")
    print("  2. Sigmoid (smooth transitions)")
    print("  3. Exponential (sticky to current cluster)")
    interp_input = safe_input(
        color_text("Select interpolation mode (1/2/3) [1]: ", Fore.YELLOW), default="1", allow_back=True
    )
    if not interp_input:
        interp_input = "1"
    if interp_input == "2":
        spc_interpolation_mode = "sigmoid"
    elif interp_input == "3":
        spc_interpolation_mode = "exponential"
    else:
        spc_interpolation_mode = "linear"

    # Min flip duration
    flip_dur_input = safe_input(
        color_text("Minimum bars in cluster before flip (1-10) [3]: ", Fore.YELLOW), default="3", allow_back=True
    )
    if flip_dur_input:
        try:
            spc_min_flip_duration = int(flip_dur_input)
            if not (1 <= spc_min_flip_duration <= 10):
                log_warn("Min flip duration must be in [1, 10], using default 3")
                spc_min_flip_duration = None
        except ValueError:
            log_warn("Invalid min flip duration, using default")
            spc_min_flip_duration = None

    # Flip confidence threshold
    conf_thresh_input = safe_input(
        color_text("Flip confidence threshold (0.0-1.0) [0.6]: ", Fore.YELLOW), default="0.6", allow_back=True
    )
    if conf_thresh_input:
        try:
            spc_flip_confidence_threshold = float(conf_thresh_input)
            if not (0.0 <= spc_flip_confidence_threshold <= 1.0):
                log_warn("Confidence threshold must be in [0.0, 1.0], using default 0.6")
                spc_flip_confidence_threshold = None
        except ValueError:
            log_warn("Invalid confidence threshold, using default")
            spc_flip_confidence_threshold = None

    # Multi-timeframe (optional)
    mtf_input = safe_input(
        color_text("Enable multi-timeframe analysis? (y/n) [n]: ", Fore.YELLOW), default="n", allow_back=True
    ).lower()
    spc_enable_mtf = mtf_input in ["y", "yes"]
    if spc_enable_mtf:
        mtf_tf_input = safe_input(
            color_text("MTF timeframes (comma-separated, e.g., 1h,4h,1d) [1h,4h]: ", Fore.YELLOW),
            default="1h,4h",
            allow_back=True,
        )
        if mtf_tf_input:
            spc_mtf_timeframes = [tf.strip() for tf in mtf_tf_input.split(",") if tf.strip()]
        else:
            spc_mtf_timeframes = ["1h", "4h"]

        align_input = safe_input(
            color_text("Require all timeframes to align? (y/n) [y]: ", Fore.YELLOW), default="y", allow_back=True
        ).lower()
        spc_mtf_require_alignment = align_input in ["y", "yes"]

    return {
        "volatility_adjustment": spc_volatility_adjustment,
        "use_correlation_weights": spc_use_correlation_weights,
        "time_decay_factor": spc_time_decay_factor,
        "interpolation_mode": spc_interpolation_mode,
        "min_flip_duration": spc_min_flip_duration,
        "flip_confidence_threshold": spc_flip_confidence_threshold,
        "enable_mtf": spc_enable_mtf,
        "mtf_timeframes": spc_mtf_timeframes,
        "mtf_require_alignment": spc_mtf_require_alignment,
    }
