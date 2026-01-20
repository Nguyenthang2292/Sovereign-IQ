"""Timeframe selection prompts for batch scanner."""

from colorama import Fore
from typing import Dict, List, Optional, Tuple

from modules.common.ui.formatting import color_text
from modules.common.ui.logging import log_error, log_warn
from modules.common.utils import normalize_timeframe, safe_input
from modules.gemini_chart_analyzer.core.utils import DEFAULT_TIMEFRAMES, normalize_timeframes


def prompt_analysis_mode(
    default: str = "2", loaded_config: Optional[Dict] = None
) -> Tuple[str, Optional[str], Optional[List[str]]]:
    """
    Prompt user to select analysis mode (single or multi-timeframe).
    """
    default_timeframes_str = None
    default_timeframe = "1h"

    if loaded_config:
        analysis_mode = loaded_config.get("analysis_mode", "multi-timeframe")
        if analysis_mode == "multi-timeframe":
            default = "2"
            loaded_tfs = loaded_config.get("timeframes", [])
            if loaded_tfs:
                default_timeframes_str = ", ".join(loaded_tfs)
        else:
            default = "1"
            default_timeframe = loaded_config.get("timeframe", "1h")

    print("\nAnalysis mode:")
    print("  1. Single timeframe")
    print("  2. Multi-timeframe (recommended)")
    mode = safe_input(color_text(f"Select mode (1/2) [{default}]: ", Fore.YELLOW), default=default)
    if not mode:
        mode = default

    timeframe = None
    timeframes = None

    if mode == "2":
        default_tf_display = default_timeframes_str if default_timeframes_str else ", ".join(DEFAULT_TIMEFRAMES)
        print(f"\nDefault timeframes: {', '.join(DEFAULT_TIMEFRAMES)}")
        print("Timeframes: 15m, 30m,1h, 4h, 1d, 1w (comma-separated)")
        timeframes_input = safe_input(
            color_text(f"Enter timeframes (comma-separated) [{default_tf_display}]: ", Fore.YELLOW),
            default=default_timeframes_str if default_timeframes_str else "",
        )
        if not timeframes_input and default_timeframes_str:
            try:
                timeframes_list = [tf.strip() for tf in default_timeframes_str.split(",") if tf.strip()]
                timeframes = normalize_timeframes(timeframes_list)
            except Exception:
                timeframes = DEFAULT_TIMEFRAMES
        elif not timeframes_input:
            timeframes = DEFAULT_TIMEFRAMES
        else:
            try:
                timeframes_list = [tf.strip() for tf in timeframes_input.split(",") if tf.strip()]
                timeframes = normalize_timeframes(timeframes_list)
                if not timeframes:
                    log_warn("No valid timeframes, using default")
                    timeframes = DEFAULT_TIMEFRAMES
            except Exception as e:
                log_warn(f"Error parsing timeframes: {e}, using default")
                timeframes = DEFAULT_TIMEFRAMES

        return "multi-timeframe", None, timeframes
    else:
        print("\nTimeframes: 15m, 30m,1h, 4h, 1d, 1w")
        timeframe = safe_input(
            color_text(f"Enter timeframe [{default_timeframe}]: ", Fore.YELLOW), default=default_timeframe
        )
        if not timeframe:
            timeframe = default_timeframe

        try:
            timeframe = normalize_timeframe(timeframe)
        except Exception as e:
            log_warn(f"Error parsing timeframe: {e}, defaulting to '{default_timeframe}'")
            timeframe = default_timeframe
            try:
                timeframe = normalize_timeframe(timeframe)
            except Exception as e2:
                log_error(f"Critical error normalizing default timeframe: {e2}")
                raise

        return "single-timeframe", timeframe, None


def prompt_max_symbols(default: Optional[int] = None, loaded_config: Optional[Dict] = None) -> Optional[int]:
    """Prompt user for max symbols."""
    if loaded_config:
        default = loaded_config.get("max_symbols", None)

    default_max_symbols_str = str(default) if default else ""
    max_symbols_prompt = f"[{default_max_symbols_str}]" if default_max_symbols_str else "[all]"
    max_symbols_input = safe_input(
        color_text(f"Max symbols to scan (press Enter for all) {max_symbols_prompt}: ", Fore.YELLOW),
        default=default_max_symbols_str if default_max_symbols_str else "",
    )
    max_symbols = None
    if max_symbols_input:
        try:
            max_symbols = int(max_symbols_input)
            if max_symbols < 1:
                log_warn(f"max_symbols ({max_symbols}) must be >= 1, resetting to all")
                max_symbols = None
        except ValueError:
            log_warn("Invalid input, scanning all symbols")
    elif default:
        max_symbols = default

    return max_symbols


def prompt_cooldown(default: float = 2.5, loaded_config: Optional[Dict] = None) -> float:
    """Prompt user for cooldown."""
    if loaded_config:
        default = loaded_config.get("cooldown", 2.5)

    default_cooldown_str = str(default)
    cooldown_input = safe_input(
        color_text(f"Cooldown between batches in seconds [{default}]: ", Fore.YELLOW), default=default_cooldown_str
    )
    if not cooldown_input and loaded_config:
        return default
    else:
        cooldown = 2.5
        if cooldown_input:
            try:
                cooldown = float(cooldown_input)
                if cooldown < 0.0:
                    log_warn(f"cooldown ({cooldown}) must be >= 0.0, clamping to 0.0")
                    cooldown = 0.0
            except ValueError:
                log_warn("Invalid input, using default 2.5s")
        return cooldown


def prompt_limit(default: int = 700, loaded_config: Optional[Dict] = None) -> int:
    """Prompt user for candle limit."""
    if loaded_config:
        default = loaded_config.get("limit", 700)

    default_limit_str = str(default)
    limit_input = safe_input(
        color_text(f"Number of candles per symbol [{default}]: ", Fore.YELLOW), default=default_limit_str
    )
    if not limit_input and loaded_config:
        return default
    else:
        limit = 700
        if limit_input:
            try:
                limit = int(limit_input)
                if limit < 1:
                    log_warn(f"limit ({limit}) must be >= 1, clamping to 1")
                    limit = 1
            except ValueError:
                log_warn("Invalid input, using default 700")
        return limit
