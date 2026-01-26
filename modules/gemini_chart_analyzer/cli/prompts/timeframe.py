"""Timeframe selection prompts for batch scanner."""

from typing import Dict, List, Optional, Tuple

from colorama import Fore

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
    mode = safe_input(color_text(f"Select mode (1/2) [{default}]: ", Fore.YELLOW), default=default, allow_back=True)
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
            allow_back=True,
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
            color_text(f"Enter timeframe [{default_timeframe}]: ", Fore.YELLOW),
            default=default_timeframe,
            allow_back=True,
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


def prompt_market_coverage(
    loaded_config: Optional[Dict] = None,
) -> Tuple[Optional[int], Optional[float], str, int, float]:
    """Prompt user for market coverage (merging max symbols and sampling).

    Args:
        loaded_config: Optional loaded configuration for defaults

    Returns:
        Tuple of (max_symbols, stage0_percentage, strategy, strata_count, hybrid_top_percentage)
    """
    default_max = None
    default_sample = None
    default_strategy = "stratified"
    default_strata_count = 3
    default_hybrid_top = 50.0

    if loaded_config:
        default_max = loaded_config.get("max_symbols")
        default_sample = loaded_config.get("stage0_sample_percentage")
        default_strategy = loaded_config.get("stage0_sampling_strategy", "stratified")
        default_strata_count = loaded_config.get("stage0_stratified_strata_count", 3)
        default_hybrid_top = loaded_config.get("stage0_hybrid_top_percentage", 50.0)

    print("\nMarket Coverage:")
    print("  1. All symbols (Full scan)")
    print("  2. Sample % (Smart sampling - RECOMMENDED for large markets)")
    print("  3. Fixed Count (Scan only the first N symbols)")

    # Determine default choice
    default_choice = "1"
    if default_sample:
        default_choice = "2"
    elif default_max:
        default_choice = "3"

    choice = safe_input(
        color_text(f"Select option (1/2/3) [{default_choice}]: ", Fore.YELLOW),
        default=default_choice,
        allow_back=True,
    )

    if choice == "2":
        print("\nStage 0: Smart Sampling (Strategic Coverage):")
        print("  Select a percentage of symbols using a specific strategy")
        sample_prompt = f"[{default_sample}%]" if default_sample else "[33%]"
        sample_input = safe_input(
            color_text(f"Percentage to sample (1-100) {sample_prompt}: ", Fore.YELLOW),
            default=str(default_sample) if default_sample else "33",
            allow_back=True,
        ).strip()

        try:
            if sample_input.endswith("%"):
                sample_input = sample_input[:-1]
            percentage = float(sample_input)
            percentage = max(1.0, min(100.0, percentage))
        except ValueError:
            log_warn("Invalid input, defaulting to 33%")
            percentage = 33.0

        print("\nSampling Strategy:")
        print("  1. Stratified (RECOMMENDED): Sample evenly from volume tiers (Top/Mid/Low)")
        print("  2. Volume-weighted: High volume symbols have higher chance")
        print("  3. Top-N Hybrid: Top N% by volume, rest random")
        print("  4. Systematic: Every n-th symbol from volume-sorted list")
        print("  5. Liquidity-weighted: Combines volume/volatility/spread")
        print("  6. Pure Random: Uniform probability for all symbols")

        # Map display order to internal names
        strategy_map = {
            "1": "stratified",
            "2": "volume_weighted",
            "3": "top_n_hybrid",
            "4": "systematic",
            "5": "liquidity_weighted",
            "6": "random",
        }

        # Reverse map for default
        reverse_strategy_map = {v: k for k, v in strategy_map.items()}
        default_strategy_choice = reverse_strategy_map.get(default_strategy, "1")

        strat_choice = safe_input(
            color_text(f"Select strategy (1-6) [{default_strategy_choice}]: ", Fore.YELLOW),
            default=default_strategy_choice,
            allow_back=True,
        )

        strategy = strategy_map.get(strat_choice, "stratified")
        strata_count = default_strata_count
        hybrid_top_percentage = default_hybrid_top

        if strategy == "stratified":
            count_input = safe_input(
                color_text(f"Number of volume tiers (strata) [{default_strata_count}]: ", Fore.YELLOW),
                default=str(default_strata_count),
                allow_back=True,
            )
            try:
                strata_count = int(count_input)
            except ValueError:
                strata_count = 3
        elif strategy == "top_n_hybrid":
            top_input = safe_input(
                color_text(f"Percentage of sample to take from top volume [{default_hybrid_top}%]: ", Fore.YELLOW),
                default=str(default_hybrid_top),
                allow_back=True,
            )
            try:
                hybrid_top_percentage = float(top_input)
            except ValueError:
                hybrid_top_percentage = 50.0

        return None, percentage, strategy, strata_count, hybrid_top_percentage

    elif choice == "3":
        default_max_str = str(default_max) if default_max else "50"
        max_input = safe_input(
            color_text(f"Number of symbols to scan [{default_max_str}]: ", Fore.YELLOW),
            default=default_max_str,
            allow_back=True,
        ).strip()

        try:
            return int(max_input), None, "random", 3, 50.0
        except ValueError:
            log_warn("Invalid input, defaulting to 50 symbols")
            return 50, None, "random", 3, 50.0

    # Default: All symbols
    return None, None, "random", 3, 50.0


def prompt_cooldown(default: float = 2.5, loaded_config: Optional[Dict] = None) -> float:
    """Prompt user for cooldown."""
    if loaded_config:
        default = loaded_config.get("cooldown", 2.5)

    default_cooldown_str = str(default)
    cooldown_input = safe_input(
        color_text(f"Cooldown between batches in seconds [{default}]: ", Fore.YELLOW),
        default=default_cooldown_str,
        allow_back=True,
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
        color_text(f"Number of candles per symbol [{default}]: ", Fore.YELLOW),
        default=default_limit_str,
        allow_back=True,
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
