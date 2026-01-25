"""Pre-filter configuration prompts for batch scanner."""

from typing import Dict, Optional

from colorama import Fore

from modules.common.utils import color_text, safe_input


def prompt_enable_pre_filter(default: bool = False, loaded_config: Optional[Dict] = None) -> bool:
    """
    Prompt user to enable or disable pre-filter.

    Args:
        default: Default value for pre-filter enable/disable
        loaded_config: Optional loaded configuration to use for defaults

    Returns:
        True if pre-filter should be enabled, False otherwise
    """
    if loaded_config:
        default = loaded_config.get("enable_pre_filter", False)

    default_pre_filter = "y" if default else "n"
    print("\nPre-filter option:")
    print("  Filter symbols using VotingAnalyzer or HybridAnalyzer before Gemini scan")
    print("  (Selects all symbols with signals)")
    pre_filter_input = safe_input(
        color_text(f"Enable pre-filter? (y/n) [{default_pre_filter}]: ", Fore.YELLOW),
        default=default_pre_filter,
        allow_back=True,
    ).lower()
    if not pre_filter_input:
        pre_filter_input = default_pre_filter
    enable_pre_filter = pre_filter_input in ["y", "yes"]

    return enable_pre_filter


def prompt_pre_filter_mode(default: str = "voting", loaded_config: Optional[Dict] = None) -> str:
    """
    Prompt user to select pre-filter mode (voting/hybrid).

    Args:
        default: Default pre-filter mode
        loaded_config: Optional loaded configuration to use for defaults

    Returns:
        Pre-filter mode string ("voting" or "hybrid")
    """
    if loaded_config:
        default = loaded_config.get("pre_filter_mode", "voting")

    print("\nPre-filter mode:")
    print("  1. Voting (Pure voting system - all indicators vote simultaneously)")
    print("  2. Hybrid (Sequential filtering: ATC → Range Oscillator → SPC → Decision Matrix)")
    default_mode_num = "1" if default == "voting" else "2"
    mode_input = safe_input(
        color_text(f"Select pre-filter mode (1/2) [{default_mode_num}]: ", Fore.YELLOW),
        default=default_mode_num,
        allow_back=True,
    )
    if not mode_input:
        mode_input = default_mode_num

    if mode_input == "2":
        return "hybrid"
    else:
        return "voting"


def prompt_fast_mode(default: bool = True, loaded_config: Optional[Dict] = None) -> bool:
    """
    Prompt user to select fast or full mode for pre-filter.

    Args:
        default: Default value for fast mode
        loaded_config: Optional loaded configuration to use for defaults

    Returns:
        True if fast mode, False if full mode
    """
    if loaded_config:
        default = loaded_config.get("fast_mode", True)

    default_fast_mode = "1" if default else "2"
    print("\nPre-filter speed mode:")
    print("  1. Fast (3-stage filtering - Recommended)")
    print("     Stage 1: ATC → Stage 2: Range Osc + SPC → Stage 3: All ML models")
    print("     Note: Stage 3 runs ML models for symbols that passed Stage 2")
    print("  2. Full (All indicators including ML models - SLOW!)")
    fast_mode_input = safe_input(
        color_text(f"Select mode (1/2) [{default_fast_mode}]: ", Fore.YELLOW),
        default=default_fast_mode,
        allow_back=True,
    )
    if not fast_mode_input:
        fast_mode_input = default_fast_mode
    fast_mode = fast_mode_input != "2"  # True if not "2", False if "2"

    return fast_mode


def prompt_stage0_random_sample(
    default: Optional[float] = None, total_symbols: Optional[int] = None, loaded_config: Optional[Dict] = None
) -> Optional[float]:
    """
    Prompt user to enable Stage 0 random sampling via percentage.

    Args:
        default: Default percentage to sample (None = disabled)
        total_symbols: Total number of available symbols (if known)
        loaded_config: Optional loaded configuration to use for defaults

    Returns:
        Percentage of symbols to randomly sample (1-100), or None to disable sampling
    """
    if loaded_config:
        default = loaded_config.get("stage0_sample_percentage", loaded_config.get("stage0_random_sample", None))

    print("\nStage 0: Random Sampling (Optional Speed Boost):")
    print("  Randomly select a percentage of symbols before running ATC scan")
    print("  This significantly speeds up the pre-filter process for large markets")
    if total_symbols:
        print(f"  Total available symbols: {total_symbols}")
    print("  Enter percentage to sample (1-100) (press Enter to skip)")

    while True:
        sample_input = safe_input(
            color_text(f"Percentage of symbols to sample [{default or 'disabled'}]: ", Fore.YELLOW),
            default=str(default) if default else "",
            allow_back=True,
        ).strip()

        if not sample_input:
            # Use default or disable
            return default

        try:
            # Handle possible % sign
            if sample_input.endswith("%"):
                sample_input = sample_input[:-1]

            percentage = float(sample_input)
            if percentage <= 0 or percentage > 100:
                print(color_text("  Invalid: Percentage must be between 1 and 100", Fore.RED))
                continue

            return percentage
        except ValueError:
            print(color_text("  Invalid: Please enter a number between 1 and 100", Fore.RED))
            continue


def prompt_pre_filter_percentage(
    default: Optional[float] = None, loaded_config: Optional[Dict] = None
) -> Optional[float]:
    """
    Prompt user to enter pre-filter percentage.

    Args:
        default: Default percentage value (None = use 10.0)
        loaded_config: Optional loaded configuration to use for defaults

    Returns:
        Percentage value (0-100) or None to use default 10.0
    """
    if loaded_config:
        default = loaded_config.get("pre_filter_percentage", None)

    if default is None:
        default = 10.0

    print("\nPre-filter percentage:")
    print("  Percentage of top-scoring symbols to select for Gemini analysis")
    print("  Lower = fewer symbols but higher quality (default: 10%)")
    print("  Higher = more symbols but may include lower quality signals")
    print("  Range: 1-100 (press Enter to use default 10%)")

    while True:
        percentage_input = safe_input(
            color_text("Pre-filter percentage (1-100) [10.0]: ", Fore.YELLOW),
            default=str(default) if default else "10.0",
            allow_back=True,
        ).strip()

        if not percentage_input:
            # Use default
            return None if default == 10.0 else default

        try:
            percentage = float(percentage_input)
            if 1.0 <= percentage <= 100.0:
                return percentage
            else:
                print(color_text("  Invalid: Percentage must be between 1.0 and 100.0", Fore.RED))
                continue
        except ValueError:
            print(color_text("  Invalid: Please enter a number between 1.0 and 100.0", Fore.RED))
            continue
