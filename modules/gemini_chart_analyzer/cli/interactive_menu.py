
from typing import Optional, Tuple
import argparse
import copy
import sys

from colorama import Fore, Style

from modules.common.ui.formatting import prompt_user_input_with_backspace
from modules.common.utils import (
from modules.common.ui.formatting import prompt_user_input_with_backspace
from modules.common.utils import (

"""
Interactive menu for Gemini Chart Analyzer CLI.

This module provides interactive configuration menu for chart analysis
with Google Gemini AI, following the pattern from other modules.
"""



    color_text,
    normalize_timeframe,
    prompt_user_input,
)
from modules.gemini_chart_analyzer.cli.argument_parser import _format_current_value

# Indicator default values - centralized to avoid duplication
INDICATOR_DEFAULTS = {
    "MA": {"periods": [20, 50, 200]},
    "RSI": {"period": 14},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "BB": {"period": 20, "std": 2},
}

# Default string values for prompts
INDICATOR_DEFAULT_STRINGS = {
    "MA": "20,50,200",
    "RSI": "14",
    "BB": "20",
}


def _display_main_menu(config):
    """Display main menu with current configuration values."""
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("GEMINI CHART ANALYZER - Configuration Menu", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print("\n" + color_text("MAIN MENU", Fore.YELLOW, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))

    # Format current values
    symbol_val = _format_current_value(getattr(config, "symbol", None))
    timeframe_val = _format_current_value(getattr(config, "timeframe", "1h"))
    timeframes_val = _format_current_value(getattr(config, "timeframes_list", None))

    # Indicators
    ma_val = "disabled"
    if hasattr(config, "indicators") and "MA" in config.indicators:
        ma_val = f"periods={config.indicators['MA']['periods']}"
    elif not getattr(config, "no_ma", False):
        ma_val = f"periods={INDICATOR_DEFAULTS['MA']['periods']}"

    rsi_val = "disabled"
    if hasattr(config, "indicators") and "RSI" in config.indicators:
        rsi_val = f"period={config.indicators['RSI']['period']}"
    elif not getattr(config, "no_rsi", False):
        rsi_val = f"period={INDICATOR_DEFAULTS['RSI']['period']}"

    macd_val = "disabled" if getattr(config, "no_macd", False) else "enabled"
    bb_val = "disabled"
    if hasattr(config, "indicators") and "BB" in config.indicators:
        bb_val = f"period={config.indicators['BB']['period']}"
    elif getattr(config, "enable_bb", False):
        bb_val = f"period={INDICATOR_DEFAULTS['BB']['period']}"

    prompt_type_val = _format_current_value(getattr(config, "prompt_type", "detailed"))
    custom_prompt_val = _format_current_value(getattr(config, "custom_prompt", None))

    if timeframes_val and timeframes_val != "not set":
        tf_display = f"{symbol_val} / Multi-TF: {timeframes_val}"
    else:
        tf_display = f"{symbol_val} / {timeframe_val}"
    print(f"  1. Symbol & Timeframe [{color_text(tf_display, Fore.GREEN)}]")
    print(
        f"  2. Indicators Configuration [{color_text(f'MA={ma_val}, RSI={rsi_val}, MACD={macd_val}, BB={bb_val}', Fore.GREEN)}]"
    )
    print(f"  3. Gemini Prompt Configuration [{color_text(f'{prompt_type_val}', Fore.GREEN)}]")
    print("  4. Review and Confirm")
    print("  5. Exit")
    print(color_text("-" * 80, Fore.CYAN))


def _prompt_with_back(prompt: str, default: str = None, allow_back: bool = True) -> Tuple[Optional[str], str]:
    """
    Prompt user with backspace key for back navigation.

    Returns:
        (value, action) where action is 'main' or 'continue'
    """
    if allow_back:
        back_prompt = f"{prompt} (press Backspace to go back): "
    else:
        back_prompt = prompt

    if allow_back:
        user_input, is_back = prompt_user_input_with_backspace(back_prompt, default=default)

        if is_back:
            return (None, "main")

        user_input = user_input.strip() if user_input is not None else None

        return (user_input, "continue")
    else:
        user_input = prompt_user_input(back_prompt, default=default).strip()
        return (user_input, "continue")


def _configure_symbol_timeframe(config):
    """Configure symbol and timeframe with back option."""
    print("\n" + color_text("1. SYMBOL & TIMEFRAME", Fore.YELLOW, Style.BRIGHT))
    print(color_text("   b) Back to main menu", Fore.CYAN))
    print()

    # Symbol
    current_symbol = getattr(config, "symbol", None)
    symbol_input, action = _prompt_with_back(
        f"Enter symbol (e.g., BTC/USDT) [{current_symbol or 'BTC/USDT'}]: ", default=current_symbol or "BTC/USDT"
    )
    if action == "main":
        return ("main", False)

    if not symbol_input:
        symbol_input = "BTC/USDT"

    # Choose single or multi-timeframe mode
    current_timeframes_list = getattr(config, "timeframes_list", None)
    use_multi_tf = current_timeframes_list is not None and len(current_timeframes_list) > 0

    print("\nAnalysis mode:")
    print("  1. Single timeframe")
    print("  2. Multi-timeframe (recommended)")
    mode_input, action = _prompt_with_back(
        f"Select mode (1/2) [{('2' if use_multi_tf else '1')}]: ", default="2" if use_multi_tf else "1"
    )
    if action == "main":
        return ("main", False)

    if not mode_input:
        mode_input = "2" if use_multi_tf else "1"

    changed = symbol_input != current_symbol

    if mode_input == "2":
        # Multi-timeframe mode
        from modules.gemini_chart_analyzer.core.utils import DEFAULT_TIMEFRAMES, normalize_timeframes

        current_tfs_str = (
            ", ".join(current_timeframes_list) if current_timeframes_list else ", ".join(DEFAULT_TIMEFRAMES)
        )
        print(f"\nDefault timeframes: {', '.join(DEFAULT_TIMEFRAMES)}")
        print("Timeframes: 15m, 30m, 1h, 4h, 1d, 1w (comma-separated)")
        timeframes_input, action = _prompt_with_back(
            f"Enter timeframes (comma-separated) [{current_tfs_str}]: ", default=current_tfs_str
        )
        if action == "main":
            return ("main", False)

        if not timeframes_input:
            timeframes_input = ", ".join(DEFAULT_TIMEFRAMES)

        try:
            timeframes_list = [tf.strip() for tf in timeframes_input.split(",") if tf.strip()]
            normalized_tfs = normalize_timeframes(timeframes_list)
            if normalized_tfs:
                config.timeframes_list = normalized_tfs
                config.timeframe = None  # Clear single timeframe
                if normalized_tfs != (current_timeframes_list or []):
                    changed = True
            else:
                print(color_text("Warning: No valid timeframes. Using single timeframe mode.", Fore.YELLOW))
                config.timeframes_list = None
                config.timeframe = "1h"
        except Exception as e:
            print(color_text(f"Warning: Error parsing timeframes: {e}. Using single timeframe mode.", Fore.YELLOW))
            config.timeframes_list = None
            config.timeframe = "1h"
    else:
        # Single timeframe mode
        current_timeframe = getattr(config, "timeframe", "1h")
        print("\nTimeframes: 15m (or m15), 30m (or m30), 1h (or h1), 4h (or h4), 1d (or d1), 1w (or w1)")
        timeframe_input, action = _prompt_with_back(
            f"Enter timeframe [{current_timeframe}]: ", default=current_timeframe
        )
        if action == "main":
            return ("main", False)

        if not timeframe_input:
            timeframe_input = "1h"

        # Normalize timeframe (accept both '15m' and 'm15', '1h' and 'h1', etc.)
        timeframe_input = normalize_timeframe(timeframe_input)

        config.timeframe = timeframe_input
        config.timeframes_list = None  # Clear multi-timeframe
        if timeframe_input != current_timeframe:
            changed = True

    config.symbol = symbol_input

    return ("main", changed)


def _configure_indicator(
    config,
    name,
    prompt_text,
    default_str,
    parse_fn,
    disable_flag_name=None,
    enable_flag_name=None,
    value_prompt_text=None,
    current_value_getter=None,
):
    """
    Helper function to configure an indicator with prompt-validate-update pattern.

    Args:
        config: Configuration object
        name: Indicator name (e.g., 'MA', 'RSI')
        prompt_text: Text for enable/disable prompt (e.g., "Use Moving Averages? (y/n)")
        default_str: Default string value for display
        parse_fn: Function to parse user input, returns parsed value or raises/returns fallback
        disable_flag_name: Name of disable flag (e.g., 'no_ma'), uses 'no_<lowername>' if None
        enable_flag_name: Name of enable flag (e.g., 'enable_bb'), overrides disable_flag_name if set
        value_prompt_text: Text for value prompt (e.g., "MA periods (comma-separated, e.g., 20,50,200)")
                          If None, indicator doesn't require additional value input
        current_value_getter: Function to get current value dict from config, returns dict or None

    Returns:
        (action, changed, parsed_value) where:
        - action: 'main' or 'continue'
        - changed: bool indicating if value changed
        - parsed_value: parsed indicator config dict or None if disabled
    """
    # Determine flag names
    lower_name = name.lower()
    if enable_flag_name:
        disable_flag = None
        enable_flag = enable_flag_name
        is_enabled = getattr(config, enable_flag, False)
    else:
        if disable_flag_name is None:
            disable_flag_name = f"no_{lower_name}"
        disable_flag = disable_flag_name
        enable_flag = None
        is_enabled = not getattr(config, disable_flag, False)

    # Get current value
    if current_value_getter:
        current_value = current_value_getter()
    else:
        if enable_flag:
            current_value = config.indicators.get(name) if is_enabled else None
        else:
            if is_enabled:
                current_value = config.indicators.get(name)
                if current_value is None:
                    # Use default based on indicator type (copy to avoid mutation)
                    if name in INDICATOR_DEFAULTS:
                        default = INDICATOR_DEFAULTS[name]
                        if "periods" in default:
                            # Copy list to avoid shared state mutation
                            current_value = {"periods": list(default["periods"])}
                        else:
                            # Copy dict to avoid shared state mutation
                            current_value = default.copy()
                    else:
                        current_value = {}
            else:
                current_value = None

    # Build enable/disable prompt
    enable_default = "y" if current_value else "n"
    use_input, action = _prompt_with_back(f"{prompt_text} [{enable_default}]: ", default=enable_default)
    if action == "main":
        return ("main", False, None)

    use_indicator = (use_input or "").lower() in ["y", "yes", ""]

    # Handle disabled case
    if not use_indicator:
        if enable_flag:
            setattr(config, enable_flag, False)
        else:
            setattr(config, disable_flag, True)
        config.indicators.pop(name, None)
        changed = current_value is not None
        return ("continue", changed, None)

    # Handle enabled case
    if enable_flag:
        setattr(config, enable_flag, True)
    else:
        setattr(config, disable_flag, False)

    # If no value prompt needed (e.g., MACD), use fixed default
    if value_prompt_text is None:
        if name == "MACD":
            # Copy to avoid shared state mutation
            parsed_value = INDICATOR_DEFAULTS["MACD"].copy()
        else:
            parsed_value = current_value or {}
    else:
        # Build value prompt with current/default value
        if current_value:
            if "periods" in current_value:
                current_str = ",".join(map(str, current_value["periods"]))
            elif "period" in current_value:
                current_str = str(current_value["period"])
            else:
                current_str = default_str
        else:
            current_str = default_str

        value_input, action = _prompt_with_back(f"{value_prompt_text} [{current_str}]: ", default=current_str)
        if action == "main":
            return ("main", False, None)

        # Parse value (empty input uses default_str)
        if not value_input:
            value_input = default_str

        try:
            parsed_value = parse_fn(value_input)
        except (KeyboardInterrupt, SystemExit):
            # Re-raise intentional interrupts to preserve normal interrupt handling
            raise
        except Exception as e:
            # Parse function raised error, use fallback with warning
            print(
                color_text(f"Error parsing input ({type(e).__name__}: {e}), using default: {default_str}", Fore.YELLOW)
            )
            parsed_value = parse_fn(default_str)

    # Update config and check if changed
    config.indicators[name] = parsed_value
    if current_value is None:
        changed = True
    elif name == "MA":
        changed = list(parsed_value.get("periods", [])) != list(current_value.get("periods", []))
    elif name in ["RSI", "BB"]:
        changed = parsed_value.get("period") != current_value.get("period")
    elif name == "MACD":
        # MACD has fixed values, so if already enabled, configuration hasn't changed
        changed = False
    else:
        changed = parsed_value != current_value

    return ("continue", changed, parsed_value)


def _configure_indicators(config):
    """Configure indicators with back option."""
    print("\n" + color_text("2. INDICATORS CONFIGURATION", Fore.YELLOW, Style.BRIGHT))
    print(color_text("   b) Back to main menu", Fore.CYAN))
    print()

    if not hasattr(config, "indicators"):
        config.indicators = {}

    changed = False

    # Moving Averages
    def get_current_ma():
        if not getattr(config, "no_ma", False):
            default_ma = INDICATOR_DEFAULTS["MA"]
            # Copy to avoid shared state mutation
            return config.indicators.get("MA", {"periods": list(default_ma["periods"])})
        return None

    def parse_ma_periods(input_str):
        periods = [int(p.strip()) for p in input_str.split(",") if p.strip()]
        if not periods:
            raise ValueError("At least one period required")
        return {"periods": periods}

    action, ma_changed, _ = _configure_indicator(
        config,
        "MA",
        "Use Moving Averages? (y/n)",
        INDICATOR_DEFAULT_STRINGS["MA"],
        parse_ma_periods,
        value_prompt_text=f"MA periods (comma-separated, e.g., {INDICATOR_DEFAULT_STRINGS['MA']})",
        current_value_getter=get_current_ma,
    )
    if action == "main":
        return ("main", False)
    changed = changed or ma_changed

    # RSI
    def get_current_rsi():
        if not getattr(config, "no_rsi", False):
            default_rsi = INDICATOR_DEFAULTS["RSI"]
            # Copy to avoid shared state mutation
            return config.indicators.get("RSI", default_rsi.copy())
        return None

    def parse_rsi_period(input_str):
        period = int(input_str.strip())
        if period <= 0:
            raise ValueError("Period must be positive")
        return {"period": period}

    action, rsi_changed, _ = _configure_indicator(
        config,
        "RSI",
        "Use RSI? (y/n)",
        INDICATOR_DEFAULT_STRINGS["RSI"],
        parse_rsi_period,
        value_prompt_text="RSI period",
        current_value_getter=get_current_rsi,
    )
    if action == "main":
        return ("main", False)
    changed = changed or rsi_changed

    # MACD
    def get_current_macd():
        if not getattr(config, "no_macd", False):
            # Copy to avoid shared state mutation
            return INDICATOR_DEFAULTS["MACD"].copy()
        return None

    def parse_macd(_input_str):
        # Copy to avoid shared state mutation
        return INDICATOR_DEFAULTS["MACD"].copy()

    action, macd_changed, _ = _configure_indicator(
        config, "MACD", "Use MACD? (y/n)", "", parse_macd, current_value_getter=get_current_macd
    )
    if action == "main":
        return ("main", False)
    changed = changed or macd_changed

    # Bollinger Bands
    def get_current_bb():
        if getattr(config, "enable_bb", False):
            bb_dict = config.indicators.get("BB")
            # Return defensive copy to avoid shared state mutation
            return copy.deepcopy(bb_dict) if bb_dict is not None else None
        return None

    def parse_bb_period(input_str):
        period = int(input_str.strip())
        if period <= 0:
            raise ValueError("Period must be positive")
        # Use std from defaults
        return {"period": period, "std": INDICATOR_DEFAULTS["BB"]["std"]}

    action, bb_changed, _ = _configure_indicator(
        config,
        "BB",
        "Use Bollinger Bands? (y/n)",
        INDICATOR_DEFAULT_STRINGS["BB"],
        parse_bb_period,
        enable_flag_name="enable_bb",
        value_prompt_text="BB period",
        current_value_getter=get_current_bb,
    )
    if action == "main":
        return ("main", False)
    changed = changed or bb_changed

    return ("main", changed)


def _configure_prompt(config):
    """Configure Gemini prompt type with back option."""
    print("\n" + color_text("3. GEMINI PROMPT CONFIGURATION", Fore.YELLOW, Style.BRIGHT))
    print("  1. Detailed - Phân tích chi tiết với cấu trúc đầy đủ (mặc định)")
    print("  2. Simple - Phân tích đơn giản")
    print("  3. Custom - Nhập prompt tùy chỉnh")
    print(color_text("  b) Back to main menu", Fore.CYAN))
    print()

    current_prompt_type = getattr(config, "prompt_type", "detailed")
    current_custom = getattr(config, "custom_prompt", None)

    choice_input, action = _prompt_with_back(
        f"Select prompt type (1/2/3) [{('1' if current_prompt_type == 'detailed' else '2' if current_prompt_type == 'simple' else '3')}]: ",
        default="1" if current_prompt_type == "detailed" else "2" if current_prompt_type == "simple" else "3",
    )
    if action == "main":
        return ("main", False)

    if not choice_input:
        choice_input = "1"

    changed = False
    if choice_input == "2":
        config.prompt_type = "simple"
        config.custom_prompt = None
        if current_prompt_type != "simple":
            changed = True
    elif choice_input == "3":
        config.prompt_type = "custom"
        custom_prompt_input, action = _prompt_with_back(
            f"Enter custom prompt [{current_custom or ''}]: ", default=current_custom or ""
        )
        if action == "main":
            return ("main", False)

        if not custom_prompt_input:
            print(color_text("Warning: Custom prompt cannot be empty. Using 'detailed' instead.", Fore.YELLOW))
            config.prompt_type = "detailed"
            config.custom_prompt = None
        else:
            config.custom_prompt = custom_prompt_input
            if custom_prompt_input != current_custom:
                changed = True
    else:
        config.prompt_type = "detailed"
        config.custom_prompt = None
        if current_prompt_type != "detailed":
            changed = True

    return ("main", changed)


def _review_and_confirm(config):
    """Review configuration and confirm."""
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("CONFIGURATION REVIEW", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

    print(f"\nSymbol: {getattr(config, 'symbol', 'not set')}")
    timeframes_list = getattr(config, "timeframes_list", None)
    if timeframes_list:
        print(f"Timeframes: {', '.join(timeframes_list)} (Multi-timeframe mode)")
    else:
        print(f"Timeframe: {getattr(config, 'timeframe', '1h')} (Single timeframe mode)")

    print("\nIndicators:")
    indicators = getattr(config, "indicators", {})
    if "MA" in indicators:
        print(f"  - Moving Averages: periods={indicators['MA']['periods']}")
    else:
        print("  - Moving Averages: disabled")

    if "RSI" in indicators:
        print(f"  - RSI: period={indicators['RSI']['period']}")
    else:
        print("  - RSI: disabled")

    if "MACD" in indicators:
        print("  - MACD: enabled")
    else:
        print("  - MACD: disabled")

    if "BB" in indicators:
        print(f"  - Bollinger Bands: period={indicators['BB']['period']}")
    else:
        print("  - Bollinger Bands: disabled")

    print(f"\nGemini Prompt Type: {getattr(config, 'prompt_type', 'detailed')}")
    if getattr(config, "prompt_type", "detailed") == "custom":
        custom_prompt = getattr(config, "custom_prompt", "")
        print(f"Custom Prompt: {custom_prompt[:50]}{'...' if len(custom_prompt) > 50 else ''}")

    print("\n" + color_text("-" * 80, Fore.CYAN))
    confirm = prompt_user_input("Confirm this configuration? (y/n) [y]: ", default="y").strip().lower()

    if confirm in ["y", "yes", ""]:
        return "done"
    else:
        return "main"


def interactive_config_menu():
    """
    Interactive menu for configuring Gemini Chart Analyzer.

    Returns:
        argparse.Namespace object with all configuration values
    """
    # Create namespace object with defaults
    config = argparse.Namespace()

    # Initialize default values
    config.symbol = None
    config.timeframe = "1h"
    config.indicators = {}
    config.no_ma = False
    config.no_rsi = False
    config.no_macd = False
    config.enable_bb = False
    config.prompt_type = "detailed"
    config.custom_prompt = None
    config.limit = 500
    config.chart_figsize_tuple = (16, 10)
    config.chart_dpi = 150
    config.no_cleanup = False

    # Track unsaved changes
    has_unsaved_changes = False

    # Main menu loop
    while True:
        _display_main_menu(config)

        choice = prompt_user_input("\nSelect option [1-5]: ").strip()

        if choice == "1":
            _, changed = _configure_symbol_timeframe(config)
            if changed:
                has_unsaved_changes = True
        elif choice == "2":
            _, changed = _configure_indicators(config)
            if changed:
                has_unsaved_changes = True
        elif choice == "3":
            _, changed = _configure_prompt(config)
            if changed:
                has_unsaved_changes = True
        elif choice == "4":
            result = _review_and_confirm(config)
            if result == "done":
                has_unsaved_changes = False  # Changes are confirmed/saved
                break
        elif choice == "5":
            # Prompt for confirmation before exiting
            if has_unsaved_changes:
                confirm_msg = color_text(
                    "\n⚠️  Are you sure you want to exit? Unsaved changes will be lost. (y/N): ", Fore.YELLOW
                )
            else:
                confirm_msg = color_text("\nAre you sure you want to exit? (y/N): ", Fore.YELLOW)

            confirm = prompt_user_input(confirm_msg, default="n").strip().lower()

            if confirm in ["y", "yes"]:
                print(color_text("\nExiting configuration menu.", Fore.YELLOW))
                sys.exit(0)
            else:
                # User chose not to exit, return to menu
                continue
        else:
            print(color_text("Invalid choice. Please select 1-5.", Fore.RED))

    return config
