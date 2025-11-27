"""
Command-line interface for pairs trading analysis.
"""

import sys
import argparse
from typing import Dict, Optional, Tuple

from colorama import Fore, Style

try:
    from modules.common.utils import color_text
except ImportError:
    def color_text(text, color=None, style=None):
        return text

try:
    from modules.config import (
        PAIRS_TRADING_WEIGHTS,
        PAIRS_TRADING_TOP_N,
        PAIRS_TRADING_MIN_SPREAD,
        PAIRS_TRADING_MAX_SPREAD,
        PAIRS_TRADING_MIN_CORRELATION,
        PAIRS_TRADING_MAX_CORRELATION,
        PAIRS_TRADING_MAX_HALF_LIFE,
        PAIRS_TRADING_WEIGHT_PRESETS,
        PAIRS_TRADING_OLS_FIT_INTERCEPT,
        PAIRS_TRADING_KALMAN_DELTA,
        PAIRS_TRADING_KALMAN_OBS_COV,
        PAIRS_TRADING_KALMAN_PRESETS,
        PAIRS_TRADING_OPPORTUNITY_PRESETS,
    )
except ImportError:
    PAIRS_TRADING_WEIGHTS = {"1d": 0.5, "3d": 0.3, "1w": 0.2}
    PAIRS_TRADING_TOP_N = 5
    PAIRS_TRADING_MIN_SPREAD = 0.01
    PAIRS_TRADING_MAX_SPREAD = 0.50
    PAIRS_TRADING_MIN_CORRELATION = 0.3
    PAIRS_TRADING_MAX_CORRELATION = 0.9
    PAIRS_TRADING_MAX_HALF_LIFE = 50
    PAIRS_TRADING_OLS_FIT_INTERCEPT = True
    PAIRS_TRADING_KALMAN_DELTA = 1e-5
    PAIRS_TRADING_KALMAN_OBS_COV = 1.0
    PAIRS_TRADING_WEIGHT_PRESETS = {
        "momentum": {"1d": 0.5, "3d": 0.3, "1w": 0.2},
        "balanced": {"1d": 0.3, "3d": 0.4, "1w": 0.3},
    }
    PAIRS_TRADING_KALMAN_PRESETS = {
        "balanced": {"delta": 1e-5, "obs_cov": 1.0, "description": "Default balanced profile"},
    }
    PAIRS_TRADING_OPPORTUNITY_PRESETS = {
        "balanced": {
            "description": "Default balanced scoring",
        }
    }


def standardize_symbol_input(symbol: str) -> str:
    """Convert raw user input into f'{base}/USDT' style if needed."""
    if not symbol:
        return ""
    cleaned = symbol.strip().upper()
    if "/" in cleaned:
        base, quote = cleaned.split("/", 1)
        base = base.strip()
        quote = quote.strip() or "USDT"
        return f"{base}/{quote}"
    if cleaned.endswith("USDT"):
        base = cleaned[:-4]
        base = base.strip()
        return f"{base}/USDT"
    return f"{cleaned}/USDT"


def prompt_interactive_mode() -> Dict[str, Optional[str]]:
    """Interactive launcher for selecting analysis mode and symbol source."""
    print(color_text("\n" + "=" * 60, Fore.CYAN, Style.BRIGHT))
    print(color_text("Pairs Trading Analysis - Interactive Launcher", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 60, Fore.CYAN, Style.BRIGHT))
    print("1) Auto mode  - analyze entire market to surface opportunities")
    print("2) Manual mode - focus on specific symbols you provide")
    print("3) Exit")

    while True:
        choice = input(color_text("\nSelect option [1-3]: ", Fore.YELLOW)).strip() or "1"
        if choice in {"1", "2", "3"}:
            break
        print(color_text("Invalid selection. Please enter 1, 2, or 3.", Fore.RED))

    if choice == "3":
        print(color_text("\nExiting...", Fore.YELLOW))
        sys.exit(0)

    manual_symbols = None
    if choice == "2":
        manual_symbols = input(
            color_text(
                "Enter symbols separated by comma/space (e.g., BTC/USDT, ETH/USDT): ",
                Fore.YELLOW,
            )
        ).strip()

    return {
        "mode": "manual" if choice == "2" else "auto",
        "symbols_raw": manual_symbols or None,
    }


def prompt_weight_preset_selection(current_preset: Optional[str]) -> str:
    """Interactive selection menu for weight presets."""
    presets = list(PAIRS_TRADING_WEIGHT_PRESETS.items())
    if not presets:
        return current_preset or "momentum"

    default_choice = None
    for idx, (key, _) in enumerate(presets, start=1):
        if key == current_preset:
            default_choice = str(idx)
            break
    if default_choice is None:
        default_choice = "1"

    print(color_text("\nSelect weight preset for calculating performance score for pairs trading:", Fore.CYAN, Style.BRIGHT))
    for idx, (key, weights) in enumerate(presets, start=1):
        weights_desc = f"1d={weights['1d']:.2f}, 3d={weights['3d']:.2f}, 1w={weights['1w']:.2f}"
        highlight = Style.BRIGHT if key == current_preset else Style.NORMAL
        print(
            color_text(
                f"{idx}) {key.capitalize()} ({weights_desc})",
                Fore.MAGENTA if key == current_preset else Fore.WHITE,
                highlight,
            )
        )

    choice_map = {str(idx + 1): key for idx, (key, _) in enumerate(presets)}

    while True:
        user_choice = input(
            color_text(
                f"\nEnter preset [1-{len(presets)}] (default {default_choice}): ",
                Fore.YELLOW,
            )
        ).strip() or default_choice
        if user_choice in choice_map:
            selected = choice_map[user_choice]
            print(
                color_text(
                    f"Using {selected.capitalize()} preset",
                    Fore.GREEN,
                    Style.BRIGHT,
                )
            )
            return selected
        print(color_text("Invalid selection. Please try again.", Fore.RED))


def prompt_kalman_preset_selection(
    current_delta: float,
    current_obs_cov: float,
) -> Tuple[float, float, Optional[str]]:
    """Interactive selection menu for Kalman parameter presets."""
    presets = list(PAIRS_TRADING_KALMAN_PRESETS.items())
    if not presets:
        return current_delta, current_obs_cov, None

    default_choice = "1"
    print(
        color_text(
            "\nSelect Kalman filter profile for hedge ratio:",
            Fore.CYAN,
            Style.BRIGHT,
        )
    )
    for idx, (key, data) in enumerate(presets, start=1):
        desc = data.get("description", "")
        delta = data.get("delta")
        obs_cov = data.get("obs_cov")
        print(
            color_text(
                f"{idx}) {key} (delta={delta:.2e}, obs_cov={obs_cov:.2f}) - {desc}",
                Fore.WHITE,
                Style.NORMAL,
            )
        )
    choice_map = {str(idx): (key, data) for idx, (key, data) in enumerate(presets, start=1)}
    while True:
        user_choice = input(
            color_text(
                f"\nEnter preset [1-{len(presets)}] (default {default_choice}): ",
                Fore.YELLOW,
            )
        ).strip() or default_choice
        if user_choice in choice_map:
            key, data = choice_map[user_choice]
            delta = float(data.get("delta", current_delta))
            obs_cov = float(data.get("obs_cov", current_obs_cov))
            print(
                color_text(
                    f"Using {key} profile (delta={delta:.2e}, obs_cov={obs_cov:.2f})",
                    Fore.GREEN,
                    Style.BRIGHT,
                )
            )
            return delta, obs_cov, key
        print(color_text("Invalid selection. Please try again.", Fore.RED))


def prompt_opportunity_preset_selection(
    current_key: Optional[str],
) -> str:
    """Interactive selection for opportunity scoring profiles."""
    presets = list(PAIRS_TRADING_OPPORTUNITY_PRESETS.items())
    if not presets:
        return current_key or "balanced"

    default_choice = None
    for idx, (key, _) in enumerate(presets, start=1):
        if key == current_key:
            default_choice = str(idx)
            break
    if default_choice is None:
        default_choice = "1"

    print(color_text("\nSelect opportunity scoring profile:", Fore.CYAN, Style.BRIGHT))
    for idx, (key, data) in enumerate(presets, start=1):
        desc = data.get("description", "")
        print(
            color_text(
                f"{idx}) {key} - {desc}",
                Fore.WHITE if key != current_key else Fore.MAGENTA,
                Style.BRIGHT if key == current_key else Style.NORMAL,
            )
        )

    choice_map = {str(idx): key for idx, (key, _) in enumerate(presets, start=1)}
    while True:
        selection = input(
            color_text(
                f"\nEnter preset [1-{len(presets)}] (default {default_choice}): ",
                Fore.YELLOW,
            )
        ).strip() or default_choice
        if selection in choice_map:
            chosen = choice_map[selection]
            print(
                color_text(
                    f"Using {chosen} scoring profile",
                    Fore.GREEN,
                    Style.BRIGHT,
                )
            )
            return chosen
        print(color_text("Invalid selection. Please try again.", Fore.RED))


def parse_weights(weights_str: Optional[str], preset_key: Optional[str] = None) -> Dict[str, float]:
    """Parse weights string into dictionary.
    
    Args:
        weights_str: Weights in format '1d:0.5,3d:0.3,1w:0.2'
        preset_key: Named preset (momentum/balanced)
        
    Returns:
        Dictionary with weights, normalized to sum to 1.0
    """
    # Highest precedence: manual weights string
    if not weights_str and preset_key:
        preset = PAIRS_TRADING_WEIGHT_PRESETS.get(preset_key)
        if preset:
            return preset.copy()

    weights = PAIRS_TRADING_WEIGHTS.copy()
    if not weights_str:
        return weights
    
    try:
        weight_parts = weights_str.split(",")
        weights = {}
        for part in weight_parts:
            key, value = part.split(":")
            weights[key.strip()] = float(value.strip())
        # Validate weights sum to 1.0
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            print(
                color_text(
                    f"Warning: Weights sum to {total:.3f}, not 1.0. Normalizing...",
                    Fore.YELLOW,
                )
            )
            weights = {k: v / total for k, v in weights.items()}
    except Exception as e:
        print(
            color_text(
                f"Error parsing weights: {e}. Using default weights.",
                Fore.RED,
            )
        )
        weights = PAIRS_TRADING_WEIGHTS.copy()
    
    return weights


def parse_symbols(symbols_str: Optional[str]):
    """Parse symbols string into display and parsed lists.
    
    Args:
        symbols_str: Comma/space separated symbols
        
    Returns:
        Tuple of (target_symbol_inputs, parsed_target_symbols)
    """
    target_symbol_inputs = []
    parsed_target_symbols = []
    
    if not symbols_str:
        return target_symbol_inputs, parsed_target_symbols
    
    raw_parts = (
        symbols_str.replace(",", " ")
        .replace(";", " ")
        .replace("|", " ")
        .split()
    )
    seen_display = set()
    seen_parsed = set()
    for part in raw_parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        display_value = cleaned.upper()
        parsed_value = standardize_symbol_input(cleaned)
        if display_value not in seen_display:
            seen_display.add(display_value)
            target_symbol_inputs.append(display_value)
        parsed_key = parsed_value.upper()
        if parsed_key not in seen_parsed:
            seen_parsed.add(parsed_key)
            parsed_target_symbols.append(parsed_value)
    
    return target_symbol_inputs, parsed_target_symbols


def parse_args():
    """Parse command-line arguments for pairs trading analysis."""
    parser = argparse.ArgumentParser(
        description="Pairs Trading Analysis - Identify trading opportunities from best/worst performers"
    )
    parser.add_argument(
        "--pairs-count",
        type=int,
        default=PAIRS_TRADING_TOP_N,
        help=f"Number of tradeable pairs to return (default: {PAIRS_TRADING_TOP_N})",
    )
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=50,
        help="Number of top/bottom symbols to consider per side when forming pairs",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Maximum number of symbols to analyze (default: all available)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Weights for timeframes in format '1d:0.5,3d:0.3,1w:0.2' (default: from config)",
    )
    parser.add_argument(
        "--weight-preset",
        type=str,
        choices=list(PAIRS_TRADING_WEIGHT_PRESETS.keys()),
        default="balanced",
        help="Choose predefined weight preset (momentum or balanced). Ignored if --weights is provided.",
    )
    parser.add_argument(
        "--ols-fit-intercept",
        dest="ols_fit_intercept",
        action="store_true",
        default=PAIRS_TRADING_OLS_FIT_INTERCEPT,
        help="Use intercept term when fitting OLS hedge ratio (default: enabled).",
    )
    parser.add_argument(
        "--no-ols-fit-intercept",
        dest="ols_fit_intercept",
        action="store_false",
        help="Disable intercept (beta forced through origin) when fitting OLS hedge ratio.",
    )
    parser.add_argument(
        "--kalman-delta",
        type=float,
        default=PAIRS_TRADING_KALMAN_DELTA,
        help=f"State noise factor delta for Kalman hedge ratio (default: {PAIRS_TRADING_KALMAN_DELTA}).",
    )
    parser.add_argument(
        "--kalman-obs-cov",
        type=float,
        default=PAIRS_TRADING_KALMAN_OBS_COV,
        help=f"Observation covariance for Kalman hedge ratio (default: {PAIRS_TRADING_KALMAN_OBS_COV}).",
    )
    parser.add_argument(
        "--kalman-preset",
        type=str,
        choices=list(PAIRS_TRADING_KALMAN_PRESETS.keys()),
        default="balanced",
        help="Choose predefined Kalman parameter preset (fast_react / balanced / stable).",
    )
    parser.add_argument(
        "--opportunity-preset",
        type=str,
        choices=list(PAIRS_TRADING_OPPORTUNITY_PRESETS.keys()),
        default="balanced",
        help="Choose opportunity scoring profile (e.g. balanced/aggressive/conservative).",
    )
    parser.add_argument(
        "--min-spread",
        type=float,
        default=PAIRS_TRADING_MIN_SPREAD,
        help=f"Minimum spread percentage (default: {PAIRS_TRADING_MIN_SPREAD*100:.2f}%)",
    )
    parser.add_argument(
        "--max-spread",
        type=float,
        default=PAIRS_TRADING_MAX_SPREAD,
        help=f"Maximum spread percentage (default: {PAIRS_TRADING_MAX_SPREAD*100:.2f}%)",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=PAIRS_TRADING_MIN_CORRELATION,
        help=f"Minimum correlation (default: {PAIRS_TRADING_MIN_CORRELATION:.2f})",
    )
    parser.add_argument(
        "--max-correlation",
        type=float,
        default=PAIRS_TRADING_MAX_CORRELATION,
        help=f"Maximum correlation (default: {PAIRS_TRADING_MAX_CORRELATION:.2f})",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=10,
        help="Maximum number of pairs to display (default: 10)",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip pairs validation (show all opportunities)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Manual mode: comma/space separated symbols to focus on (e.g., 'BTC/USDT,ETH/USDT')",
    )
    parser.add_argument(
        "--no-menu",
        action="store_true",
        help="Skip interactive launcher (retain legacy CLI flag workflow)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["opportunity_score", "quantitative_score"],
        default="opportunity_score",
        help="Sort pairs by opportunity_score or quantitative_score (default: opportunity_score)",
    )
    parser.add_argument(
        "--require-cointegration",
        action="store_true",
        help="Only accept cointegrated pairs (filter out non-cointegrated pairs)",
    )
    parser.add_argument(
        "--max-half-life",
        type=float,
        default=PAIRS_TRADING_MAX_HALF_LIFE,
        help=f"Maximum acceptable half-life for mean reversion (default: {PAIRS_TRADING_MAX_HALF_LIFE})",
    )
    parser.add_argument(
        "--min-quantitative-score",
        type=float,
        default=None,
        help="Minimum quantitative score (0-100) to accept a pair (default: no threshold)",
    )
    
    return parser.parse_args()

