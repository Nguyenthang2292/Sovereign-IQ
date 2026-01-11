"""
Argument parser for ATC + Range Oscillator + SPC Hybrid and Pure Voting.

This module contains functions for parsing command-line arguments
and interactive configuration menu for both Hybrid and Pure Voting approaches.
"""

import sys
import argparse
from colorama import Fore, Style

from config import (
    DEFAULT_TIMEFRAME,
    DECISION_MATRIX_VOTING_THRESHOLD,
    DECISION_MATRIX_MIN_VOTES,
    SPC_STRATEGY_PARAMETERS,
    SPC_LOOKBACK,
    SPC_P_LOW,
    SPC_P_HIGH,
    RANGE_OSCILLATOR_LENGTH,
    RANGE_OSCILLATOR_MULTIPLIER,
    HMM_WINDOW_SIZE_DEFAULT,
    HMM_WINDOW_KAMA_DEFAULT,
    HMM_FAST_KAMA_DEFAULT,
    HMM_SLOW_KAMA_DEFAULT,
    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
)
from modules.common.utils import (
    color_text,
    safe_input,
    prompt_user_input,
)
from modules.common.ui.formatting import prompt_user_input_with_backspace
from modules.adaptive_trend.cli import prompt_timeframe


def _format_current_value(value) -> str:
    """Format current value for display in menu."""
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    if isinstance(value, str):
        return value
    return str(value)


def _display_main_menu(config, mode="hybrid"):
    """Display main menu with current configuration values."""
    if mode == "voting":
        title = "ATC + Range Oscillator + SPC Pure Voting - Configuration Menu"
    else:
        title = "ATC + Range Oscillator + SPC Hybrid - Configuration Menu"
    
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text(title, Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print("\n" + color_text("MAIN MENU", Fore.YELLOW, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    
    # Format current values
    timeframe_val = _format_current_value(getattr(config, 'timeframe', DEFAULT_TIMEFRAME))
    spc_k_val = _format_current_value(getattr(config, 'spc_k', 2))
    spc_status = "enabled" if getattr(config, 'enable_spc', True) else "disabled"
    # Use True as default for XGBoost, HMM, Random Forest (new defaults)
    xgb_enabled = getattr(config, 'enable_xgboost', True)
    hmm_enabled = getattr(config, 'enable_hmm', True)
    rf_enabled = getattr(config, 'enable_random_forest', True)
    xgb_status = "enabled" if xgb_enabled else "disabled"
    hmm_status = "enabled" if hmm_enabled else "disabled"
    rf_status = "enabled" if rf_enabled else "disabled"
    # Decision Matrix - show current values instead of enabled/disabled
    voting_threshold = getattr(config, 'voting_threshold', DECISION_MATRIX_VOTING_THRESHOLD)
    min_votes = getattr(config, 'min_votes', DECISION_MATRIX_MIN_VOTES)
    dm_status = f"threshold={voting_threshold}, min_votes={min_votes}"
    
    print(f"  1. Timeframe Selection [{color_text(timeframe_val, Fore.GREEN)}]")
    print(f"  2. SPC Configuration [{color_text(f'{spc_status}, k={spc_k_val}', Fore.GREEN)}]")
    print(f"  3. XGBoost Configuration [{color_text(xgb_status, Fore.GREEN)}]")
    print(f"  4. HMM Configuration [{color_text(hmm_status, Fore.GREEN)}]")
    print(f"  5. Random Forest Configuration [{color_text(rf_status, Fore.GREEN)}]")
    print(f"  6. Decision Matrix Configuration [{color_text(dm_status, Fore.GREEN)}]")
    print(f"  7. Review and Confirm")
    print(f"  8. Exit")
    print(color_text("-" * 80, Fore.CYAN))


def _prompt_with_back(prompt: str, default: str = None, allow_back: bool = True) -> tuple:
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
            return (None, 'main')
        
        # Convert to lowercase for consistency
        if user_input:
            user_input = user_input.strip().lower()
        
        return (user_input, 'continue')
    else:
        user_input = prompt_user_input(back_prompt, default=default).strip().lower()
        return (user_input, 'continue')

def _configure_timeframe(config, mode="hybrid"):
    """Configure timeframe with back option.
    
    Returns:
        tuple: (action, changed) where action is 'main' and changed indicates if changes were made
    """
    print("\n" + color_text("1. TIMEFRAME SELECTION", Fore.YELLOW, Style.BRIGHT))
    print()
    
    current_tf = getattr(config, 'timeframe', DEFAULT_TIMEFRAME)
    new_tf = prompt_timeframe(default_timeframe=current_tf)
    changed = (new_tf != current_tf)
    config.timeframe = new_tf
    config.no_menu = True  # Already selected, skip menu
    return ('main', changed)


def _configure_spc(config, mode="hybrid"):
    """Configure SPC with back option.
    
    Returns:
        tuple: (action, changed) where action is 'main' and changed indicates if changes were made
    """
    print("\n" + color_text("2. SPC (Simplified Percentile Clustering) CONFIGURATION", Fore.YELLOW, Style.BRIGHT))
    if mode == "voting":
        print("Note: All 3 SPC strategies will be calculated and aggregated into 1 vote")
    else:
        print("Note: All 3 SPC strategies will be calculated and used in Decision Matrix")
    print(color_text("   b) Back to main menu", Fore.CYAN))
    print()
    
    current_k = getattr(config, 'spc_k', 2)
    default_k_str = str(current_k)
    k_input, action = _prompt_with_back(f"Number of clusters (2 or 3) [{current_k}]: ", default=default_k_str)
    if action == 'main':
        return ('main', False)
    
    config.enable_spc = True  # Always enabled
    
    # Validate k_input - handle three cases explicitly:
    # 1. Valid input: '2' or '3'
    # 2. Empty/Enter: matches default (user pressed Enter)
    # 3. Invalid: anything else
    if k_input in ['2', '3']:
        new_k = int(k_input)
    elif k_input == default_k_str:
        # User pressed Enter - use current value
        new_k = current_k
    else:
        # Non-empty but invalid input - warn and keep current value
        print(color_text(f"Invalid input for number of clusters; keeping current value: {current_k}", Fore.YELLOW))
        new_k = current_k
    
    changed = (new_k != current_k)
    config.spc_k = new_k
    
    # SPC lookback and percentiles: use defaults from config
    config.spc_lookback = SPC_LOOKBACK
    config.spc_p_low = SPC_P_LOW
    config.spc_p_high = SPC_P_HIGH
    
    # Strategy-specific parameters
    config.spc_min_signal_strength = SPC_STRATEGY_PARAMETERS['cluster_transition']['min_signal_strength']
    config.spc_min_rel_pos_change = SPC_STRATEGY_PARAMETERS['cluster_transition']['min_rel_pos_change']
    config.spc_min_regime_strength = SPC_STRATEGY_PARAMETERS['regime_following']['min_regime_strength']
    config.spc_min_cluster_duration = SPC_STRATEGY_PARAMETERS['regime_following']['min_cluster_duration']
    config.spc_extreme_threshold = SPC_STRATEGY_PARAMETERS['mean_reversion']['extreme_threshold']
    config.spc_min_extreme_duration = SPC_STRATEGY_PARAMETERS['mean_reversion']['min_extreme_duration']
    
    # Set SPC strategy explicitly based on mode
    if mode == "voting":
        config.spc_strategy = "all"
    else:
        config.spc_strategy = "hybrid"
    
    return ('main', changed)


def _configure_xgboost(config, mode="hybrid"):
    """Configure XGBoost with back option.
    
    Returns:
        tuple: (action, changed) where action is 'main' and changed indicates if changes were made
    """
    while True:
        print("\n" + color_text("3. XGBOOST PREDICTION CONFIGURATION (Optional)", Fore.YELLOW, Style.BRIGHT))
        print(color_text("   b) Back to main menu", Fore.CYAN))
        print()
        
        current_enabled = getattr(config, 'enable_xgboost', True)  # Default enabled
        default_val = "y" if current_enabled else "n"
        enable_xgb_input, action = _prompt_with_back(f"Enable XGBoost prediction? (y/n) [{default_val}]: ", default=default_val)
        if action == 'main':
            return ('main', False)
        
        new_enabled = enable_xgb_input.lower() in ['y', 'yes']
        changed = (new_enabled != current_enabled)
        config.enable_xgboost = new_enabled
        return ('main', changed)


def _configure_hmm(config, mode="hybrid"):
    """Configure HMM with back option.
    
    Returns:
        tuple: (action, changed) where action is 'main' and changed indicates if changes were made
    """
    while True:
        print("\n" + color_text("4. HMM (Hidden Markov Model) CONFIGURATION (Optional)", Fore.YELLOW, Style.BRIGHT))
        print(color_text("   b) Back to main menu", Fore.CYAN))
        print()
        
        current_enabled = getattr(config, 'enable_hmm', True)  # Default enabled
        default_val = "y" if current_enabled else "n"
        enable_hmm_input, action = _prompt_with_back(f"Enable HMM signal? (y/n) [{default_val}]: ", default=default_val)
        if action == 'main':
            return ('main', False)
        
        new_enabled = enable_hmm_input.lower() in ['y', 'yes']
        changed = (new_enabled != current_enabled)
        config.enable_hmm = new_enabled
        
        # HMM parameters: use defaults from config
        config.hmm_window_size = HMM_WINDOW_SIZE_DEFAULT
        config.hmm_window_kama = HMM_WINDOW_KAMA_DEFAULT
        config.hmm_fast_kama = HMM_FAST_KAMA_DEFAULT
        config.hmm_slow_kama = HMM_SLOW_KAMA_DEFAULT
        config.hmm_orders_argrelextrema = HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
        config.hmm_strict_mode = HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
        
        return ('main', changed)


def _configure_random_forest(config, mode="hybrid"):
    """Configure Random Forest with back option.
    
    Returns:
        tuple: (action, changed) where action is 'main' and changed indicates if changes were made
    """
    while True:
        print("\n" + color_text("5. RANDOM FOREST CONFIGURATION (Optional)", Fore.YELLOW, Style.BRIGHT))
        print(color_text("   b) Back to main menu", Fore.CYAN))
        print()
        
        current_enabled = getattr(config, 'enable_random_forest', True)  # Default enabled
        default_val = "y" if current_enabled else "n"
        enable_rf_input, action = _prompt_with_back(f"Enable Random Forest prediction? (y/n) [{default_val}]: ", default=default_val)
        if action == 'main':
            return ('main', False)
        
        new_enabled = enable_rf_input.lower() in ['y', 'yes']
        changed = (new_enabled != current_enabled)
        config.enable_random_forest = new_enabled
        config.random_forest_model_path = None  # Uses default path from config
        
        return ('main', changed)


def _configure_decision_matrix(config, mode="hybrid"):
    """Configure Decision Matrix parameters (Voting Threshold and Min Votes).
    
    Returns:
        tuple: (action, changed) where action is 'main' and changed indicates if changes were made
    """
    print("\n" + color_text("6. DECISION MATRIX CONFIGURATION", Fore.YELLOW, Style.BRIGHT))
    if mode == "hybrid":
        print("Note: Decision Matrix is recommended when using all 3 SPC strategies")
    print(color_text("   b) Back to main menu", Fore.CYAN))
    print()
    
    # Decision Matrix is always enabled, just configure parameters
    config.use_decision_matrix = True
    
    # Get current values or use defaults
    current_threshold = getattr(config, 'voting_threshold', DECISION_MATRIX_VOTING_THRESHOLD)
    current_min_votes = getattr(config, 'min_votes', DECISION_MATRIX_MIN_VOTES)
    
    # Configure Voting Threshold
    threshold_str, action = _prompt_with_back(
        f"Voting Threshold (0.0-1.0) [{current_threshold}]: ", 
        default=str(current_threshold)
    )
    if action == 'main':
        return ('main', False)
    
    try:
        new_threshold = float(threshold_str) if threshold_str else current_threshold
        if not (0.0 <= new_threshold <= 1.0):
            print(color_text("Warning: Voting Threshold should be between 0.0 and 1.0. Preserving current value.", Fore.YELLOW))
            new_threshold = current_threshold
        config.voting_threshold = new_threshold
    except ValueError:
        print(color_text(f"Invalid input. Preserving current value: {current_threshold}", Fore.YELLOW))
        config.voting_threshold = current_threshold
    
    threshold_changed = (config.voting_threshold != current_threshold)
    
    # Configure Min Votes
    min_votes_str, action = _prompt_with_back(
        f"Min Votes (minimum number of indicators that must agree) [{current_min_votes}]: ", 
        default=str(current_min_votes)
    )
    if action == 'main':
        # Rollback threshold change if user chose back
        config.voting_threshold = current_threshold
        return ('main', False)
    
    try:
        new_min_votes = int(min_votes_str) if min_votes_str else current_min_votes
        if new_min_votes < 1:
            print(color_text("Warning: Min Votes should be at least 1. Preserving current value.", Fore.YELLOW))
            new_min_votes = current_min_votes
        config.min_votes = new_min_votes
    except ValueError:
        print(color_text(f"Invalid input. Preserving current value: {current_min_votes}", Fore.YELLOW))
        config.min_votes = current_min_votes
    
    min_votes_changed = (config.min_votes != current_min_votes)
    changed = threshold_changed or min_votes_changed
    
    print(color_text("\nDecision Matrix configuration updated:", Fore.GREEN))
    print(f"  Voting Threshold: {config.voting_threshold}")
    print(f"  Min Votes: {config.min_votes}")
    
    safe_input("\nPress Enter to return to main menu...", default='')
    return ('main', changed)


def _review_and_confirm(config, mode="hybrid"):
    """Review configuration and confirm."""
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("CONFIGURATION REVIEW", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    
    print(f"\nTimeframe: {getattr(config, 'timeframe', DEFAULT_TIMEFRAME)}")
    print(f"SPC: {'enabled' if getattr(config, 'enable_spc', True) else 'disabled'} (k={getattr(config, 'spc_k', 2)})")
    print(f"XGBoost: {'enabled' if getattr(config, 'enable_xgboost', True) else 'disabled'}")
    print(f"HMM: {'enabled' if getattr(config, 'enable_hmm', True) else 'disabled'}")
    print(f"Random Forest: {'enabled' if getattr(config, 'enable_random_forest', True) else 'disabled'}")
    voting_threshold = getattr(config, 'voting_threshold', DECISION_MATRIX_VOTING_THRESHOLD)
    min_votes = getattr(config, 'min_votes', DECISION_MATRIX_MIN_VOTES)
    print(f"Decision Matrix: threshold={voting_threshold}, min_votes={min_votes}")
    
    print("\n" + color_text("-" * 80, Fore.CYAN))
    confirm = prompt_user_input("Confirm this configuration? (y/n) [y]: ", default="y").strip().lower()
    
    if confirm in ['y', 'yes', '']:
        return 'done'
    else:
        return 'main'


def interactive_config_menu(mode="hybrid"):
    """
    Interactive menu for configuring ATC + Range Oscillator + SPC.
    
    Args:
        mode: "hybrid" or "voting" - determines menu title and behavior
    
    Returns:
        argparse.Namespace object with all configuration values
    """
    # Create namespace object with defaults
    config = argparse.Namespace()
    
    # Initialize default values
    config.timeframe = DEFAULT_TIMEFRAME
    config.no_menu = True
    config.limit = 500
    config.max_workers = 10
    config.osc_length = RANGE_OSCILLATOR_LENGTH
    config.osc_mult = RANGE_OSCILLATOR_MULTIPLIER
    config.enable_spc = True
    config.spc_k = 2
    config.spc_lookback = SPC_LOOKBACK
    config.spc_p_low = SPC_P_LOW
    config.spc_p_high = SPC_P_HIGH
    config.spc_min_signal_strength = SPC_STRATEGY_PARAMETERS['cluster_transition']['min_signal_strength']
    config.spc_min_rel_pos_change = SPC_STRATEGY_PARAMETERS['cluster_transition']['min_rel_pos_change']
    config.spc_min_regime_strength = SPC_STRATEGY_PARAMETERS['regime_following']['min_regime_strength']
    config.spc_min_cluster_duration = SPC_STRATEGY_PARAMETERS['regime_following']['min_cluster_duration']
    config.spc_extreme_threshold = SPC_STRATEGY_PARAMETERS['mean_reversion']['extreme_threshold']
    config.spc_min_extreme_duration = SPC_STRATEGY_PARAMETERS['mean_reversion']['min_extreme_duration']
    # Set SPC strategy explicitly based on mode
    if mode == "voting":
        config.spc_strategy = "all"
    else:
        config.spc_strategy = "hybrid"
    config.enable_xgboost = True  # Default enabled
    config.enable_hmm = True  # Default enabled
    config.hmm_window_size = HMM_WINDOW_SIZE_DEFAULT
    config.hmm_window_kama = HMM_WINDOW_KAMA_DEFAULT
    config.hmm_fast_kama = HMM_FAST_KAMA_DEFAULT
    config.hmm_slow_kama = HMM_SLOW_KAMA_DEFAULT
    config.hmm_orders_argrelextrema = HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
    config.hmm_strict_mode = HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
    config.enable_random_forest = True  # Default enabled
    config.random_forest_model_path = None
    config.use_decision_matrix = True  # Always enabled, only parameters are configurable
    config.voting_threshold = DECISION_MATRIX_VOTING_THRESHOLD
    config.min_votes = DECISION_MATRIX_MIN_VOTES
    config.ema_len = 28
    config.hma_len = 28
    config.wma_len = 28
    config.dema_len = 28
    config.lsma_len = 28
    config.kama_len = 28
    config.robustness = "Medium"
    config.lambda_param = 0.5
    config.decay = 0.1
    config.cutout = 5
    config.min_signal = 0.01
    config.max_symbols = None
    config.osc_strategies = None
    
    # Track unsaved changes
    has_unsaved_changes = False
    
    # Main menu loop
    while True:
        _display_main_menu(config, mode)
        
        choice = prompt_user_input("\nSelect option [1-8]: ").strip()
        
        if choice == '1':
            _, changed = _configure_timeframe(config, mode)
            if changed:
                has_unsaved_changes = True
        elif choice == '2':
            _, changed = _configure_spc(config, mode)
            if changed:
                has_unsaved_changes = True
        elif choice == '3':
            _, changed = _configure_xgboost(config, mode)
            if changed:
                has_unsaved_changes = True
        elif choice == '4':
            _, changed = _configure_hmm(config, mode)
            if changed:
                has_unsaved_changes = True
        elif choice == '5':
            _, changed = _configure_random_forest(config, mode)
            if changed:
                has_unsaved_changes = True
        elif choice == '6':
            _, changed = _configure_decision_matrix(config, mode)
            if changed:
                has_unsaved_changes = True
        elif choice == '7':
            result = _review_and_confirm(config, mode)
            if result == 'done':
                has_unsaved_changes = False  # Changes are confirmed/saved
                break
        elif choice == '8':
            # Prompt for confirmation before exiting
            if has_unsaved_changes:
                confirm_msg = color_text(
                    "\n⚠️  Are you sure you want to exit? Unsaved changes will be lost. (y/N): ",
                    Fore.YELLOW
                )
            else:
                confirm_msg = color_text(
                    "\nAre you sure you want to exit? (y/N): ",
                    Fore.YELLOW
                )
            
            confirm = prompt_user_input(confirm_msg, default="n").strip().lower()
            
            if confirm in ['y', 'yes']:
                print(color_text("\nExiting configuration menu.", Fore.YELLOW))
                sys.exit(0)
            else:
                # User chose not to exit, return to menu
                continue
        else:
            print(color_text("Invalid choice. Please select 1-8.", Fore.RED))
    
    return config


def parse_args(mode="hybrid", force_enable_spc=True, force_enable_decision_matrix=True):
    """
    Parse command-line arguments or use interactive menu.
    
    Args:
        mode: "hybrid" or "voting" - determines parser description and behavior
        force_enable_spc: If True, force enable SPC regardless of flag
        force_enable_decision_matrix: If True, force enable Decision Matrix regardless of flag
    
    If no arguments provided, shows interactive menu.
    Otherwise, parses command-line arguments.
    """
    # Check if any arguments were provided
    if len(sys.argv) == 1:
        # No arguments, use interactive menu
        return interactive_config_menu(mode=mode)
    
    # Parse command-line arguments
    if mode == "voting":
        description = "ATC + Range Oscillator + SPC Pure Voting Signal Filter"
    else:
        description = "ATC + Range Oscillator + SPC Hybrid Signal Filter"
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Copy all arguments from range_oscillator/cli/argument_parser.py
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME,
        help=f"Timeframe for analysis (default: {DEFAULT_TIMEFRAME})")
    parser.add_argument("--no-menu", action="store_true",
        help="Disable interactive timeframe menu")
    parser.add_argument("--limit", type=int, default=500,
        help="Number of candles to fetch (default: 500)")
    parser.add_argument("--ema-len", type=int, default=28, help="EMA length (default: 28)")
    parser.add_argument("--hma-len", type=int, default=28, help="HMA length (default: 28)")
    parser.add_argument("--wma-len", type=int, default=28, help="WMA length (default: 28)")
    parser.add_argument("--dema-len", type=int, default=28, help="DEMA length (default: 28)")
    parser.add_argument("--lsma-len", type=int, default=28, help="LSMA length (default: 28)")
    parser.add_argument("--kama-len", type=int, default=28, help="KAMA length (default: 28)")
    parser.add_argument("--robustness", type=str, choices=["Narrow", "Medium", "Wide"],
        default="Medium", help="Robustness setting (default: Medium)")
    parser.add_argument("--lambda", type=float, default=0.5, dest="lambda_param",
        help="Lambda parameter (default: 0.5)")
    parser.add_argument("--decay", type=float, default=0.1, help="Decay rate (default: 0.1)")
    parser.add_argument("--cutout", type=int, default=5,
        help="Number of bars to skip at start (default: 5)")
    parser.add_argument("--min-signal", type=float, default=0.01,
        help="Minimum signal strength to display (default: 0.01)")
    parser.add_argument("--max-symbols", type=int, default=None,
        help="Maximum number of symbols to scan (default: None = all)")
    parser.add_argument("--osc-length", type=int, default=50,
        help="Range Oscillator length parameter (default: 50)")
    parser.add_argument("--osc-mult", type=float, default=2.0,
        help="Range Oscillator multiplier (default: 2.0)")
    parser.add_argument("--max-workers", type=int, default=10,
        help="Maximum number of parallel workers for Range Oscillator filtering (default: 10)")
    parser.add_argument("--osc-strategies", type=int, nargs="+", default=None,
        help="Range Oscillator strategies to use (e.g., --osc-strategies 5 6 7 8 9). Default: all [5, 6, 7, 8, 9]")
    
    # SPC configuration
    if mode == "voting":
        parser.add_argument("--enable-spc", action="store_true", help="Enable SPC (default: False)")
        parser.add_argument("--spc-strategy", type=str, default="cluster_transition",
            choices=["cluster_transition", "regime_following", "mean_reversion"],
            help="SPC strategy (default: cluster_transition)")
    else:
        parser.add_argument(
            "--enable-spc",
            action="store_true",
            help="Enable SPC (always enabled, all 3 strategies used). Note: This flag is always enabled internally.",
        )
    parser.add_argument(
        "--spc-k",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of clusters for SPC (default: 2)",
    )
    parser.add_argument(
        "--spc-lookback",
        type=int,
        default=SPC_LOOKBACK,
        help=f"Historical bars for SPC (default: {SPC_LOOKBACK})",
    )
    parser.add_argument(
        "--spc-p-low",
        type=float,
        default=SPC_P_LOW,
        help=f"Lower percentile for SPC (default: {SPC_P_LOW})",
    )
    parser.add_argument(
        "--spc-p-high",
        type=float,
        default=SPC_P_HIGH,
        help=f"Upper percentile for SPC (default: {SPC_P_HIGH})",
    )
    parser.add_argument(
        "--spc-min-signal-strength",
        type=float,
        default=0.3,
        help="Minimum signal strength for SPC cluster transition (default: 0.3)",
    )
    parser.add_argument(
        "--spc-min-rel-pos-change",
        type=float,
        default=0.1,
        help="Minimum relative position change for SPC cluster transition (default: 0.1)",
    )
    parser.add_argument(
        "--spc-min-regime-strength",
        type=float,
        default=0.7,
        help="Minimum regime strength for SPC regime following (default: 0.7)",
    )
    parser.add_argument(
        "--spc-min-cluster-duration",
        type=int,
        default=2,
        help="Minimum bars in same cluster for SPC regime following (default: 2)",
    )
    parser.add_argument(
        "--spc-extreme-threshold",
        type=float,
        default=0.2,
        help="Real_clust threshold for extreme in SPC mean reversion (default: 0.2)",
    )
    parser.add_argument(
        "--spc-min-extreme-duration",
        type=int,
        default=3,
        help="Minimum bars in extreme for SPC mean reversion (default: 3)",
    )
    
    # XGBoost configuration (optional)
    parser.add_argument(
        "--enable-xgboost",
        action="store_true",
        help="Enable XGBoost prediction in Decision Matrix (default: False)",
    )
    
    # HMM configuration (optional)
    parser.add_argument(
        "--enable-hmm",
        action="store_true",
        help="Enable HMM (Hidden Markov Model) signal in Decision Matrix (default: False)",
    )
    parser.add_argument(
        "--hmm-window-size",
        type=int,
        default=HMM_WINDOW_SIZE_DEFAULT,
        help=f"HMM rolling window size (default: {HMM_WINDOW_SIZE_DEFAULT})",
    )
    parser.add_argument(
        "--hmm-window-kama",
        type=int,
        default=HMM_WINDOW_KAMA_DEFAULT,
        help=f"HMM KAMA window size (default: {HMM_WINDOW_KAMA_DEFAULT})",
    )
    parser.add_argument(
        "--hmm-fast-kama",
        type=int,
        default=HMM_FAST_KAMA_DEFAULT,
        help=f"HMM fast KAMA parameter (default: {HMM_FAST_KAMA_DEFAULT})",
    )
    parser.add_argument(
        "--hmm-slow-kama",
        type=int,
        default=HMM_SLOW_KAMA_DEFAULT,
        help=f"HMM slow KAMA parameter (default: {HMM_SLOW_KAMA_DEFAULT})",
    )
    parser.add_argument(
        "--hmm-orders-argrelextrema",
        type=int,
        default=HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
        help=f"HMM order for swing detection (default: {HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT})",
    )
    parser.add_argument(
        "--hmm-strict-mode",
        action="store_const",
        const=True,
        default=None,
        help=f"Use strict mode for HMM swing-to-state conversion (default: {HMM_HIGH_ORDER_STRICT_MODE_DEFAULT})",
    )
    
    # Random Forest configuration (optional)
    parser.add_argument(
        "--enable-random-forest",
        action="store_true",
        help="Enable Random Forest prediction in Decision Matrix (default: False)",
    )
    parser.add_argument(
        "--random-forest-model-path",
        type=str,
        default=None,
        help="Path to Random Forest model file (default: uses default path from config)",
    )
    
    # Decision Matrix options
    if mode == "hybrid":
        parser.add_argument(
            "--use-decision-matrix",
            action="store_true",
            help="Use decision matrix voting system (always enabled with SPC). Note: This flag is always enabled internally.",
        )
    
    parser.add_argument(
        "--voting-threshold",
        type=float,
        default=DECISION_MATRIX_VOTING_THRESHOLD,
        help=f"Minimum weighted score for positive vote (default: {DECISION_MATRIX_VOTING_THRESHOLD})",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=DECISION_MATRIX_MIN_VOTES,
        help=f"Minimum number of indicators that must agree (default: {DECISION_MATRIX_MIN_VOTES})",
    )
    
    args = parser.parse_args()
    
    # Force enable SPC and Decision Matrix if requested
    if force_enable_spc:
        args.enable_spc = True
    if force_enable_decision_matrix:
        args.use_decision_matrix = True
    
    return args

