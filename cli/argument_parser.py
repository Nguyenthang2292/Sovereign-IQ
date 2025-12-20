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
    prompt_user_input,
)
from modules.adaptive_trend.cli import prompt_timeframe


def interactive_config_menu(mode="hybrid"):
    """
    Interactive menu for configuring ATC + Range Oscillator + SPC.
    
    Args:
        mode: "hybrid" or "voting" - determines menu title and behavior
    
    Returns:
        argparse.Namespace object with all configuration values
    """
    if mode == "voting":
        title = "ATC + Range Oscillator + SPC Pure Voting - Configuration Menu"
        spc_note = "Note: All 3 SPC strategies will be calculated and aggregated into 1 vote"
        decision_matrix_title = "6. DECISION MATRIX CONFIGURATION (Required)"
        decision_matrix_note = None
        use_decision_matrix = False  # Not needed for voting mode
    else:
        title = "ATC + Range Oscillator + SPC Hybrid - Configuration Menu"
        spc_note = "Note: All 3 SPC strategies will be calculated and used in Decision Matrix"
        decision_matrix_title = "6. DECISION MATRIX CONFIGURATION"
        decision_matrix_note = "Note: Decision Matrix is required when using all 3 SPC strategies"
        use_decision_matrix = True  # Always enabled for hybrid
    
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text(title, Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    
    # Create namespace object
    class Config:
        pass
    
    config = Config()
    
    # 1. Timeframe selection
    print("\n" + color_text("1. TIMEFRAME SELECTION", Fore.YELLOW, Style.BRIGHT))
    config.timeframe = prompt_timeframe(default_timeframe=DEFAULT_TIMEFRAME)
    config.no_menu = True  # Already selected, skip menu
    
    # 2. Set default values (not shown in menu, can be changed in modules/config.py)
    config.limit = 500  # Default: 500 candles
    config.max_workers = 10  # Default: 10 parallel workers
    
    # 3. Range Oscillator parameters (loaded from config, not shown in menu)
    # Adjust these values in modules/config.py if needed
    config.osc_length = RANGE_OSCILLATOR_LENGTH
    config.osc_mult = RANGE_OSCILLATOR_MULTIPLIER
    
    # 4. SPC configuration (always enabled, all 3 strategies will be used)
    print("\n" + color_text("4. SPC (Simplified Percentile Clustering) CONFIGURATION", Fore.YELLOW, Style.BRIGHT))
    print(spc_note)
    
    config.enable_spc = True  # Always enabled
    
    k_input = prompt_user_input("Number of clusters (2 or 3) [2]: ", default="2")
    config.spc_k = int(k_input) if k_input in ['2', '3'] else 2
    
    # SPC lookback and percentiles: use defaults from config (not shown in menu)
    config.spc_lookback = SPC_LOOKBACK
    config.spc_p_low = SPC_P_LOW
    config.spc_p_high = SPC_P_HIGH
    
    # Strategy-specific parameters (loaded from config, not shown in menu)
    # Adjust these values in modules/config.py if needed
    config.spc_min_signal_strength = SPC_STRATEGY_PARAMETERS['cluster_transition']['min_signal_strength']
    config.spc_min_rel_pos_change = SPC_STRATEGY_PARAMETERS['cluster_transition']['min_rel_pos_change']
    config.spc_min_regime_strength = SPC_STRATEGY_PARAMETERS['regime_following']['min_regime_strength']
    config.spc_min_cluster_duration = SPC_STRATEGY_PARAMETERS['regime_following']['min_cluster_duration']
    config.spc_extreme_threshold = SPC_STRATEGY_PARAMETERS['mean_reversion']['extreme_threshold']
    config.spc_min_extreme_duration = SPC_STRATEGY_PARAMETERS['mean_reversion']['min_extreme_duration']
    if mode == "voting":
        config.spc_strategy = "all"  # Indicates all 3 strategies will be used
    
    # 5. XGBoost configuration (optional)
    print("\n" + color_text("5. XGBOOST PREDICTION CONFIGURATION (Optional)", Fore.YELLOW, Style.BRIGHT))
    enable_xgb_input = prompt_user_input("Enable XGBoost prediction? (y/n) [n]: ", default="n")
    config.enable_xgboost = enable_xgb_input.lower() in ['y', 'yes']
    
    # 5b. HMM configuration (optional)
    print("\n" + color_text("5b. HMM (Hidden Markov Model) CONFIGURATION (Optional)", Fore.YELLOW, Style.BRIGHT))
    enable_hmm_input = prompt_user_input("Enable HMM signal? (y/n) [n]: ", default="n")
    config.enable_hmm = enable_hmm_input.lower() in ['y', 'yes']
    
    # HMM parameters: use defaults from config (not shown in menu)
    config.hmm_window_size = HMM_WINDOW_SIZE_DEFAULT
    config.hmm_window_kama = HMM_WINDOW_KAMA_DEFAULT
    config.hmm_fast_kama = HMM_FAST_KAMA_DEFAULT
    config.hmm_slow_kama = HMM_SLOW_KAMA_DEFAULT
    config.hmm_orders_argrelextrema = HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
    config.hmm_strict_mode = HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
    
    # 5c. Random Forest configuration (optional)
    print("\n" + color_text("5c. RANDOM FOREST CONFIGURATION (Optional)", Fore.YELLOW, Style.BRIGHT))
    enable_rf_input = prompt_user_input("Enable Random Forest prediction? (y/n) [n]: ", default="n")
    config.enable_random_forest = enable_rf_input.lower() in ['y', 'yes']
    
    # Random Forest model path: use default (not shown in menu)
    config.random_forest_model_path = None  # Uses default path from config
    
    # 6. Decision Matrix configuration
    print("\n" + color_text(decision_matrix_title, Fore.YELLOW, Style.BRIGHT))
    if decision_matrix_note:
        print(decision_matrix_note)
    if use_decision_matrix:
        config.use_decision_matrix = True  # Always enabled for hybrid
    
    # Voting threshold and min votes: use defaults from config (not shown in menu)
    config.voting_threshold = DECISION_MATRIX_VOTING_THRESHOLD
    config.min_votes = DECISION_MATRIX_MIN_VOTES
    
    # Set default values for other parameters
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
    config.spc_strategy = "all"  # Indicates all 3 strategies will be used
    
    # Set default HMM values if not enabled
    if not hasattr(config, 'enable_hmm'):
        config.enable_hmm = False
    if not hasattr(config, 'hmm_window_size'):
        config.hmm_window_size = HMM_WINDOW_SIZE_DEFAULT
    if not hasattr(config, 'hmm_window_kama'):
        config.hmm_window_kama = HMM_WINDOW_KAMA_DEFAULT
    if not hasattr(config, 'hmm_fast_kama'):
        config.hmm_fast_kama = HMM_FAST_KAMA_DEFAULT
    if not hasattr(config, 'hmm_slow_kama'):
        config.hmm_slow_kama = HMM_SLOW_KAMA_DEFAULT
    if not hasattr(config, 'hmm_orders_argrelextrema'):
        config.hmm_orders_argrelextrema = HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
    if not hasattr(config, 'hmm_strict_mode'):
        config.hmm_strict_mode = HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
    
    # Set default Random Forest values if not enabled
    if not hasattr(config, 'enable_random_forest'):
        config.enable_random_forest = False
    if not hasattr(config, 'random_forest_model_path'):
        config.random_forest_model_path = None
    
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
    if force_enable_decision_matrix and mode == "hybrid":
        args.use_decision_matrix = True
    
    return args

