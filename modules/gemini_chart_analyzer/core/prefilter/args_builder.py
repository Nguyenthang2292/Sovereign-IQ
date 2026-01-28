"""Argument builder for VotingAnalyzer used by prefilter workflow."""

import argparse
from typing import Any, Dict, Optional

from config import (
    DECISION_MATRIX_MIN_VOTES,
    DECISION_MATRIX_VOTING_THRESHOLD,
    HMM_FAST_KAMA_DEFAULT,
    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
    HMM_SLOW_KAMA_DEFAULT,
    HMM_WINDOW_KAMA_DEFAULT,
    HMM_WINDOW_SIZE_DEFAULT,
    RANGE_OSCILLATOR_LENGTH,
    RANGE_OSCILLATOR_MULTIPLIER,
    SPC_LOOKBACK,
    SPC_P_HIGH,
    SPC_P_LOW,
    SPC_STRATEGY_PARAMETERS,
)


def build_voting_analyzer_args(
    *,
    timeframe: str,
    limit: int,
    fast_mode: bool,
    spc_config: Optional[Dict[str, Any]],
    rf_model_path: Optional[str],
    atc_performance: Optional[Dict[str, Any]],
    approximate_ma_scanner: Optional[Dict[str, Any]],
    use_atc_performance: bool,
) -> argparse.Namespace:
    """Build argparse.Namespace for VotingAnalyzer with consistent defaults."""
    args = argparse.Namespace()
    args.use_atc_performance = use_atc_performance
    args.timeframe = timeframe
    args.no_menu = True
    args.limit = limit
    args.max_workers = 10
    args.osc_length = RANGE_OSCILLATOR_LENGTH
    args.osc_mult = RANGE_OSCILLATOR_MULTIPLIER
    args.osc_strategies = None  # Use all strategies
    args.enable_spc = True
    args.spc_k = 2

    # Use spc_config if provided
    if spc_config:
        args.spc_preset = spc_config.get("preset")
        args.spc_volatility_adjustment = spc_config.get("volatility_adjustment")
        args.spc_use_correlation_weights = spc_config.get("use_correlation_weights")
        args.spc_time_decay_factor = spc_config.get("time_decay_factor")
        args.spc_interpolation_mode = spc_config.get("interpolation_mode")
        args.spc_min_flip_duration = spc_config.get("min_flip_duration")
        args.spc_flip_confidence_threshold = spc_config.get("flip_confidence_threshold")

        # MTF parameters
        args.spc_enable_mtf = spc_config.get("enable_mtf", False)
        args.spc_mtf_timeframes = spc_config.get("mtf_timeframes")
        args.spc_mtf_require_alignment = spc_config.get("mtf_require_alignment")

    args.spc_lookback = SPC_LOOKBACK
    args.spc_p_low = SPC_P_LOW
    args.spc_p_high = SPC_P_HIGH
    args.spc_min_signal_strength = SPC_STRATEGY_PARAMETERS["cluster_transition"]["min_signal_strength"]
    args.spc_min_rel_pos_change = SPC_STRATEGY_PARAMETERS["cluster_transition"]["min_rel_pos_change"]
    args.spc_min_regime_strength = SPC_STRATEGY_PARAMETERS["regime_following"]["min_regime_strength"]
    args.spc_min_cluster_duration = SPC_STRATEGY_PARAMETERS["regime_following"]["min_cluster_duration"]
    args.spc_extreme_threshold = SPC_STRATEGY_PARAMETERS["mean_reversion"]["extreme_threshold"]
    args.spc_min_extreme_duration = SPC_STRATEGY_PARAMETERS["mean_reversion"]["min_extreme_duration"]
    args.spc_strategy = "all"

    # In fast mode, ML models are disabled initially but enabled in Stage 3
    # This allows Stage 1-2 to run without ML overhead
    args.enable_xgboost = not fast_mode
    args.enable_hmm = not fast_mode
    args.hmm_window_size = HMM_WINDOW_SIZE_DEFAULT
    args.hmm_window_kama = HMM_WINDOW_KAMA_DEFAULT
    args.hmm_fast_kama = HMM_FAST_KAMA_DEFAULT
    args.hmm_slow_kama = HMM_SLOW_KAMA_DEFAULT
    args.hmm_orders_argrelextrema = HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
    args.hmm_strict_mode = HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
    args.enable_random_forest = True if rf_model_path else False
    args.random_forest_model_path = rf_model_path
    args.use_decision_matrix = True
    args.voting_threshold = DECISION_MATRIX_VOTING_THRESHOLD
    args.min_votes = DECISION_MATRIX_MIN_VOTES

    args.ema_len = 28
    args.hma_len = 28
    args.wma_len = 28
    args.dema_len = 28
    args.lsma_len = 28
    args.kama_len = 28
    args.robustness = "Medium"
    args.lambda_param = 0.5
    args.decay = 0.1
    args.cutout = 5
    args.min_signal = 0.01
    args.max_symbols = None

    # Full parallel settings for adaptive_trend_LTS
    if atc_performance:
        # Use provided performance configuration (mapping from snake_case standard config to args)
        # Standard config keys: batch_processing, use_cuda, parallel_l1, parallel_l2, prefer_gpu,
        # use_cache, fast_mode, precision, use_dask, npartitions, use_incremental,
        # use_memory_mapped, use_compression, compression_level, warm_cache, use_codegen_specialization

        # Core Performance Settings
        args.batch_processing = atc_performance.get("batch_processing", True)
        args.use_cuda = atc_performance.get("use_cuda", False)
        args.parallel_l1 = atc_performance.get("parallel_l1", True)
        args.parallel_l2 = atc_performance.get("parallel_l2", True)
        args.prefer_gpu = atc_performance.get("prefer_gpu", True)
        args.use_cache = atc_performance.get("use_cache", True)
        args.fast_mode = atc_performance.get("fast_mode", True)
        args.precision = atc_performance.get("precision", "float32")

        # Rust backend flag (often mapped from batch_processing)
        args.use_rust_backend = atc_performance.get("batch_processing", True)

        # Dask settings (Out-of-Core Processing)
        args.use_dask = atc_performance.get("use_dask", False)
        args.npartitions = atc_performance.get("npartitions")
        if args.use_dask:
            args.execution_mode = "dask"

        # Incremental Updates (for live trading)
        args.use_incremental = atc_performance.get("use_incremental", False)

        # Memory Optimizations
        args.use_memory_mapped = atc_performance.get("use_memory_mapped", False)
        args.use_compression = atc_performance.get("use_compression", False)
        args.compression_level = atc_performance.get("compression_level", 5)

        # Advanced Settings
        args.warm_cache = atc_performance.get("warm_cache", False)
        args.use_codegen_specialization = atc_performance.get("use_codegen_specialization", False)
    else:
        # Default fallback if not provided
        args.parallel_l1 = True
        args.parallel_l2 = True
        args.use_rust_backend = True  # Enable Rust backend for ATC (CPU parallelism)
        args.use_cuda = False

    # Approximate MA Scanner settings
    if approximate_ma_scanner and approximate_ma_scanner.get("enabled", False):
        args.use_approximate = True
        args.use_adaptive_approximate = approximate_ma_scanner.get("use_adaptive", False)
        args.approximate_volatility_window = approximate_ma_scanner.get("volatility_window", 20)
        args.approximate_volatility_factor = approximate_ma_scanner.get("volatility_factor", 1.0)
    else:
        args.use_approximate = False
        args.use_adaptive_approximate = False

    return args
