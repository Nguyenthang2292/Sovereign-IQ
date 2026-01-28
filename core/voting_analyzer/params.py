"""Parameter extraction helpers for VotingAnalyzer."""

from config.spc import SPC_STRATEGY_PARAMETERS


class VotingParamsMixin:
    """Mixin for extracting oscillator and SPC parameters."""

    def get_oscillator_params(self) -> dict:
        """Extract Range Oscillator parameters."""
        return {
            "osc_length": self.args.osc_length,
            "osc_mult": self.args.osc_mult,
            "max_workers": self.args.max_workers,
            "strategies": self.args.osc_strategies,
        }

    def get_spc_params(self) -> dict:
        """Extract SPC parameters for all 3 strategies."""
        # Import enhancement parameters
        try:
            from config.spc_enhancements import (
                SPC_ENABLE_MTF,
                SPC_FLIP_CONFIDENCE_THRESHOLD,
                SPC_INTERPOLATION_MODE,
                SPC_MIN_FLIP_DURATION,
                SPC_MTF_REQUIRE_ALIGNMENT,
                SPC_MTF_TIMEFRAMES,
                SPC_PRESET_AGGRESSIVE,
                SPC_PRESET_BALANCED,
                SPC_PRESET_CONSERVATIVE,
                SPC_TIME_DECAY_FACTOR,
                SPC_USE_CORRELATION_WEIGHTS,
                SPC_VOLATILITY_ADJUSTMENT,
            )
        except ImportError:
            # Fallback to defaults if spc_enhancements module not available
            SPC_VOLATILITY_ADJUSTMENT = False
            SPC_USE_CORRELATION_WEIGHTS = False
            SPC_TIME_DECAY_FACTOR = 1.0
            SPC_INTERPOLATION_MODE = "linear"
            SPC_MIN_FLIP_DURATION = 3
            SPC_FLIP_CONFIDENCE_THRESHOLD = 0.6
            SPC_ENABLE_MTF = False
            SPC_MTF_TIMEFRAMES = ["1h", "4h"]
            SPC_MTF_REQUIRE_ALIGNMENT = True
            SPC_PRESET_CONSERVATIVE = {}
            SPC_PRESET_BALANCED = {}
            SPC_PRESET_AGGRESSIVE = {}

        # Check if preset is specified
        preset = getattr(self.args, "spc_preset", None)
        if preset:
            if preset == "conservative":
                preset_config = SPC_PRESET_CONSERVATIVE
            elif preset == "balanced":
                preset_config = SPC_PRESET_BALANCED
            elif preset == "aggressive":
                preset_config = SPC_PRESET_AGGRESSIVE
            else:
                preset_config = {}

            # Use preset values as defaults, but allow CLI overrides
            volatility_adjustment = preset_config.get("volatility_adjustment", SPC_VOLATILITY_ADJUSTMENT)
            use_correlation_weights = preset_config.get("use_correlation_weights", SPC_USE_CORRELATION_WEIGHTS)
            time_decay_factor = preset_config.get("time_decay_factor", SPC_TIME_DECAY_FACTOR)
            interpolation_mode = preset_config.get("interpolation_mode", SPC_INTERPOLATION_MODE)
            min_flip_duration = preset_config.get("min_flip_duration", SPC_MIN_FLIP_DURATION)
            flip_confidence_threshold = preset_config.get("flip_confidence_threshold", SPC_FLIP_CONFIDENCE_THRESHOLD)
        else:
            # Use config defaults
            volatility_adjustment = SPC_VOLATILITY_ADJUSTMENT
            use_correlation_weights = SPC_USE_CORRELATION_WEIGHTS
            time_decay_factor = SPC_TIME_DECAY_FACTOR
            interpolation_mode = SPC_INTERPOLATION_MODE
            min_flip_duration = SPC_MIN_FLIP_DURATION
            flip_confidence_threshold = SPC_FLIP_CONFIDENCE_THRESHOLD

        # Override with CLI arguments if provided
        if hasattr(self.args, "spc_volatility_adjustment") and self.args.spc_volatility_adjustment:
            volatility_adjustment = True
        if hasattr(self.args, "spc_use_correlation_weights") and self.args.spc_use_correlation_weights:
            use_correlation_weights = True
        if hasattr(self.args, "spc_time_decay_factor") and self.args.spc_time_decay_factor is not None:
            time_decay_factor = self.args.spc_time_decay_factor
        if hasattr(self.args, "spc_interpolation_mode") and self.args.spc_interpolation_mode is not None:
            interpolation_mode = self.args.spc_interpolation_mode
        if hasattr(self.args, "spc_min_flip_duration") and self.args.spc_min_flip_duration is not None:
            min_flip_duration = self.args.spc_min_flip_duration
        if hasattr(self.args, "spc_flip_confidence_threshold") and self.args.spc_flip_confidence_threshold is not None:
            flip_confidence_threshold = self.args.spc_flip_confidence_threshold

        # MTF parameters
        enable_mtf = getattr(self.args, "spc_enable_mtf", False) or SPC_ENABLE_MTF
        mtf_timeframes = getattr(self.args, "spc_mtf_timeframes", None) or SPC_MTF_TIMEFRAMES
        mtf_require_alignment = getattr(self.args, "spc_mtf_require_alignment", None)
        if mtf_require_alignment is None:
            mtf_require_alignment = SPC_MTF_REQUIRE_ALIGNMENT

        # Use values from config if not provided in args
        cluster_transition_params = SPC_STRATEGY_PARAMETERS["cluster_transition"].copy()
        regime_following_params = SPC_STRATEGY_PARAMETERS["regime_following"].copy()
        mean_reversion_params = SPC_STRATEGY_PARAMETERS["mean_reversion"].copy()

        # Override with args if provided (for command-line usage)
        if hasattr(self.args, "spc_min_signal_strength"):
            cluster_transition_params["min_signal_strength"] = self.args.spc_min_signal_strength
        if hasattr(self.args, "spc_min_rel_pos_change"):
            cluster_transition_params["min_rel_pos_change"] = self.args.spc_min_rel_pos_change
        if hasattr(self.args, "spc_min_regime_strength"):
            regime_following_params["min_regime_strength"] = self.args.spc_min_regime_strength
        if hasattr(self.args, "spc_min_cluster_duration"):
            regime_following_params["min_cluster_duration"] = self.args.spc_min_cluster_duration
        if hasattr(self.args, "spc_extreme_threshold"):
            mean_reversion_params["extreme_threshold"] = self.args.spc_extreme_threshold
        if hasattr(self.args, "spc_min_extreme_duration"):
            mean_reversion_params["min_extreme_duration"] = self.args.spc_min_extreme_duration

        return {
            "k": self.args.spc_k,
            "lookback": self.args.spc_lookback,
            "p_low": self.args.spc_p_low,
            "p_high": self.args.spc_p_high,
            "cluster_transition_params": cluster_transition_params,
            "regime_following_params": regime_following_params,
            "mean_reversion_params": mean_reversion_params,
            # Enhancement parameters (with CLI override support)
            "volatility_adjustment": volatility_adjustment,
            "use_correlation_weights": use_correlation_weights,
            "time_decay_factor": time_decay_factor,
            "interpolation_mode": interpolation_mode,
            "min_flip_duration": min_flip_duration,
            "flip_confidence_threshold": flip_confidence_threshold,
            # MTF parameters
            "enable_mtf": enable_mtf,
            "mtf_timeframes": mtf_timeframes,
            "mtf_require_alignment": mtf_require_alignment,
        }
