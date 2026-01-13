"""
HMM (Hidden Markov Model) module.

This module provides Hidden Markov Model implementations for cryptocurrency trading:
- Basic HMM with swing detection
- High-Order HMM
- HMM-KAMA
- Signal processing and combination
"""

from modules.hmm.core.high_order import (
    # Core HMM models
    TrueHighOrderHMM,
    decode_expanded_state,
    expand_state_sequence,
    get_expanded_state_count,
    map_expanded_to_base_state,
    true_high_order_hmm,
)
from modules.hmm.core.kama import (
    HMM_KAMA,
    hmm_kama,
    prepare_observations,
)
from modules.hmm.core.swings import (
    BEARISH,
    BULLISH,
    HMM_SWINGS,
    NEUTRAL,
    SwingsHMM,
    average_swing_distance,
    convert_swing_to_state,
    hmm_swings,
    safe_forward_backward,
)

# Signal processing
from modules.hmm.signals.combiner import HMMSignalCombiner, combine_signals
from modules.hmm.signals.confidence import (
    calculate_combined_confidence,
    calculate_kama_confidence,
)
from modules.hmm.signals.resolution import (
    HOLD,
    LONG,
    SHORT,
    Signal,
    calculate_dynamic_threshold,
    resolve_signal_conflict,
)
from modules.hmm.signals.scoring import normalize_scores
from modules.hmm.signals.utils import (
    validate_dataframe,
)

__all__ = [
    # Core HMM models - Basic
    "hmm_swings",
    "SwingsHMM",
    "HMM_SWINGS",
    "BULLISH",
    "NEUTRAL",
    "BEARISH",
    "convert_swing_to_state",
    "average_swing_distance",
    "safe_forward_backward",
    # Core HMM models - High-Order
    "true_high_order_hmm",
    "TrueHighOrderHMM",
    "get_expanded_state_count",
    "expand_state_sequence",
    "decode_expanded_state",
    "map_expanded_to_base_state",
    # Core HMM models - KAMA
    "hmm_kama",
    "HMM_KAMA",
    "prepare_observations",
    # Signal processing
    "combine_signals",
    "HMMSignalCombiner",
    "calculate_kama_confidence",
    "calculate_combined_confidence",
    "normalize_scores",
    "calculate_dynamic_threshold",
    "resolve_signal_conflict",
    "Signal",
    "LONG",
    "HOLD",
    "SHORT",
    "validate_dataframe",
]
