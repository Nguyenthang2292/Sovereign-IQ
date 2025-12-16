"""
Core HMM models module.

This module contains the core Hidden Markov Model implementations:
- Basic HMM with swing detection
- High-Order HMM
- HMM-KAMA
"""

# Basic HMM with swings
from modules.hmm.core.swings import (
    hmm_swings,
    SwingsHMM,
    HMM_SWINGS,
    BULLISH,
    NEUTRAL,
    BEARISH,
    convert_swing_to_state,
    average_swing_distance,
    safe_forward_backward,
)

# High-Order HMM
from modules.hmm.core.high_order import (
    true_high_order_hmm,
    TrueHighOrderHMM,
    get_expanded_state_count,
    expand_state_sequence,
    decode_expanded_state,
    map_expanded_to_base_state,
)

# HMM-KAMA
from modules.hmm.core.kama import (
    hmm_kama,
    HMM_KAMA,
    prepare_observations,
)

__all__ = [
    # Basic HMM
    "hmm_swings",
    "SwingsHMM",
    "HMM_SWINGS",
    "BULLISH",
    "NEUTRAL",
    "BEARISH",
    "convert_swing_to_state",
    "average_swing_distance",
    "safe_forward_backward",
    # High-Order HMM
    "true_high_order_hmm",
    "TrueHighOrderHMM",
    "get_expanded_state_count",
    "expand_state_sequence",
    "decode_expanded_state",
    "map_expanded_to_base_state",
    # HMM-KAMA
    "hmm_kama",
    "HMM_KAMA",
    "prepare_observations",
]

