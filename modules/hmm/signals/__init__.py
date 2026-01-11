
from modules.hmm.signals.combiner import HMMSignalCombiner, combine_signals
from modules.hmm.signals.confidence import (

"""
Signal processing and combination module.

This module handles signal processing, scoring, confidence calculation,
and signal resolution for HMM models.
"""

    calculate_combined_confidence,
    calculate_kama_confidence,
)
from modules.hmm.signals.resolution import (
    HOLD,
    LONG,
    SHORT,
    Signal,
    calculate_dynamic_threshold,
    resolve_multi_strategy_conflicts,
    resolve_signal_conflict,
)
from modules.hmm.signals.scoring import normalize_scores
from modules.hmm.signals.utils import (
    validate_dataframe,
)

__all__ = [
    "combine_signals",
    "HMMSignalCombiner",
    "calculate_kama_confidence",
    "calculate_combined_confidence",
    "normalize_scores",
    "calculate_dynamic_threshold",
    "resolve_signal_conflict",
    "resolve_multi_strategy_conflicts",
    "Signal",
    "LONG",
    "HOLD",
    "SHORT",
    "validate_dataframe",
]
