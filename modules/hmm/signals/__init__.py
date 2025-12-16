"""
Signal processing and combination module.

This module handles signal processing, scoring, confidence calculation,
and signal resolution for HMM models.
"""

from modules.hmm.signals.combiner import combine_signals, HMMSignalCombiner
from modules.hmm.signals.confidence import (
    calculate_kama_confidence,
    calculate_combined_confidence,
)
from modules.hmm.signals.scoring import normalize_scores
from modules.hmm.signals.resolution import (
    calculate_dynamic_threshold,
    resolve_signal_conflict,
    resolve_multi_strategy_conflicts,
    Signal,
    LONG,
    HOLD,
    SHORT,
)
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

