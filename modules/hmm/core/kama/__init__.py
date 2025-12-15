"""
HMM-KAMA Module.

This module provides HMM-KAMA (Kaufman Adaptive Moving Average based HMM) implementation
with Association Rule Mining and K-Means clustering for secondary analysis.
"""

# Models
from modules.hmm.core.kama.models import (
    HMM_KAMA,
    reorder_hmm_model,
    train_hmm,
    apply_hmm_model,
)

# Features
from modules.hmm.core.kama.features import (
    prepare_observations,
)

# Analysis
from modules.hmm.core.kama.analysis import (
    calculate_all_state_durations,
    compute_state_using_standard_deviation,
    compute_state_using_hmm,
    compute_state_using_association_rule_mining,
    calculate_composite_scores_association_rule_mining,
    compute_state_using_k_means,
)

# Workflow
from modules.hmm.core.kama.workflow import (
    hmm_kama,
)

# Strategy
from modules.hmm.core.kama.strategy import (
    KamaHMMStrategy,
)

# Utils
from modules.hmm.core.kama.utils import (
    prevent_infinite_loop,
    timeout_context,
)

__all__ = [
    # Models
    "HMM_KAMA",
    "reorder_hmm_model",
    "train_hmm",
    "apply_hmm_model",
    # Features
    "prepare_observations",
    # Analysis
    "calculate_all_state_durations",
    "compute_state_using_standard_deviation",
    "compute_state_using_hmm",
    "compute_state_using_association_rule_mining",
    "calculate_composite_scores_association_rule_mining",
    "compute_state_using_k_means",
    # Workflow
    "hmm_kama",
    # Strategy
    "KamaHMMStrategy",
    # Utils
    "prevent_infinite_loop",
    "timeout_context",
]

