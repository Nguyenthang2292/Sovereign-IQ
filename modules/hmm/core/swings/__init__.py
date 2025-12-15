"""
HMM-Swings Module.

This module provides Basic HMM with swing detection implementation for market state prediction.
"""

# Models
from modules.hmm.core.swings.models import (
    HMM_SWINGS,
    HighOrderHMM,
    BULLISH,
    NEUTRAL,
    BEARISH,
)

# State Conversion
from modules.hmm.core.swings.state_conversion import (
    convert_swing_to_state,
)

# Optimization
from modules.hmm.core.swings.optimization import (
    optimize_n_states,
    _calculate_hmm_parameters,
)

# Model Creation
from modules.hmm.core.swings.model_creation import (
    create_hmm_model,
    train_model,
    compute_transition_matrix_from_data,
    compute_emission_probabilities_from_data,
    compute_start_probabilities_from_data,
    _map_observed_to_hidden_state,
)

# Prediction
from modules.hmm.core.swings.prediction import (
    predict_next_hidden_state_forward_backward,
    predict_next_observation,
    evaluate_model_accuracy,
)

# Swing Utils
from modules.hmm.core.swings.swing_utils import (
    average_swing_distance,
    safe_forward_backward,
    timeout,
)

# Workflow
from modules.hmm.core.swings.workflow import (
    hmm_swings,
)

# Strategy
from modules.hmm.core.swings.strategy import (
    SwingsHMMStrategy,
)

__all__ = [
    # Models
    "HMM_SWINGS",
    "HighOrderHMM",
    "BULLISH",
    "NEUTRAL",
    "BEARISH",
    # State Conversion
    "convert_swing_to_state",
    # Optimization
    "optimize_n_states",
    "_calculate_hmm_parameters",
    # Model Creation
    "create_hmm_model",
    "train_model",
    "compute_transition_matrix_from_data",
    "compute_emission_probabilities_from_data",
    "compute_start_probabilities_from_data",
    "_map_observed_to_hidden_state",
    # Prediction
    "predict_next_hidden_state_forward_backward",
    "predict_next_observation",
    "evaluate_model_accuracy",
    # Swing Utils
    "average_swing_distance",
    "safe_forward_backward",
    "timeout",
    # Workflow
    "hmm_swings",
    # Strategy
    "SwingsHMMStrategy",
]

