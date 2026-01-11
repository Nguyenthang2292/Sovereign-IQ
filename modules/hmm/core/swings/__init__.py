
from modules.hmm.core.swings.model_creation import (

"""
HMM-Swings Module.

This module provides Basic HMM with swing detection implementation for market state prediction.
"""

# Models
# Model Creation
    _map_observed_to_hidden_state,
    compute_emission_probabilities_from_data,
    compute_start_probabilities_from_data,
    compute_transition_matrix_from_data,
    create_hmm_model,
    train_model,
)
from modules.hmm.core.swings.models import (
    BEARISH,
    BULLISH,
    HMM_SWINGS,
    NEUTRAL,
    SwingsHMM,
)

# Optimization
from modules.hmm.core.swings.optimization import (
    _calculate_hmm_parameters,
    optimize_n_states,
)

# Prediction
from modules.hmm.core.swings.prediction import (
    evaluate_model_accuracy,
    predict_next_hidden_state_forward_backward,
    predict_next_observation,
)

# State Conversion
from modules.hmm.core.swings.state_conversion import (
    convert_swing_to_state,
)

# Strategy
from modules.hmm.core.swings.strategy import (
    SwingsHMMStrategy,
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

__all__ = [
    # Models
    "HMM_SWINGS",
    "SwingsHMM",
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
