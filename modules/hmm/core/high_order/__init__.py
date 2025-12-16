"""
High-Order HMM Module.

This module provides True High-Order HMM implementation using state space expansion
for market state prediction based on k previous states.
"""

# State Expansion
from modules.hmm.core.high_order.state_expansion import (
    get_expanded_state_count,
    expand_state_sequence,
    decode_expanded_state,
    map_expanded_to_base_state,
)

# Constants
from config.hmm import (
    HMM_HIGH_ORDER_N_BASE_STATES as N_BASE_STATES,
    HMM_HIGH_ORDER_N_SYMBOLS as N_SYMBOLS,
)

# Models
from modules.hmm.core.high_order.models import (
    TrueHighOrderHMM,
)

# Model Creation
from modules.hmm.core.high_order.model_creation import (
    create_high_order_hmm_model,
    train_model,
    compute_transition_matrix_from_data_high_order,
    compute_emission_probabilities_from_data_high_order,
    compute_start_probabilities_from_data_high_order,
    _map_observed_to_hidden_state,
)

# Optimization
from modules.hmm.core.high_order.optimization import (
    optimize_n_states_high_order,
    optimize_order_k,
    _calculate_hmm_parameters,
)

# Prediction
from modules.hmm.core.high_order.prediction import (
    predict_next_hidden_state_forward_backward_high_order,
    predict_next_observation_high_order,
)

# Workflow
from modules.hmm.core.high_order.workflow import (
    true_high_order_hmm,
)

# Strategy
from modules.hmm.core.high_order.strategy import (
    TrueHighOrderHMMStrategy,
)

__all__ = [
    # State Expansion
    "get_expanded_state_count",
    "expand_state_sequence",
    "decode_expanded_state",
    "map_expanded_to_base_state",
    # Models
    "TrueHighOrderHMM",
    "N_BASE_STATES",
    "N_SYMBOLS",
    # Model Creation
    "create_high_order_hmm_model",
    "train_model",
    "compute_transition_matrix_from_data_high_order",
    "compute_emission_probabilities_from_data_high_order",
    "compute_start_probabilities_from_data_high_order",
    # Optimization
    "optimize_n_states_high_order",
    "optimize_order_k",
    "_calculate_hmm_parameters",
    "_map_observed_to_hidden_state",  # From model_creation
    # Prediction
    "predict_next_hidden_state_forward_backward_high_order",
    "predict_next_observation_high_order",
    # Workflow
    "true_high_order_hmm",
    # Strategy
    "TrueHighOrderHMMStrategy",
]

