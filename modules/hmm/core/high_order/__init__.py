
from config.hmm import (

"""
High-Order HMM Module.

This module provides True High-Order HMM implementation using state space expansion
for market state prediction based on k previous states.
"""

# State Expansion
# Constants
    HMM_HIGH_ORDER_N_BASE_STATES as N_BASE_STATES,
)
from config.hmm import (
    HMM_HIGH_ORDER_N_SYMBOLS as N_SYMBOLS,
)

# Model Creation
from modules.hmm.core.high_order.model_creation import (
    _map_observed_to_hidden_state,
    compute_emission_probabilities_from_data_high_order,
    compute_start_probabilities_from_data_high_order,
    compute_transition_matrix_from_data_high_order,
    create_high_order_hmm_model,
    train_model,
)

# Models
from modules.hmm.core.high_order.models import (
    TrueHighOrderHMM,
)

# Optimization
from modules.hmm.core.high_order.optimization import (
    _calculate_hmm_parameters,
    optimize_n_states_high_order,
    optimize_order_k,
)

# Prediction
from modules.hmm.core.high_order.prediction import (
    predict_next_hidden_state_forward_backward_high_order,
    predict_next_observation_high_order,
)
from modules.hmm.core.high_order.state_expansion import (
    decode_expanded_state,
    expand_state_sequence,
    get_expanded_state_count,
    map_expanded_to_base_state,
)

# Strategy
from modules.hmm.core.high_order.strategy import (
    TrueHighOrderHMMStrategy,
)

# Workflow
from modules.hmm.core.high_order.workflow import (
    true_high_order_hmm,
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
