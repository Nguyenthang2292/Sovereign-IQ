"""
High-Order HMM Model Creation.

This module handles HMM model creation, training, and data-driven parameter computation for high-order HMM.
"""

from typing import List, Optional

import numpy as np
from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM

from config.hmm import HMM_HIGH_ORDER_N_BASE_STATES as N_BASE_STATES
from modules.common.utils import log_info, log_warn
from modules.hmm.core.high_order.state_expansion import (
    expand_state_sequence,
    get_expanded_state_count,
)


def _map_observed_to_hidden_state(observed_state: int, n_states: int, n_observed_states: int = 3) -> int:
    """
    Map observed state value (0, 1, 2) to hidden state index (0, ..., n_states-1).

    This handles the mismatch between observed states (always 0, 1, 2) and
    optimized hidden states (can be 2, 3, or more).

    Args:
        observed_state: Observed state value (0=Down, 1=Side, 2=Up)
        n_states: Number of hidden states in the model
        n_observed_states: Number of possible observed states (default: 3)

    Returns:
        Hidden state index in range [0, n_states-1]
    """
    observed_state = max(0, min(observed_state, n_observed_states - 1))

    if n_states == n_observed_states:
        return observed_state
    elif n_states == 2:
        return 0 if observed_state <= 1 else 1
    else:
        hidden_idx = (observed_state * n_states) // n_observed_states
        return max(0, min(hidden_idx, n_states - 1))


# N_BASE_STATES imported from config.hmm


def compute_transition_matrix_from_data_high_order(
    states: List[float], n_states: int, order: int = 1, n_base_states: int = N_BASE_STATES
) -> np.ndarray:
    """
    Compute transition matrix from historical state transitions with order support.

    For order > 1, uses state space expansion to count transitions between
    expanded states (sequences of k base states).

    Args:
        states: List of observed states (0, 1, or 2)
        n_states: Number of hidden states in the model
        order: Order of the HMM (k)
        n_base_states: Number of base states (default: 3)

    Returns:
        Transition matrix of shape (n_states, n_states)
    """
    if len(states) < 2:
        return np.ones((n_states, n_states), dtype=np.float32) / n_states

    if order > 1:
        # Use state space expansion
        expanded_states = expand_state_sequence(states, order, n_base_states)

        if len(expanded_states) < 2:
            return np.ones((n_states, n_states), dtype=np.float32) / n_states

        # Initialize transition count matrix for expanded states
        transition_counts = np.zeros((n_states, n_states), dtype=np.float32)

        # Count transitions between consecutive expanded states
        # NOTE: Expanded states are clamped to [0, n_states-1] to handle edge cases
        #       where expanded state index might exceed n_states (shouldn't happen in normal cases)
        for i in range(len(expanded_states) - 1):
            from_expanded = expanded_states[i]
            to_expanded = expanded_states[i + 1]

            # Map expanded states to hidden states if needed
            # For high-order HMM, n_states = n_base_states^order
            # So we can use expanded states directly as hidden states
            # However, expanded states can exceed n_states if the sequence is long
            # We need to clamp them to valid range [0, n_states-1]
            from_expanded = max(0, min(from_expanded, n_states - 1))
            to_expanded = max(0, min(to_expanded, n_states - 1))

            transition_counts[from_expanded, to_expanded] += 1
    else:
        # Order = 1: use original logic
        transition_counts = np.zeros((n_states, n_states), dtype=np.float32)

        for i in range(len(states) - 1):
            observed_from = int(states[i])
            observed_to = int(states[i + 1])

            from_hidden = _map_observed_to_hidden_state(observed_from, n_states)
            to_hidden = _map_observed_to_hidden_state(observed_to, n_states)

            transition_counts[from_hidden, to_hidden] += 1

    # Convert counts to probabilities (add small smoothing to avoid zeros)
    smoothing = 0.01
    transition_matrix = transition_counts + smoothing

    # Normalize rows to sum to 1
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_matrix / row_sums

    return transition_matrix.astype(np.float32)


def compute_emission_probabilities_from_data_high_order(
    states: List[float], n_states: int, n_symbols: int, order: int = 1, n_base_states: int = N_BASE_STATES
) -> List:
    """
    Compute emission probabilities from historical state-observation pairs with order support.

    For order > 1, each expanded state emits symbols based on the last state
    in its sequence.

    Args:
        states: List of observed states (0, 1, or 2) - these are the observations
        n_states: Number of hidden states (can be expanded states when order > 1)
        n_symbols: Number of observable symbols (always 3: 0=Down, 1=Side, 2=Up)
        order: Order of the HMM (k)
        n_base_states: Number of base states (default: 3)

    Returns:
        List of Categorical distributions for each hidden state
    """
    if len(states) < n_states:
        log_warn(f"Insufficient data ({len(states)} states) for {n_states} hidden states. Using uniform emissions.")
        return [Categorical([[1 / n_symbols] * n_symbols]) for _ in range(n_states)]

    emission_counts = np.zeros((n_states, n_symbols), dtype=np.float32)

    if order > 1:
        # Use state space expansion
        expanded_states = expand_state_sequence(states, order, n_base_states)

        # For each expanded state, emit symbol based on the last state in sequence
        # NOTE: Expanded states are clamped to [0, n_states-1] to prevent index out of bounds
        for i, expanded_state in enumerate(expanded_states):
            # The observation is the state at position i + order - 1 (last in sequence)
            if i + order - 1 < len(states):
                symbol_idx = int(states[i + order - 1])
                symbol_idx = max(0, min(symbol_idx, n_symbols - 1))

                # Clamp expanded_state to valid range [0, n_states-1]
                expanded_state_clamped = max(0, min(expanded_state, n_states - 1))

                if 0 <= expanded_state_clamped < n_states and 0 <= symbol_idx < n_symbols:
                    emission_counts[expanded_state_clamped, symbol_idx] += 1
    else:
        # Order = 1: use original logic
        for observed_state_val in states:
            observed_state = int(observed_state_val)
            symbol_idx = observed_state

            hidden_state_idx = _map_observed_to_hidden_state(observed_state, n_states)

            if 0 <= hidden_state_idx < n_states and 0 <= symbol_idx < n_symbols:
                emission_counts[hidden_state_idx, symbol_idx] += 1

    # Convert to probabilities with smoothing
    smoothing = 0.1 if n_states >= n_symbols else 0.2

    emission_probs = []
    for state_idx in range(n_states):
        state_emissions = emission_counts[state_idx] + smoothing

        if state_emissions.sum() == 0:
            state_emissions = np.ones(n_symbols, dtype=np.float32) / n_symbols
        else:
            state_emissions = state_emissions / state_emissions.sum()

        emission_probs.append(Categorical([state_emissions.tolist()]))

    if order > 1:
        log_info(
            f"Computed data-driven emissions for order={order} HMM "
            f"with n_states={n_states} (from {len(states)} observations)"
        )

    return emission_probs


def compute_start_probabilities_from_data_high_order(
    states: List[float], n_states: int, order: int = 1, n_base_states: int = N_BASE_STATES
) -> np.ndarray:
    """
    Compute initial state probabilities from historical data with order support.

    Args:
        states: List of observed states (0, 1, or 2)
        n_states: Number of hidden states
        order: Order of the HMM (k)
        n_base_states: Number of base states (default: 3)

    Returns:
        Array of initial state probabilities
    """
    if len(states) == 0:
        return np.ones(n_states, dtype=np.float32) / n_states

    start_counts = np.zeros(n_states, dtype=np.float32)

    if order > 1:
        # Use state space expansion
        expanded_states = expand_state_sequence(states, order, n_base_states)
        if len(expanded_states) > 0:
            first_expanded = expanded_states[0]
            if 0 <= first_expanded < n_states:
                start_counts[first_expanded] += 1
    else:
        # Order = 1: use original logic
        first_observed_state = int(states[0])
        first_hidden_state = _map_observed_to_hidden_state(first_observed_state, n_states)
        start_counts[first_hidden_state] += 1

    # Add smoothing and normalize
    smoothing = 0.1
    start_probs = start_counts + smoothing
    start_probs = start_probs / start_probs.sum()

    return start_probs.astype(np.float32)


def create_high_order_hmm_model(
    n_symbols: int = 3,
    n_states: int = 2,
    order: int = 1,
    states_data: Optional[List[float]] = None,
    use_data_driven: bool = True,
    n_base_states: int = N_BASE_STATES,
) -> DenseHMM:
    """
    Create an HMM model with order support using state space expansion.

    When order > 1, uses state space expansion to create expanded states.
    The actual number of states in the model will be n_base_states^order.

    Args:
        n_symbols: Number of observable symbols (0, 1, 2)
        n_states: Number of base hidden states (when order=1) or expanded states (when order>1)
        order: Order of the HMM (k)
        states_data: Historical state sequence for data-driven initialization
        use_data_driven: If True, compute parameters from data; if False, use hardcoded values
        n_base_states: Number of base states (default: 3)

    Returns:
        DenseHMM: The configured HMM model
    """
    # FIX: Validate n_states matches expected expanded state count
    # Issue: When order > 1, n_states must equal n_base_states^order for correct model creation
    #        Mismatch can cause shape errors in transition/emission matrices
    # Solution: Check and correct n_states to match expected_expanded_states
    # When order > 1, n_states should already be the expanded state count
    # But we need to ensure it matches
    if order > 1:
        expected_expanded_states = get_expanded_state_count(n_base_states, order)

        if n_states != expected_expanded_states:
            log_warn(
                f"n_states ({n_states}) doesn't match expected expanded states "
                f"({expected_expanded_states}) for order={order}. Using expected value."
            )
            n_states = expected_expanded_states

    if use_data_driven and states_data is not None and len(states_data) >= 2:
        try:
            transition_matrix = compute_transition_matrix_from_data_high_order(
                states_data, n_states, order, n_base_states
            )
            emission_distributions = compute_emission_probabilities_from_data_high_order(
                states_data, n_states, n_symbols, order, n_base_states
            )
            start_probs = compute_start_probabilities_from_data_high_order(states_data, n_states, order, n_base_states)
            end_probs = np.ones(n_states, dtype=np.float32) * 0.01

            log_info(
                f"Using data-driven High-Order HMM initialization "
                f"(order={order}, n_states={n_states}, from {len(states_data)} states)"
            )

            return DenseHMM(
                emission_distributions,
                edges=transition_matrix.tolist(),
                starts=start_probs.tolist(),
                ends=end_probs.tolist(),
                verbose=False,
            )
        except Exception as e:
            log_warn(f"Data-driven initialization failed: {e}. Falling back to hardcoded values.")
            use_data_driven = False

    # Fallback to hardcoded initialization
    if n_states == 2:
        distributions = [Categorical([[0.25, 0.25, 0.50]]), Categorical([[0.50, 0.25, 0.25]])]
        edges = [[0.85, 0.15], [0.15, 0.85]]
        starts = [0.5, 0.5]
        ends = [0.01, 0.01]
    else:
        distributions = [Categorical([[1 / n_symbols] * n_symbols]) for _ in range(n_states)]
        edges = np.ones((n_states, n_states), dtype=np.float32) / n_states
        starts = np.ones(n_states, dtype=np.float32) / n_states
        ends = np.ones(n_states, dtype=np.float32) * 0.01

    if not use_data_driven:
        log_info(f"Using hardcoded HMM initialization (order={order}, n_states={n_states})")

    return DenseHMM(distributions, edges=edges, starts=starts, ends=ends, verbose=False)


def train_model(model: DenseHMM, observations: List) -> DenseHMM:
    """
    Train the HMM model with observation data.

    Args:
        model: The HMM model to be trained
        observations: List of observation arrays

    Returns:
        The trained HMM model
    """
    model.fit(observations)
    return model
