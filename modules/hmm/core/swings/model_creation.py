
from typing import List, Optional

import numpy as np

from modules.common.utils import log_data, log_info, log_warn
from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM

"""
HMM-Swings Model Creation.

This module handles HMM model creation, training, and data-driven parameter computation.
"""





def _map_observed_to_hidden_state(observed_state: int, n_states: int, n_observed_states: int = 3) -> int:
    """
    Map observed state value (0, 1, 2) to hidden state index (0, ..., n_states-1).

    This handles the mismatch between observed states (always 0, 1, 2) and
    optimized hidden states (can be 2, 3, or more).

    Uses balanced mapping to avoid data imbalance:
    - Divides observed states into roughly equal groups
    - Avoids round() edge cases that can cause uneven distribution

    Args:
        observed_state: Observed state value (0=Down, 1=Side, 2=Up)
        n_states: Number of hidden states in the model
        n_observed_states: Number of possible observed states (default: 3)

    Returns:
        Hidden state index in range [0, n_states-1]
    """
    # Clamp observed_state to valid range
    observed_state = max(0, min(observed_state, n_observed_states - 1))

    if n_states == n_observed_states:
        # Direct mapping: 0->0, 1->1, 2->2
        return observed_state
    elif n_states == 2:
        # Map 3 observed states to 2 hidden states with balanced distribution:
        # 0 (Down) -> 0 (Bearish state)
        # 1 (Side) -> 0 (Bearish state, neutral can be bearish)
        # 2 (Up) -> 1 (Bullish state)
        # This gives State 0: 2 states, State 1: 1 state (acceptable for semantic grouping)
        return 0 if observed_state <= 1 else 1
    else:
        # Balanced mapping: divide observed states into roughly equal groups
        # Use integer division to avoid round() edge cases and ensure balanced distribution
        # Map [0, n_observed_states-1] to [0, n_states-1]

        # Strategy: Divide the observed state range into n_states equal segments
        # Each hidden state gets a roughly equal portion of observed states

        # Alternative: Use simple integer division with proper scaling
        # Map observed_state to hidden_state using: hidden = (observed * n_states) // n_observed_states
        # This ensures balanced distribution
        hidden_idx = (observed_state * n_states) // n_observed_states

        # Clamp to valid range (shouldn't be needed but safety check)
        return max(0, min(hidden_idx, n_states - 1))


def compute_transition_matrix_from_data(states: List[float], n_states: int) -> np.ndarray:
    """
    Compute transition matrix from historical state transitions.

    This creates a data-driven transition matrix by counting actual state transitions
    in the training data, making it adaptive to different market conditions.

    Args:
        states: List of observed states (0, 1, or 2)
        n_states: Number of hidden states in the model

    Returns:
        Transition matrix of shape (n_states, n_states)
    """
    if len(states) < 2:
        # Fallback to uniform transitions if insufficient data
        return np.ones((n_states, n_states), dtype=np.float32) / n_states

    # Initialize transition count matrix
    transition_counts = np.zeros((n_states, n_states), dtype=np.float32)

    # Count transitions between consecutive states
    for i in range(len(states) - 1):
        observed_from = int(states[i])
        observed_to = int(states[i + 1])

        # Map observed states to hidden states
        from_hidden = _map_observed_to_hidden_state(observed_from, n_states)
        to_hidden = _map_observed_to_hidden_state(observed_to, n_states)

        transition_counts[from_hidden, to_hidden] += 1

    # Convert counts to probabilities (add small smoothing to avoid zeros)
    smoothing = 0.01
    transition_matrix = transition_counts + smoothing

    # Normalize rows to sum to 1
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = transition_matrix / row_sums

    return transition_matrix.astype(np.float32)


def compute_emission_probabilities_from_data(states: List[float], n_states: int, n_symbols: int) -> List:
    """
    Compute emission probabilities from historical state-observation pairs.

    This creates data-driven emission probabilities by counting which observations
    (symbols) are most likely to be emitted from each hidden state.

    Even when n_states < n_symbols, we can still compute data-driven emissions by:
    - Mapping observed states to hidden states using _map_observed_to_hidden_state
    - Counting which symbols are actually emitted from each mapped hidden state
    - This preserves information from real data instead of using hardcoded values

    Args:
        states: List of observed states (0, 1, or 2) - these are the observations
        n_states: Number of hidden states (can be 2, 3, or more)
        n_symbols: Number of observable symbols (always 3: 0=Down, 1=Side, 2=Up)

    Returns:
        List of Categorical distributions for each hidden state, computed from actual data
    """
    if len(states) < n_states:
        # Fallback to uniform emissions if insufficient data
        log_warn(f"Insufficient data ({len(states)} states) for {n_states} hidden states. Using uniform emissions.")
        return [Categorical([[1 / n_symbols] * n_symbols]) for _ in range(n_states)]

    # Initialize emission count matrix: emission_counts[hidden_state][symbol] = count
    # Shape: (n_states, n_symbols) - hidden states can emit any of the 3 observable symbols
    emission_counts = np.zeros((n_states, n_symbols), dtype=np.float32)

    # Count emissions: map observed states to hidden states, then count which symbols are emitted
    # This works even when n_states < n_symbols by using the mapping function
    for observed_state_val in states:
        observed_state = int(observed_state_val)
        symbol_idx = observed_state  # The observed state value IS the observation symbol (0, 1, or 2)

        # Map observed state to hidden state (handles n_states < n_symbols case)
        hidden_state_idx = _map_observed_to_hidden_state(observed_state, n_states)

        # Ensure valid indices
        if 0 <= hidden_state_idx < n_states and 0 <= symbol_idx < n_symbols:
            emission_counts[hidden_state_idx, symbol_idx] += 1

    # Convert to probabilities with smoothing
    # Use adaptive smoothing: more smoothing when n_states < n_symbols to handle sparsity
    smoothing = 0.1 if n_states >= n_symbols else 0.2

    emission_probs = []

    for state_idx in range(n_states):
        state_emissions = emission_counts[state_idx] + smoothing

        # Check if this hidden state has any emissions (after mapping)
        if state_emissions.sum() == 0:
            # If no emissions mapped to this state, use uniform distribution
            state_emissions = np.ones(n_symbols, dtype=np.float32) / n_symbols
        else:
            # Normalize to sum to 1
            state_emissions = state_emissions / state_emissions.sum()

        emission_probs.append(Categorical([state_emissions.tolist()]))

    # Log information about the computed emissions
    if n_states < n_symbols:
        log_info(
            f"Computed data-driven emissions for n_states={n_states} < n_symbols={n_symbols} "
            f"using state mapping (from {len(states)} observations)"
        )

    return emission_probs


def compute_start_probabilities_from_data(states: List[float], n_states: int) -> np.ndarray:
    """
    Compute initial state probabilities from historical data.

    Args:
        states: List of observed states (0, 1, or 2)
        n_states: Number of hidden states

    Returns:
        Array of initial state probabilities
    """
    if len(states) == 0:
        # Fallback to uniform
        return np.ones(n_states, dtype=np.float32) / n_states

    # Count initial states (first state in sequence)
    start_counts = np.zeros(n_states, dtype=np.float32)
    if len(states) > 0:
        first_observed_state = int(states[0])
        # Map observed state to hidden state
        first_hidden_state = _map_observed_to_hidden_state(first_observed_state, n_states)
        start_counts[first_hidden_state] += 1

    # Add smoothing and normalize
    smoothing = 0.1
    start_probs = start_counts + smoothing
    start_probs = start_probs / start_probs.sum()

    return start_probs.astype(np.float32)


def create_hmm_model(n_symbols=3, n_states=2, states_data: Optional[List[float]] = None, use_data_driven: bool = True):
    """
    Create an HMM model with data-driven or hardcoded initialization.

    Args:
        n_symbols (int): Number of observable symbols (0, 1, 2)
        n_states (int): Number of hidden states
        states_data (Optional[List[float]]): Historical state sequence for data-driven initialization
        use_data_driven (bool): If True, compute parameters from data; if False, use hardcoded values

    Returns:
        DenseHMM: The configured HMM model.
    """
    if use_data_driven and states_data is not None and len(states_data) >= 2:
        # Data-driven initialization: compute from actual historical transitions
        try:
            transition_matrix = compute_transition_matrix_from_data(states_data, n_states)
            emission_distributions = compute_emission_probabilities_from_data(states_data, n_states, n_symbols)
            start_probs = compute_start_probabilities_from_data(states_data, n_states)
            end_probs = np.ones(n_states, dtype=np.float32) * 0.01  # Small end probabilities

            # Emission distributions are now always computed from data (even when n_states < n_symbols)
            log_info(f"Using data-driven HMM initialization (from {len(states_data)} states)")

            log_data(f"Transition matrix:\n{transition_matrix}")

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
        # Optimized configuration for 2 hidden states (hardcoded fallback)
        distributions = [
            Categorical([[0.25, 0.25, 0.50]]),  # Mixed trend, biased toward increase
            Categorical([[0.50, 0.25, 0.25]]),  # Mixed trend, biased toward decrease
        ]
        edges = [[0.85, 0.15], [0.15, 0.85]]
        starts = [0.5, 0.5]
        ends = [0.01, 0.01]
    else:
        # Configuration for custom number of hidden states
        distributions = [Categorical([[1 / n_symbols] * n_symbols]) for _ in range(n_states)]
        edges = np.ones((n_states, n_states), dtype=np.float32) / n_states
        starts = np.ones(n_states, dtype=np.float32) / n_states
        ends = np.ones(n_states, dtype=np.float32) * 0.01

    if not use_data_driven:
        log_info("Using hardcoded HMM initialization (fallback mode)")

    return DenseHMM(distributions, edges=edges, starts=starts, ends=ends, verbose=False)


def train_model(model, observations):
    """
    Train the HMM model with observation data.

    Args:
        model (DenseHMM): The HMM model to be trained.
        observations (list): List of observation arrays.

    Returns:
        DenseHMM: The trained HMM model.
    """
    model.fit(observations)
    return model
