"""
High-Order Hidden Markov Model Implementation.

This module implements a true High-Order HMM using state space expansion,
allowing predictions based on k previous states instead of just one.
"""

import numpy as np
import pandas as pd
import warnings
from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM
from scipy.signal import argrelextrema
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.hmm.signals.strategy import HMMStrategyResult

from modules.common.utils import log_info, log_warn, log_error
from modules.hmm.core.swings import (
    convert_swing_to_state,
    average_swing_distance,
    BULLISH,
    NEUTRAL,
    BEARISH,
    HMM_SWINGS,
    safe_forward_backward,
)
from config import (
    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
    HMM_HIGH_ORDER_USE_DATA_DRIVEN_INIT,
    HMM_HIGH_ORDER_MIN_ORDER_DEFAULT,
    HMM_HIGH_ORDER_MAX_ORDER_DEFAULT,
)

# Base number of states (0=Down, 1=Side, 2=Up)
N_BASE_STATES = 3
N_SYMBOLS = 3

# ============================================================================
# State Space Expansion Functions
# ============================================================================

def get_expanded_state_count(n_base_states: int, order: int) -> int:
    """
    Calculate the number of expanded states for a given order.
    
    With order k, we expand from n_base_states to n_base_states^k states.
    Each expanded state represents a sequence of k base states.
    
    Args:
        n_base_states: Number of base states (typically 3: 0, 1, 2)
        order: Order of the HMM (k)
        
    Returns:
        Number of expanded states = n_base_states^order
    """
    return n_base_states ** order

def expand_state_sequence(states: List[float], order: int, n_base_states: int = N_BASE_STATES) -> List[int]:
    """
    Convert a sequence of base states to expanded states using state space expansion.
    
    For order k, each expanded state represents a sequence of k consecutive base states.
    Example with order=2, n_base_states=3:
        Base states: [0, 1, 2]
        Expanded: [0*3^1 + 0*3^0, 0*3^1 + 1*3^0, 1*3^1 + 2*3^0] = [0, 1, 5]
        Which represents: [(0,0), (0,1), (1,2)]
    
    Args:
        states: List of base state values (0, 1, or 2)
        order: Order of the HMM (k)
        n_base_states: Number of base states (default: 3)
        
    Returns:
        List of expanded state indices
    """
    if len(states) < order:
        log_warn(f"Sequence length ({len(states)}) is less than order ({order}). Returning empty list.")
        return []
    
    expanded = []
    for i in range(len(states) - order + 1):
        # Get sequence of k states
        sequence = states[i:i+order]
        # Convert to expanded state index
        expanded_state = 0
        for j, state in enumerate(sequence):
            state_int = int(state)
            # Ensure state is in valid range
            state_int = max(0, min(state_int, n_base_states - 1))
            expanded_state += state_int * (n_base_states ** (order - 1 - j))
        expanded.append(expanded_state)
    
    return expanded

def decode_expanded_state(expanded_state: int, order: int, n_base_states: int = N_BASE_STATES) -> Tuple[int, ...]:
    """
    Decode an expanded state back to its constituent base states.
    
    Args:
        expanded_state: The expanded state index
        order: Order of the HMM (k)
        n_base_states: Number of base states (default: 3)
        
    Returns:
        Tuple of k base states
    """
    states = []
    remaining = expanded_state
    for i in range(order):
        power = n_base_states ** (order - 1 - i)
        state = remaining // power
        states.append(state)
        remaining = remaining % power
    return tuple(states)

def map_expanded_to_base_state(expanded_state: int, order: int, n_base_states: int = N_BASE_STATES) -> int:
    """
    Map an expanded state to its corresponding base state.
    
    For prediction purposes, we take the last state in the sequence,
    as that represents the current state we're transitioning from.
    
    Args:
        expanded_state: The expanded state index
        order: Order of the HMM (k)
        n_base_states: Number of base states (default: 3)
        
    Returns:
        Base state value (0, 1, or 2)
    """
    decoded = decode_expanded_state(expanded_state, order, n_base_states)
    # Return the last state in the sequence (most recent)
    return decoded[-1]

# ============================================================================
# High-Order HMM Parameter Computation
# ============================================================================

def _calculate_hmm_parameters(n_states: int, n_symbols: int = 3) -> int:
    """
    Calculate the number of free parameters in an HMM model.
    
    Args:
        n_states: Number of hidden states
        n_symbols: Number of observable symbols (default: 3)
        
    Returns:
        Number of free parameters
    """
    transition_params = n_states * (n_states - 1)
    emission_params = n_states * (n_symbols - 1)
    start_params = n_states - 1
    end_params = n_states - 1
    
    return transition_params + emission_params + start_params + end_params

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

def compute_transition_matrix_from_data_high_order(
    states: List[float], 
    n_states: int, 
    order: int = 1, 
    n_base_states: int = N_BASE_STATES
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
        for i in range(len(expanded_states) - 1):
            from_expanded = expanded_states[i]
            to_expanded = expanded_states[i + 1]
            
            # Map expanded states to hidden states if needed
            # For high-order HMM, n_states = n_base_states^order
            # So we can use expanded states directly as hidden states
            if from_expanded < n_states and to_expanded < n_states:
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
    states: List[float], 
    n_states: int, 
    n_symbols: int, 
    order: int = 1, 
    n_base_states: int = N_BASE_STATES
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
        return [Categorical([[1/n_symbols] * n_symbols]) for _ in range(n_states)]
    
    emission_counts = np.zeros((n_states, n_symbols), dtype=np.float32)
    
    if order > 1:
        # Use state space expansion
        expanded_states = expand_state_sequence(states, order, n_base_states)
        
        # For each expanded state, emit symbol based on the last state in sequence
        for i, expanded_state in enumerate(expanded_states):
            # The observation is the state at position i + order - 1 (last in sequence)
            if i + order - 1 < len(states):
                symbol_idx = int(states[i + order - 1])
                symbol_idx = max(0, min(symbol_idx, n_symbols - 1))
                
                if 0 <= expanded_state < n_states and 0 <= symbol_idx < n_symbols:
                    emission_counts[expanded_state, symbol_idx] += 1
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
    states: List[float], 
    n_states: int, 
    order: int = 1, 
    n_base_states: int = N_BASE_STATES
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

# ============================================================================
# High-Order HMM Model Creation and Optimization
# ============================================================================

def create_high_order_hmm_model(
    n_symbols: int = 3, 
    n_states: int = 2, 
    order: int = 1, 
    states_data: Optional[List[float]] = None, 
    use_data_driven: bool = True,
    n_base_states: int = N_BASE_STATES
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
            start_probs = compute_start_probabilities_from_data_high_order(
                states_data, n_states, order, n_base_states
            )
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
                verbose=False
            )
        except Exception as e:
            log_warn(f"Data-driven initialization failed: {e}. Falling back to hardcoded values.")
            use_data_driven = False
    
    # Fallback to hardcoded initialization
    if n_states == 2:
        distributions = [
            Categorical([[0.25, 0.25, 0.50]]),
            Categorical([[0.50, 0.25, 0.25]])
        ]
        edges = [[0.85, 0.15], [0.15, 0.85]]
        starts = [0.5, 0.5]
        ends = [0.01, 0.01]
    else:
        distributions = [Categorical([[1/n_symbols] * n_symbols]) for _ in range(n_states)]
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

def optimize_n_states_high_order(
    observations, 
    order: int = 1, 
    min_states: int = 2, 
    max_states: int = 10, 
    n_folds: int = 3, 
    use_bic: bool = True,
    n_base_states: int = N_BASE_STATES
) -> int:
    """
    Optimize number of states for High-Order HMM using TimeSeriesSplit cross-validation.
    
    When order > 1, optimizes the number of base states, then expands to n_base_states^order.
    
    Args:
        observations: A list containing a single observation sequence (2D array)
        order: Order of the HMM (k)
        min_states: Minimum number of base hidden states
        max_states: Maximum number of base hidden states
        n_folds: Number of folds in cross-validation
        use_bic: If True, use BIC for model selection
        n_base_states: Number of base states (default: 3)
        
    Returns:
        The best number of hidden states (expanded if order > 1)
    """
    if len(observations) != 1:
        raise ValueError("Expected a single observation sequence.")
    
    seq = observations[0]
    seq_length = len(seq)
    n_symbols = 3
    
    if seq_length < n_folds * 2:
        raise ValueError(f"Sequence length ({seq_length}) too short for {n_folds} folds.")
    
    if order > 1:
        # For high-order HMM, we optimize base states, then expand
        # But actually, with state space expansion, n_states = n_base_states^order
        # So we need to optimize differently - we optimize the base state count
        # But this is complex, so for now we'll use a fixed expansion
        # The actual optimization should be done in optimize_order_k
        expanded_states = get_expanded_state_count(n_base_states, order)
        log_info(f"For order={order}, using expanded states={expanded_states}")
        return expanded_states
    
    # For order=1, use standard optimization
    best_n_states = min_states
    best_score = np.inf if use_bic else -np.inf
    
    tscv = TimeSeriesSplit(n_splits=n_folds)
    indices = np.arange(seq_length)
    
    for n_states in range(min_states, max_states + 1):
        log_likelihoods = []
        test_sizes = []
        
        for train_idx, test_idx in tscv.split(indices):
            train_seq = seq[train_idx]
            test_seq = seq[test_idx]
            
            if len(train_seq) == 0 or len(test_seq) == 0:
                continue
            
            model = create_high_order_hmm_model(
                n_symbols=n_symbols, 
                n_states=n_states, 
                order=order,
                states_data=train_seq.tolist() if len(train_seq) > 0 else None,
                use_data_driven=True
            )
            model = train_model(model, [train_seq])
            
            try:
                log_likelihood = model.log_probability([test_seq])
                log_likelihoods.append(log_likelihood)
                test_sizes.append(len(test_seq))
            except Exception as e:
                log_warn(f"Error in log_probability: {type(e).__name__}: {e}")
        
        if log_likelihoods:
            avg_log_likelihood = np.mean(log_likelihoods)
            total_test_samples = sum(test_sizes)
            
            if use_bic:
                k = _calculate_hmm_parameters(n_states, n_symbols)
                bic = -2 * avg_log_likelihood + k * np.log(total_test_samples)
                
                if bic < best_score:
                    best_score = bic
                    best_n_states = n_states
                    log_info(f"New best n_states={n_states} with BIC={bic:.2f}")
            else:
                if avg_log_likelihood > best_score:
                    best_score = avg_log_likelihood
                    best_n_states = n_states
    
    if use_bic:
        log_info(f"Selected n_states={best_n_states} with best BIC={best_score:.2f}")
    else:
        log_info(f"Selected n_states={best_n_states} with best log-likelihood={best_score:.2f}")
    
    return best_n_states

def optimize_order_k(
    observations,
    min_order: int = 2,
    max_order: int = 4,
    n_folds: int = 3,
    use_bic: bool = True,
    min_states: int = 2,
    max_states: int = 10,
    n_base_states: int = N_BASE_STATES
) -> Tuple[int, int]:
    """
    Optimize both order k and number of states for High-Order HMM.
    
    Tries different combinations of (order, n_states) and selects the best
    based on BIC or log-likelihood.
    
    Args:
        observations: A list containing a single observation sequence (2D array)
        min_order: Minimum order to try
        max_order: Maximum order to try
        n_folds: Number of folds in cross-validation
        use_bic: If True, use BIC for model selection
        min_states: Minimum number of base states (only used when order=1)
        max_states: Maximum number of base states (only used when order=1)
        n_base_states: Number of base states (default: 3)
        
    Returns:
        Tuple of (best_order, best_n_states)
    """
    if len(observations) != 1:
        raise ValueError("Expected a single observation sequence.")
    
    seq = observations[0]
    seq_length = len(seq)
    n_symbols = 3
    
    if seq_length < n_folds * 2:
        raise ValueError(f"Sequence length ({seq_length}) too short for {n_folds} folds.")
    
    best_order = 1
    best_n_states = min_states
    best_score = np.inf if use_bic else -np.inf
    
    tscv = TimeSeriesSplit(n_splits=n_folds)
    indices = np.arange(seq_length)
    
    # Try each order
    for order in range(min_order, max_order + 1):
        # For order > 1, n_states is determined by expansion: n_base_states^order
        if order > 1:
            n_states = get_expanded_state_count(n_base_states, order)
            
            # Check if we have enough data
            if seq_length < n_states * 2:
                log_warn(
                    f"Skipping order={order} (n_states={n_states}): "
                    f"insufficient data ({seq_length} < {n_states * 2})"
                )
                continue
            n_states_range = [n_states]
        else:
            # For order=1, try different n_states
            n_states_range = range(min_states, max_states + 1)
        
        # Try each n_states for this order
        for n_states in n_states_range:
            log_likelihoods = []
            test_sizes = []
            
            for train_idx, test_idx in tscv.split(indices):
                train_seq = seq[train_idx]
                test_seq = seq[test_idx]
                
                if len(train_seq) == 0 or len(test_seq) == 0:
                    continue
                
                # For high-order HMM, we need to handle observations differently
                # The model uses expanded states as hidden states, but observations
                # must still be base states (0, 1, 2) because emission distributions
                # only accept values in [0, n_symbols-1]
                if order > 1:
                    train_states = train_seq.flatten().tolist()
                    test_states = test_seq.flatten().tolist()
                    
                    # For high-order HMM, we use base states as observations
                    # The expanded states are used internally for transition matrix
                    # but observations remain as base states
                    train_obs = [np.array(train_states).reshape(-1, 1)]
                    test_obs = [np.array(test_states).reshape(-1, 1)]
                    
                    # Check if we have enough data for expansion
                    if len(train_states) < order or len(test_states) < 1:
                        continue
                else:
                    train_obs = [train_seq]
                    test_obs = [test_seq]
                
                # Create and train model
                try:
                    # For data-driven initialization, we need the base states
                    train_states_base = train_seq.flatten().tolist()
                    model = create_high_order_hmm_model(
                        n_symbols=n_symbols,
                        n_states=n_states,
                        order=order,
                        states_data=train_states_base,
                        use_data_driven=True,
                        n_base_states=n_base_states
                    )
                    model = train_model(model, train_obs)
                    
                    # Evaluate on test sequence
                    log_likelihood = model.log_probability(test_obs)
                    log_likelihoods.append(log_likelihood)
                    test_sizes.append(len(test_obs[0]))
                except Exception as e:
                    log_warn(f"Error with order={order}, n_states={n_states}: {type(e).__name__}: {e}")
                    continue
            
            if log_likelihoods:
                avg_log_likelihood = np.mean(log_likelihoods)
                total_test_samples = sum(test_sizes)
                
                if use_bic:
                    k = _calculate_hmm_parameters(n_states, n_symbols)
                    bic = -2 * avg_log_likelihood + k * np.log(total_test_samples)
                    
                    if bic < best_score:
                        best_score = bic
                        best_order = order
                        best_n_states = n_states
                        log_info(
                            f"New best: order={order}, n_states={n_states} "
                            f"with BIC={bic:.2f} (k={k}, N={total_test_samples})"
                        )
                else:
                    if avg_log_likelihood > best_score:
                        best_score = avg_log_likelihood
                        best_order = order
                        best_n_states = n_states
                        log_info(
                            f"New best: order={order}, n_states={n_states} "
                            f"with log-likelihood={best_score:.2f}"
                        )
    
    if use_bic:
        log_info(
            f"Selected order={best_order}, n_states={best_n_states} "
            f"with best BIC={best_score:.2f}"
        )
    else:
        log_info(
            f"Selected order={best_order}, n_states={best_n_states} "
            f"with best log-likelihood={best_score:.2f}"
        )
    
    return best_order, best_n_states

# ============================================================================
# Prediction Functions
# ============================================================================

def predict_next_hidden_state_forward_backward_high_order(
    model: DenseHMM, 
    observations: list, 
    order: int = 1,
    n_base_states: int = N_BASE_STATES
) -> List[float]:
    """
    Compute the hidden state distribution for step T+1 given T observations.
    
    For high-order HMM, maps expanded states back to base states.
    
    Args:
        model: The trained HMM model
        observations: List of observations (expanded states if order > 1)
        order: Order of the HMM (k)
        n_base_states: Number of base states (default: 3)
        
    Returns:
        The probability distribution of the hidden state at step T+1
    """
    _, log_alpha, _, _, _ = safe_forward_backward(model, observations)
    log_alpha_last = log_alpha[-1]
    
    with np.errstate(over='ignore', under='ignore'):
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        alpha_last = np.exp(log_alpha_last)
    
    alpha_last /= alpha_last.sum()
    transition_matrix = np.array(model.edges)
    
    next_hidden_proba = alpha_last @ transition_matrix
    # Ensure probabilities are non-negative and normalized
    next_hidden_proba = np.maximum(next_hidden_proba, 0)
    if next_hidden_proba.sum() > 0:
        next_hidden_proba /= next_hidden_proba.sum()
    
    if order > 1:
        # Map expanded states to base states
        n_expanded = len(next_hidden_proba)
        base_proba = np.zeros(n_base_states)
        
        for expanded_idx in range(n_expanded):
            base_state = map_expanded_to_base_state(expanded_idx, order, n_base_states)
            base_proba[base_state] += next_hidden_proba[expanded_idx]
        
        return base_proba.tolist()
    else:
        # For order=1, return directly
        if next_hidden_proba.ndim == 1:
            return next_hidden_proba.tolist()
        else:
            sum_left = next_hidden_proba[:, 0].sum()
            sum_right = next_hidden_proba[:, 1].sum()
            return [sum_left, sum_right]

def predict_next_observation_high_order(
    model: DenseHMM, 
    observations: list, 
    order: int = 1,
    n_base_states: int = N_BASE_STATES
):
    """
    Return an array (n_symbols,) representing P( O_{T+1} = i ), for i=0..n_symbols-1.
    
    Args:
        model: The trained HMM model
        observations: List of observations
        order: Order of the HMM (k)
        n_base_states: Number of base states (default: 3)
        
    Returns:
        Array of probabilities for each observation symbol
    """
    next_hidden_proba = predict_next_hidden_state_forward_backward_high_order(
        model, observations, order, n_base_states
    )
    distributions = model.distributions
    
    params = list(distributions[0].parameters())
    n_symbols = params[1].shape[1]
    next_obs_proba = np.zeros(n_symbols)
    
    emission_probs_list = []
    for dist in distributions:
        params = list(dist.parameters())
        emission_tensor = params[1]
        emission_probs_list.append(emission_tensor.flatten())
    
    # For high-order HMM, we need to aggregate emissions from expanded states
    if order > 1:
        # Map expanded state emissions to base state emissions
        n_expanded = len(emission_probs_list)
        base_emission_probs = [np.zeros(n_symbols) for _ in range(n_base_states)]
        
        for expanded_idx in range(n_expanded):
            base_state = map_expanded_to_base_state(expanded_idx, order, n_base_states)
            if expanded_idx < len(emission_probs_list):
                base_emission_probs[base_state] += emission_probs_list[expanded_idx]
        
        # Normalize base emission probabilities
        for base_state in range(n_base_states):
            if base_emission_probs[base_state].sum() > 0:
                base_emission_probs[base_state] /= base_emission_probs[base_state].sum()
        
        # Calculate next observation probability using base states
        for o in range(n_symbols):
            for base_state in range(n_base_states):
                if base_state < len(next_hidden_proba):
                    next_obs_proba[o] += next_hidden_proba[base_state] * base_emission_probs[base_state][o]
    else:
        # Order = 1: use original logic
        for o in range(n_symbols):
            for z in range(len(next_hidden_proba)):
                next_obs_proba[o] += next_hidden_proba[z] * emission_probs_list[z][o]
    
    # Normalize to ensure valid probabilities
    prob_sum = next_obs_proba.sum()
    if prob_sum > 0:
        next_obs_proba = next_obs_proba / prob_sum
    else:
        # Fallback to uniform distribution if sum is zero
        next_obs_proba = np.ones(n_symbols) / n_symbols
    
    # Ensure all probabilities are non-negative
    next_obs_proba = np.maximum(next_obs_proba, 0)
    # Renormalize after clamping
    prob_sum = next_obs_proba.sum()
    if prob_sum > 0:
        next_obs_proba = next_obs_proba / prob_sum
    
    return next_obs_proba

# ============================================================================
# TrueHighOrderHMM Class
# ============================================================================

class TrueHighOrderHMM:
    """
    True High-Order Hidden Markov Model for market state prediction.
    
    Implements a real High-Order HMM using state space expansion, allowing
    predictions based on k previous states instead of just one.
    """
    
    def __init__(
        self,
        orders_argrelextrema: Optional[int] = None,
        strict_mode: Optional[bool] = None,
        use_data_driven: bool = True,
        train_ratio: float = 0.8,
        min_order: int = HMM_HIGH_ORDER_MIN_ORDER_DEFAULT,
        max_order: int = HMM_HIGH_ORDER_MAX_ORDER_DEFAULT,
        min_states: int = 2,
        max_states: int = 10,
        n_folds: int = 3,
        use_bic: bool = True,
    ):
        """
        Initialize True High-Order HMM analyzer.
        
        Args:
            orders_argrelextrema: Order parameter for swing detection (default: from config)
            strict_mode: Whether to use strict mode for swing-to-state conversion (default: from config)
            use_data_driven: Use data-driven initialization for HMM parameters
            train_ratio: Ratio of data to use for training
            min_order: Minimum order k to try during optimization
            max_order: Maximum order k to try during optimization
            min_states: Minimum number of base hidden states (for order=1)
            max_states: Maximum number of base hidden states (for order=1)
            n_folds: Number of folds for cross-validation
            use_bic: Use BIC for model selection (default: True)
        """
        self.orders_argrelextrema = (
            orders_argrelextrema
            if orders_argrelextrema is not None
            else HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
        )
        self.strict_mode = (
            strict_mode
            if strict_mode is not None
            else HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
        )
        self.use_data_driven = use_data_driven
        self.train_ratio = train_ratio
        self.min_order = min_order
        self.max_order = max_order
        self.min_states = min_states
        self.max_states = max_states
        self.n_folds = n_folds
        self.use_bic = use_bic
        
        # Model state
        self.model: Optional[DenseHMM] = None
        self.optimal_order: Optional[int] = None
        self.optimal_n_states: Optional[int] = None
        self.swing_highs_info: Optional[pd.DataFrame] = None
        self.swing_lows_info: Optional[pd.DataFrame] = None
        self.states: Optional[List[float]] = None
        self.train_states: Optional[List[float]] = None
        self.test_states: Optional[List[float]] = None
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate input DataFrame."""
        required_columns = ['open', 'high', 'low', 'close']
        if df is None or df.empty or not all(col in df.columns for col in required_columns):
            log_error("Invalid dataframe provided - missing required columns")
            return False
        
        try:
            for col in required_columns:
                pd.to_numeric(df[col], errors='raise')
        except (ValueError, TypeError):
            log_error("Invalid dataframe provided - non-numeric data detected")
            return False
        
        return True
    
    def _determine_interval(self, df: pd.DataFrame) -> str:
        """Determine data interval from DataFrame index."""
        if len(df) > 1 and isinstance(df.index, pd.DatetimeIndex):
            time_diff = df.index[1] - df.index[0]
            total_minutes = int(time_diff.total_seconds() / 60)
            return f"h{total_minutes // 60}" if total_minutes % 60 == 0 else f"m{total_minutes}"
        elif len(df) > 1:
            log_warn("DataFrame index is not DatetimeIndex. Using default interval.")
            return "h1"
        return "h1"
    
    def detect_swings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect swing highs and lows from price data.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Tuple of (swing_highs_info, swing_lows_info) DataFrames
        """
        swing_highs = argrelextrema(df['high'].values, np.greater, order=self.orders_argrelextrema)[0]
        swing_lows = argrelextrema(df['low'].values, np.less, order=self.orders_argrelextrema)[0]
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            log_warn("Not enough swing points detected for reliable prediction")
            return pd.DataFrame(), pd.DataFrame()
        
        swing_highs_info = df.iloc[swing_highs][['open', 'high', 'low', 'close']]
        swing_lows_info = df.iloc[swing_lows][['open', 'high', 'low', 'close']]
        
        return swing_highs_info, swing_lows_info
    
    def convert_to_states(self, swing_highs_info: pd.DataFrame, swing_lows_info: pd.DataFrame) -> List[float]:
        """
        Convert swing points to state sequence.
        
        Args:
            swing_highs_info: DataFrame with swing highs
            swing_lows_info: DataFrame with swing lows
            
        Returns:
            List of state values (0, 1, or 2)
        """
        return convert_swing_to_state(swing_highs_info, swing_lows_info, strict_mode=self.strict_mode)
    
    def optimize_and_create_model(self, train_states: List[float]) -> DenseHMM:
        """
        Optimize order k and number of states, then create HMM model.
        
        Args:
            train_states: Training state sequence
            
        Returns:
            Trained HMM model
        """
        train_observations = [np.array(train_states).reshape(-1, 1)]
        
        # Optimize order k and n_states together
        try:
            self.optimal_order, self.optimal_n_states = optimize_order_k(
                train_observations,
                min_order=self.min_order,
                max_order=self.max_order,
                n_folds=self.n_folds,
                use_bic=self.use_bic,
                min_states=self.min_states,
                max_states=self.max_states,
            )
        except Exception as e:
            log_warn(f"Order optimization failed: {e}. Using default order=2, n_states=9.")
            self.optimal_order = 2
            self.optimal_n_states = get_expanded_state_count(N_BASE_STATES, 2)
        
        # Prepare observations based on order
        # For high-order HMM, observations are still base states (0, 1, 2)
        # Expanded states are only used for transition matrix computation
        # This is because pomegranate requires observations in [0, n_symbols-1]
        train_obs = train_observations
        
        # Create model with data-driven initialization
        model = create_high_order_hmm_model(
            n_symbols=N_SYMBOLS,
            n_states=self.optimal_n_states,
            order=self.optimal_order,
            states_data=train_states if self.use_data_driven else None,
            use_data_driven=self.use_data_driven and HMM_HIGH_ORDER_USE_DATA_DRIVEN_INIT,
        )
        
        # Train model
        model = train_model(model, train_obs)
        
        return model
    
    def predict_next_state(self, model: DenseHMM, states: List[float]) -> Tuple[int, float]:
        """
        Predict next state from current state sequence.
        
        Args:
            model: Trained HMM model
            states: Current state sequence
            
        Returns:
            Tuple of (predicted_state_index, probability)
        """
        # Prepare observations - always use base states (0, 1, 2)
        # Expanded states are only used internally for transition matrix
        full_observations = [np.array(states).reshape(-1, 1)]
        
        next_obs_proba = predict_next_observation_high_order(
            model, 
            full_observations, 
            order=self.optimal_order or 1,
            n_base_states=N_BASE_STATES
        )
        next_obs_proba = np.nan_to_num(next_obs_proba, nan=1/3, posinf=1/3, neginf=1/3)
        
        if not np.isfinite(next_obs_proba).all() or np.sum(next_obs_proba) == 0:
            return 1, 0.33  # Default to NEUTRAL
        
        max_index = int(np.argmax(next_obs_proba))
        max_value = float(next_obs_proba[max_index])
        
        return max_index, max_value
    
    def _calculate_duration(self, swing_highs_info: pd.DataFrame, swing_lows_info: pd.DataFrame, interval_str: str) -> int:
        """Calculate predicted state duration."""
        if isinstance(swing_highs_info.index, pd.DatetimeIndex) and isinstance(swing_lows_info.index, pd.DatetimeIndex):
            average_distance = average_swing_distance(swing_highs_info, swing_lows_info) or 3600
        else:
            log_warn("Non-datetime index detected. Using default swing distance.")
            average_distance = 3600
        
        if interval_str.startswith("h"):
            converted_distance = average_distance / 3600
        elif interval_str.startswith("m"):
            converted_distance = average_distance / 60
        else:
            converted_distance = average_distance
        
        return int(converted_distance)
    
    def analyze(
        self,
        df: pd.DataFrame,
        eval_mode: bool = True,
    ) -> HMM_SWINGS:
        """
        Main analysis pipeline: detect swings, convert to states, train model, and predict.
        
        Args:
            df: DataFrame containing price data
            eval_mode: If True, evaluates model performance on test set (not implemented yet)
            
        Returns:
            HMM_SWINGS: Prediction result
        """
        # Validate input
        if not self._validate_dataframe(df):
            return HMM_SWINGS(next_state_with_high_order_hmm=NEUTRAL, next_state_duration=1, next_state_probability=0.33)
        
        # Determine interval
        interval_str = self._determine_interval(df)
        
        # Detect swings
        swing_highs_info, swing_lows_info = self.detect_swings(df)
        if swing_highs_info.empty or swing_lows_info.empty:
            return HMM_SWINGS(next_state_with_high_order_hmm=NEUTRAL, next_state_duration=1, next_state_probability=0.33)
        
        self.swing_highs_info = swing_highs_info
        self.swing_lows_info = swing_lows_info
        
        # Convert to states
        states = self.convert_to_states(swing_highs_info, swing_lows_info)
        if not states:
            log_warn("No states detected from swing points")
            return HMM_SWINGS(next_state_with_high_order_hmm=NEUTRAL, next_state_duration=1, next_state_probability=0.33)
        
        self.states = states
        
        # Split data
        train_size = int(len(states) * self.train_ratio)
        if train_size < 2:
            train_states, test_states = states, []
        else:
            train_states, test_states = states[:train_size], states[train_size:]
        
        self.train_states = train_states
        self.test_states = test_states
        
        # Build and train model
        model = self.optimize_and_create_model(train_states)
        self.model = model
        
        # Calculate duration
        duration = self._calculate_duration(swing_highs_info, swing_lows_info, interval_str)
        
        # Predict next state
        max_index, max_value = self.predict_next_state(model, states)
        
        # Map index to signal
        signal_map = {0: BEARISH, 1: NEUTRAL, 2: BULLISH}
        signal = signal_map.get(max_index, NEUTRAL)
        
        return HMM_SWINGS(
            next_state_with_high_order_hmm=signal,
            next_state_duration=duration,
            next_state_probability=max_value
        )

# ============================================================================
# Wrapper Function
# ============================================================================

def true_high_order_hmm(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    eval_mode: bool = True,
    orders_argrelextrema: Optional[int] = None,
    strict_mode: Optional[bool] = None,
    min_order: int = 2,
    max_order: int = 4,
) -> HMM_SWINGS:
    """
    Generates and trains a true High-Order Hidden Markov Model using swing points.
    
    This is a wrapper function that uses the TrueHighOrderHMM class internally.
    
    Parameters:
        df: DataFrame containing price data with at least the columns 'open', 'high', 'low', 'close'.
        train_ratio: The ratio of data to use for training (default: 0.8).
        eval_mode: If True, evaluates model performance on the test set.
        orders_argrelextrema: Order parameter for swing detection (default: from config).
        strict_mode: Whether to use strict mode for swing-to-state conversion (default: from config).
        min_order: Minimum order k to try during optimization (default: 2).
        max_order: Maximum order k to try during optimization (default: 4).
    
    Returns:
        HMM_SWINGS: Instance containing the predicted market state.
    """
    analyzer = TrueHighOrderHMM(
        orders_argrelextrema=orders_argrelextrema,
        strict_mode=strict_mode,
        use_data_driven=HMM_HIGH_ORDER_USE_DATA_DRIVEN_INIT,
        train_ratio=train_ratio,
        min_order=min_order,
        max_order=max_order,
    )
    return analyzer.analyze(df, eval_mode=eval_mode)

# ============================================================================
# Strategy Interface Implementation
# ============================================================================

class TrueHighOrderHMMStrategy:
    """
    HMM Strategy wrapper for True High-Order HMM.
    
    Implements HMMStrategy interface to enable registry-based management.
    """
    
    def __init__(
        self,
        name: str = "true_high_order",
        weight: float = 1.0,
        enabled: bool = True,
        orders_argrelextrema: Optional[int] = None,
        strict_mode: Optional[bool] = None,
        min_order: int = 2,
        max_order: int = 4,
        **kwargs
    ):
        """
        Initialize True High-Order HMM Strategy.
        
        Args:
            name: Strategy name (default: "true_high_order")
            weight: Strategy weight for voting (default: 1.0)
            enabled: Whether strategy is enabled (default: True)
            orders_argrelextrema: Order parameter for swing detection
            strict_mode: Use strict mode for swing-to-state conversion
            min_order: Minimum order k for optimization
            max_order: Maximum order k for optimization
            **kwargs: Additional parameters (train_ratio, eval_mode, etc.)
        """
        from modules.hmm.signals.strategy import HMMStrategy, HMMStrategyResult
        from modules.hmm.signals.resolution import LONG, HOLD, SHORT
        
        self.name = name
        self.weight = weight
        self.enabled = enabled
        self.orders_argrelextrema = (
            orders_argrelextrema
            if orders_argrelextrema is not None
            else HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT
        )
        self.strict_mode = (
            strict_mode
            if strict_mode is not None
            else HMM_HIGH_ORDER_STRICT_MODE_DEFAULT
        )
        self.min_order = min_order
        self.max_order = max_order
        self.params = kwargs
    
    def analyze(self, df: pd.DataFrame, **kwargs) -> 'HMMStrategyResult':
        """
        Analyze market data using True High-Order HMM.
        
        Args:
            df: DataFrame containing OHLCV data
            **kwargs: Additional parameters (may override self.params)
            
        Returns:
            HMMStrategyResult with signal, probability, state, and metadata
        """
        from modules.hmm.signals.strategy import HMMStrategyResult
        from modules.hmm.signals.resolution import LONG, HOLD, SHORT
        
        # Merge params with kwargs (kwargs take precedence)
        params = {**self.params, **kwargs}
        train_ratio = params.get("train_ratio", 0.8)
        eval_mode = params.get("eval_mode", True)
        
        # Run True High-Order HMM analysis
        result = true_high_order_hmm(
            df,
            train_ratio=train_ratio,
            eval_mode=eval_mode,
            orders_argrelextrema=self.orders_argrelextrema,
            strict_mode=self.strict_mode,
            min_order=self.min_order,
            max_order=self.max_order,
        )
        
        # Convert to HMMStrategyResult
        signal = result.next_state_with_high_order_hmm
        probability = result.next_state_probability
        state = result.next_state_with_high_order_hmm
        
        metadata = {
            "duration": result.next_state_duration,
            "train_ratio": train_ratio,
            "eval_mode": eval_mode,
            "min_order": self.min_order,
            "max_order": self.max_order,
        }
        
        return HMMStrategyResult(
            signal=signal,
            probability=probability,
            state=state,
            metadata=metadata
        )
