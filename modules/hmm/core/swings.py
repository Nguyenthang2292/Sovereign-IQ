import numpy as np
import pandas as pd
import threading
import warnings
from dataclasses import dataclass
from functools import wraps
from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM
from scipy.signal import argrelextrema
from sklearn.model_selection import TimeSeriesSplit
from typing import Any, List, Literal, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.hmm.signals.strategy import HMMStrategyResult
from colorama import init
init(autoreset=True)

from modules.common.utils import log_data, log_info, log_error, log_warn, log_model, log_analysis
from config import (
    HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT,
    HMM_HIGH_ORDER_STRICT_MODE_DEFAULT,
    HMM_HIGH_ORDER_USE_DATA_DRIVEN_INIT,
)

@dataclass
class HMM_SWINGS:
    next_state_with_high_order_hmm: Literal[-1, 0, 1]
    next_state_duration: int
    next_state_probability: float

BULLISH, NEUTRAL, BEARISH = 1, 0, -1

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result: List[Any] = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        
        return wrapper
    return decorator

@timeout(30)
def safe_forward_backward(model, observations):
    return model.forward_backward(observations)

def convert_swing_to_state(swing_highs_info: pd.DataFrame, swing_lows_info: pd.DataFrame, strict_mode: bool = False) -> List[float]:
    """
    Convert swing high and low points to market state sequence.
    
    States:
    - 0: Downtrend
    - 1: Sideways/Consolidation
    - 2: Uptrend
    
    Methods:
    - strict_mode=True: Compares consecutive swing values requiring equal counts
    - strict_mode=False: Uses chronological transitions between highs and lows
    
    Parameters
    ----------
    swing_highs_info : DataFrame
        Swing highs with 'high' column and datetime index
    swing_lows_info : DataFrame
        Swing lows with 'low' column and datetime index
    strict_mode : bool, default=False
        Whether to use strict comparison mode
        
    Returns
    -------
    List[float]
        Market state values (0, 1, or 2)
    """
    if swing_highs_info.empty or swing_lows_info.empty:
        log_warn("One of the swing DataFrames is empty. Returning empty list.")
        return []
    
    if strict_mode:
        """
        Strict mode: Match swing highs and lows by timestamp proximity,
        then compare consecutive matched pairs to determine state transitions.
        
        This ensures we compare swings that actually occurred close in time,
        rather than just using array indices which can be misaligned.
        """
        states = []
        
        # Ensure DataFrames are sorted by timestamp
        swing_highs_info = swing_highs_info.sort_index()
        swing_lows_info = swing_lows_info.sort_index()
        
        # Create lists of swing points with timestamps
        high_points = []
        for idx, row in swing_highs_info.iterrows():
            high_points.append({'time': idx, 'type': 'high', 'value': row['high']})
        
        low_points = []
        for idx, row in swing_lows_info.iterrows():
            low_points.append({'time': idx, 'type': 'low', 'value': row['low']})
        
        # Linear O(N) algorithm: Maintain state of nearest peak/trough while traversing
        # Strategy: Duyệt tuyến tính, duy trì peak/trough gần nhất, tạo pair khi gặp opposite
        
        # Merge and sort all swings by time
        all_swings = sorted(high_points + low_points, key=lambda x: x['time'])
        
        if len(all_swings) < 2:
            log_warn("Not enough swing points for strict mode comparison")
            return []
        
        matched_pairs = []
        last_high = None  # Track last unmatched high (highest so far)
        last_low = None   # Track last unmatched low (lowest so far)
        
        # Linear traversal: O(N)
        for swing in all_swings:
            if swing['type'] == 'high':
                # If new high is higher than last high (and no low intervened), update
                if last_high is None or swing['value'] > last_high['value']:
                    last_high = swing
                # If we have a low waiting, create pair with current high
                if last_low is not None:
                    pair_time = max(last_high['time'], last_low['time'])
                    matched_pairs.append({
                        'time': pair_time,
                        'high': last_high['value'],
                        'low': last_low['value'],
                    })
                    # Reset low after pairing, keep high for potential next pair
                    last_low = None
            else:  # swing['type'] == 'low'
                # If new low is lower than last low (and no high intervened), update
                if last_low is None or swing['value'] < last_low['value']:
                    last_low = swing
                # If we have a high waiting, create pair with current low
                if last_high is not None:
                    pair_time = max(last_high['time'], last_low['time'])
                    matched_pairs.append({
                        'time': pair_time,
                        'high': last_high['value'],
                        'low': last_low['value'],
                    })
                    # Reset high after pairing, keep low for potential next pair
                    last_high = None
        
        # Sort pairs by timestamp - O(N log N) where N = number of pairs
        unique_pairs = sorted(matched_pairs, key=lambda x: x['time'])
        
        # Compare consecutive pairs to determine states
        for i in range(1, len(unique_pairs)):
            current_pair = unique_pairs[i]
            previous_pair = unique_pairs[i - 1]
            
            current_high = current_pair['high']
            previous_high = previous_pair['high']
            current_low = current_pair['low']
            previous_low = previous_pair['low']
            
            if current_high < previous_high and current_low < previous_low:
                state = 0  # Downtrend
            elif current_high > previous_high and current_low > previous_low:
                state = 2  # Uptrend
            else:
                state = 1  # Sideways/Consolidation
            
            states.append(state)
        
        if len(states) == 0:
            log_warn("No valid state transitions found in strict mode")
        
        return states
    else:
        # Remove rows with NaN values
        swing_highs_info = swing_highs_info.dropna(subset=['high'])
        swing_lows_info = swing_lows_info.dropna(subset=['low'])
        
        # Combine high and low swing points
        swings = []
        for idx in swing_highs_info.index:
            swings.append({'time': idx, 'type': 'high', 'value': swing_highs_info.loc[idx, 'high']})
        for idx in swing_lows_info.index:
            swings.append({'time': idx, 'type': 'low', 'value': swing_lows_info.loc[idx, 'low']})
        
        # Sort and remove duplicates
        swings.sort(key=lambda x: x['time'])
        unique_swings, prev_time = [], None
        for swing in swings:
            if swing['time'] != prev_time:
                unique_swings.append(swing)
                prev_time = swing['time']
        
        # Determine states
        states, prev_swing = [], None
        for swing in unique_swings:
            if prev_swing is None:
                prev_swing = swing
                continue
            
            if prev_swing['type'] == 'low' and swing['type'] == 'high':
                state = 2  # price increase
            elif prev_swing['type'] == 'high' and swing['type'] == 'low':
                state = 0  # price decrease
            else:
                state = 1  # unchanged or mixed
            
            states.append(state)
            prev_swing = swing
        
        return states

def _calculate_hmm_parameters(n_states: int, n_symbols: int = 3) -> int:
    """
    Calculate the number of free parameters in an HMM model.
    
    For an HMM with n_states hidden states and n_symbols observable symbols:
    - Transition matrix: n_states * (n_states - 1) (each row sums to 1)
    - Emission probabilities: n_states * (n_symbols - 1) (each row sums to 1)
    - Start probabilities: n_states - 1 (sums to 1)
    - End probabilities: n_states - 1 (sums to 1)
    
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

def optimize_n_states(observations, min_states=2, max_states=10, n_folds=3, use_bic=True):
    """
    Automatically optimize the number of hidden states using TimeSeriesSplit cross-validation.
    
    Uses TimeSeriesSplit instead of KFold to ensure:
    - Training data is always from the past
    - Test data is always from the immediate future
    - No artificial jumps in the sequence (avoids concatenating non-adjacent segments)
    
    Uses BIC (Bayesian Information Criterion) instead of raw log-likelihood to prevent overfitting:
    - BIC = -2 * log_likelihood + k * log(N)
    - k: number of free parameters
    - N: number of samples
    - Lower BIC is better (penalizes model complexity)
    
    Parameters:
    - observations: A list containing a single observation sequence (2D array).
    - min_states: Minimum number of hidden states.
    - max_states: Maximum number of hidden states.
    - n_folds: Number of folds in cross-validation.
    - use_bic: If True, use BIC for model selection; if False, use log-likelihood (default: True).
    
    Returns:
    - The best number of hidden states.
    """
    
    # Check if observations are in the correct format
    if len(observations) != 1:
        raise ValueError("Expected a single observation sequence.")
    
    seq = observations[0]  # Get the single observation sequence
    seq_length = len(seq)  # Length of the sequence
    n_symbols = 3  # Fixed number of observable symbols
    
    if seq_length < n_folds * 2:
        raise ValueError(f"Sequence length ({seq_length}) too short for {n_folds} folds. Need at least {n_folds * 2} points.")
    
    # For BIC: lower is better; for log-likelihood: higher is better
    best_n_states = min_states
    best_score = np.inf if use_bic else -np.inf
    
    # Try each number of hidden states
    for n_states in range(min_states, max_states + 1):
        log_likelihoods = []
        test_sizes = []  # Track test set sizes for BIC calculation
        
        # Use TimeSeriesSplit for time series cross-validation
        # This ensures train is always from past, test is always from immediate future
        # No artificial jumps in the sequence
        tscv = TimeSeriesSplit(n_splits=n_folds)
        
        # TimeSeriesSplit works on 1D arrays, so we use indices
        indices = np.arange(seq_length)
        
        for train_idx, test_idx in tscv.split(indices):
            # Create train and test sequences using indices
            # This preserves temporal order: train is always before test
            train_seq = seq[train_idx]
            test_seq = seq[test_idx]
            
            if len(train_seq) == 0 or len(test_seq) == 0:
                continue
            
            # Create and train model
            model = create_hmm_model(n_symbols=n_symbols, n_states=n_states)
            model = train_model(model, [train_seq])
            
            # Evaluate on test sequence (log probability calculation)
            try:
                log_likelihood = model.log_probability([test_seq])
                log_likelihoods.append(log_likelihood)
                test_sizes.append(len(test_seq))
            except Exception as e:
                log_warn(f"Error in log_probability: {type(e).__name__}: {e}")
        
        # Compute the average score
        if log_likelihoods:
            avg_log_likelihood = np.mean(log_likelihoods)
            total_test_samples = sum(test_sizes)
            
            if use_bic:
                # Calculate BIC: BIC = -2 * log_likelihood + k * log(N)
                # Lower BIC is better
                k = _calculate_hmm_parameters(n_states, n_symbols)
                bic = -2 * avg_log_likelihood + k * np.log(total_test_samples)
                
                if bic < best_score:
                    best_score = bic
                    best_n_states = n_states
                    log_info(f"New best n_states={n_states} with BIC={bic:.2f} (k={k}, N={total_test_samples})")
            else:
                # Use log-likelihood (higher is better)
                if avg_log_likelihood > best_score:
                    best_score = avg_log_likelihood
                    best_n_states = n_states
    
    if use_bic:
        log_info(f"Selected n_states={best_n_states} with best BIC={best_score:.2f}")
    else:
        log_info(f"Selected n_states={best_n_states} with best log-likelihood={best_score:.2f}")
    
    return best_n_states

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
        
        # Calculate segment boundaries
        # For n_observed_states=3, n_states=4: segments are [0-0.75), [0.75-1.5), [1.5-2.25), [2.25-3)
        # But we want integer boundaries, so we use: [0-1), [1-2), [2-3) mapped to [0, 1, 2, 3]
        
        # Use ceiling of (observed_state + 1) * n_states / n_observed_states - 1
        # This ensures each observed state maps to exactly one hidden state
        # and distribution is as balanced as possible
        
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
        return [Categorical([[1/n_symbols] * n_symbols]) for _ in range(n_states)]
    
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
                verbose=False
            )
        except Exception as e:
            log_warn(f"Data-driven initialization failed: {e}. Falling back to hardcoded values.")
            use_data_driven = False
    
    # Fallback to hardcoded initialization
    if n_states == 2:
        # Optimized configuration for 2 hidden states (hardcoded fallback)
        distributions = [
            Categorical([[0.25, 0.25, 0.50]]),  # Mixed trend, biased toward increase
            Categorical([[0.50, 0.25, 0.25]])   # Mixed trend, biased toward decrease
        ]
        edges = [[0.85, 0.15], [0.15, 0.85]]
        starts = [0.5, 0.5]
        ends = [0.01, 0.01]
    else:
        # Configuration for custom number of hidden states
        distributions = [Categorical([[1/n_symbols] * n_symbols]) for _ in range(n_states)]
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

def predict_next_hidden_state_forward_backward(model: DenseHMM, observations: list) -> List[float]:
    """
    Compute the hidden state distribution for step T+1 given T observations.
    
    Args:
        model (DenseHMM): The trained HMM model.
        observations (list): List of observations.
        
    Returns:
        list: The probability distribution of the hidden state at step T+1.

    Explanation of the alpha and beta variables (from the forward-backward algorithm):

    - Alpha (forward variable):
    Represents the probability of observing the sequence from the beginning up to time t 
    and being in a specific state at time t.
    
    - Beta (backward variable):
    Represents the probability of observing the sequence from time t+1 to the end 
    given that the system is in a specific state at time t.
    
    Combining alpha and beta allows computation of the posterior probabilities of the states.
    """
    # Get forward probabilities
    _, log_alpha, _, _, _ = safe_forward_backward(model, observations)
    log_alpha_last = log_alpha[-1]
    
    # Convert to standard probabilities
    with np.errstate(over='ignore', under='ignore'):
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        alpha_last = np.exp(log_alpha_last)
    
    alpha_last /= alpha_last.sum()
    transition_matrix = model.edges
    
    # Compute distribution for step T+1
    next_hidden_proba = alpha_last @ transition_matrix
    
    # Handle both 1D and 2D arrays
    if next_hidden_proba.ndim == 1:
        # For 2-state models, return the probabilities directly
        return next_hidden_proba.tolist()
    else:
        # For multi-state models, sum the probabilities per column
        sum_left = next_hidden_proba[:, 0].sum()
        sum_right = next_hidden_proba[:, 1].sum()
        return [sum_left, sum_right]

def predict_next_observation(model, observations):
    """
    Return an array (n_symbols,) representing P( O_{T+1} = i ), for i=0..n_symbols-1.
    """
    next_hidden_proba = predict_next_hidden_state_forward_backward(model, observations)
    distributions = model.distributions
    
    # Get emission distributions
    params = list(distributions[0].parameters())
    n_symbols = params[1].shape[1]
    next_obs_proba = np.zeros(n_symbols)
    
    emission_probs_list = []
    for dist in distributions:
        params = list(dist.parameters())
        emission_tensor = params[1]
        emission_probs_list.append(emission_tensor.flatten())
    
    # Calculate next observation probability
    for o in range(n_symbols):
        for z in range(len(next_hidden_proba)):
            next_obs_proba[o] += next_hidden_proba[z] * emission_probs_list[z][o]
    
    return next_obs_proba / next_obs_proba.sum()

def average_swing_distance(swing_highs_info, swing_lows_info):
    """
    Calculate the average time interval (in seconds) between consecutive swing highs and swing lows,
    and return the overall average.

    Args:
        swing_highs_info (pd.DataFrame): DataFrame containing swing high information with datetime index.
        swing_lows_info (pd.DataFrame): DataFrame containing swing low information with datetime index.

    Returns:
        float: The average time distance between swing points in seconds.
    """
    # Calculate high intervals
    swing_high_times = swing_highs_info.index
    intervals_seconds_high = [(swing_high_times[i] - swing_high_times[i - 1]).total_seconds() 
                            for i in range(1, len(swing_high_times))]
    avg_distance_high = np.mean(intervals_seconds_high) if intervals_seconds_high else 0

    # Calculate low intervals
    swing_low_times = swing_lows_info.index
    intervals_seconds_low = [(swing_low_times[i] - swing_low_times[i - 1]).total_seconds() 
                            for i in range(1, len(swing_low_times))]
    avg_distance_low = np.mean(intervals_seconds_low) if intervals_seconds_low else 0

    # Return average
    if avg_distance_high and avg_distance_low:
        return (avg_distance_high + avg_distance_low) / 2
    return avg_distance_high or avg_distance_low

def evaluate_model_accuracy(model, train_states, test_states):
    """
    Evaluate the accuracy of the HMM model on the test set.
    
    Parameters:
        model: The trained HMM model.
        train_states: Sequence of states used for training.
        test_states: Sequence of states used for testing.
        
    Returns:
        float: The accuracy of the model.
    """
    correct_predictions = 0
    
    # Predict each state in the test set
    for i in range(len(test_states)):
        # Create an observation sequence using training data and known test states so far
        current_states = train_states + test_states[:i]
        observations = [np.array(current_states).reshape(-1, 1)]
        
        # Predict the next state
        next_obs_proba = predict_next_observation(model, observations)
        predicted_state = np.argmax(next_obs_proba)
        
        # Compare with the actual state
        if predicted_state == test_states[i]:
            correct_predictions += 1
    
    # Calculate accuracy
    return correct_predictions / len(test_states) if test_states else 0.0

class HighOrderHMM:
    """
    High-Order Hidden Markov Model for market state prediction.
    
    Encapsulates HMM model creation, training, and prediction logic with
    data-driven initialization and optimized state selection.
    """
    
    def __init__(
        self,
        orders_argrelextrema: Optional[int] = None,
        strict_mode: Optional[bool] = None,
        use_data_driven: bool = True,
        train_ratio: float = 0.8,
        min_states: int = 2,
        max_states: int = 10,
        n_folds: int = 3,
        use_bic: bool = True,
    ):
        """
        Initialize High-Order HMM analyzer.
        
        Args:
            orders_argrelextrema: Order parameter for swing detection (default: from config)
            strict_mode: Whether to use strict mode for swing-to-state conversion (default: from config)
            use_data_driven: Use data-driven initialization for HMM parameters
            train_ratio: Ratio of data to use for training
            min_states: Minimum number of hidden states for optimization
            max_states: Maximum number of hidden states for optimization
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
        self.min_states = min_states
        self.max_states = max_states
        self.n_folds = n_folds
        self.use_bic = use_bic
        
        # Model state
        self.model: Optional[DenseHMM] = None
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
            return "h1"  # Default to 1 hour
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
        Optimize number of states and create HMM model.
        
        Args:
            train_states: Training state sequence
            
        Returns:
            Trained HMM model
        """
        train_observations = [np.array(train_states).reshape(-1, 1)]
        
        # Optimize number of states
        try:
            self.optimal_n_states = optimize_n_states(
                train_observations,
                min_states=self.min_states,
                max_states=self.max_states,
                n_folds=self.n_folds,
                use_bic=self.use_bic
            )
        except Exception as e:
            log_warn(f"State optimization failed: {e}. Using default n_states=2.")
            self.optimal_n_states = 2
        
        # Create model with data-driven initialization
        model = create_hmm_model(
            n_symbols=3,
            n_states=self.optimal_n_states,
            states_data=train_states if self.use_data_driven else None,
            use_data_driven=self.use_data_driven and HMM_HIGH_ORDER_USE_DATA_DRIVEN_INIT
        )
        
        # Train model
        model = train_model(model, train_observations)
        
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
        full_observations = [np.array(states).reshape(-1, 1)]
        next_obs_proba = predict_next_observation(model, full_observations)
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
            average_distance = 3600  # Default to 1 hour in seconds
        
        # Convert time units
        if interval_str.startswith("h"):
            converted_distance = average_distance / 3600  # to hours
        elif interval_str.startswith("m"):
            converted_distance = average_distance / 60    # to minutes
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
            eval_mode: If True, evaluates model performance on test set
            
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
        
        # Evaluate model if requested
        accuracy = evaluate_model_accuracy(model, train_states, test_states) if eval_mode and test_states else 0.0
        
        # Calculate duration
        duration = self._calculate_duration(swing_highs_info, swing_lows_info, interval_str)
        
        # Return NEUTRAL if accuracy is too low
        # Threshold increased from 0.3 to 0.33 to ensure higher model quality
        if accuracy <= 0.33:
            return HMM_SWINGS(
                next_state_with_high_order_hmm=NEUTRAL,
                next_state_duration=duration,
                next_state_probability=max(accuracy, 0.33)
            )
        
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


def hmm_swings(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    eval_mode: bool = True,
    orders_argrelextrema: Optional[int] = None,
    strict_mode: Optional[bool] = None,
) -> HMM_SWINGS:
    """
    Generates and trains a Hidden Markov Model (HMM) using swing points extracted from market price data.
    
    This is a backward-compatible wrapper function that uses the HighOrderHMM class internally.
    
    Parameters:
        df: DataFrame containing price data with at least the columns 'open', 'high', 'low', 'close'.
        train_ratio: The ratio of data to use for training (default: 0.8).
        eval_mode: If True, evaluates model performance on the test set.
        orders_argrelextrema: Order parameter for swing detection (default: from config).
        strict_mode: Whether to use strict mode for swing-to-state conversion (default: from config).
    
        Returns:
            HMM_SWINGS: Instance containing the predicted market state.
    """
    analyzer = HighOrderHMM(
        orders_argrelextrema=orders_argrelextrema,
        strict_mode=strict_mode,
        use_data_driven=HMM_HIGH_ORDER_USE_DATA_DRIVEN_INIT,
        train_ratio=train_ratio,
    )
    return analyzer.analyze(df, eval_mode=eval_mode)


# ============================================================================
# Strategy Interface Implementation
# ============================================================================

class SwingsHMMStrategy:
    """
    HMM Strategy wrapper for Basic HMM with swings.
    
    Implements HMMStrategy interface to enable registry-based management.
    """
    
    def __init__(
        self,
        name: str = "swings",
        weight: float = 1.0,
        enabled: bool = True,
        orders_argrelextrema: Optional[int] = None,
        strict_mode: Optional[bool] = None,
        **kwargs
    ):
        """
        Initialize Swings HMM Strategy.
        
        Args:
            name: Strategy name (default: "swings")
            weight: Strategy weight for voting (default: 1.0)
            enabled: Whether strategy is enabled (default: True)
            orders_argrelextrema: Order parameter for swing detection
            strict_mode: Use strict mode for swing-to-state conversion
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
        self.params = kwargs
    
    def analyze(self, df: pd.DataFrame, **kwargs) -> 'HMMStrategyResult':
        """
        Analyze market data using Basic HMM with swings.
        
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
        
        # Run HMM swings analysis
        result = hmm_swings(
            df,
            train_ratio=train_ratio,
            eval_mode=eval_mode,
            orders_argrelextrema=self.orders_argrelextrema,
            strict_mode=self.strict_mode,
        )
        
        # Convert to HMMStrategyResult
        signal = result.next_state_with_high_order_hmm
        probability = result.next_state_probability
        state = result.next_state_with_high_order_hmm
        
        metadata = {
            "duration": result.next_state_duration,
            "train_ratio": train_ratio,
            "eval_mode": eval_mode,
        }
        
        return HMMStrategyResult(
            signal=signal,
            probability=probability,
            state=state,
            metadata=metadata
        )
