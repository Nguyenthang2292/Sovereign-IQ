"""
High-Order HMM Prediction Functions.

This module handles prediction of next states for high-order HMM.
"""

import warnings
from typing import List
import numpy as np
from pomegranate.hmm import DenseHMM

from modules.hmm.core.swings.swing_utils import safe_forward_backward
from modules.hmm.core.high_order.state_expansion import map_expanded_to_base_state

# Base number of states (0=Down, 1=Side, 2=Up)
N_BASE_STATES = 3


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
    
    # Convert to numpy array if it's a PyTorch tensor or other type
    if hasattr(next_hidden_proba, 'cpu'):
        # PyTorch tensor
        next_hidden_proba = next_hidden_proba.cpu().detach().numpy()
    elif not isinstance(next_hidden_proba, np.ndarray):
        next_hidden_proba = np.asarray(next_hidden_proba)
    
    # Ensure probabilities are non-negative and normalized
    next_hidden_proba = np.maximum(next_hidden_proba, 0)
    if next_hidden_proba.sum() > 0:
        next_hidden_proba /= next_hidden_proba.sum()
    
    if order > 1:
        # Map expanded states to base states
        n_expanded = len(next_hidden_proba)
        base_proba = np.zeros(n_base_states)
        
        # Ensure next_hidden_proba is a numpy array
        next_hidden_proba = np.asarray(next_hidden_proba).flatten()
        
        for expanded_idx in range(min(n_expanded, len(next_hidden_proba))):
            base_state = map_expanded_to_base_state(expanded_idx, order, n_base_states)
            # Get probability value and ensure it's a scalar
            prob_value = next_hidden_proba[expanded_idx]
            if isinstance(prob_value, np.ndarray):
                if prob_value.size == 1:
                    prob_value = float(prob_value.item())
                else:
                    prob_value = float(prob_value.sum())
            elif not isinstance(prob_value, (int, float)):
                try:
                    prob_value = float(prob_value)
                except (ValueError, TypeError):
                    prob_value = 0.0
            base_proba[base_state] += prob_value
        
        # Normalize base probabilities
        if base_proba.sum() > 0:
            base_proba /= base_proba.sum()
        
        return base_proba.tolist()
    else:
        # For order=1, return directly
        if next_hidden_proba.ndim == 1:
            return next_hidden_proba.tolist()
        else:
            # 2D array: sum along first dimension to get probabilities for each state
            # Shape is typically (sequence_length, n_states) or (n_states, n_states)
            # We need to sum along axis 0 to get (n_states,) distribution
            try:
                # Sum along first dimension to aggregate probabilities
                # Result should have shape (n_states,)
                if next_hidden_proba.shape[1] == n_base_states:
                    # Shape is (sequence_length, n_states) - sum along axis 0
                    result = next_hidden_proba.sum(axis=0)
                elif next_hidden_proba.shape[0] == n_base_states:
                    # Shape is (n_states, n_states) - sum along axis 0
                    result = next_hidden_proba.sum(axis=0)
                else:
                    # Fallback: flatten and take first n_base_states elements
                    flattened = next_hidden_proba.flatten()
                    result = flattened[:n_base_states] if len(flattened) >= n_base_states else flattened
                
                # Normalize to ensure valid probabilities
                result = np.asarray(result)
                if result.sum() > 0:
                    result = result / result.sum()
                else:
                    result = np.ones(n_base_states) / n_base_states
                
                return result.tolist()
            except Exception as e:
                # Fallback to 1D handling
                flattened = next_hidden_proba.flatten()
                if len(flattened) >= n_base_states:
                    result = flattened[:n_base_states]
                    if result.sum() > 0:
                        result = result / result.sum()
                    else:
                        result = np.ones(n_base_states) / n_base_states
                    return result.tolist()
                else:
                    return flattened.tolist()


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
                # Ensure emission_probs_list[expanded_idx] is a numpy array
                emission_array = np.asarray(emission_probs_list[expanded_idx]).flatten()
                # Take only first n_symbols elements
                if len(emission_array) >= n_symbols:
                    base_emission_probs[base_state] += emission_array[:n_symbols]
                else:
                    # Pad with zeros if needed
                    padded = np.zeros(n_symbols)
                    padded[:len(emission_array)] = emission_array
                    base_emission_probs[base_state] += padded
        
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

