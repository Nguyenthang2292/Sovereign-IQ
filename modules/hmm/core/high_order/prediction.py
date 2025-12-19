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
from config.hmm import HMM_HIGH_ORDER_N_BASE_STATES as N_BASE_STATES


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
    
    # FIX: Handle PyTorch Tensor conversion
    # Issue: log_alpha from pomegranate can be a PyTorch Tensor, not numpy array
    # Solution: Convert to numpy before processing to avoid shape/indexing errors
    # Debug evidence: log_alpha_shape was "(1, 93, 9)" (3D Tensor) causing next_hidden_proba to have wrong shape
    # Convert log_alpha to numpy array if it's a tensor
    if hasattr(log_alpha, 'cpu'):
        log_alpha = log_alpha.cpu().detach().numpy()
    elif hasattr(log_alpha, 'numpy'):
        log_alpha = log_alpha.numpy()
    else:
        log_alpha = np.asarray(log_alpha)
    
    # FIX: Handle 3D log_alpha shape correctly
    # Issue: log_alpha can have shape (batch_size, sequence_length, n_states) = (1, 93, 9)
    #        Using log_alpha[-1] incorrectly took the last batch instead of last time step
    # Solution: Use log_alpha[-1, -1, :] to get last batch AND last time step
    # Debug evidence: log_alpha[-1] returned shape (93, 9) instead of (9,), causing next_hidden_proba to have 837 elements
    # Get the last row (last time step) of log_alpha
    # log_alpha shape can be:
    # - 1D: (n_states,) - already the last state
    # - 2D: (sequence_length, n_states) - take last row
    # - 3D: (batch_size, sequence_length, n_states) - take last batch, last row
    # Convert edges to numpy array, handling deprecated copy keyword
    if isinstance(model.edges, np.ndarray):
        transition_matrix = model.edges.copy()
    else:
        transition_matrix = np.asarray(model.edges)
    n_states = transition_matrix.shape[0]
    
    # FIX: Convert transition matrix from log probabilities to probabilities if needed
    # Issue: pomegranate stores transition matrix as log probabilities (negative values)
    #        When multiplying alpha_last @ transition_matrix, we need probabilities, not log probabilities
    # Solution: Check if transition matrix is in log space (has negative values) and convert
    # Debug evidence: transition_matrix_min=-66.44, transition_matrix_max=-0.0018, all negative
    if np.any(transition_matrix < 0):
        # Transition matrix is in log space, convert to probability space
        with np.errstate(over='ignore', under='ignore'):
            transition_matrix = np.exp(transition_matrix)
        # Normalize rows to ensure valid probability distribution
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
    
    if log_alpha.ndim == 3:
        # 3D array: (batch_size, sequence_length, n_states)
        # Take the last batch (usually batch_size=1, so [-1]) and last time step
        log_alpha_last = log_alpha[-1, -1, :]  # Last batch, last row: shape (n_states,)
    elif log_alpha.ndim == 2:
        # 2D array: (sequence_length, n_states) - take last row
        log_alpha_last = log_alpha[-1, :]  # Last row: shape (n_states,)
    elif log_alpha.ndim == 1:
        # 1D array: take last n_states elements
        if len(log_alpha) >= n_states:
            log_alpha_last = log_alpha[-n_states:]
        else:
            log_alpha_last = log_alpha
    else:
        # Flatten and take last n_states elements
        log_alpha_flat = log_alpha.flatten()
        if len(log_alpha_flat) >= n_states:
            log_alpha_last = log_alpha_flat[-n_states:]
        else:
            log_alpha_last = log_alpha_flat
    
    with np.errstate(over='ignore', under='ignore'):
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        alpha_last = np.exp(log_alpha_last)
    
    # FIX: Ensure alpha_last has correct shape and length
    # Issue: After exp(), alpha_last might have wrong shape or length mismatch with transition_matrix
    # Solution: Flatten and ensure length matches n_states before matrix multiplication
    # Ensure alpha_last is 1D vector with correct length
    alpha_last = np.asarray(alpha_last).flatten()
    
    # Ensure alpha_last matches transition_matrix dimensions
    if len(alpha_last) != n_states:
        # If mismatch, take only the last n_states elements or pad
        if len(alpha_last) > n_states:
            alpha_last = alpha_last[-n_states:]
        elif len(alpha_last) < n_states:
            # Pad with zeros
            padded = np.zeros(n_states)
            padded[:len(alpha_last)] = alpha_last
            alpha_last = padded
    
    # Normalize
    if alpha_last.sum() > 0:
        alpha_last /= alpha_last.sum()
    else:
        alpha_last = np.ones(n_states) / n_states
    
    next_hidden_proba = alpha_last @ transition_matrix
    
    # Convert to numpy array if it's a PyTorch tensor or other type
    if hasattr(next_hidden_proba, 'cpu'):
        # PyTorch tensor
        next_hidden_proba = next_hidden_proba.cpu().detach().numpy()
    elif not isinstance(next_hidden_proba, np.ndarray):
        next_hidden_proba = np.asarray(next_hidden_proba)
    
    # Ensure probabilities are non-negative and normalized
    next_hidden_proba = np.maximum(next_hidden_proba, 0)
    
    # FIX: Handle zero-sum probability distribution
    # Issue: When log probabilities are very negative, exp() can result in all zeros
    #        causing next_hidden_proba.sum() = 0, leading to base_proba_sum = 0.0
    # Solution: Use uniform distribution as fallback when sum is zero
    # Debug evidence: next_hidden_proba_sum was 0.0, causing base_proba_sum = 0.0 before fix
    if next_hidden_proba.sum() > 0:
        next_hidden_proba /= next_hidden_proba.sum()
    else:
        # If sum is 0, use uniform distribution
        next_hidden_proba = np.ones(len(next_hidden_proba)) / len(next_hidden_proba)
    
    if order > 1:
        # Map expanded states to base states
        n_expanded = len(next_hidden_proba)
        base_proba = np.zeros(n_base_states)
        
        # Ensure next_hidden_proba is a numpy array
        next_hidden_proba = np.asarray(next_hidden_proba).flatten()
        
        # FIX: Properly map expanded states to base states
        # Issue: When next_hidden_proba has wrong shape (e.g., 837 elements instead of 9),
        #        mapping fails and base_proba_sum becomes 0.0
        # Solution: After fixing log_alpha extraction, n_expanded should match expected_expanded
        #           and mapping should work correctly
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

