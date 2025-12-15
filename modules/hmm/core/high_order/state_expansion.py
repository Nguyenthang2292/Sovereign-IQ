"""
High-Order HMM State Expansion.

This module handles state space expansion for high-order HMM, converting base states
to expanded states that represent sequences of k previous states.
"""

from typing import List, Tuple
from modules.common.utils import log_warn

# Base number of states (0=Down, 1=Side, 2=Up)
N_BASE_STATES = 3


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

