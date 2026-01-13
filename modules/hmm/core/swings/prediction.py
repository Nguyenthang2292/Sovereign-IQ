import warnings
from typing import List

import numpy as np
from pomegranate.hmm import DenseHMM

from modules.hmm.core.swings.swing_utils import safe_forward_backward

"""
HMM-Swings Prediction Functions.

This module handles prediction of next states and model evaluation.
"""


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
    with np.errstate(over="ignore", under="ignore"):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        alpha_last = np.exp(log_alpha_last)

    # Handle 2D alpha_last: if it's (N, n_states), sum along axis 0 to get (n_states,)
    # This happens when pomegranate returns batch format
    if alpha_last.ndim == 2:
        # Sum along axis 0 to aggregate across batch/time dimension
        alpha_last = alpha_last.sum(axis=0)

    # Normalize to get probability distribution
    alpha_last = alpha_last / alpha_last.sum()
    transition_matrix = model.edges

    # Compute distribution for step T+1
    # alpha_last is now guaranteed to be 1D (n_states,)
    # transition_matrix is (n_states, n_states)
    # Result will be 1D (n_states,)
    next_hidden_proba = alpha_last @ transition_matrix

    # Return as list - should always be 1D now
    return next_hidden_proba.tolist()


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
