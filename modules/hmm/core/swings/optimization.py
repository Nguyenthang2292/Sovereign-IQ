
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

from modules.common.utils import log_info, log_warn
from modules.hmm.core.swings.model_creation import create_hmm_model, train_model
from modules.hmm.core.swings.model_creation import create_hmm_model, train_model

"""
HMM-Swings State Optimization.

This module handles optimization of the number of hidden states using cross-validation.
"""




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
        raise ValueError(
            f"Sequence length ({seq_length}) too short for {n_folds} folds. Need at least {n_folds * 2} points."
        )

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
