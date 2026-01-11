
from typing import TYPE_CHECKING, Tuple

from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import numpy as np

"""
High-Order HMM Optimization.

This module handles optimization of order k and number of states for high-order HMM.
"""



if TYPE_CHECKING:
    from modules.hmm.core.high_order.model_creation import create_high_order_hmm_model, train_model

from config.hmm import HMM_HIGH_ORDER_N_BASE_STATES as N_BASE_STATES
from modules.common.utils import log_info, log_warn
from modules.hmm.core.high_order.state_expansion import get_expanded_state_count


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


def optimize_n_states_high_order(
    observations,
    order: int = 1,
    min_states: int = 2,
    max_states: int = 10,
    n_folds: int = 3,
    use_bic: bool = True,
    n_base_states: int = N_BASE_STATES,
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

            # Flatten train_seq to 1D list for states_data
            train_states_flat = train_seq.flatten().tolist() if len(train_seq) > 0 else None
            model = create_high_order_hmm_model(
                n_symbols=n_symbols, n_states=n_states, order=order, states_data=train_states_flat, use_data_driven=True
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
    n_base_states: int = N_BASE_STATES,
) -> Tuple[int, int]:
    """
    Optimize both order k and number of states for High-Order HMM.

    This function tries different combinations of order k and base states,
    then selects the best combination based on BIC or log-likelihood.

    Args:
        observations: A list containing a single observation sequence (2D array)
        min_order: Minimum order k to try
        max_order: Maximum order k to try
        n_folds: Number of folds in cross-validation
        use_bic: If True, use BIC for model selection
        min_states: Minimum number of base states (for order=1)
        max_states: Maximum number of base states (for order=1)
        n_base_states: Number of base states (default: 3)

    Returns:
        Tuple of (best_order, best_n_states) where best_n_states is expanded if order > 1
    """
    if len(observations) != 1:
        raise ValueError("Expected a single observation sequence.")

    seq = observations[0]
    seq_length = len(seq)
    n_symbols = 3

    if seq_length < n_folds * 2:
        raise ValueError(f"Sequence length ({seq_length}) too short for {n_folds} folds.")

    best_order = min_order
    best_n_states = get_expanded_state_count(n_base_states, min_order)
    best_score = np.inf if use_bic else -np.inf

    tscv = TimeSeriesSplit(n_splits=n_folds)
    indices = np.arange(seq_length)

    # Try each order k
    for order in range(min_order, max_order + 1):
        # For order > 1, n_states = n_base_states^order (expanded states)
        # For order = 1, we optimize n_states separately
        if order == 1:
            # Optimize n_states for order=1
            n_states = optimize_n_states_high_order(
                observations,
                order=1,
                min_states=min_states,
                max_states=max_states,
                n_folds=n_folds,
                use_bic=use_bic,
                n_base_states=n_base_states,
            )
        else:
            # For order > 1, use expanded states
            n_states = get_expanded_state_count(n_base_states, order)

        log_likelihoods = []
        test_sizes = []

        for train_idx, test_idx in tscv.split(indices):
            train_seq = seq[train_idx]
            test_seq = seq[test_idx]

            if len(train_seq) == 0 or len(test_seq) == 0:
                continue

            # Create and train model
            from modules.hmm.core.high_order.model_creation import create_high_order_hmm_model, train_model

            # Flatten train_seq to 1D list for states_data
            train_states_flat = train_seq.flatten().tolist() if len(train_seq) > 0 else None
            model = create_high_order_hmm_model(
                n_symbols=n_symbols, n_states=n_states, order=order, states_data=train_states_flat, use_data_driven=True
            )
            model = train_model(model, [train_seq])

            # Evaluate on test sequence
            try:
                log_likelihood = model.log_probability([test_seq])
                log_likelihoods.append(log_likelihood)
                test_sizes.append(len(test_seq))
            except Exception as e:
                log_warn(f"Error in log_probability for order={order}: {type(e).__name__}: {e}")

        # Compute the average score
        if log_likelihoods:
            avg_log_likelihood = np.mean(log_likelihoods)
            total_test_samples = sum(test_sizes)

            if use_bic:
                # Calculate BIC
                k = _calculate_hmm_parameters(n_states, n_symbols)
                bic = -2 * avg_log_likelihood + k * np.log(total_test_samples)

                if bic < best_score:
                    best_score = bic
                    best_order = order
                    best_n_states = n_states
                    log_info(
                        f"New best order={order}, n_states={n_states} with BIC={bic:.2f} (k={k}, N={total_test_samples})"
                    )
            else:
                # Use log-likelihood (higher is better)
                if avg_log_likelihood > best_score:
                    best_score = avg_log_likelihood
                    best_order = order
                    best_n_states = n_states

    if use_bic:
        log_info(f"Selected order={best_order}, n_states={best_n_states} with best BIC={best_score:.2f}")
    else:
        log_info(f"Selected order={best_order}, n_states={best_n_states} with best log-likelihood={best_score:.2f}")

    return best_order, best_n_states
