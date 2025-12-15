"""
HMM-KAMA Model Operations.

This module contains the HMM_KAMA dataclass and all HMM model training/application operations.
"""

from dataclasses import dataclass
from typing import Literal, Tuple, cast
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from modules.common.utils import log_info, log_error, log_warn, log_model, log_data

# Fix KMeans memory leak on Windows with MKL
import os
import warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings(
    "ignore", message="KMeans is known to have a memory leak on Windows with MKL"
)


@dataclass
class HMM_KAMA:
    """Result dataclass for HMM-KAMA analysis."""
    next_state_with_hmm_kama: Literal[-1, 0, 1, 2, 3]
    current_state_of_state_using_std: Literal[-1, 0, 1]
    current_state_of_state_using_hmm: Literal[-1, 0, 1]
    state_high_probabilities_using_arm_apriori: Literal[-1, 0, 1, 2, 3]
    state_high_probabilities_using_arm_fpgrowth: Literal[-1, 0, 1, 2, 3]
    current_state_of_state_using_kmeans: Literal[-1, 0, 1]


def reorder_hmm_model(model: GaussianHMM) -> GaussianHMM:
    """
    Reorder HMM states based on the mean of the first feature (Returns).
    State 0: Lowest Returns (Strong Bearish)
    State 3: Highest Returns (Strong Bullish)
    """
    if not hasattr(model, "means_"):
        return model

    # Sort by the first feature (Returns)
    # We want state 0 to be the most negative return (Bearish Strong)
    # and state N to be the most positive return (Bullish Strong)
    order = np.argsort(model.means_[:, 0])

    # If already sorted, return
    if np.array_equal(order, np.arange(model.n_components)):
        return model

    log_info(f"Reordering HMM states. New order mapping: {order}")

    # Reorder means
    model.means_ = model.means_[order]

    # Reorder covariances
    if hasattr(model, "covars_"):
        # For diag covariance type, covars_ shape should be (n_components, n_features)
        # But sometimes hmmlearn returns (n_components, n_features, n_features) for full cov
        if model.covars_.ndim == 2:
            # Normal case: (n_components, n_features)
            model.covars_ = model.covars_[order, :]
        elif model.covars_.ndim == 3:
            # Full covariance case: extract diagonal first, then reorder
            diag_covars = np.array([
                np.diag(model.covars_[i]) for i in range(model.covars_.shape[0])
            ])
            model.covars_ = diag_covars[order, :]
        else:
            # Fallback for unexpected shape
            log_warn(f"Unexpected covars_ shape: {model.covars_.shape}, skipping reorder")

    # Reorder start probabilities
    model.startprob_ = model.startprob_[order]

    # Reorder transition matrix
    # The transition matrix is (n_states, n_states)
    # We need to reorder both rows and columns
    model.transmat_ = model.transmat_[order][:, order]

    return model


def train_hmm(
    observations: np.ndarray,
    n_components: int = 4,
    n_iter: int = 10,
    random_state: int = 36,
) -> GaussianHMM:
    """Train HMM with robust error handling and data validation"""
    if observations.size == 0:
        raise ValueError("Empty observations array")

    n_features = observations.shape[1] if len(observations.shape) > 1 else 1

    # Data cleaning
    if not np.isfinite(observations).all():
        log_warn("Observations contain invalid values. Cleaning...")
        for col in range(observations.shape[1]):  # type: ignore
            col_data = observations[:, col]
            finite_mask = np.isfinite(col_data)
            if finite_mask.any():
                observations[:, col] = np.where(
                    finite_mask, col_data, np.median(col_data[finite_mask])
                )
            else:
                observations[:, col] = 0.0

    # Variance check
    variances = np.var(observations, axis=0)
    low_var_mask = variances < 1e-12
    if low_var_mask.any():
        log_data(f"Low variance detected in columns {np.where(low_var_mask)[0]}. Adding noise.")
        noise = np.random.RandomState(random_state).normal(0, 1e-6, observations.shape)
        observations[:, low_var_mask] += noise[:, low_var_mask]

    # Scaling to prevent overflow
    obs_max = np.max(np.abs(observations))
    scale_factor = 1.0
    if obs_max > 1e6:
        scale_factor = 1e6 / obs_max
        observations = observations * scale_factor
        log_data(f"Scaled observations by factor {scale_factor} to prevent overflow")

    log_model(
        f"Training HMM - Observations shape: {observations.shape}, "
        f"Range: [{np.min(observations):.2e}, {np.max(observations):.2e}]"
    )

    try:
        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=min(n_iter, 50),
            random_state=random_state,
            tol=1e-3,
        )

        with np.errstate(all="ignore"):
            model.fit(observations)

        # Validation
        if (
            hasattr(model, "transmat_")
            and not np.isfinite(model.transmat_).all()
            or hasattr(model, "means_")
            and not np.isfinite(model.means_).all()
        ):
            raise ValueError("Invalid transition matrix or means after fitting")
        
        # Validate covars_ shape before reordering
        if hasattr(model, "covars_"):
            expected_shape = (n_components, n_features)
            if model.covars_.shape != expected_shape:
                log_warn(
                    f"Invalid covars_ shape after fit: {model.covars_.shape}, "
                    f"expected {expected_shape}. Attempting to fix."
                )
                # Try to fix covars_ shape
                try:
                    if model.covars_.ndim == 3:
                        # If 3D (full covariance matrix), extract diagonal
                        # Shape: (n_components, n_features, n_features) -> (n_components, n_features)
                        model.covars_ = np.array([
                            np.diag(model.covars_[i]) for i in range(min(n_components, model.covars_.shape[0]))
                        ])
                        # Ensure we have the right number of components
                        if model.covars_.shape[0] < n_components:
                            model.covars_ = np.tile(
                                model.covars_[:1, :], (n_components, 1)
                            )
                        elif model.covars_.shape[0] > n_components:
                            model.covars_ = model.covars_[:n_components, :]
                        # Ensure positive values
                        model.covars_ = np.maximum(model.covars_, 1e-6)
                    elif model.covars_.ndim == 1:
                        # If 1D, reshape to (n_components, n_features)
                        model.covars_ = np.tile(
                            model.covars_.reshape(-1, 1), (1, n_features)
                        )[:n_components, :]
                        model.covars_ = np.maximum(model.covars_, 1e-6)
                    elif model.covars_.ndim == 2:
                        # If 2D but wrong shape, adjust
                        if model.covars_.shape[0] != n_components:
                            if model.covars_.shape[0] < n_components:
                                # Tile if too few components
                                model.covars_ = np.tile(
                                    model.covars_[:1, :], (n_components, 1)
                                )
                            else:
                                # Slice if too many components
                                model.covars_ = model.covars_[:n_components, :]
                        if model.covars_.shape[1] != n_features:
                            if model.covars_.shape[1] < n_features:
                                # Tile if too few features
                                model.covars_ = np.tile(
                                    model.covars_[:, :1], (1, n_features)
                                )
                            else:
                                # Slice if too many features
                                model.covars_ = model.covars_[:, :n_features]
                        # Ensure positive values
                        model.covars_ = np.maximum(model.covars_, 1e-6)
                except Exception as fix_error:
                    log_warn(f"Failed to fix covars_ shape: {fix_error}. Using default.")
                    model.covars_ = np.ones((n_components, n_features), dtype=np.float64) * 0.01
        
        # Reorder states to ensure semantic consistency (0=Bearish, 3=Bullish)
        model = reorder_hmm_model(model)
        
        # Final validation after reordering
        if hasattr(model, "covars_"):
            if model.covars_.ndim == 3:
                # Extract diagonal from full covariance matrices
                log_warn(
                    f"covars_ is 3D (full covariance): {model.covars_.shape}. "
                    f"Extracting diagonal to get shape ({n_components}, {n_features})."
                )
                diag_covars = np.array([
                    np.diag(model.covars_[i]) for i in range(min(n_components, model.covars_.shape[0]))
                ])
                # Ensure we have the right number of components
                if diag_covars.shape[0] < n_components:
                    diag_covars = np.tile(diag_covars[:1, :], (n_components, 1))
                elif diag_covars.shape[0] > n_components:
                    diag_covars = diag_covars[:n_components, :]
                # Ensure positive values
                diag_covars = np.maximum(diag_covars, 1e-6)
                # Set using object.__setattr__ to bypass property setter
                object.__setattr__(model, 'covars_', diag_covars)
            elif model.covars_.shape != (n_components, n_features):
                log_warn(
                    f"covars_ shape incorrect after reordering: {model.covars_.shape}, "
                    f"expected ({n_components}, {n_features}). Fixing."
                )
                try:
                    if model.covars_.ndim == 2 and model.covars_.size > 0:
                        # Use first component's covars and tile
                        first_covar = model.covars_[0, :] if model.covars_.shape[0] > 0 else model.covars_.flatten()[:n_features]
                        fixed_covars = np.tile(
                            first_covar.reshape(1, -1), (n_components, 1)
                        )
                        fixed_covars = np.maximum(fixed_covars, 1e-6)
                        object.__setattr__(model, 'covars_', fixed_covars)
                    else:
                        fixed_covars = np.ones((n_components, n_features), dtype=np.float64) * 0.01
                        object.__setattr__(model, 'covars_', fixed_covars)
                except Exception as e:
                    log_warn(f"Failed to fix covars_ shape: {e}. Using default.")
                    fixed_covars = np.ones((n_components, n_features), dtype=np.float64) * 0.01
                    object.__setattr__(model, 'covars_', fixed_covars)
        
        log_info("HMM training completed successfully")

    except Exception as e:
        log_error(f"HMM training failed: {str(e)}. Creating default model.")

        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=1,
            random_state=random_state,
        )

        # Initialize with safe defaults
        model.startprob_ = np.ones(n_components, dtype=np.float64) / n_components
        model.transmat_ = (
            np.eye(n_components, dtype=np.float64) * 0.7
            + np.ones((n_components, n_components), dtype=np.float64)
            * 0.3
            / n_components
        )

        # Create synthetic means sorted from low to high
        model.means_ = np.zeros((n_components, n_features), dtype=np.float64)
        for i in range(n_components):
            # Synthetic means: spread out based on quantiles
            model.means_[i] = np.quantile(
                observations, (i + 1) / (n_components + 1), axis=0
            )

        try:
            # Calculate variance for each feature
            variances = np.var(observations, axis=0)
            # Ensure variances is 1D array
            if variances.ndim == 0:
                variances = np.array([variances])
            
            # For diag covariance type, shape must be (n_components, n_features)
            # Broadcast variances to (n_components, n_features)
            model.covars_ = np.tile(
                variances.reshape(1, -1), (n_components, 1)
            ).astype(np.float64)
            
            # Ensure minimum variance to avoid numerical issues
            model.covars_ = np.maximum(model.covars_, 1e-6)
        except Exception as e:
            log_warn(f"Failed to set covars from data: {e}. Using default variance.")
            # Fallback: use uniform variance
            model.covars_ = np.ones((n_components, n_features), dtype=np.float64) * 0.01

    return model


def apply_hmm_model(
    model: GaussianHMM, data: pd.DataFrame, observations: np.ndarray
) -> Tuple[pd.DataFrame, int]:
    """Apply the trained HMM model to the data and predict hidden states."""
    predicted_states = model.predict(observations)

    # Pad or truncate to match data length
    if len(predicted_states) != len(data):
        if len(predicted_states) < len(data):
            last_state = predicted_states[-1] if len(predicted_states) > 0 else 0
            predicted_states = np.concatenate(
                [
                    predicted_states,
                    np.full(len(data) - len(predicted_states), last_state),
                ]
            )
        else:
            predicted_states = predicted_states[: len(data)]

    # Refactor: mapping respects sorted states (0 = lowest return, 3 = highest).
    state_mapping = {
        0: "bearish strong",
        1: "bearish weak",
        2: "bullish weak",
        3: "bullish strong",
    }

    data = data.copy()
    data["state"] = [state_mapping.get(s, f"State {s}") for s in predicted_states]

    last_state = predicted_states[-1] if len(predicted_states) > 0 else 0

    try:
        next_state_probs = model.transmat_[last_state]
        # Handle invalid probabilities
        if np.isnan(next_state_probs).any() or np.isinf(next_state_probs).any():
            next_state_probs = np.ones(len(next_state_probs)) / len(next_state_probs)

        next_state = int(np.argmax(next_state_probs))
    except (AttributeError, IndexError):
        next_state = 0

    return data, next_state

