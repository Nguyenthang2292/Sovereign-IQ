from contextlib import contextmanager
from dataclasses import dataclass
import functools
import os
import threading
import warnings
from typing import Literal, Optional, Tuple, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.hmm.signals.strategy import HMMStrategyResult

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Fix KMeans memory leak on Windows with MKL
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings(
    "ignore", message="KMeans is known to have a memory leak on Windows with MKL"
)

from modules.common.indicators import calculate_kama
from modules.common.utils import log_data, log_info, log_error, log_warn, log_model, log_analysis
from config import (
    HMM_WINDOW_KAMA_DEFAULT,
    HMM_FAST_KAMA_DEFAULT,
    HMM_SLOW_KAMA_DEFAULT,
    HMM_WINDOW_SIZE_DEFAULT,
)

@dataclass
class HMM_KAMA:
    next_state_with_hmm_kama: Literal[-1, 0, 1, 2, 3]
    current_state_of_state_using_std: Literal[-1, 0, 1]
    current_state_of_state_using_hmm: Literal[-1, 0, 1]
    state_high_probabilities_using_arm_apriori: Literal[-1, 0, 1, 2, 3]
    state_high_probabilities_using_arm_fpgrowth: Literal[-1, 0, 1, 2, 3]
    current_state_of_state_using_kmeans: Literal[-1, 0, 1]

def prepare_observations(
    data: pd.DataFrame,
    window_kama: Optional[int] = None,
    fast_kama: Optional[int] = None,
    slow_kama: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Generate crypto-optimized observation features.

    Uses price minus KAMA deviation to keep inputs closer to stationarity.
    
    Args:
        data: DataFrame with OHLCV data
        window_kama: KAMA window size (default: from config)
        fast_kama: Fast KAMA parameter (default: from config)
        slow_kama: Slow KAMA parameter (default: from config)
    """
    if data.empty or "close" not in data.columns or len(data) < 10:
        raise ValueError(
            f"Invalid data: empty={data.empty}, has close={'close' in data.columns}, len={len(data)}"
        )

    close_prices = data["close"].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if close_prices.isna().any():  # type: ignore
        close_prices = close_prices.fillna(close_prices.median())

    # Normalize prices to 0-1000 scale for consistency (UPDATED)
    price_range = close_prices.max() - close_prices.min()
    unique_prices = close_prices.nunique()

    if price_range == 0 or unique_prices < 3:
        log_data(f"Problematic price data: range={price_range}, unique_prices={unique_prices}")
        close_prices = pd.Series(
            np.linspace(
                close_prices.mean() * 0.95,
                close_prices.mean() * 1.05,
                len(close_prices),
            )
        )
        price_range = close_prices.max() - close_prices.min()

    close_prices_norm = (
        ((close_prices - close_prices.min()) / price_range * 1000)
        if price_range > 0
        else pd.Series(np.linspace(450, 550, len(close_prices)))
    )
    close_prices_array = close_prices_norm.values.astype(np.float64)

    # 1. Calculate KAMA
    try:
        window_param = int(window_kama if window_kama is not None else HMM_WINDOW_KAMA_DEFAULT)
        fast = int(fast_kama if fast_kama is not None else HMM_FAST_KAMA_DEFAULT)
        slow_raw = int(slow_kama if slow_kama is not None else HMM_SLOW_KAMA_DEFAULT)
        slow = max(slow_raw, fast + 5)

        window = max(2, min(window_param, len(close_prices_array) // 2))

        kama_values = calculate_kama(
            close_prices_array, window=window, fast=fast, slow=slow
        )

        if np.max(kama_values) - np.min(kama_values) < 1e-10:
            log_data("KAMA has zero variance. Adding gradient.")
            kama_values = np.linspace(
                kama_values[0] - 0.5, kama_values[0] + 0.5, len(kama_values)
            )

    except Exception as e:
        log_error(f"KAMA calculation failed: {e}. Using EMA fallback.")
        kama_values = (
            pd.Series(close_prices_array)
            .ewm(alpha=2.0 / (window_param + 1), adjust=False)
            .mean()
            .values
        )

    # 2. Calculate Features

    # Feature 1: Returns (Stationary)
    returns = np.diff(close_prices_array, prepend=close_prices_array[0])
    if np.std(returns) < 1e-10:
        log_warn("Returns have zero variance. Returning None (Neutral).")
        return None

    # Feature 2: Price Deviation from KAMA (Stationary-ish)
    kama_deviation = close_prices_array - kama_values

    # Feature 3: Volatility (Stationary)
    # Using change in KAMA as a proxy for trend strength/volatility
    volatility = np.abs(np.diff(np.array(kama_values), prepend=kama_values[0]))
    if np.std(volatility) < 1e-10:
        log_warn("Volatility has zero variance. Returning None (Neutral).")
        return None

    rolling_vol = (
        pd.Series(returns).rolling(window=5, min_periods=1).std().fillna(0.01).values
    )
    volatility = (volatility + np.asarray(rolling_vol)) / 2

    # Cleaning
    def _clean_crypto_array(arr, name="array", default_val=0.0):
        arr = np.where(np.isfinite(arr), arr, default_val)
        valid_values = arr[arr != default_val]
        q_range = (
            max(
                float(abs(np.percentile(valid_values, 95))),
                float(abs(np.percentile(valid_values, 5))),
            )
            * 1.5
            if np.any(valid_values)
            else 1000
        )
        return np.clip(arr, -q_range, q_range).astype(np.float64)

    returns = _clean_crypto_array(returns, "returns", 0.0)
    kama_deviation = _clean_crypto_array(kama_deviation, "kama_deviation", 0.0)
    volatility = _clean_crypto_array(volatility, "volatility", 0.01)

    # Final variance check
    if np.std(returns) == 0:
        returns[0] = 0.01
    if np.std(kama_deviation) == 0:
        kama_deviation[-1] += 0.01
    if np.std(volatility) == 0:
        volatility[0], volatility[-1] = 0.005, 0.015

    feature_matrix = np.column_stack([returns, kama_deviation, volatility])

    if not np.isfinite(feature_matrix).all():
        log_error("Feature matrix contains invalid values. Returning None (Neutral).")
        return None

    log_analysis(
        f"Crypto-optimized features - Shape: {feature_matrix.shape}, "
        f"Returns range: [{returns.min():.6f}, {returns.max():.6f}], "
        f"Deviation range: [{kama_deviation.min():.6f}, {kama_deviation.max():.6f}], "
        f"Volatility range: [{volatility.min():.6f}, {volatility.max():.6f}]"
    )

    return feature_matrix

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

# -----------------------------------------------------------------------------
# Secondary Analysis Functions (Duration, ARM, Clustering)
# -----------------------------------------------------------------------------

def compute_state_using_standard_deviation(durations: pd.DataFrame) -> int:
    """Return 0 if last duration stays within mean ± std, else 1; empty -> 0."""
    if durations.empty:
        return 0
    mean_duration, std_duration = (
        durations["duration"].mean(),
        durations["duration"].std(),
    )
    last_duration = durations.iloc[-1]["duration"]
    # If duration is within 1 std dev, return 0, else 1
    return (
        0
        if (
            mean_duration - std_duration
            <= last_duration
            <= mean_duration + std_duration
        )
        else 1
    )

def compute_state_using_hmm(durations: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Computes hidden states from duration data using a Gaussian HMM."""
    if len(durations) < 2:
        durations_copy = durations.copy()
        durations_copy["hidden_state"] = 0
        return durations_copy, 0

    try:
        model = GaussianHMM(
            n_components=min(2, len(durations)),
            covariance_type="diag",
            n_iter=10,
            random_state=36,
        )
        model.fit(durations[["duration"]].values)
        hidden_states = model.predict(durations[["duration"]].values)
        durations_copy = durations.copy()
        durations_copy["hidden_state"] = hidden_states
        return durations_copy, int(hidden_states[-1])

    except Exception as e:
        log_model(f"Duration HMM fitting failed: {e}. Using default state assignment.")
        durations_copy = durations.copy()
        durations_copy["hidden_state"] = 0
        return durations_copy, 0

def calculate_composite_scores_association_rule_mining(
    rules: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate composite score with better infinity handling"""
    if rules.empty:
        return rules

    numeric_cols = rules.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in rules.columns:
            rules[col] = rules[col].replace([np.inf, -np.inf], np.nan)
            fill_value = rules[col].median() if rules[col].notna().any() else 0.0  # type: ignore
            rules[col] = rules[col].fillna(fill_value)

    metrics = [
        m
        for m in [
            "antecedent support",
            "consequent support",
            "support",
            "confidence",
            "lift",
            "representativity",
            "leverage",
            "conviction",
            "zhangs_metric",
            "jaccard",
            "certainty",
            "kulczynski",
        ]
        if m in rules.columns
    ]

    if not metrics:
        rules["composite_score"] = 0.0
        return rules

    rules_normalized = rules.copy()

    if len(rules_normalized) > 0:
        try:
            for metric in metrics:
                values = np.where(
                    np.isfinite(rules_normalized[metric].values.astype(np.float64)),
                    rules_normalized[metric].values.astype(np.float64),
                    0.0,
                )
                mean_val, std_val = np.mean(values), np.std(values)

                if std_val > 0 and np.isfinite(std_val):
                    rules_normalized[metric] = np.clip(
                        (values - mean_val) / std_val, -5, 5
                    )
                else:
                    rules_normalized[metric] = 0.0

        except Exception as e:
            log_data(f"Manual normalization failed: {e}. Using raw values.")
            pass

    rules_normalized["composite_score"] = (
        rules_normalized[metrics].mean(axis=1) if metrics else 0.0
    )

    return rules_normalized.sort_values(by="composite_score", ascending=False)

def compute_state_using_association_rule_mining(
    durations: pd.DataFrame,
) -> Tuple[int, int]:
    """Return (apriori_state, fpgrowth_state) derived from ARM on durations."""
    if durations.empty:
        return 0, 0

    bins, labels = [0, 15, 30, 100], ["state_1", "state_2", "state_3"]
    # Handle outliers in duration
    max_duration = durations["duration"].max()
    if max_duration > 100:
        bins = [0, 15, 30, max_duration + 1]

    durations["duration_bin"] = pd.cut(
        durations["duration"], bins=bins, labels=labels, right=False
    )
    durations["transaction"] = durations[["state", "duration_bin"]].apply(
        lambda x: [str(x["state"]), str(x["duration_bin"])], axis=1
    )

    te = TransactionEncoder()
    try:
        te_ary = te.fit(durations["transaction"]).transform(durations["transaction"])
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)  # type: ignore
    except Exception as e:
        log_warn(f"TransactionEncoder failed: {e}")
        return 0, 0

    # Helper for mining
    def mine_rules(method_func, df_trans):
        frequent_itemsets = pd.DataFrame()
        for min_support_val in [0.2, 0.15, 0.1, 0.05]:
            try:
                frequent_itemsets = method_func(
                    df_trans, min_support=min_support_val, use_colnames=True
                )
                if not frequent_itemsets.empty:
                    break
            except Exception:
                continue

        if frequent_itemsets.empty:
            return pd.DataFrame()

        try:
            return association_rules(
                frequent_itemsets, metric="confidence", min_threshold=0.6
            )
        except Exception:
            return pd.DataFrame()

    rules_apriori = mine_rules(apriori, df_transactions)
    rules_apriori_sorted = calculate_composite_scores_association_rule_mining(
        rules_apriori
    )
    top_antecedents_apriori = (
        rules_apriori_sorted.iloc[0]["antecedents"]
        if not rules_apriori_sorted.empty
        else frozenset()
    )

    rules_fpgrowth = mine_rules(fpgrowth, df_transactions)
    rules_fpgrowth_sorted = calculate_composite_scores_association_rule_mining(
        rules_fpgrowth
    )
    top_antecedents_fpgrowth = (
        rules_fpgrowth_sorted.iloc[0]["antecedents"]
        if not rules_fpgrowth_sorted.empty
        else frozenset()
    )

    top_apriori, top_fpgrowth = 0, 0
    STATE_MAPPING = {
        "bearish weak": 1,
        "bullish weak": 2,
        "bearish strong": 0,
        "bullish strong": 3,
    }

    for item in top_antecedents_apriori:
        if item in STATE_MAPPING:
            top_apriori = STATE_MAPPING[item]
            break

    for item in top_antecedents_fpgrowth:
        if item in STATE_MAPPING:
            top_fpgrowth = STATE_MAPPING[item]
            break

    return top_apriori, top_fpgrowth

def compute_state_using_k_means(durations: pd.DataFrame) -> int:
    """Cluster durations via K-Means; return latest cluster label, fallback 0."""
    if len(durations) < 3:
        return 0

    try:
        kmeans = KMeans(n_clusters=2, random_state=42, max_iter=300)
        durations["cluster"] = kmeans.fit_predict(durations[["duration"]])
    except Exception as e:
        log_model(f"K-Means clustering failed: {e}. Using default cluster 0.")
        durations["cluster"] = 0

    return int(durations.iloc[-1]["cluster"])

def calculate_all_state_durations(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the duration of all consecutive state segments."""
    df = data.copy()
    df["group"] = (df["state"] != df["state"].shift()).cumsum()

    return (
        df.groupby("group")
        .agg(  # type: ignore
            state=("state", "first"),
            start_time=("state", lambda s: s.index[0]),
            duration=("state", "size"),
        )
        .reset_index(drop=True)
    )

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

_thread_local = threading.local()

def prevent_infinite_loop(max_calls=3):
    """Decorator to prevent infinite loops in function calls"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(_thread_local, "call_counts"):
                _thread_local.call_counts = {}

            func_name = func.__name__
            if func_name not in _thread_local.call_counts:
                _thread_local.call_counts[func_name] = 0

            _thread_local.call_counts[func_name] += 1

            try:
                if _thread_local.call_counts[func_name] > 1:
                    log_warn(
                        f"Multiple calls detected for {func_name} ({_thread_local.call_counts[func_name]}). Possible infinite loop."
                    )
                    if _thread_local.call_counts[func_name] > max_calls:
                        log_error(
                            f"Too many recursive calls for {func_name}. Breaking to prevent infinite loop."
                        )
                        return HMM_KAMA(-1, -1, -1, -1, -1, -1)

                return func(*args, **kwargs)
            finally:
                _thread_local.call_counts[func_name] = 0

        return wrapper

    return decorator

@contextmanager
def timeout_context(seconds):
    """Cross-platform timeout context manager"""
    timeout_occurred = threading.Event()
    timer = threading.Timer(seconds, timeout_occurred.set)
    timer.start()

    try:
        yield
        if timeout_occurred.is_set():
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
    finally:
        timer.cancel()

@prevent_infinite_loop(max_calls=3)
def hmm_kama(
    df: pd.DataFrame,
    window_kama: Optional[int] = None,
    fast_kama: Optional[int] = None,
    slow_kama: Optional[int] = None,
    window_size: Optional[int] = None,
) -> HMM_KAMA:
    """Run the full HMM-KAMA workflow on the provided dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        window_kama: KAMA window size (default: from config)
        fast_kama: Fast KAMA parameter (default: from config)
        slow_kama: Slow KAMA parameter (default: from config)
        window_size: Rolling window size (default: from config)
    """
    try:
        with timeout_context(30):
            # 1. Validation
            if df is None or df.empty or "close" not in df.columns or len(df) < 20:
                raise ValueError(
                    f"Invalid DataFrame: empty={df.empty if df is not None else True}, has close={'close' in df.columns if df is not None else False}, len={len(df) if df is not None else 0}"
                )

            if df["close"].std() == 0 or pd.isna(df["close"].std()):
                raise ValueError("Price data has no variance")

            window_param = int(window_kama if window_kama is not None else HMM_WINDOW_KAMA_DEFAULT)
            window_size_val = int(window_size if window_size is not None else HMM_WINDOW_SIZE_DEFAULT)
            min_required = max(window_param, window_size_val, 10)
            if len(df) < min_required:
                raise ValueError(
                    f"Insufficient data: got {len(df)}, need at least {min_required}"
                )

            # 2. Preprocessing
            df_clean = df.copy()
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                if pd.notna(df_clean[col].quantile(0.99)) and pd.notna(
                    df_clean[col].quantile(0.01)
                ):
                    df_clean[col] = df_clean[col].clip(
                        lower=df_clean[col].quantile(0.01) * 10,
                        upper=df_clean[col].quantile(0.99) * 10,
                    )

            df_clean = df_clean.ffill().bfill()

            for col in numeric_cols:
                if df_clean[col].isna().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

            hmm_kama_result = HMM_KAMA(-1, -1, -1, -1, -1, -1)

            # 3. Model Training & Prediction
            observations = prepare_observations(df_clean, window_kama, fast_kama, slow_kama)
            if observations is None:
                log_warn("Insufficient data variance for HMM. Returning Neutral state.")
                return HMM_KAMA(-1, -1, -1, -1, -1, -1)

            model = train_hmm(observations, n_components=4)
            data, next_state = apply_hmm_model(model, df_clean, observations)
            hmm_kama_result.next_state_with_hmm_kama = cast(
                Literal[0, 1, 2, 3], next_state
            )

            # 4. Secondary Analysis (Duration, ARM, Clustering)
            all_duration = calculate_all_state_durations(data)

            hmm_kama_result.current_state_of_state_using_std = cast(
                Literal[0, 1], compute_state_using_standard_deviation(all_duration)
            )

            if all_duration["state"].nunique() <= 1:
                all_duration["state_encoded"] = 0
            else:
                all_duration["state_encoded"] = LabelEncoder().fit_transform(
                    all_duration["state"]
                )

            all_duration, last_hidden_state = compute_state_using_hmm(all_duration)
            hmm_kama_result.current_state_of_state_using_hmm = cast(
                Literal[0, 1], min(1, max(0, last_hidden_state))
            )

            top_apriori, top_fpgrowth = compute_state_using_association_rule_mining(
                all_duration
            )
            hmm_kama_result.state_high_probabilities_using_arm_apriori = cast(
                Literal[0, 1, 2, 3], top_apriori
            )
            hmm_kama_result.state_high_probabilities_using_arm_fpgrowth = cast(
                Literal[0, 1, 2, 3], top_fpgrowth
            )

            hmm_kama_result.current_state_of_state_using_kmeans = cast(
                Literal[0, 1], compute_state_using_k_means(all_duration)
            )

            return hmm_kama_result

    except Exception as e:
        log_error(f"Error in hmm_kama: {str(e)}")
        # Return safe default
        return HMM_KAMA(-1, -1, -1, -1, -1, -1)

# ============================================================================
# Strategy Interface Implementation
# ============================================================================

class KamaHMMStrategy:
    """
    HMM Strategy wrapper for HMM-KAMA.
    
    Implements HMMStrategy interface to enable registry-based management.
    """
    
    def __init__(
        self,
        name: str = "kama",
        weight: float = 1.5,
        enabled: bool = True,
        window_kama: Optional[int] = None,
        fast_kama: Optional[int] = None,
        slow_kama: Optional[int] = None,
        window_size: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize KAMA HMM Strategy.
        
        Args:
            name: Strategy name (default: "kama")
            weight: Strategy weight for voting (default: 1.5)
            enabled: Whether strategy is enabled (default: True)
            window_kama: KAMA window size
            fast_kama: Fast KAMA parameter
            slow_kama: Slow KAMA parameter
            window_size: Rolling window size
            **kwargs: Additional parameters
        """
        from modules.hmm.signals.strategy import HMMStrategy, HMMStrategyResult
        from modules.hmm.signals.resolution import LONG, HOLD, SHORT
        
        self.name = name
        self.weight = weight
        self.enabled = enabled
        self.window_kama = (
            window_kama if window_kama is not None else HMM_WINDOW_KAMA_DEFAULT
        )
        self.fast_kama = (
            fast_kama if fast_kama is not None else HMM_FAST_KAMA_DEFAULT
        )
        self.slow_kama = (
            slow_kama if slow_kama is not None else HMM_SLOW_KAMA_DEFAULT
        )
        self.window_size = (
            window_size if window_size is not None else HMM_WINDOW_SIZE_DEFAULT
        )
        self.params = kwargs
    
    def analyze(self, df: pd.DataFrame, **kwargs) -> 'HMMStrategyResult':
        """
        Analyze market data using HMM-KAMA.
        
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
        
        # Run HMM-KAMA analysis
        result = hmm_kama(
            df,
            window_kama=self.window_kama,
            fast_kama=self.fast_kama,
            slow_kama=self.slow_kama,
            window_size=self.window_size,
        )
        
        # Map HMM_KAMA state to Signal
        # States: 0,1,2,3 -> 0,2 are bearish, 1,3 are bullish
        primary_state = result.next_state_with_hmm_kama
        
        # Convert state to signal
        # States 1,3 = bullish (LONG), States 0,2 = bearish (SHORT)
        if primary_state in {1, 3}:
            signal = LONG
        elif primary_state in {0, 2}:
            signal = SHORT
        else:
            signal = HOLD
        
        # Calculate confidence from transition states and ARM states
        # Use a simple heuristic: count bullish/bearish indicators
        bullish_indicators = 0
        bearish_indicators = 0
        
        # Transition states
        if result.current_state_of_state_using_std == 1:
            bullish_indicators += 1
        elif result.current_state_of_state_using_std == -1:
            bearish_indicators += 1
        
        if result.current_state_of_state_using_hmm == 1:
            bullish_indicators += 1
        elif result.current_state_of_state_using_hmm == -1:
            bearish_indicators += 1
        
        if result.current_state_of_state_using_kmeans == 1:
            bullish_indicators += 1
        elif result.current_state_of_state_using_kmeans == -1:
            bearish_indicators += 1
        
        # ARM states
        if result.state_high_probabilities_using_arm_apriori in {1, 3}:
            bullish_indicators += 1
        elif result.state_high_probabilities_using_arm_apriori in {0, 2}:
            bearish_indicators += 1
        
        if result.state_high_probabilities_using_arm_fpgrowth in {1, 3}:
            bullish_indicators += 1
        elif result.state_high_probabilities_using_arm_fpgrowth in {0, 2}:
            bearish_indicators += 1
        
        # Calculate probability based on indicator agreement
        total_indicators = bullish_indicators + bearish_indicators
        if total_indicators > 0:
            if signal == LONG:
                probability = max(0.5, bullish_indicators / total_indicators)
            elif signal == SHORT:
                probability = max(0.5, bearish_indicators / total_indicators)
            else:
                probability = 0.5
        else:
            probability = 0.5
        
        metadata = {
            "primary_state": primary_state,
            "transition_std": result.current_state_of_state_using_std,
            "transition_hmm": result.current_state_of_state_using_hmm,
            "transition_kmeans": result.current_state_of_state_using_kmeans,
            "arm_apriori": result.state_high_probabilities_using_arm_apriori,
            "arm_fpgrowth": result.state_high_probabilities_using_arm_fpgrowth,
            "bullish_indicators": bullish_indicators,
            "bearish_indicators": bearish_indicators,
        }
        
        return HMMStrategyResult(
            signal=signal,
            probability=probability,
            state=primary_state,
            metadata=metadata
        )