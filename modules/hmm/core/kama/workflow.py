
from typing import Literal, Optional, cast

from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

from config import (

from config import (

"""
HMM-KAMA Main Workflow.

This module contains the main hmm_kama function that orchestrates the entire workflow.
"""



    HMM_WINDOW_KAMA_DEFAULT,
    HMM_WINDOW_SIZE_DEFAULT,
)
from modules.common.utils import log_error, log_warn
from modules.hmm.core.kama.analysis import (
    calculate_all_state_durations,
    compute_state_using_association_rule_mining,
    compute_state_using_hmm,
    compute_state_using_k_means,
    compute_state_using_standard_deviation,
)
from modules.hmm.core.kama.features import prepare_observations
from modules.hmm.core.kama.models import HMM_KAMA, apply_hmm_model, train_hmm
from modules.hmm.core.kama.utils import prevent_infinite_loop, timeout_context


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
                raise ValueError(f"Insufficient data: got {len(df)}, need at least {min_required}")

            # 2. Preprocessing
            df_clean = df.copy()
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                if pd.notna(df_clean[col].quantile(0.99)) and pd.notna(df_clean[col].quantile(0.01)):
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
            hmm_kama_result.next_state_with_hmm_kama = cast(Literal[0, 1, 2, 3], next_state)

            # 4. Secondary Analysis (Duration, ARM, Clustering)
            all_duration = calculate_all_state_durations(data)

            std_state_result = compute_state_using_standard_deviation(all_duration)
            hmm_kama_result.current_state_of_state_using_std = cast(Literal[0, 1], std_state_result)

            if all_duration["state"].nunique() <= 1:
                all_duration["state_encoded"] = 0
            else:
                all_duration["state_encoded"] = LabelEncoder().fit_transform(all_duration["state"])

            all_duration, last_hidden_state = compute_state_using_hmm(all_duration)
            hmm_state_clamped = min(1, max(0, last_hidden_state))
            hmm_kama_result.current_state_of_state_using_hmm = cast(Literal[0, 1], hmm_state_clamped)

            top_apriori, top_fpgrowth = compute_state_using_association_rule_mining(all_duration)
            hmm_kama_result.state_high_probabilities_using_arm_apriori = cast(Literal[0, 1, 2, 3], top_apriori)
            hmm_kama_result.state_high_probabilities_using_arm_fpgrowth = cast(Literal[0, 1, 2, 3], top_fpgrowth)

            kmeans_state_result = compute_state_using_k_means(all_duration)
            hmm_kama_result.current_state_of_state_using_kmeans = cast(Literal[0, 1], kmeans_state_result)

            return hmm_kama_result

    except Exception as e:
        log_error(f"Error in hmm_kama: {str(e)}")
        # Return safe default
        return HMM_KAMA(-1, -1, -1, -1, -1, -1)
