"""
HMM-KAMA Secondary Analysis.

This module contains secondary analysis functions for duration, ARM, and clustering.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from modules.common.utils import log_warn, log_model, log_data


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

