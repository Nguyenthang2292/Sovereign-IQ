"""Voting and aggregation helpers for VotingAnalyzer."""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.decision_matrix import DECISION_MATRIX_INDICATOR_ACCURACIES
from modules.decision_matrix.core.classifier import DecisionMatrixClassifier


class VotingVotingMixin:
    """Mixin for vote aggregation and decision matrix evaluation."""

    @contextmanager
    def _temporary_mode(self, new_mode: str):
        """Context manager to temporarily change spc_aggregator.config.mode (thread-safe)."""
        with self._mode_lock:
            original_mode = self.spc_aggregator.config.mode
            self.spc_aggregator.config.mode = new_mode
            try:
                yield
            finally:
                self.spc_aggregator.config.mode = original_mode

    def _aggregate_spc_votes(
        self,
        symbol_data: Dict[str, Any],
        signal_type: str,
        use_threshold_fallback: bool = False,
    ) -> Tuple[int, float]:
        """
        Aggregate 3 SPC strategy votes into a single vote.

        Uses SPCVoteAggregator with improved voting logic similar to Range Oscillator:
        - Separate LONG/SHORT weight calculation
        - Configurable consensus modes (threshold/weighted)
        - Optional adaptive weights based on performance
        - Signal strength filtering
        - Fallback to threshold mode if weighted mode gives no vote
        - Fallback to simple mode if both weighted and threshold give no vote

        Args:
            symbol_data: Symbol data with SPC signals
            signal_type: "LONG" or "SHORT"
            use_threshold_fallback: If True, force use threshold mode

        Returns:
            (vote, strength) where vote is 1 if matches expected signal_type, 0 otherwise
        """
        expected_signal = 1 if signal_type == "LONG" else -1

        # Use threshold mode if fallback requested
        if use_threshold_fallback or self.spc_aggregator.config.mode == "threshold":
            # Temporarily switch to threshold mode
            with self._temporary_mode("threshold"):
                vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)

            # If threshold mode also gives no vote, try simple mode fallback
            if vote == 0 and self.spc_aggregator.config.enable_simple_fallback:
                with self._temporary_mode("simple"):
                    vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)
        else:
            # Try weighted mode first
            vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)

            # If weighted mode gives no vote (vote = 0), fallback to threshold mode
            if vote == 0:
                with self._temporary_mode("threshold"):
                    vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)

                # If threshold mode also gives no vote, try simple mode fallback
                if vote == 0 and self.spc_aggregator.config.enable_simple_fallback:
                    with self._temporary_mode("simple"):
                        vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)

        # Convert vote to 1/0 format for Decision Matrix compatibility
        # Only accept vote if it matches the expected signal direction
        final_vote = 1 if vote == expected_signal else 0
        return (final_vote, strength)

    def _get_indicator_accuracy(self, indicator: str, signal_type: str) -> float:
        """Get historical accuracy for an indicator from config."""
        return DECISION_MATRIX_INDICATOR_ACCURACIES.get(indicator, 0.5)

    def apply_voting_system(
        self,
        signals_df: pd.DataFrame,
        signal_type: str,
        indicators_to_include: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply pure voting system to all signals.

        This is the core of Phương án 2 - no sequential filtering,
        just calculate all signals and vote.

        Args:
            signals_df: DataFrame with calculated signals
            signal_type: "LONG" or "SHORT"
            indicators_to_include: Optional list of indicator names to include in voting.
                If None, includes all enabled indicators.
                Valid names: "atc", "oscillator", "spc", "xgboost", "hmm", "random_forest"
        """
        if signals_df.empty:
            return pd.DataFrame()

        if indicators_to_include is None:
            # Include all enabled indicators
            indicators = ["atc", "oscillator"]
            if self.args.enable_spc:
                indicators.append("spc")
            if hasattr(self.args, "enable_xgboost") and self.args.enable_xgboost:
                indicators.append("xgboost")
            if hasattr(self.args, "enable_hmm") and self.args.enable_hmm:
                indicators.append("hmm")
            if hasattr(self.args, "enable_random_forest") and self.args.enable_random_forest:
                indicators.append("random_forest")
        else:
            # Only include specified indicators
            indicators = indicators_to_include.copy()

        results = []

        for _, row in signals_df.iterrows():
            # Build dynamic indicators list based on actual vote data availability
            # This prevents errors when an indicator is in the list but has no vote data
            available_indicators = []

            # Check which indicators actually have vote data for this row
            if "atc" in indicators and row.get("atc_vote") is not None:
                available_indicators.append("atc")
            if "oscillator" in indicators and row.get("osc_vote") is not None:
                available_indicators.append("oscillator")
            if "spc" in indicators and (
                row.get("spc_cluster_transition_signal") is not None
                or row.get("spc_regime_following_signal") is not None
                or row.get("spc_mean_reversion_signal") is not None
            ):
                available_indicators.append("spc")
            if "xgboost" in indicators and row.get("xgboost_vote") is not None:
                available_indicators.append("xgboost")
            if "hmm" in indicators and row.get("hmm_vote") is not None:
                available_indicators.append("hmm")
            if "random_forest" in indicators and row.get("random_forest_vote") is not None:
                available_indicators.append("random_forest")

            # Skip this row if no indicators have vote data
            if not available_indicators:
                continue

            classifier = DecisionMatrixClassifier(indicators=available_indicators)

            # Get votes from all indicators (only include those in available_indicators list)
            if "atc" in available_indicators:
                atc_vote = row.get("atc_vote", 0)
                atc_strength = row.get("atc_strength", 0.0)
                classifier.add_node_vote(
                    "atc", atc_vote, atc_strength, self._get_indicator_accuracy("atc", signal_type)
                )

            if "oscillator" in available_indicators:
                osc_vote = row.get("osc_vote", 0)
                osc_strength = row.get("osc_confidence", 0.0)
                classifier.add_node_vote(
                    "oscillator", osc_vote, osc_strength, self._get_indicator_accuracy("oscillator", signal_type)
                )

            if "spc" in available_indicators:
                # Aggregate SPC votes from 3 strategies
                spc_vote, spc_strength = self._aggregate_spc_votes(row.to_dict(), signal_type)
                classifier.add_node_vote(
                    "spc", spc_vote, spc_strength, self._get_indicator_accuracy("spc", signal_type)
                )

            if "xgboost" in available_indicators:
                # XGBoost vote
                xgb_vote = row.get("xgboost_vote", 0)
                xgb_strength = row.get("xgboost_confidence", 0.0)
                classifier.add_node_vote(
                    "xgboost", xgb_vote, xgb_strength, self._get_indicator_accuracy("xgboost", signal_type)
                )

            if "hmm" in available_indicators:
                # HMM vote
                hmm_vote = row.get("hmm_vote", 0)
                hmm_strength = row.get("hmm_confidence", 0.0)
                classifier.add_node_vote(
                    "hmm", hmm_vote, hmm_strength, self._get_indicator_accuracy("hmm", signal_type)
                )

            if "random_forest" in available_indicators:
                # Random Forest vote
                rf_vote = row.get("random_forest_vote", 0)
                rf_strength = row.get("random_forest_confidence", 0.0)
                classifier.add_node_vote(
                    "random_forest", rf_vote, rf_strength, self._get_indicator_accuracy("random_forest", signal_type)
                )

            classifier.calculate_weighted_impact()

            cumulative_vote, weighted_score, voting_breakdown = classifier.calculate_cumulative_vote(
                threshold=self.args.voting_threshold,
                min_votes=self.args.min_votes,
            )
            # Only keep if cumulative vote is positive
            if cumulative_vote == 1:
                result = row.to_dict()
                result["cumulative_vote"] = cumulative_vote
                result["weighted_score"] = weighted_score
                result["voting_breakdown"] = voting_breakdown

                metadata = classifier.get_metadata()
                result["feature_importance"] = metadata["feature_importance"]
                result["weighted_impact"] = metadata["weighted_impact"]
                result["independent_accuracy"] = metadata["independent_accuracy"]

                votes_count = sum(v for v in classifier.node_votes.values())
                if votes_count == len(indicators):
                    result["source"] = "ALL_INDICATORS"
                elif votes_count >= self.args.min_votes:
                    result["source"] = "MAJORITY_VOTE"
                else:
                    result["source"] = "WEIGHTED_VOTE"

                results.append(result)

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("weighted_score", ascending=False).reset_index(drop=True)

        return result_df
