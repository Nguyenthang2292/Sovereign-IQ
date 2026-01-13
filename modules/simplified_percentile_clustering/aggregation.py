"""
SPC Vote Aggregation Module.

This module provides aggregation logic for combining votes from multiple SPC strategies
into a single vote, similar to Range Oscillator's combined strategy approach.

Features:
- Weighted voting system (similar to Range Oscillator)
- Separate LONG/SHORT weight calculation
- Adaptive weights based on performance (optional)
- Confidence score calculation
- Signal strength filtering
- Multiple consensus modes (threshold, weighted)
"""

from typing import Dict, Optional, Tuple

import numpy as np

from config import (
    AGREEMENT_WEIGHT,
    DECISION_MATRIX_SPC_STRATEGY_ACCURACIES,
    STRENGTH_WEIGHT,
)
from modules.simplified_percentile_clustering.config import (
    SPCAggregationConfig,
)


class SPCVoteAggregator:
    """
    Aggregates votes from 3 SPC strategies into a single vote.

    Similar to Range Oscillator's CombinedStrategy, but specifically designed
    for SPC strategy aggregation.
    """

    def __init__(self, config: Optional[SPCAggregationConfig] = None):
        """
        Initialize SPC Vote Aggregator.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or SPCAggregationConfig()

        # Strategy names
        self.strategy_names = ["cluster_transition", "regime_following", "mean_reversion"]

        # Base weights from accuracy (can be overridden by adaptive or custom weights)
        self.base_weights = DECISION_MATRIX_SPC_STRATEGY_ACCURACIES.copy()

    def _calculate_adaptive_weights(
        self,
        signals_history: Dict[str, list],
        strengths_history: Dict[str, list],
    ) -> Dict[str, float]:
        """
        Calculate adaptive weights based on recent performance.

        Similar to Range Oscillator's adaptive weight calculation.

        Args:
            signals_history: Dict mapping strategy name to list of recent signals (-1, 0, 1)
            strengths_history: Dict mapping strategy name to list of recent strengths

        Returns:
            Dict mapping strategy name to adaptive weight
        """
        # Find first strategy with history to determine window size
        valid_strategies = [s for s in self.strategy_names if s in signals_history]
        if not valid_strategies:
            return self.base_weights.copy()

        window = min(self.config.adaptive_performance_window, len(signals_history[valid_strategies[0]]))

        if window < 5:  # Not enough data
            return self.base_weights.copy()

        # Calculate consensus (majority vote) for each historical point
        consensus_history = []
        for i in range(window):
            votes = []
            for strategy in self.strategy_names:
                if strategy in signals_history and i < len(signals_history[strategy]):
                    votes.append(signals_history[strategy][i])
            if votes:
                consensus = np.sign(sum(votes))  # -1, 0, or 1
                consensus_history.append(consensus)
            else:
                consensus_history.append(0)

        strategy_scores = {}

        for strategy in self.strategy_names:
            if strategy not in signals_history or strategy not in strengths_history:
                strategy_scores[strategy] = self.base_weights.get(strategy, 0.5)
                continue

            signals = signals_history[strategy][-window:]
            strengths = strengths_history[strategy][-window:]

            # Agreement: how often strategy agrees with consensus
            matches = [s == c for s, c in zip(signals, consensus_history) if c != 0]
            agreement = np.mean(matches) if matches else 0.5

            # Average strength when strategy is active
            active_mask = [abs(s) > 0 for s in signals]
            avg_strength = (
                np.mean([s for s, active in zip(strengths, active_mask) if active]) if any(active_mask) else 0.0
            )

            # Combined score (similar to Range Oscillator)
            score = (agreement * AGREEMENT_WEIGHT) + (avg_strength * STRENGTH_WEIGHT)
            strategy_scores[strategy] = max(0.1, score)  # Minimum weight

        # Normalize weights to sum to total of base weights
        total_base = sum(self.base_weights.values())
        total_score = sum(strategy_scores.values())
        if total_score > 0:
            strategy_scores = {k: v / total_score * total_base for k, v in strategy_scores.items()}

        return strategy_scores

    def _simple_mode_vote(
        self,
        strategy_signals: Dict[str, int],
        strategy_strengths: Dict[str, float],
        signal_type: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> int:
        """
        Simple mode voting: calculate signal based on accuracy sum of active strategies.

        This mode doesn't distinguish between LONG/SHORT - it just sums accuracy
        of all strategies that have a signal (non-zero), regardless of direction.

        Args:
            strategy_signals: Dict mapping strategy name to signal (-1, 0, 1)
            strategy_strengths: Dict mapping strategy name to strength (0.0-1.0)
            signal_type: "LONG" or "SHORT" (for determining expected direction)
            weights: Optional weights to use. If None, uses base_weights.

        Returns:
            Vote: 1 if LONG signal, -1 if SHORT signal, 0 if no signal
        """
        use_weights = weights or self.base_weights

        # Calculate accuracy sum for LONG strategies (signal == 1)
        long_accuracy_sum = sum(
            use_weights.get(strategy, 0.0) for strategy, signal in strategy_signals.items() if signal == 1
        )

        # Calculate accuracy sum for SHORT strategies (signal == -1)
        short_accuracy_sum = sum(
            use_weights.get(strategy, 0.0) for strategy, signal in strategy_signals.items() if signal == -1
        )

        # Total accuracy of all active strategies (regardless of direction)
        total_accuracy = long_accuracy_sum + short_accuracy_sum

        # Check if total accuracy meets minimum threshold
        if total_accuracy < self.config.simple_min_accuracy_total:
            return 0

        # Determine signal based on which direction has higher accuracy
        # If equal, prefer the expected signal_type
        if long_accuracy_sum > short_accuracy_sum:
            return 1
        elif short_accuracy_sum > long_accuracy_sum:
            return -1
        else:
            # Equal accuracy: return based on signal_type preference
            return 1 if signal_type == "LONG" else -1

    def aggregate(
        self,
        symbol_data: Dict,
        signal_type: str,
        signals_history: Optional[Dict[str, list]] = None,
        strengths_history: Optional[Dict[str, list]] = None,
    ) -> Tuple[int, float, float]:
        """
        Aggregate 3 SPC strategy votes into a single vote.

        Similar to Range Oscillator's voting logic but for SPC strategies.

        Args:
            symbol_data: Dict containing SPC strategy signals and strengths
            signal_type: "LONG" or "SHORT"
            signals_history: Optional history of signals for adaptive weights
            strengths_history: Optional history of strengths for adaptive weights

        Returns:
            Tuple of (vote, strength, confidence)
            - vote: 1 if LONG signal, -1 if SHORT signal, 0 if no signal
            - strength: Aggregated signal strength (0.0-1.0)
            - confidence: Confidence score (0.0-1.0)
        """
        expected_signal = 1 if signal_type == "LONG" else -1

        # Get signals and strengths from all 3 strategies
        strategy_signals = {}
        strategy_strengths = {}
        strategy_votes = {}  # For threshold mode: 1 if matches expected_signal, 0 otherwise

        for strategy in self.strategy_names:
            signal_key = f"spc_{strategy}_signal"
            strength_key = f"spc_{strategy}_strength"

            signal = symbol_data.get(signal_key, 0)
            strength = symbol_data.get(strength_key, 0.0)

            strategy_signals[strategy] = signal
            strategy_strengths[strategy] = max(0.0, min(1.0, strength))

            # Vote: 1 if matches expected signal, 0 otherwise (for threshold mode)
            vote = 1 if signal == expected_signal else 0
            strategy_votes[strategy] = vote

        # Calculate weights (raw, before normalization)
        if self.config.strategy_weights:
            raw_weights = self.config.strategy_weights.copy()
        elif self.config.enable_adaptive_weights and signals_history and strengths_history:
            raw_weights = self._calculate_adaptive_weights(signals_history, strengths_history)
        else:
            raw_weights = self.base_weights.copy()

        # Normalize weights for weighted/threshold modes
        total_weight = sum(raw_weights.values())
        if total_weight == 0:
            return (0, 0.0, 0.0)

        weights = {k: v / total_weight for k, v in raw_weights.items()}

        # Calculate LONG and SHORT weights separately (like Range Oscillator)
        # LONG weight: sum of weights for strategies with signal == 1
        long_weight = sum(weights[strategy] for strategy, signal in strategy_signals.items() if signal == 1)

        # SHORT weight: sum of weights for strategies with signal == -1
        short_weight = sum(weights[strategy] for strategy, signal in strategy_signals.items() if signal == -1)

        # Voting logic based on mode
        if self.config.mode == "simple":
            # Simple mode: accuracy-based voting without LONG/SHORT distinction
            # Use raw_weights (not normalized) for accuracy calculation
            final_vote = self._simple_mode_vote(strategy_signals, strategy_strengths, signal_type, raw_weights)

            # Calculate weighted strength for simple mode: average of all active strategies
            active_strengths = [
                strategy_strengths[strategy] for strategy in self.strategy_names if strategy_signals[strategy] != 0
            ]
            weighted_strength = np.mean(active_strengths) if active_strengths else 0.0
        else:
            # Calculate weighted strength (average of agreeing strategies)
            agreeing_strengths = []
            for strategy in self.strategy_names:
                if strategy_votes[strategy] == 1:  # Agrees with expected signal
                    agreeing_strengths.append(strategy_strengths[strategy])

            weighted_strength = np.mean(agreeing_strengths) if agreeing_strengths else 0.0
            # Voting logic based on specific non-simple mode
            if self.config.mode == "weighted":
                # Weighted mode: similar to Range Oscillator
                min_tot = self.config.weighted_min_total
                min_diff = self.config.weighted_min_diff

                if signal_type == "LONG":
                    is_signal = (
                        (long_weight > short_weight)
                        and (long_weight >= min_tot)
                        and ((long_weight - short_weight) >= min_diff)
                    )
                    final_vote = 1 if is_signal else 0
                else:  # SHORT
                    is_signal = (
                        (short_weight > long_weight)
                        and (short_weight >= min_tot)
                        and ((short_weight - long_weight) >= min_diff)
                    )
                    final_vote = -1 if is_signal else 0
            else:
                # Threshold mode: minimum number of strategies must agree
                n_strategies = len(self.strategy_names)
                min_agree = int(np.ceil(n_strategies * self.config.threshold))

                # Count strategies that actually vote LONG or SHORT
                long_count = sum(1 for s in strategy_signals.values() if s == 1)
                short_count = sum(1 for s in strategy_signals.values() if s == -1)

                if signal_type == "LONG":
                    if long_count >= min_agree and long_count > short_count:
                        final_vote = 1
                    else:
                        final_vote = 0
                else:  # SHORT
                    if short_count >= min_agree and short_count > long_count:
                        final_vote = -1
                    else:
                        final_vote = 0

        # Special logic: If 1 strategy has strength > 0.7 and others = 0, accept it
        # This handles cases where only one strong strategy signals while others are silent
        if final_vote == 0:
            active_strategies = [
                (strategy, strategy_signals[strategy], strategy_strengths[strategy])
                for strategy in self.strategy_names
                if strategy_signals[strategy] != 0
            ]

            # Check if exactly 1 strategy is active with strong signal
            if len(active_strategies) == 1:
                strategy_name, signal, strength = active_strategies[0]
                if strength > 0.7:
                    # Accept the single strong strategy's signal
                    expected_signal = 1 if signal_type == "LONG" else -1
                    if signal == expected_signal:
                        final_vote = expected_signal
                        weighted_strength = strength

        # Apply signal strength filtering
        if self.config.min_signal_strength > 0 and weighted_strength < self.config.min_signal_strength:
            final_vote = 0

        # Track if we used simple mode fallback
        used_simple_fallback = False

        # Simple mode fallback: if weighted/threshold produce no signal, try simple accuracy-based voting
        if final_vote == 0 and self.config.enable_simple_fallback and self.config.mode != "simple":
            fallback_vote = self._simple_mode_vote(
                strategy_signals,
                strategy_strengths,
                signal_type,
                raw_weights,  # Use raw_weights (not normalized) for accuracy calculation
            )
            if fallback_vote != 0:
                final_vote = fallback_vote
                used_simple_fallback = True
                # Recalculate weighted_strength for simple mode: average of all active strategies
                active_strengths = [
                    strategy_strengths[strategy] for strategy in self.strategy_names if strategy_signals[strategy] != 0
                ]
                weighted_strength = np.mean(active_strengths) if active_strengths else 0.0

        # Calculate confidence score (similar to Range Oscillator)
        if final_vote != 0:
            # Check if we're in simple mode (either directly or as fallback)
            is_simple_mode = self.config.mode == "simple" or used_simple_fallback

            if is_simple_mode:
                # Simple mode confidence: based on total accuracy and average strength
                # Get accuracy sum for the winning direction (use raw_weights)
                if final_vote == 1:
                    accuracy_sum = sum(
                        raw_weights.get(strategy, 0.0) for strategy, signal in strategy_signals.items() if signal == 1
                    )
                else:  # final_vote == -1
                    accuracy_sum = sum(
                        raw_weights.get(strategy, 0.0) for strategy, signal in strategy_signals.items() if signal == -1
                    )

                # Normalize accuracy sum (max possible is sum of all raw_weights)
                max_accuracy = sum(raw_weights.values())
                accuracy_level = accuracy_sum / max_accuracy if max_accuracy > 0 else 0.0

                # Combined confidence for simple mode
                confidence = (accuracy_level * AGREEMENT_WEIGHT) + (weighted_strength * STRENGTH_WEIGHT)
            else:
                # Standard confidence calculation for weighted/threshold modes
                # Agreement level: fraction of strategies that agree
                # strategy_votes is 1 if it matches the expected signal, so sum is count of agreeing strategies
                agree_count = sum(strategy_votes.values())
                agree_level = agree_count / len(self.strategy_names)

                # Average strength of agreeing strategies
                avg_strength = weighted_strength

                # Combined confidence (similar to Range Oscillator)
                confidence = (agree_level * AGREEMENT_WEIGHT) + (avg_strength * STRENGTH_WEIGHT)
        else:
            confidence = 0.0

        return (final_vote, min(weighted_strength, 1.0), min(confidence, 1.0))


__all__ = [
    "SPCVoteAggregator",
]
