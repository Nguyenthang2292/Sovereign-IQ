
from typing import Dict, List

from modules.hmm.signals.resolution import HOLD, LONG, SHORT, Signal
from modules.hmm.signals.strategy import HMMStrategy, HMMStrategyResult
from modules.hmm.signals.strategy import HMMStrategy, HMMStrategyResult

"""
HMM Voting Mechanisms Module

Implements various voting mechanisms for combining signals from multiple HMM strategies.
"""




class VotingMechanism:
    """
    Voting mechanisms for combining HMM strategy signals.
    """

    @staticmethod
    def simple_majority(strategies: List[HMMStrategy], results: Dict[str, HMMStrategyResult]) -> Signal:
        """
        Simple majority voting: signal with most votes wins.

        Args:
            strategies: List of strategies (for weights if needed)
            results: Dictionary mapping strategy names to results

        Returns:
            Combined signal (LONG, HOLD, or SHORT)
        """
        votes = {LONG: 0, SHORT: 0, HOLD: 0}

        for strategy in strategies:
            if strategy.name in results:
                signal = results[strategy.name].signal
                votes[signal] += 1

        # Find signal with most votes
        max_votes = max(votes.values())
        if max_votes == 0:
            return HOLD

        # Return signal with most votes (prefer LONG/SHORT over HOLD)
        if votes[LONG] == max_votes:
            return LONG
        elif votes[SHORT] == max_votes:
            return SHORT
        else:
            return HOLD

    @staticmethod
    def weighted_voting(strategies: List[HMMStrategy], results: Dict[str, HMMStrategyResult]) -> Signal:
        """
        Weighted voting: each strategy's vote is weighted by its weight.

        Args:
            strategies: List of strategies with weights
            results: Dictionary mapping strategy names to results

        Returns:
            Combined signal (LONG, HOLD, or SHORT)
        """
        weighted_votes = {LONG: 0.0, SHORT: 0.0, HOLD: 0.0}

        for strategy in strategies:
            if strategy.name in results:
                signal = results[strategy.name].signal
                weight = strategy.weight
                weighted_votes[signal] += weight

        # Find signal with highest weighted votes
        max_weight = max(weighted_votes.values())
        if max_weight == 0:
            return HOLD

        # Return signal with highest weighted votes (prefer LONG/SHORT over HOLD)
        if weighted_votes[LONG] == max_weight:
            return LONG
        elif weighted_votes[SHORT] == max_weight:
            return SHORT
        else:
            return HOLD

    @staticmethod
    def confidence_weighted(strategies: List[HMMStrategy], results: Dict[str, HMMStrategyResult]) -> Signal:
        """
        Confidence-weighted voting: votes weighted by strategy weight and signal probability.

        Args:
            strategies: List of strategies with weights
            results: Dictionary mapping strategy names to results

        Returns:
            Combined signal (LONG, HOLD, or SHORT)
        """
        weighted_votes = {LONG: 0.0, SHORT: 0.0, HOLD: 0.0}

        for strategy in strategies:
            if strategy.name in results:
                result = results[strategy.name]
                signal = result.signal
                # Weight = strategy_weight * signal_probability
                weight = strategy.weight * result.probability
                weighted_votes[signal] += weight

        # Find signal with highest weighted votes
        max_weight = max(weighted_votes.values())
        if max_weight == 0:
            return HOLD

        # Return signal with highest weighted votes (prefer LONG/SHORT over HOLD)
        if weighted_votes[LONG] == max_weight:
            return LONG
        elif weighted_votes[SHORT] == max_weight:
            return SHORT
        else:
            return HOLD

    @staticmethod
    def threshold_based(
        strategies: List[HMMStrategy], results: Dict[str, HMMStrategyResult], threshold: float = 0.5
    ) -> Signal:
        """
        Threshold-based voting: only strategies with probability >= threshold vote.

        Args:
            strategies: List of strategies
            results: Dictionary mapping strategy names to results
            threshold: Minimum probability threshold for voting (default: 0.5)

        Returns:
            Combined signal (LONG, HOLD, or SHORT)
        """
        votes = {LONG: 0, SHORT: 0, HOLD: 0}

        for strategy in strategies:
            if strategy.name in results:
                result = results[strategy.name]
                # Only count votes from strategies above threshold
                if result.probability >= threshold:
                    signal = result.signal
                    votes[signal] += strategy.weight

        # Find signal with most votes
        max_votes = max(votes.values())
        if max_votes == 0:
            return HOLD

        # Return signal with most votes (prefer LONG/SHORT over HOLD)
        if votes[LONG] == max_votes:
            return LONG
        elif votes[SHORT] == max_votes:
            return SHORT
        else:
            return HOLD


__all__ = ["VotingMechanism"]
