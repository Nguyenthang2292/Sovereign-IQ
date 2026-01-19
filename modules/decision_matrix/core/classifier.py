"""
Decision Matrix Classifier.

Simple Decision Matrix Classification Algorithm inspired by Random Forest.
Uses voting system with weighted impact and feature importance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from modules.decision_matrix.config.config import (
    MAX_CAP_ITERATIONS,
    MAX_WEIGHT_CAP_N2,
    MAX_WEIGHT_CAP_N3_PLUS,
)
from modules.decision_matrix.core.random_forest_core import RandomForestCore
from modules.decision_matrix.utils.threshold import ThresholdCalculator
from modules.decision_matrix.utils.training_data import TrainingDataStorage


@dataclass
class DecisionMatrixClassifier:
    """
    Simple Decision Matrix Classification Algorithm.

    Inspired by Random Forest voting system from Document1.pdf.

    Architecture:
    - Node 1: ATC vote (0 or 1)
    - Node 2: Range Oscillator vote (0 or 1)
    - Node 3: SPC vote (0 or 1) [optional]
    - Cumulative Vote: Weighted combination of all votes

    Args:
        random_seed: Optional seed for reproducible random shuffling.
                     If None, uses unpredictable random state.
    """

    indicators: List[str] = field(default_factory=lambda: ["atc", "oscillator"])
    node_votes: Dict[str, int] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    independent_accuracy: Dict[str, float] = field(default_factory=dict)
    weighted_impact: Dict[str, float] = field(default_factory=dict)
    signal_strengths: Dict[str, float] = field(default_factory=dict)

    random_seed: Optional[int] = None
    random_forest: RandomForestCore = field(default_factory=lambda: RandomForestCore())
    training_data: TrainingDataStorage = field(default_factory=TrainingDataStorage)
    threshold_calculator: ThresholdCalculator = field(default_factory=ThresholdCalculator)

    rf_results: Dict[str, Dict] = field(default_factory=dict)
    pass_fail_counts: Dict[str, Dict] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize random forest with seed if provided."""
        # Type validation for random_seed
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise TypeError(f"random_seed must be int or None, got {type(self.random_seed)}")

        if self.random_seed is not None:
            self.random_forest = RandomForestCore(random_seed=self.random_seed)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"DecisionMatrixClassifier(indicators={self.indicators}, seed={self.random_seed})"

    def add_node_vote(
        self,
        indicator: str,
        vote: int,
        signal_strength: float = 0.5,
        accuracy: Optional[float] = None,
    ) -> None:
        """
        Add vote from an indicator node.

        Args:
            indicator: Indicator name ('atc', 'oscillator', 'spc')
            vote: Vote value (0 or 1)
            signal_strength: Signal strength (0.0 to 1.0) - used for feature importance
            accuracy: Independent accuracy (0.0 to 1.0) - optional, will use signal_strength if not provided
        """
        # Validate indicator is in the active list
        if indicator not in self.indicators:
            raise ValueError(
                f"Indicator '{indicator}' is not in the active indicators list: {self.indicators}. "
                f"Available indicators are: {self.indicators}"
            )

        # Input validation
        if vote not in (0, 1):
            raise ValueError(f"Vote must be 0 or 1, got {vote}")

        if not (0.0 <= signal_strength <= 1.0):
            raise ValueError(f"Signal strength must be between 0.0 and 1.0, got {signal_strength}")

        if accuracy is not None and not (0.0 <= accuracy <= 1.0):
            raise ValueError(f"Accuracy must be between 0.0 and 1.0, got {accuracy}")

        self.node_votes[indicator] = vote
        self.signal_strengths[indicator] = signal_strength

        # Feature importance based on historical accuracy (not signal strength)
        # This ensures balanced importance regardless of signal magnitude
        if accuracy is not None:
            self.feature_importance[indicator] = accuracy
            self.independent_accuracy[indicator] = accuracy
        else:
            # Fallback: use signal strength if accuracy not provided
            self.feature_importance[indicator] = signal_strength
            self.independent_accuracy[indicator] = signal_strength

    def calculate_weighted_impact(self) -> None:
        """
        Calculate weighted impact for each indicator.

        Weighted impact = how much each indicator contributes to the voting scheme.
        Should be balanced (not let one indicator dominate >30-40%).
        """
        # Handle empty indicators list
        if len(self.indicators) == 0:
            return

        missing_indicators = [ind for ind in self.indicators if ind not in self.feature_importance]

        if missing_indicators:
            raise ValueError(
                f"Missing feature importance data for indicators: {missing_indicators}. "
                f"Call add_node_vote() for each indicator before calculate_weighted_impact()."
            )

        # Calculate total importance
        total_importance = sum(self.feature_importance.values())

        if total_importance == 0:
            # Equal weights if no importance data
            equal_weight = 1.0 / len(self.indicators)
            for indicator in self.indicators:
                self.weighted_impact[indicator] = equal_weight
        else:
            # Weighted by feature importance
            for indicator in self.indicators:
                importance = self.feature_importance.get(indicator, 0.0)
                self.weighted_impact[indicator] = importance / total_importance

            # Check for over-representation and normalize weights
            # For N=1: No capping needed (only 1 indicator)
            # For N=2: Cap at 60% to prevent one indicator from dominating
            # For N>=3: Cap at 40% to prevent over-representation
            if len(self.indicators) == 1:
                pass
            elif len(self.indicators) == 2:
                cap_value = MAX_WEIGHT_CAP_N2
                max_weight = max(self.weighted_impact.values()) if self.weighted_impact else 0.0
                if max_weight > cap_value:
                    max_indicator = max(self.weighted_impact.items(), key=lambda x: x[1])[0]
                    excess = self.weighted_impact[max_indicator] - cap_value
                    self.weighted_impact[max_indicator] = cap_value

                    other_indicators = [ind for ind in self.indicators if ind != max_indicator]
                    if other_indicators and excess > 0:
                        self.weighted_impact[other_indicators[0]] += excess

                    # Normalize to ensure sum = 1.0
                    total = sum(self.weighted_impact.values())
                    if total > 0:
                        for indicator in self.indicators:
                            self.weighted_impact[indicator] /= total
            elif len(self.indicators) >= 3:
                cap_value = MAX_WEIGHT_CAP_N3_PLUS

                # Iterative capping: cap all indicators exceeding cap_value
                # Then normalize to ensure sum = 1.0
                # Repeat until no indicator exceeds cap_value
                max_iterations = MAX_CAP_ITERATIONS
                for iteration in range(max_iterations):
                    max_weight = max(self.weighted_impact.values()) if self.weighted_impact else 0.0

                    if max_weight <= cap_value:
                        break

                    # Find all indicators exceeding cap_value
                    capped_indicators = []
                    total_excess = 0.0
                    for indicator in self.indicators:
                        if self.weighted_impact[indicator] > cap_value:
                            excess = self.weighted_impact[indicator] - cap_value
                            self.weighted_impact[indicator] = cap_value
                            capped_indicators.append((indicator, excess))
                            total_excess += excess

                    # Redistribute excess to indicators below cap_value
                    below_cap = [ind for ind in self.indicators if self.weighted_impact[ind] < cap_value]
                    if below_cap and total_excess > 0:
                        for indicator in below_cap:
                            self.weighted_impact[indicator] += total_excess / len(below_cap)

                    # Normalize to ensure sum = 1.0
                    total = sum(self.weighted_impact.values())
                    if total > 0:
                        for indicator in self.indicators:
                            self.weighted_impact[indicator] /= total

        # Final normalization correction to ensure sum is exactly 1.0
        # This corrects for any accumulated floating point errors
        total = sum(self.weighted_impact.values())
        if total > 0:
            for indicator in self.indicators:
                self.weighted_impact[indicator] /= total

            # Final check to ensure sum is exactly 1.0
            actual_sum = sum(self.weighted_impact.values())
            if abs(actual_sum - 1.0) > 1e-10:
                # Apply correction to largest weight to maintain precision
                max_ind = max(self.weighted_impact.items(), key=lambda x: x[1])[0]
                self.weighted_impact[max_ind] += 1.0 - actual_sum

    def calculate_cumulative_vote(
        self,
        threshold: float = 0.5,
        min_votes: int = 2,
    ) -> Tuple[int, float, Dict[str, Dict]]:
        """
        Calculate cumulative vote from all nodes.

        Args:
            threshold: Minimum weighted score for positive vote (default: 0.5)
            min_votes: Minimum number of indicators that must vote positive (default: 2)

        Returns:
            Tuple of:
            - cumulative_vote: 1 if weighted score >= threshold, 0 otherwise
            - weighted_score: Calculated weighted score (0.0 to 1.0)
            - voting_breakdown: Dictionary with individual votes and weights
        """
        # Validate that weighted impact has been calculated
        if not self.weighted_impact:
            raise ValueError("Weighted impact not calculated. Call calculate_weighted_impact() first.")

        # Validate that all indicators have weighted impact
        missing_indicators = []
        for indicator in self.indicators:
            if indicator not in self.weighted_impact:
                missing_indicators.append(indicator)

        if missing_indicators:
            raise ValueError(
                f"Missing weighted impact for indicators: {missing_indicators}. Call calculate_weighted_impact() first."
            )

        # Calculate weighted score
        weighted_score = 0.0
        voting_breakdown = {}
        positive_votes = 0

        # Calculate total weight of indicators that voted positive
        total_positive_weight = 0.0
        for indicator in self.indicators:
            vote = self.node_votes.get(indicator, 0)
            weight = self.weighted_impact.get(indicator, 1.0 / len(self.indicators))
            contribution = vote * weight
            weighted_score += contribution

            voting_breakdown[indicator] = {
                "vote": vote,
                "weight": weight,
                "contribution": contribution,
            }

            if vote == 1:
                positive_votes += 1
                total_positive_weight += weight

        # Check minimum votes requirement
        if positive_votes < min_votes:
            return (0, weighted_score, voting_breakdown)

        # Use threshold based on min_proportion only
        # This aligns with Pine Script logic: passes > fails ? 1 : 0
        # min_proportion represents the minimum proportion of indicators that must vote positive
        if len(self.indicators) > 0:
            min_proportion = min_votes / len(self.indicators)
            effective_threshold = min(threshold, min_proportion)
        else:
            effective_threshold = threshold

        # Final vote based on effective threshold
        cumulative_vote = 1 if weighted_score >= effective_threshold else 0

        return (cumulative_vote, weighted_score, voting_breakdown)

    def get_metadata(self) -> Dict:
        """Get all metadata for display."""
        metadata = {
            "node_votes": self.node_votes.copy(),
            "feature_importance": self.feature_importance.copy(),
            "independent_accuracy": self.independent_accuracy.copy(),
            "weighted_impact": self.weighted_impact.copy(),
            "signal_strengths": self.signal_strengths.copy(),
        }
        if self.rf_results:
            metadata["random_forest"] = self.rf_results
        if self.pass_fail_counts:
            metadata["pass_fail_counts"] = self.pass_fail_counts
        return metadata

    def classify_with_random_forest(
        self,
        x1: float,
        x2: float,
        y: int,
        x1_type: str,
        x2_type: str,
        historical_x1: Optional[List[float]] = None,
        historical_x2: Optional[List[float]] = None,
    ) -> Dict:
        """
        Classify using Random Forest algorithm.

        Args:
            x1: Current value for feature 1
            x2: Current value for feature 2
            y: Current label (0 or 1)
            x1_type: Type of feature 1 ("Volume", "Z-Score", "Stochastic", "RSI", "MFI", "EMA", "SMA")
            x2_type: Type of feature 2
            historical_x1: Historical values for feature 1 (for Volume threshold)
            historical_x2: Historical values for feature 2 (for Volume threshold)

        Returns:
            Dictionary with Random Forest results
        """
        x1_matrix = self.training_data.get_x1_matrix()
        x2_matrix = self.training_data.get_x2_matrix()

        if len(x1_matrix) == 0:
            return {
                "vote": 0,
                "accuracy": 0.0,
                "y1_pass": 0,
                "y1_fail": 0,
                "y2_pass": 0,
                "y2_fail": 0,
                "x1_vote": 0,
                "x2_vote": 0,
                "x1_accuracy": 0.0,
                "x2_accuracy": 0.0,
            }

        x1_threshold = self.threshold_calculator.calculate_threshold(x1_type, historical_x1)
        x2_threshold = self.threshold_calculator.calculate_threshold(x2_type, historical_x2)

        results = self.random_forest.classify(x1_matrix, x2_matrix, x1, x2, x1_threshold, x2_threshold)

        self.rf_results = results
        self.pass_fail_counts = {
            "x1": {"pass": results["y1_pass"], "fail": results["y1_fail"]},
            "x2": {"pass": results["y2_pass"], "fail": results["y2_fail"]},
        }

        return results

    def add_training_sample(self, x1: float, x2: float, y: int) -> None:
        """
        Add training sample to storage.

        Args:
            x1: Feature 1 value
            x2: Feature 2 value
            y: Label (0 or 1)
        """
        self.training_data.add_sample(x1, x2, y)

    def reset(self) -> None:
        """Reset classifier for next symbol."""
        self.node_votes.clear()
        self.feature_importance.clear()
        self.independent_accuracy.clear()
        self.weighted_impact.clear()
        self.signal_strengths.clear()


__all__ = ["DecisionMatrixClassifier"]
