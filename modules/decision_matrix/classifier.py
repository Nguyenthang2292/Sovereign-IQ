"""
Decision Matrix Classifier.

Simple Decision Matrix Classification Algorithm inspired by Random Forest.
Uses voting system with weighted impact and feature importance.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


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
    """
    
    indicators: List[str] = field(default_factory=lambda: ['atc', 'oscillator'])
    node_votes: Dict[str, int] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    independent_accuracy: Dict[str, float] = field(default_factory=dict)
    weighted_impact: Dict[str, float] = field(default_factory=dict)
    signal_strengths: Dict[str, float] = field(default_factory=dict)
    
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
            # No indicators, no weights to calculate
            return
        
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
            
            # Check for over-representation (>40%)
            # Note: This cap strictly works for N >= 3 indicators.
            # For N=2, the minimum even weight is 50%, so 40% cap is mathematically impossible
            # without discarding weight. Skip normalization for N=2 to avoid incorrect results.
            if len(self.indicators) >= 3:
                max_weight = max(self.weighted_impact.values()) if self.weighted_impact else 0.0
                if max_weight > 0.4:
                    # Normalize to prevent over-representation
                    # Find the indicator with max weight
                    max_indicator = max(self.weighted_impact.items(), key=lambda x: x[1])[0]
                    
                    # Cap max weight at 40%
                    excess = self.weighted_impact[max_indicator] - 0.4
                    self.weighted_impact[max_indicator] = 0.4
                    
                    # Redistribute excess weight equally among other indicators
                    other_indicators = [ind for ind in self.indicators if ind != max_indicator]
                    if other_indicators and excess > 0:
                        equal_addition = excess / len(other_indicators)
                        for indicator in other_indicators:
                            self.weighted_impact[indicator] += equal_addition
                    
                    # Final check: if max is still > 0.4 after redistribution, 
                    # it means we need to cap again (shouldn't happen, but safety check)
                    max_weight_after = max(self.weighted_impact.values())
                    if max_weight_after > 0.4:
                        # This should be rare, but handle it by scaling all weights proportionally
                        scale_factor = 0.4 / max_weight_after
                        for indicator in self.indicators:
                            self.weighted_impact[indicator] *= scale_factor
                        # Redistribute remaining
                        remaining = 1.0 - sum(self.weighted_impact.values())
                        if remaining > 0:
                            equal_addition = remaining / len(self.indicators)
                            for indicator in self.indicators:
                                self.weighted_impact[indicator] += equal_addition
    
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
                'vote': vote,
                'weight': weight,
                'contribution': contribution,
            }
            
            if vote == 1:
                positive_votes += 1
                total_positive_weight += weight
        
        # Check minimum votes requirement
        if positive_votes < min_votes:
            return (0, weighted_score, voting_breakdown)
        
        # Adjust threshold based on the actual weights of indicators that voted positive
        # Problem: When we have many indicators (e.g., 5) but only min_votes (e.g., 2) vote=1,
        # weighted_score = sum of weights of positive votes (e.g., 0.193 + 0.196 = 0.389)
        # But threshold=0.5 is too high in this case.
        # 
        # Solution: Use a dynamic threshold based on the total weight of positive indicators
        # If we have min_votes=2 and they have total weight=0.389, effective threshold should be <= 0.389
        # We use the minimum of:
        #   1. Original threshold (0.5)
        #   2. Minimum proportion based on min_votes (2/5 = 0.4)
        #   3. Total weight of positive indicators * 0.95 (allow 5% margin for precision)
        if len(self.indicators) > 0:
            min_proportion = min_votes / len(self.indicators)
            # Use total_positive_weight if available to account for uneven weights
            if total_positive_weight > 0:
                # Use 95% of total_positive_weight as threshold to allow for floating point precision
                # This ensures that if all positive indicators vote=1, weighted_score will pass
                weight_based_threshold = total_positive_weight * 0.95
                # Use the minimum of threshold, min_proportion, and weight_based_threshold
                effective_threshold = min(threshold, min_proportion, weight_based_threshold)
            else:
                # Fallback: use min of threshold and min_proportion
                effective_threshold = min(threshold, min_proportion)
        else:
            effective_threshold = threshold
        
        # Final vote based on adjusted threshold
        cumulative_vote = 1 if weighted_score >= effective_threshold else 0
        
        return (cumulative_vote, weighted_score, voting_breakdown)
    
    def get_metadata(self) -> Dict:
        """Get all metadata for display."""
        return {
            'node_votes': self.node_votes.copy(),
            'feature_importance': self.feature_importance.copy(),
            'independent_accuracy': self.independent_accuracy.copy(),
            'weighted_impact': self.weighted_impact.copy(),
            'signal_strengths': self.signal_strengths.copy(),
        }
    
    def reset(self) -> None:
        """Reset classifier for next symbol."""
        self.node_votes.clear()
        self.feature_importance.clear()
        self.independent_accuracy.clear()
        self.weighted_impact.clear()
        self.signal_strengths.clear()


__all__ = ["DecisionMatrixClassifier"]

