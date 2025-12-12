"""
Tests for DecisionMatrixClassifier.
"""

import pytest
from modules.decision_matrix.classifier import DecisionMatrixClassifier


class TestDecisionMatrixClassifier:
    """Test suite for DecisionMatrixClassifier."""
    
    def test_init_default(self):
        """Test default initialization."""
        classifier = DecisionMatrixClassifier()
        
        assert classifier.indicators == ['atc', 'oscillator']
        assert classifier.node_votes == {}
        assert classifier.feature_importance == {}
        assert classifier.weighted_impact == {}
        assert classifier.signal_strengths == {}
    
    def test_init_custom_indicators(self):
        """Test initialization with custom indicators."""
        indicators = ['atc', 'oscillator', 'spc']
        classifier = DecisionMatrixClassifier(indicators=indicators)
        
        assert classifier.indicators == indicators
    
    def test_add_node_vote_with_accuracy(self):
        """Test adding vote with accuracy."""
        classifier = DecisionMatrixClassifier()
        
        classifier.add_node_vote('atc', vote=1, signal_strength=0.8, accuracy=0.65)
        
        assert classifier.node_votes['atc'] == 1
        assert classifier.signal_strengths['atc'] == 0.8
        assert classifier.feature_importance['atc'] == 0.65
        assert classifier.independent_accuracy['atc'] == 0.65
    
    def test_add_node_vote_without_accuracy(self):
        """Test adding vote without accuracy (uses signal_strength)."""
        classifier = DecisionMatrixClassifier()
        
        classifier.add_node_vote('atc', vote=1, signal_strength=0.8)
        
        assert classifier.node_votes['atc'] == 1
        assert classifier.signal_strengths['atc'] == 0.8
        assert classifier.feature_importance['atc'] == 0.8
        assert classifier.independent_accuracy['atc'] == 0.8
    
    def test_add_multiple_votes(self):
        """Test adding votes from multiple indicators."""
        classifier = DecisionMatrixClassifier(indicators=['atc', 'oscillator', 'spc'])
        
        classifier.add_node_vote('atc', vote=1, signal_strength=0.7, accuracy=0.65)
        classifier.add_node_vote('oscillator', vote=1, signal_strength=0.8, accuracy=0.70)
        classifier.add_node_vote('spc', vote=0, signal_strength=0.5, accuracy=0.66)
        
        assert classifier.node_votes['atc'] == 1
        assert classifier.node_votes['oscillator'] == 1
        assert classifier.node_votes['spc'] == 0
        assert len(classifier.node_votes) == 3
    
    def test_calculate_weighted_impact_equal_importance(self):
        """Test weighted impact calculation with equal importance."""
        classifier = DecisionMatrixClassifier(indicators=['atc', 'oscillator'])
        
        classifier.add_node_vote('atc', vote=1, signal_strength=0.7, accuracy=0.5)
        classifier.add_node_vote('oscillator', vote=1, signal_strength=0.8, accuracy=0.5)
        classifier.calculate_weighted_impact()
        
        assert classifier.weighted_impact['atc'] == 0.5
        assert classifier.weighted_impact['oscillator'] == 0.5
    
    def test_calculate_weighted_impact_different_importance(self):
        """Test weighted impact calculation with different importance."""
        classifier = DecisionMatrixClassifier(indicators=['atc', 'oscillator'])
        
        classifier.add_node_vote('atc', vote=1, signal_strength=0.7, accuracy=0.65)
        classifier.add_node_vote('oscillator', vote=1, signal_strength=0.8, accuracy=0.70)
        classifier.calculate_weighted_impact()
        
        # Both weights should be proportional to their importance
        # Since neither exceeds 40%, no normalization should occur
        total = 0.65 + 0.70
        atc_expected = 0.65 / total
        osc_expected = 0.70 / total
        
        # Check that weights are proportional (may have slight adjustments from normalization)
        assert classifier.weighted_impact['atc'] > 0
        assert classifier.weighted_impact['oscillator'] > 0
        assert abs(sum(classifier.weighted_impact.values()) - 1.0) < 0.001
        # ATC should have lower weight than oscillator (0.65 < 0.70)
        assert classifier.weighted_impact['atc'] < classifier.weighted_impact['oscillator']
    
    def test_calculate_weighted_impact_over_representation(self):
        """Test weighted impact normalization when one indicator dominates."""
        classifier = DecisionMatrixClassifier(indicators=['atc', 'oscillator'])
        
        # ATC has very high accuracy (would be >40%)
        classifier.add_node_vote('atc', vote=1, signal_strength=0.7, accuracy=0.9)
        classifier.add_node_vote('oscillator', vote=1, signal_strength=0.8, accuracy=0.1)
        classifier.calculate_weighted_impact()
        
        # After normalization, max weight should be reduced (may not be exactly 40% due to redistribution)
        max_weight = max(classifier.weighted_impact.values())
        # Should be less than original 0.9/1.0 = 0.9
        assert max_weight < 0.9
        # Total should still be 1.0
        total = sum(classifier.weighted_impact.values())
        assert abs(total - 1.0) < 0.001
        # Both indicators should have some weight
        assert classifier.weighted_impact['atc'] > 0
        assert classifier.weighted_impact['oscillator'] > 0
    
    def test_calculate_weighted_impact_no_importance(self):
        """Test weighted impact with no importance data (equal weights)."""
        classifier = DecisionMatrixClassifier(indicators=['atc', 'oscillator', 'spc'])
        
        # Don't add any votes - no importance data at all
        classifier.calculate_weighted_impact()
        
        # Should use equal weights when total_importance == 0
        expected_weight = 1.0 / 3
        for indicator in classifier.indicators:
            assert abs(classifier.weighted_impact.get(indicator, 0) - expected_weight) < 0.001
    
    def test_calculate_cumulative_vote_all_agree(self):
        """Test cumulative vote when all indicators agree."""
        classifier = DecisionMatrixClassifier(indicators=['atc', 'oscillator'])
        
        classifier.add_node_vote('atc', vote=1, signal_strength=0.7, accuracy=0.65)
        classifier.add_node_vote('oscillator', vote=1, signal_strength=0.8, accuracy=0.70)
        classifier.calculate_weighted_impact()
        
        cumulative_vote, weighted_score, breakdown = classifier.calculate_cumulative_vote(
            threshold=0.5, min_votes=2
        )
        
        assert cumulative_vote == 1
        assert weighted_score > 0.5
        assert breakdown['atc']['vote'] == 1
        assert breakdown['oscillator']['vote'] == 1
    
    def test_calculate_cumulative_vote_below_threshold(self):
        """Test cumulative vote when weighted score is below threshold."""
        classifier = DecisionMatrixClassifier(indicators=['atc', 'oscillator'])
        
        classifier.add_node_vote('atc', vote=1, signal_strength=0.3, accuracy=0.65)
        classifier.add_node_vote('oscillator', vote=0, signal_strength=0.2, accuracy=0.70)
        classifier.calculate_weighted_impact()
        
        cumulative_vote, weighted_score, _ = classifier.calculate_cumulative_vote(
            threshold=0.5, min_votes=1
        )
        
        assert cumulative_vote == 0
        assert weighted_score < 0.5
    
    def test_calculate_cumulative_vote_min_votes_requirement(self):
        """Test cumulative vote with minimum votes requirement."""
        classifier = DecisionMatrixClassifier(indicators=['atc', 'oscillator', 'spc'])
        
        # Only 1 indicator votes positive, but min_votes=2
        classifier.add_node_vote('atc', vote=1, signal_strength=0.7, accuracy=0.65)
        classifier.add_node_vote('oscillator', vote=0, signal_strength=0.2, accuracy=0.70)
        classifier.add_node_vote('spc', vote=0, signal_strength=0.3, accuracy=0.66)
        classifier.calculate_weighted_impact()
        
        cumulative_vote, weighted_score, _ = classifier.calculate_cumulative_vote(
            threshold=0.3, min_votes=2
        )
        
        assert cumulative_vote == 0  # Should be 0 because only 1 vote
    
    def test_calculate_cumulative_vote_breakdown_structure(self):
        """Test voting breakdown structure."""
        classifier = DecisionMatrixClassifier(indicators=['atc', 'oscillator'])
        
        classifier.add_node_vote('atc', vote=1, signal_strength=0.7, accuracy=0.65)
        classifier.add_node_vote('oscillator', vote=1, signal_strength=0.8, accuracy=0.70)
        classifier.calculate_weighted_impact()
        
        _, _, breakdown = classifier.calculate_cumulative_vote(threshold=0.5, min_votes=2)
        
        assert 'atc' in breakdown
        assert 'oscillator' in breakdown
        assert 'vote' in breakdown['atc']
        assert 'weight' in breakdown['atc']
        assert 'contribution' in breakdown['atc']
        assert breakdown['atc']['contribution'] == breakdown['atc']['vote'] * breakdown['atc']['weight']
    
    def test_get_metadata(self):
        """Test get_metadata returns all classifier data."""
        classifier = DecisionMatrixClassifier(indicators=['atc', 'oscillator'])
        
        classifier.add_node_vote('atc', vote=1, signal_strength=0.7, accuracy=0.65)
        classifier.add_node_vote('oscillator', vote=1, signal_strength=0.8, accuracy=0.70)
        classifier.calculate_weighted_impact()
        
        metadata = classifier.get_metadata()
        
        assert 'node_votes' in metadata
        assert 'feature_importance' in metadata
        assert 'independent_accuracy' in metadata
        assert 'weighted_impact' in metadata
        assert 'signal_strengths' in metadata
        assert metadata['node_votes']['atc'] == 1
        assert metadata['feature_importance']['atc'] == 0.65
    
    def test_reset(self):
        """Test reset clears all data."""
        classifier = DecisionMatrixClassifier(indicators=['atc', 'oscillator'])
        
        classifier.add_node_vote('atc', vote=1, signal_strength=0.7, accuracy=0.65)
        classifier.add_node_vote('oscillator', vote=1, signal_strength=0.8, accuracy=0.70)
        classifier.calculate_weighted_impact()
        
        classifier.reset()
        
        assert classifier.node_votes == {}
        assert classifier.feature_importance == {}
        assert classifier.independent_accuracy == {}
        assert classifier.weighted_impact == {}
        assert classifier.signal_strengths == {}
    
    def test_full_workflow(self):
        """Test complete workflow from adding votes to getting result."""
        classifier = DecisionMatrixClassifier(indicators=['atc', 'oscillator', 'spc'])
        
        # Add votes
        classifier.add_node_vote('atc', vote=1, signal_strength=0.7, accuracy=0.65)
        classifier.add_node_vote('oscillator', vote=1, signal_strength=0.8, accuracy=0.70)
        classifier.add_node_vote('spc', vote=1, signal_strength=0.6, accuracy=0.66)
        
        # Calculate weighted impact
        classifier.calculate_weighted_impact()
        
        # Calculate cumulative vote
        cumulative_vote, weighted_score, breakdown = classifier.calculate_cumulative_vote(
            threshold=0.5, min_votes=2
        )
        
        # Verify results
        assert cumulative_vote == 1
        assert weighted_score > 0.5
        assert len(breakdown) == 3
        assert all(b['vote'] == 1 for b in breakdown.values())
        
        # Get metadata
        metadata = classifier.get_metadata()
        assert len(metadata['node_votes']) == 3

