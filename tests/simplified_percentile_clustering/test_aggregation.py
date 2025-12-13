"""
Tests for SPC Vote Aggregation module.
"""

import pytest
import numpy as np
from modules.simplified_percentile_clustering.aggregation import SPCVoteAggregator
from modules.simplified_percentile_clustering.config.aggregation_config import SPCAggregationConfig


class TestSPCAggregationConfig:
    """Test suite for SPCAggregationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SPCAggregationConfig()
        
        assert config.mode == "weighted"
        assert config.threshold == 0.5
        assert config.weighted_min_total == 0.5
        assert config.weighted_min_diff == 0.1
        assert config.enable_adaptive_weights is False
        assert config.adaptive_performance_window == 10
        assert config.min_signal_strength == 0.0
        assert config.strategy_weights is None
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SPCAggregationConfig(
            mode="threshold",
            threshold=0.6,
            weighted_min_total=0.7,
            weighted_min_diff=0.2,
            enable_adaptive_weights=True,
            adaptive_performance_window=20,
            min_signal_strength=0.3,
            strategy_weights={'cluster_transition': 0.7, 'regime_following': 0.6, 'mean_reversion': 0.5}
        )
        
        assert config.mode == "threshold"
        assert config.threshold == 0.6
        assert config.weighted_min_total == 0.7
        assert config.weighted_min_diff == 0.2
        assert config.enable_adaptive_weights is True
        assert config.adaptive_performance_window == 20
        assert config.min_signal_strength == 0.3
        assert config.strategy_weights is not None


class TestSPCVoteAggregator:
    """Test suite for SPCVoteAggregator."""
    
    def test_init_default(self):
        """Test default initialization."""
        aggregator = SPCVoteAggregator()
        
        assert aggregator.config.mode == "weighted"
        assert len(aggregator.strategy_names) == 3
        assert 'cluster_transition' in aggregator.strategy_names
        assert 'regime_following' in aggregator.strategy_names
        assert 'mean_reversion' in aggregator.strategy_names
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = SPCAggregationConfig(mode="threshold", threshold=0.6)
        aggregator = SPCVoteAggregator(config)
        
        assert aggregator.config.mode == "threshold"
        assert aggregator.config.threshold == 0.6
    
    def test_validate_config_invalid_mode(self):
        """Test validation fails for invalid mode."""
        with pytest.raises(ValueError, match="Invalid mode"):
            SPCAggregationConfig(mode="invalid")
    
    def test_validate_config_invalid_threshold(self):
        """Test validation fails for invalid threshold."""
        with pytest.raises(ValueError, match="threshold must be in"):
            SPCAggregationConfig(threshold=1.5)  # > 1.0
    
    def test_validate_config_negative_weighted_min_total(self):
        """Test validation fails for negative weighted_min_total."""
        with pytest.raises(ValueError, match="weighted_min_total must be"):
            SPCAggregationConfig(weighted_min_total=-0.1)
    
    def test_validate_config_invalid_adaptive_window(self):
        """Test validation fails for invalid adaptive_performance_window."""
        with pytest.raises(ValueError, match="adaptive_performance_window must be"):
            SPCAggregationConfig(enable_adaptive_weights=True, adaptive_performance_window=0)
    
    def test_aggregate_all_strategies_agree_long(self):
        """Test aggregation when all strategies agree on LONG."""
        aggregator = SPCVoteAggregator()
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': 1,
            'spc_mean_reversion_strength': 0.6,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        assert vote == 1
        assert strength > 0
        assert confidence > 0
    
    def test_aggregate_all_strategies_agree_short(self):
        """Test aggregation when all strategies agree on SHORT."""
        aggregator = SPCVoteAggregator()
        
        symbol_data = {
            'spc_cluster_transition_signal': -1,
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': -1,
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': -1,
            'spc_mean_reversion_strength': 0.6,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "SHORT")
        
        assert vote == -1
        assert strength > 0
        assert confidence > 0
    
    def test_aggregate_mixed_signals(self):
        """Test aggregation when strategies disagree."""
        aggregator = SPCVoteAggregator()
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': -1,  # Disagrees
            'spc_mean_reversion_strength': 0.6,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        # Should still vote LONG if weighted mode conditions are met
        assert vote in [0, 1]
    
    def test_aggregate_no_signals(self):
        """Test aggregation when no strategies have signals."""
        aggregator = SPCVoteAggregator()
        
        symbol_data = {
            'spc_cluster_transition_signal': 0,
            'spc_cluster_transition_strength': 0.0,
            'spc_regime_following_signal': 0,
            'spc_regime_following_strength': 0.0,
            'spc_mean_reversion_signal': 0,
            'spc_mean_reversion_strength': 0.0,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        assert vote == 0
        assert strength == 0.0
        assert confidence == 0.0
    
    def test_aggregate_threshold_mode(self):
        """Test aggregation in threshold mode."""
        # Threshold 0.5 means need at least 50% = ceil(3 * 0.5) = 2 strategies
        config = SPCAggregationConfig(mode="threshold", threshold=0.5)
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': 0,  # No signal
            'spc_mean_reversion_strength': 0.0,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        # 2 out of 3 strategies agree, threshold is 0.5 (need 2)
        assert vote == 1
    
    def test_aggregate_threshold_mode_insufficient(self):
        """Test threshold mode when insufficient strategies agree."""
        config = SPCAggregationConfig(mode="threshold", threshold=0.67)  # Need 2/3 strategies
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': 0,  # No signal
            'spc_regime_following_strength': 0.0,
            'spc_mean_reversion_signal': 0,  # No signal
            'spc_mean_reversion_strength': 0.0,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        # Only 1 out of 3 strategies agree, need 2
        assert vote == 0
    
    def test_aggregate_weighted_mode(self):
        """Test aggregation in weighted mode."""
        config = SPCAggregationConfig(
            mode="weighted",
            weighted_min_total=0.5,
            weighted_min_diff=0.1
        )
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': 1,
            'spc_mean_reversion_strength': 0.6,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        assert vote == 1
        assert strength > 0
    
    def test_aggregate_signal_strength_filtering(self):
        """Test signal strength filtering."""
        config = SPCAggregationConfig(min_signal_strength=0.5)
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.3,  # Low strength
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.2,  # Low strength
            'spc_mean_reversion_signal': 1,
            'spc_mean_reversion_strength': 0.4,  # Low strength
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        # If weighted strength < 0.5, vote should be 0
        if strength < 0.5:
            assert vote == 0
    
    def test_aggregate_custom_strategy_weights(self):
        """Test aggregation with custom strategy weights."""
        custom_weights = {
            'cluster_transition': 0.8,
            'regime_following': 0.6,
            'mean_reversion': 0.4,
        }
        config = SPCAggregationConfig(strategy_weights=custom_weights)
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': 0,
            'spc_regime_following_strength': 0.0,
            'spc_mean_reversion_signal': 0,
            'spc_mean_reversion_strength': 0.0,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        # Cluster transition has high weight, should influence result
        assert vote in [0, 1]
    
    def test_aggregate_adaptive_weights(self):
        """Test aggregation with adaptive weights."""
        config = SPCAggregationConfig(enable_adaptive_weights=True, adaptive_performance_window=10)
        aggregator = SPCVoteAggregator(config)
        
        # Create history where cluster_transition performs well
        signals_history = {
            'cluster_transition': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'regime_following': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'mean_reversion': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
        strengths_history = {
            'cluster_transition': [0.8] * 10,
            'regime_following': [0.6] * 10,
            'mean_reversion': [0.5] * 10,
        }
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': 1,
            'spc_mean_reversion_strength': 0.6,
        }
        
        vote, strength, confidence = aggregator.aggregate(
            symbol_data, "LONG",
            signals_history=signals_history,
            strengths_history=strengths_history
        )
        
        assert vote in [0, 1]
        assert strength > 0
    
    def test_aggregate_adaptive_weights_insufficient_data(self):
        """Test adaptive weights with insufficient history data."""
        config = SPCAggregationConfig(enable_adaptive_weights=True, adaptive_performance_window=10)
        aggregator = SPCVoteAggregator(config)
        
        # Only 3 data points (need at least 5)
        signals_history = {
            'cluster_transition': [1, 1, 1],
            'regime_following': [1, 0, 1],
            'mean_reversion': [0, 1, 0],
        }
        strengths_history = {
            'cluster_transition': [0.8, 0.8, 0.8],
            'regime_following': [0.6, 0.6, 0.6],
            'mean_reversion': [0.5, 0.5, 0.5],
        }
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': 1,
            'spc_mean_reversion_strength': 0.6,
        }
        
        # Should fallback to base weights
        vote, strength, confidence = aggregator.aggregate(
            symbol_data, "LONG",
            signals_history=signals_history,
            strengths_history=strengths_history
        )
        
        assert vote in [0, 1]
    
    def test_aggregate_missing_strategy_data(self):
        """Test aggregation when some strategy data is missing."""
        aggregator = SPCVoteAggregator()
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.8,
            # Missing regime_following and mean_reversion
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        # Should handle missing data gracefully
        assert vote in [0, 1]
    
    def test_aggregate_confidence_calculation(self):
        """Test confidence score calculation."""
        aggregator = SPCVoteAggregator()
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.9,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.8,
            'spc_mean_reversion_signal': 1,
            'spc_mean_reversion_strength': 0.7,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        if vote != 0:
            assert confidence > 0
            assert confidence <= 1.0
            # Higher agreement and strength should yield higher confidence
            assert confidence > 0.5
    
    def test_aggregate_confidence_zero_when_no_vote(self):
        """Test confidence is zero when no vote."""
        aggregator = SPCVoteAggregator()
        
        symbol_data = {
            'spc_cluster_transition_signal': 0,
            'spc_cluster_transition_strength': 0.0,
            'spc_regime_following_signal': 0,
            'spc_regime_following_strength': 0.0,
            'spc_mean_reversion_signal': 0,
            'spc_mean_reversion_strength': 0.0,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        assert vote == 0
        assert confidence == 0.0

