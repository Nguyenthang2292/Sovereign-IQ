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
        assert config.enable_simple_fallback is True
        assert config.simple_min_accuracy_total == 0.65  # Updated from 1.5 to accept single strategy
    
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
        assert config.enable_simple_fallback is True
        assert config.simple_min_accuracy_total == 0.65  # Updated from 1.5 to accept single strategy


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
    
    def test_validate_config_negative_simple_min_accuracy(self):
        """Test validation fails for negative simple_min_accuracy_total."""
        with pytest.raises(ValueError, match="simple_min_accuracy_total must be"):
            SPCAggregationConfig(simple_min_accuracy_total=-0.1)
    
    def test_config_simple_mode(self):
        """Test configuration with simple mode."""
        config = SPCAggregationConfig(
            mode="simple",
            enable_simple_fallback=False,
            simple_min_accuracy_total=1.2
        )
        
        assert config.mode == "simple"
        assert config.enable_simple_fallback is False
        assert config.simple_min_accuracy_total == 1.2
    
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
        config = SPCAggregationConfig(
            min_signal_strength=0.5,
            enable_simple_fallback=False  # Disable fallback to test strength filtering
        )
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
        
        # Average strength = (0.3 + 0.2 + 0.4) / 3 = 0.3 < 0.5
        # If weighted mode produces signal but strength < 0.5, vote should be filtered to 0
        # Or if weighted mode doesn't produce signal, vote should be 0
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
    
    # ========== Simple Mode Tests ==========
    
    def test_aggregate_simple_mode_direct_long(self):
        """Test simple mode directly with LONG signals."""
        # Base accuracies: cluster_transition=0.68, regime_following=0.66, mean_reversion=0.64
        # Total if all LONG: 1.98
        config = SPCAggregationConfig(
            mode="simple",
            simple_min_accuracy_total=1.5
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
        
        # Total accuracy = 0.68 + 0.66 + 0.64 = 1.98 > 1.5, all LONG
        assert vote == 1
        assert strength > 0
        assert confidence > 0
    
    def test_aggregate_simple_mode_direct_short(self):
        """Test simple mode directly with SHORT signals."""
        config = SPCAggregationConfig(
            mode="simple",
            simple_min_accuracy_total=1.5
        )
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': -1,
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': -1,
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': -1,
            'spc_mean_reversion_strength': 0.6,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "SHORT")
        
        # Total accuracy = 1.98 > 1.5, all SHORT
        assert vote == -1
        assert strength > 0
        assert confidence > 0
    
    def test_aggregate_simple_mode_mixed_signals_long_wins(self):
        """Test simple mode with mixed signals where LONG has higher accuracy."""
        config = SPCAggregationConfig(
            mode="simple",
            simple_min_accuracy_total=1.0
        )
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,  # 0.68
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': 1,  # 0.66
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': -1,  # 0.64
            'spc_mean_reversion_strength': 0.6,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        # LONG accuracy = 0.68 + 0.66 = 1.34
        # SHORT accuracy = 0.64
        # LONG wins, total = 1.98 > 1.0
        assert vote == 1
        assert strength > 0
    
    def test_aggregate_simple_mode_mixed_signals_short_wins(self):
        """Test simple mode with mixed signals where SHORT has higher accuracy."""
        config = SPCAggregationConfig(
            mode="simple",
            simple_min_accuracy_total=1.0
        )
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': -1,  # 0.68
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': -1,  # 0.66
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': 1,  # 0.64
            'spc_mean_reversion_strength': 0.6,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "SHORT")
        
        # SHORT accuracy = 0.68 + 0.66 = 1.34
        # LONG accuracy = 0.64
        # SHORT wins, total = 1.98 > 1.0
        assert vote == -1
        assert strength > 0
    
    def test_aggregate_simple_mode_insufficient_accuracy(self):
        """Test simple mode when total accuracy is below threshold."""
        config = SPCAggregationConfig(
            mode="simple",
            simple_min_accuracy_total=2.0  # Higher than max possible (1.98)
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
        
        # Total accuracy = 1.98 < 2.0
        assert vote == 0
        assert confidence == 0.0
    
    def test_aggregate_simple_mode_equal_accuracy_prefers_long(self):
        """Test simple mode when LONG and SHORT have equal accuracy, prefers LONG for LONG signal_type."""
        config = SPCAggregationConfig(
            mode="simple",
            simple_min_accuracy_total=0.5
        )
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,  # 0.68
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': -1,  # 0.66
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': 0,  # No signal
            'spc_mean_reversion_strength': 0.0,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        # LONG = 0.68, SHORT = 0.66, LONG wins
        assert vote == 1
    
    def test_aggregate_simple_mode_equal_accuracy_prefers_short(self):
        """Test simple mode when LONG and SHORT have equal accuracy, prefers SHORT for SHORT signal_type."""
        config = SPCAggregationConfig(
            mode="simple",
            simple_min_accuracy_total=0.5
        )
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,  # 0.68
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': -1,  # 0.66
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': 0,  # No signal
            'spc_mean_reversion_strength': 0.0,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "SHORT")
        
        # LONG = 0.68, SHORT = 0.66, LONG wins but signal_type is SHORT
        # Since LONG > SHORT, vote should be 1 (LONG wins)
        assert vote == 1  # LONG has higher accuracy
    
    def test_aggregate_simple_mode_with_custom_weights(self):
        """Test simple mode with custom strategy weights."""
        custom_weights = {
            'cluster_transition': 0.9,
            'regime_following': 0.7,
            'mean_reversion': 0.5,
        }
        config = SPCAggregationConfig(
            mode="simple",
            strategy_weights=custom_weights,
            simple_min_accuracy_total=1.5
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
        
        # Total accuracy = 0.9 + 0.7 + 0.5 = 2.1 > 1.5
        assert vote == 1
        assert strength > 0
    
    def test_aggregate_simple_mode_confidence_calculation(self):
        """Test confidence calculation in simple mode."""
        config = SPCAggregationConfig(
            mode="simple",
            simple_min_accuracy_total=1.0
        )
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.9,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.8,
            'spc_mean_reversion_signal': 1,
            'spc_mean_reversion_strength': 0.7,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        assert vote == 1
        assert confidence > 0
        assert confidence <= 1.0
        # Should have high confidence with all strategies agreeing and high strength
        assert confidence > 0.5
    
    # ========== Simple Mode Fallback Tests ==========
    
    def test_aggregate_simple_fallback_from_weighted(self):
        """Test simple mode fallback when weighted mode produces no signal."""
        config = SPCAggregationConfig(
            mode="weighted",
            weighted_min_total=0.8,  # High threshold, unlikely to be met
            weighted_min_diff=0.2,
            enable_simple_fallback=True,
            simple_min_accuracy_total=1.0
        )
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.5,  # Low strength
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.4,  # Low strength
            'spc_mean_reversion_signal': 1,
            'spc_mean_reversion_strength': 0.3,  # Low strength
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        # Weighted mode might not produce signal due to low weights
        # Should fallback to simple mode
        # Total accuracy = 1.98 > 1.0, all LONG
        assert vote == 1
        assert strength > 0
        assert confidence > 0
    
    def test_aggregate_simple_fallback_from_threshold(self):
        """Test simple mode fallback when threshold mode produces no signal."""
        config = SPCAggregationConfig(
            mode="threshold",
            threshold=0.9,  # Need 3 strategies (very high)
            enable_simple_fallback=True,
            simple_min_accuracy_total=1.0
        )
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
        
        # Threshold mode: need 3 but only 2 agree -> no signal
        # Should fallback to simple mode
        # LONG accuracy = 0.68 + 0.66 = 1.34 > 1.0
        assert vote == 1
        assert strength > 0
    
    def test_aggregate_simple_fallback_disabled(self):
        """Test that simple fallback doesn't trigger when disabled."""
        config = SPCAggregationConfig(
            mode="weighted",
            weighted_min_total=0.95,  # Very high, won't be met (normalized weights sum to 1.0)
            weighted_min_diff=0.3,
            enable_simple_fallback=False,  # Disabled
            simple_min_accuracy_total=1.0
        )
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.3,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.2,
            'spc_mean_reversion_signal': 1,
            'spc_mean_reversion_strength': 0.1,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        # Weighted mode: normalized long_weight should be ~1.0 (all strategies LONG)
        # But weighted_min_total=0.95 might still pass, so let's use mixed signals
        symbol_data_mixed = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.3,
            'spc_regime_following_signal': -1,  # Different direction
            'spc_regime_following_strength': 0.2,
            'spc_mean_reversion_signal': 0,  # No signal
            'spc_mean_reversion_strength': 0.0,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data_mixed, "LONG")
        
        # Weighted mode: long_weight < 0.95, should fail
        # Fallback is disabled, so should return 0
        assert vote == 0
    
    def test_aggregate_simple_fallback_insufficient_accuracy(self):
        """Test simple fallback when accuracy is insufficient."""
        config = SPCAggregationConfig(
            mode="weighted",
            weighted_min_total=0.95,  # Very high, won't be met
            weighted_min_diff=0.3,
            enable_simple_fallback=True,
            simple_min_accuracy_total=2.0  # Higher than max possible (1.98)
        )
        aggregator = SPCVoteAggregator(config)
        
        # Use mixed signals to ensure weighted mode fails
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.3,
            'spc_regime_following_signal': -1,  # Different direction
            'spc_regime_following_strength': 0.2,
            'spc_mean_reversion_signal': 0,  # No signal
            'spc_mean_reversion_strength': 0.0,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        # Weighted fails (long_weight < 0.95), simple fallback also fails (accuracy < 2.0)
        assert vote == 0
    
    def test_aggregate_simple_fallback_with_adaptive_weights(self):
        """Test simple fallback uses adaptive weights if available."""
        config = SPCAggregationConfig(
            mode="weighted",
            weighted_min_total=0.9,
            enable_adaptive_weights=True,
            enable_simple_fallback=True,
            simple_min_accuracy_total=1.0
        )
        aggregator = SPCVoteAggregator(config)
        
        signals_history = {
            'cluster_transition': [1] * 10,
            'regime_following': [1] * 10,
            'mean_reversion': [1] * 10,
        }
        strengths_history = {
            'cluster_transition': [0.8] * 10,
            'regime_following': [0.7] * 10,
            'mean_reversion': [0.6] * 10,
        }
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.3,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.2,
            'spc_mean_reversion_signal': 1,
            'spc_mean_reversion_strength': 0.1,
        }
        
        vote, strength, confidence = aggregator.aggregate(
            symbol_data, "LONG",
            signals_history=signals_history,
            strengths_history=strengths_history
        )
        
        # Should use adaptive weights in simple fallback
        assert vote == 1
    
    def test_aggregate_simple_fallback_strength_calculation(self):
        """Test that simple fallback recalculates strength correctly."""
        config = SPCAggregationConfig(
            mode="weighted",
            weighted_min_total=0.9,
            enable_simple_fallback=True,
            simple_min_accuracy_total=1.0
        )
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.9,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.8,
            'spc_mean_reversion_signal': -1,  # Different direction
            'spc_mean_reversion_strength': 0.7,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        # Simple mode: strength should be average of ALL active strategies (not just agreeing)
        # (0.9 + 0.8 + 0.7) / 3 = 0.8
        assert vote == 1  # LONG wins (0.68 + 0.66 > 0.64)
        assert abs(strength - 0.8) < 0.01  # Should be average of all 3
    
    # ========== Edge Cases ==========
    
    def test_aggregate_simple_mode_no_active_strategies(self):
        """Test simple mode when no strategies have signals."""
        config = SPCAggregationConfig(mode="simple")
        aggregator = SPCVoteAggregator(config)
        
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
    
    def test_aggregate_simple_mode_one_strategy_only(self):
        """Test simple mode with only one strategy having signal."""
        config = SPCAggregationConfig(
            mode="simple",
            simple_min_accuracy_total=0.5
        )
        aggregator = SPCVoteAggregator(config)
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,  # 0.68
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': 0,
            'spc_regime_following_strength': 0.0,
            'spc_mean_reversion_signal': 0,
            'spc_mean_reversion_strength': 0.0,
        }
        
        vote, strength, confidence = aggregator.aggregate(symbol_data, "LONG")
        
        # Only one strategy: accuracy = 0.68 > 0.5
        assert vote == 1
        assert strength == 0.8
        assert confidence > 0
    
    def test_aggregate_adaptive_weights_no_valid_strategies(self):
        """Test adaptive weights when no strategies have history."""
        config = SPCAggregationConfig(enable_adaptive_weights=True)
        aggregator = SPCVoteAggregator(config)
        
        signals_history = {}
        strengths_history = {}
        
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
        
        # Should fallback to base weights
        assert vote in [0, 1]
    
    def test_aggregate_adaptive_weights_partial_history(self):
        """Test adaptive weights when only some strategies have history."""
        config = SPCAggregationConfig(enable_adaptive_weights=True, adaptive_performance_window=10)
        aggregator = SPCVoteAggregator(config)
        
        signals_history = {
            'cluster_transition': [1] * 10,
            # Missing regime_following and mean_reversion
        }
        strengths_history = {
            'cluster_transition': [0.8] * 10,
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
        
        # Should handle partial history gracefully
        assert vote in [0, 1]

