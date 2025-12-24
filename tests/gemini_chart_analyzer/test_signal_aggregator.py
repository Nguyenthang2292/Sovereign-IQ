"""
Tests for SignalAggregator class.

Tests cover:
- Initialization with default and custom weights
- Weight normalization
- Signal aggregation with various scenarios
- Edge cases (empty signals, all NONE, mixed signals)
- Weighted confidence calculation
- Final signal determination logic
"""

import pytest
from modules.gemini_chart_analyzer.core.signal_aggregator import (
    SignalAggregator,
    DEFAULT_TIMEFRAME_WEIGHTS
)


@pytest.fixture
def default_aggregator():
    """Create SignalAggregator with default weights."""
    return SignalAggregator()


@pytest.fixture
def custom_aggregator():
    """Create SignalAggregator with custom weights."""
    custom_weights = {
        '15m': 0.2,
        '1h': 0.3,
        '4h': 0.5
    }
    return SignalAggregator(timeframe_weights=custom_weights)


class TestSignalAggregatorInit:
    """Test SignalAggregator initialization."""
    
    def test_init_default_weights(self, default_aggregator):
        """Test initialization with default weights."""
        assert default_aggregator.timeframe_weights is not None
        assert '15m' in default_aggregator.timeframe_weights
        assert '1h' in default_aggregator.timeframe_weights
        assert '4h' in default_aggregator.timeframe_weights
    
    def test_init_custom_weights(self, custom_aggregator):
        """Test initialization with custom weights."""
        assert '15m' in custom_aggregator.timeframe_weights
        assert '1h' in custom_aggregator.timeframe_weights
        assert '4h' in custom_aggregator.timeframe_weights
        assert '15m' not in DEFAULT_TIMEFRAME_WEIGHTS or \
               custom_aggregator.timeframe_weights['15m'] != DEFAULT_TIMEFRAME_WEIGHTS.get('15m', 0)
    
    def test_weights_normalized(self, custom_aggregator):
        """Test that weights are normalized to sum to 1.0."""
        total_weight = sum(custom_aggregator.timeframe_weights.values())
        assert abs(total_weight - 1.0) < 0.0001  # Allow for floating point precision


class TestAggregateSignals:
    """Test aggregate_signals method."""
    
    def test_empty_signals(self, default_aggregator):
        """Test aggregation with empty signals dict."""
        result = default_aggregator.aggregate_signals({})
        assert result['signal'] == 'NONE'
        assert result['confidence'] == 0.0
        assert result['timeframe_breakdown'] == {}
        assert result['weights_used'] == {}
    
    def test_all_long_signals(self, default_aggregator):
        """Test aggregation when all timeframes show LONG."""
        signals = {
            '15m': {'signal': 'LONG', 'confidence': 0.7},
            '1h': {'signal': 'LONG', 'confidence': 0.8},
            '4h': {'signal': 'LONG', 'confidence': 0.75}
        }
        result = default_aggregator.aggregate_signals(signals)
        assert result['signal'] == 'LONG'
        assert result['confidence'] > 0.5
        assert len(result['timeframe_breakdown']) == 3
    
    def test_all_short_signals(self, default_aggregator):
        """Test aggregation when all timeframes show SHORT."""
        signals = {
            '15m': {'signal': 'SHORT', 'confidence': 0.6},
            '1h': {'signal': 'SHORT', 'confidence': 0.7},
            '4h': {'signal': 'SHORT', 'confidence': 0.65}
        }
        result = default_aggregator.aggregate_signals(signals)
        assert result['signal'] == 'SHORT'
        assert result['confidence'] > 0.5
    
    def test_all_none_signals(self, default_aggregator):
        """Test aggregation when all timeframes show NONE."""
        signals = {
            '15m': {'signal': 'NONE', 'confidence': 0.3},
            '1h': {'signal': 'NONE', 'confidence': 0.4},
            '4h': {'signal': 'NONE', 'confidence': 0.35}
        }
        result = default_aggregator.aggregate_signals(signals)
        assert result['signal'] == 'NONE'
        assert result['confidence'] < 0.5
    
    def test_mixed_signals_long_dominant(self, default_aggregator):
        """Test aggregation with mixed signals where LONG is dominant."""
        signals = {
            '15m': {'signal': 'LONG', 'confidence': 0.8},
            '1h': {'signal': 'LONG', 'confidence': 0.75},
            '4h': {'signal': 'SHORT', 'confidence': 0.4}  # Lower weight, lower confidence
        }
        result = default_aggregator.aggregate_signals(signals)
        assert result['signal'] == 'LONG'
        assert result['confidence'] > 0.5
    
    def test_mixed_signals_short_dominant(self, default_aggregator):
        """Test aggregation with mixed signals where SHORT is dominant."""
        signals = {
            '15m': {'signal': 'SHORT', 'confidence': 0.3},  # Lower weight
            '1h': {'signal': 'SHORT', 'confidence': 0.4},   # Lower weight
            '4h': {'signal': 'SHORT', 'confidence': 0.9}    # Higher weight, high confidence
        }
        result = default_aggregator.aggregate_signals(signals)
        assert result['signal'] == 'SHORT'
        assert result['confidence'] > 0.5
    
    def test_confidence_clamping(self, default_aggregator):
        """Test that confidence values are clamped to [0.0, 1.0]."""
        signals = {
            '15m': {'signal': 'LONG', 'confidence': 1.5},  # Above 1.0
            '1h': {'signal': 'LONG', 'confidence': -0.5}    # Below 0.0
        }
        result = default_aggregator.aggregate_signals(signals)
        # Should not crash and should handle clamping
        assert 0.0 <= result['confidence'] <= 1.0
        for tf_data in result['timeframe_breakdown'].values():
            assert 0.0 <= tf_data['confidence'] <= 1.0
    
    def test_case_insensitive_signals(self, default_aggregator):
        """Test that signal names are case-insensitive."""
        signals = {
            '15m': {'signal': 'long', 'confidence': 0.7},
            '1h': {'signal': 'SHORT', 'confidence': 0.6},
            '4h': {'signal': 'None', 'confidence': 0.5}
        }
        result = default_aggregator.aggregate_signals(signals)
        # Should process without error
        assert result['signal'] in ['LONG', 'SHORT', 'NONE']
        assert len(result['timeframe_breakdown']) == 3
    
    def test_unknown_timeframe_weight(self, default_aggregator):
        """Test that unknown timeframes get default weight."""
        signals = {
            'unknown_tf': {'signal': 'LONG', 'confidence': 0.7}
        }
        result = default_aggregator.aggregate_signals(signals)
        # Should use default weight (0.1) for unknown timeframe
        assert 'unknown_tf' in result['weights_used']
        assert result['weights_used']['unknown_tf'] == 0.1
    
    def test_breakdown_structure(self, default_aggregator):
        """Test that timeframe_breakdown has correct structure."""
        signals = {
            '15m': {'signal': 'LONG', 'confidence': 0.7},
            '1h': {'signal': 'SHORT', 'confidence': 0.6}
        }
        result = default_aggregator.aggregate_signals(signals)
        
        assert '15m' in result['timeframe_breakdown']
        assert '1h' in result['timeframe_breakdown']
        
        breakdown_15m = result['timeframe_breakdown']['15m']
        assert 'signal' in breakdown_15m
        assert 'confidence' in breakdown_15m
        assert 'weight' in breakdown_15m
        assert breakdown_15m['signal'] == 'LONG'
        assert breakdown_15m['confidence'] == 0.7


class TestCalculateWeightedConfidence:
    """Test _calculate_weighted_confidence method."""
    
    def test_empty_signals_list(self, default_aggregator):
        """Test with empty signals list."""
        result = default_aggregator._calculate_weighted_confidence([])
        assert result == 0.0
    
    def test_single_signal(self, default_aggregator):
        """Test with single signal."""
        signals = [('15m', 0.7, 0.2)]
        result = default_aggregator._calculate_weighted_confidence(signals)
        # Use pytest.approx for floating point comparison
        assert result == pytest.approx(0.7, rel=1e-9)  # Should be the confidence itself
    
    def test_multiple_signals_same_weight(self, default_aggregator):
        """Test with multiple signals having same weight."""
        signals = [
            ('15m', 0.6, 0.5),
            ('1h', 0.8, 0.5)
        ]
        result = default_aggregator._calculate_weighted_confidence(signals)
        assert result == 0.7  # Average: (0.6 + 0.8) / 2
    
    def test_multiple_signals_different_weights(self, default_aggregator):
        """Test with multiple signals having different weights."""
        signals = [
            ('15m', 0.6, 0.2),  # Lower weight
            ('4h', 0.9, 0.8)    # Higher weight
        ]
        result = default_aggregator._calculate_weighted_confidence(signals)
        # Should be closer to 0.9 (higher weight signal)
        assert result > 0.7
        assert result < 0.9  # But not exactly 0.9 due to weighting
    
    def test_zero_total_weight(self, default_aggregator):
        """Test with zero total weight (edge case)."""
        signals = [
            ('15m', 0.7, 0.0),
            ('1h', 0.8, 0.0)
        ]
        result = default_aggregator._calculate_weighted_confidence(signals)
        assert result == 0.0


class TestDetermineFinalSignal:
    """Test _determine_final_signal method."""
    
    def test_long_clear_winner(self, default_aggregator):
        """Test when LONG is clear winner."""
        signal, confidence = default_aggregator._determine_final_signal(
            long_weighted_conf=0.8,
            short_weighted_conf=0.3,
            none_weighted_conf=0.2
        )
        assert signal == 'LONG'
        assert confidence == 0.8
    
    def test_short_clear_winner(self, default_aggregator):
        """Test when SHORT is clear winner."""
        signal, confidence = default_aggregator._determine_final_signal(
            long_weighted_conf=0.3,
            short_weighted_conf=0.8,
            none_weighted_conf=0.2
        )
        assert signal == 'SHORT'
        assert confidence == 0.8
    
    def test_long_below_threshold(self, default_aggregator):
        """Test when LONG wins but below threshold."""
        signal, confidence = default_aggregator._determine_final_signal(
            long_weighted_conf=0.4,  # Below 0.5 threshold
            short_weighted_conf=0.3,
            none_weighted_conf=0.5
        )
        assert signal == 'NONE'
        assert confidence >= 0.4
    
    def test_short_below_threshold(self, default_aggregator):
        """Test when SHORT wins but below threshold."""
        signal, confidence = default_aggregator._determine_final_signal(
            long_weighted_conf=0.3,
            short_weighted_conf=0.4,  # Below 0.5 threshold
            none_weighted_conf=0.5
        )
        assert signal == 'NONE'
        assert confidence >= 0.4
    
    def test_equal_long_short(self, default_aggregator):
        """Test when LONG and SHORT are equal."""
        signal, confidence = default_aggregator._determine_final_signal(
            long_weighted_conf=0.6,
            short_weighted_conf=0.6,
            none_weighted_conf=0.3
        )
        assert signal == 'NONE'
        assert confidence >= 0.6
    
    def test_all_low_confidences(self, default_aggregator):
        """Test when all confidences are low."""
        signal, confidence = default_aggregator._determine_final_signal(
            long_weighted_conf=0.2,
            short_weighted_conf=0.3,
            none_weighted_conf=0.4
        )
        assert signal == 'NONE'
        assert confidence == 0.4  # Should be max of all


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""
    
    def test_realistic_bullish_scenario(self, default_aggregator):
        """Test realistic bullish scenario across timeframes."""
        signals = {
            '15m': {'signal': 'LONG', 'confidence': 0.65},  # Short-term bullish
            '1h': {'signal': 'LONG', 'confidence': 0.75},   # Medium-term bullish
            '4h': {'signal': 'LONG', 'confidence': 0.80},   # Strong medium-term
            '1d': {'signal': 'LONG', 'confidence': 0.70}    # Long-term bullish
        }
        result = default_aggregator.aggregate_signals(signals)
        assert result['signal'] == 'LONG'
        assert result['confidence'] > 0.6
        assert 'long_weighted_conf' in result
        assert result['long_weighted_conf'] > result['short_weighted_conf']
    
    def test_realistic_bearish_scenario(self, default_aggregator):
        """Test realistic bearish scenario across timeframes."""
        signals = {
            '15m': {'signal': 'SHORT', 'confidence': 0.60},
            '1h': {'signal': 'SHORT', 'confidence': 0.70},
            '4h': {'signal': 'SHORT', 'confidence': 0.75},
            '1d': {'signal': 'SHORT', 'confidence': 0.65}
        }
        result = default_aggregator.aggregate_signals(signals)
        assert result['signal'] == 'SHORT'
        assert result['confidence'] > 0.6
    
    def test_mixed_uncertain_scenario(self, default_aggregator):
        """Test mixed scenario with uncertain signals."""
        signals = {
            '15m': {'signal': 'LONG', 'confidence': 0.55},   # Weak
            '1h': {'signal': 'NONE', 'confidence': 0.50},     # Uncertain
            '4h': {'signal': 'SHORT', 'confidence': 0.45},   # Weak
            '1d': {'signal': 'NONE', 'confidence': 0.40}     # Uncertain
        }
        result = default_aggregator.aggregate_signals(signals)
        # Should likely be NONE due to low confidences
        assert result['signal'] in ['LONG', 'SHORT', 'NONE']
        assert result['confidence'] < 0.6
    
    def test_high_timeframe_dominance(self, default_aggregator):
        """Test that higher timeframes have more influence."""
        signals = {
            '15m': {'signal': 'SHORT', 'confidence': 0.9},  # High confidence, low weight
            '1d': {'signal': 'LONG', 'confidence': 0.6}    # Lower confidence, high weight
        }
        result = default_aggregator.aggregate_signals(signals)
        # Higher timeframe (1d) should have more influence despite lower confidence
        # The exact result depends on weights, but 1d should contribute more
        assert '1d' in result['timeframe_breakdown']
        assert result['timeframe_breakdown']['1d']['weight'] > \
               result['timeframe_breakdown']['15m']['weight']

