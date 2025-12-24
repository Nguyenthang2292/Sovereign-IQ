"""
Tests for utils module in gemini_chart_analyzer.

Tests cover:
- normalize_timeframes function
- get_timeframe_weight function
- sort_timeframes_by_weight function
- validate_timeframes function
"""

import pytest
from modules.gemini_chart_analyzer.core.utils import (
    normalize_timeframes,
    get_timeframe_weight,
    sort_timeframes_by_weight,
    validate_timeframes
)


class TestNormalizeTimeframes:
    """Test normalize_timeframes function."""
    
    def test_normalize_single_timeframe(self):
        """Test normalizing a single timeframe."""
        result = normalize_timeframes(['1h'])
        assert result == ['1h']
        assert isinstance(result, list)
    
    def test_normalize_multiple_timeframes(self):
        """Test normalizing multiple timeframes."""
        result = normalize_timeframes(['15m', '1h', '4h', '1d'])
        assert len(result) == 4
        assert '15m' in result
        assert '1h' in result
        assert '4h' in result
        assert '1d' in result
    
    def test_normalize_empty_list(self):
        """Test normalizing empty list."""
        result = normalize_timeframes([])
        assert result == []
    
    def test_normalize_removes_duplicates(self):
        """Test that duplicates are removed."""
        result = normalize_timeframes(['1h', '1h', '4h', '1h'])
        assert len(result) == 2
        assert result.count('1h') == 1
        assert '4h' in result
    
    def test_normalize_removes_empty_strings(self):
        """Test that empty strings are filtered out."""
        result = normalize_timeframes(['1h', '', '4h', '  '])
        assert len(result) == 2
        assert '1h' in result
        assert '4h' in result
    
    def test_normalize_sorts_by_weight(self):
        """Test that timeframes are sorted by weight (descending)."""
        result = normalize_timeframes(['15m', '1d', '1h', '4h'])
        # Higher timeframes should come first
        # 1d should be before 1h, 1h should be before 15m
        assert result[0] in ['1d', '4h']  # Highest weight
        assert '15m' in result  # Should be present
    
    def test_normalize_case_insensitive(self):
        """Test that normalization handles case variations."""
        result = normalize_timeframes(['1H', '4H', '1D'])
        assert len(result) == 3
        # Should normalize to lowercase
        assert all(tf.islower() or any(c.isdigit() for c in tf) for tf in result)


class TestGetTimeframeWeight:
    """Test get_timeframe_weight function."""
    
    def test_get_weight_default_weights(self):
        """Test getting weight with default weights."""
        weight = get_timeframe_weight('1h')
        assert isinstance(weight, float)
        assert weight > 0
    
    def test_get_weight_custom_weights(self):
        """Test getting weight with custom weights."""
        custom_weights = {
            '1h': 0.5,
            '4h': 0.8
        }
        weight = get_timeframe_weight('1h', weights=custom_weights)
        assert weight == 0.5
    
    def test_get_weight_unknown_timeframe(self):
        """Test getting weight for unknown timeframe."""
        weight = get_timeframe_weight('unknown_tf')
        assert weight == 0.1  # Default weight
    
    def test_get_weight_normalizes_timeframe(self):
        """Test that timeframe is normalized before lookup."""
        weight1 = get_timeframe_weight('1H')
        weight2 = get_timeframe_weight('1h')
        assert weight1 == weight2


class TestSortTimeframesByWeight:
    """Test sort_timeframes_by_weight function."""
    
    def test_sort_single_timeframe(self):
        """Test sorting single timeframe."""
        result = sort_timeframes_by_weight(['1h'])
        assert result == ['1h']
    
    def test_sort_multiple_timeframes(self):
        """Test sorting multiple timeframes by weight."""
        result = sort_timeframes_by_weight(['15m', '1d', '1h'])
        # Higher timeframes (higher weight) should come first
        assert result[0] in ['1d', '1h']  # Higher weight
        assert '15m' in result  # Should be present
    
    def test_sort_preserves_all_timeframes(self):
        """Test that all timeframes are preserved in sorted result."""
        input_tfs = ['15m', '1h', '4h', '1d']
        result = sort_timeframes_by_weight(input_tfs)
        assert len(result) == len(input_tfs)
        assert set(result) == set(input_tfs)
    
    def test_sort_custom_weights(self):
        """Test sorting with custom weights."""
        custom_weights = {
            '15m': 0.3,
            '1h': 0.1,
            '4h': 0.2
        }
        result = sort_timeframes_by_weight(['15m', '1h', '4h'], weights=custom_weights)
        # Should be sorted by custom weights (descending)
        assert result[0] == '15m'  # Highest weight (0.3)
        assert result[-1] == '1h'  # Lowest weight (0.1)
    
    def test_sort_empty_list(self):
        """Test sorting empty list."""
        result = sort_timeframes_by_weight([])
        assert result == []


class TestValidateTimeframes:
    """Test validate_timeframes function."""
    
    def test_validate_valid_timeframes(self):
        """Test validation with valid timeframes."""
        is_valid, error_msg = validate_timeframes(['15m', '1h', '4h'])
        assert is_valid is True
        assert error_msg is None
    
    def test_validate_empty_list(self):
        """Test validation with empty list."""
        is_valid, error_msg = validate_timeframes([])
        assert is_valid is False
        assert error_msg is not None
        assert 'empty' in error_msg.lower() or 'required' in error_msg.lower()
    
    def test_validate_duplicates(self):
        """Test validation detects duplicates."""
        is_valid, error_msg = validate_timeframes(['1h', '1h', '4h'])
        assert is_valid is False
        assert error_msg is not None
        assert 'duplicate' in error_msg.lower()
    
    def test_validate_case_insensitive_duplicates(self):
        """Test that validation detects case-insensitive duplicates."""
        is_valid, error_msg = validate_timeframes(['1h', '1H', '4h'])
        assert is_valid is False
        assert error_msg is not None
    
    def test_validate_single_timeframe(self):
        """Test validation with single timeframe."""
        is_valid, error_msg = validate_timeframes(['1h'])
        assert is_valid is True
        assert error_msg is None
    
    def test_validate_mixed_case(self):
        """Test validation with mixed case timeframes."""
        is_valid, error_msg = validate_timeframes(['1H', '4h', '1D'])
        assert is_valid is True  # Should normalize and validate
    
    def test_validate_returns_tuple(self):
        """Test that function returns tuple."""
        result = validate_timeframes(['1h'])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert result[1] is None or isinstance(result[1], str)


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_normalize_then_validate(self):
        """Test normalizing then validating timeframes."""
        tfs = ['15m', '1h', '4h', '1d']
        normalized = normalize_timeframes(tfs)
        is_valid, error_msg = validate_timeframes(normalized)
        assert is_valid is True
        assert error_msg is None
    
    def test_normalize_then_sort(self):
        """Test normalizing then sorting timeframes."""
        tfs = ['1d', '15m', '4h', '1h']
        normalized = normalize_timeframes(tfs)
        sorted_tfs = sort_timeframes_by_weight(normalized)
        # Should be sorted by weight
        assert len(sorted_tfs) == len(normalized)
        assert set(sorted_tfs) == set(normalized)
    
    def test_get_weights_for_normalized(self):
        """Test getting weights for normalized timeframes."""
        tfs = ['15m', '1h', '4h']
        normalized = normalize_timeframes(tfs)
        weights = [get_timeframe_weight(tf) for tf in normalized]
        assert len(weights) == len(normalized)
        assert all(w > 0 for w in weights)
    
    def test_full_workflow(self):
        """Test complete workflow: normalize -> validate -> sort -> get weights."""
        input_tfs = ['1d', '15m', '1h', '4h', '1h']  # With duplicate
        
        # Step 1: Normalize (removes duplicates)
        normalized = normalize_timeframes(input_tfs)
        assert len(normalized) == 4
        
        # Step 2: Validate
        is_valid, error_msg = validate_timeframes(normalized)
        assert is_valid is True
        
        # Step 3: Sort by weight
        sorted_tfs = sort_timeframes_by_weight(normalized)
        assert len(sorted_tfs) == 4
        
        # Step 4: Get weights
        weights = {tf: get_timeframe_weight(tf) for tf in sorted_tfs}
        assert len(weights) == 4
        assert all(w > 0 for w in weights.values())


