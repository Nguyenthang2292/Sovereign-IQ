"""
Tests for Pivot class in smart money concept module.
"""

import pytest
from datetime import datetime, timedelta, timezone
from modules.smart_money_concept.pivot import Pivot


class TestPivotInitialization:
    """Test Pivot class initialization and validation."""
    
    def test_default_initialization(self):
        """Test creating pivot with default values."""
        pivot = Pivot()
        assert pivot.level == 0.0
        assert pivot.bar_time is not None
        assert pivot.bar_time.tzinfo is not None  # Should have timezone
        assert pivot.pivot_type == 'other'
        assert pivot.strength == 1
        assert pivot.pip_size == 0.0001
    
    def test_custom_initialization(self):
        """Test creating pivot with custom values."""
        test_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        pivot = Pivot(
            level=50000.0,
            bar_time=test_time,
            pivot_type='support',
            strength=5,
            pip_size=0.01
        )
        assert pivot.level == 50000.0
        assert pivot.bar_time == test_time
        assert pivot.pivot_type == 'support'
        assert pivot.strength == 5
        assert pivot.pip_size == 0.01
    
    def test_negative_level_validation(self):
        """Test that negative levels are corrected to 0.0."""
        pivot = Pivot(level=-100.0)
        assert pivot.level == 0.0
    
    def test_invalid_level_type_validation(self):
        """Test that invalid level types are corrected."""
        pivot = Pivot(level="invalid")
        assert pivot.level == 0.0
    
    def test_none_timezone_handling(self):
        """Test that None timezone is set to UTC."""
        test_time = datetime(2024, 1, 1, 12, 0)  # No timezone
        pivot = Pivot(bar_time=test_time)
        assert pivot.bar_time.tzinfo == timezone.utc
    
    def test_strength_validation(self):
        """Test that strength less than 1 is corrected."""
        pivot = Pivot(strength=0)
        assert pivot.strength == 1
        
        pivot = Pivot(strength=-5)
        assert pivot.strength == 1
    
    def test_invalid_pip_size_validation(self):
        """Test that invalid pip_size is corrected."""
        pivot = Pivot(pip_size=0)
        assert pivot.pip_size == 0.0001
        
        pivot = Pivot(pip_size=-0.001)
        assert pivot.pip_size == 0.0001


class TestPivotStringRepresentation:
    """Test string representation methods."""
    
    def test_str_representation(self):
        """Test __str__ method."""
        test_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        pivot = Pivot(level=50000.0, bar_time=test_time, pivot_type='support', strength=3)
        str_repr = str(pivot)
        assert 'Support' in str_repr
        assert '50000.0' in str_repr
        assert '2024-01-01 12:00' in str_repr
        assert 'Strength: 3' in str_repr
    
    def test_repr_representation(self):
        """Test __repr__ method."""
        test_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        pivot = Pivot(level=50000.0, bar_time=test_time, pivot_type='resistance')
        repr_str = repr(pivot)
        assert 'Pivot(' in repr_str
        assert 'level=50000.0' in repr_str
        assert "pivot_type='resistance'" in repr_str


class TestPivotDistanceMethods:
    """Test distance calculation methods."""
    
    def test_distance_to(self):
        """Test absolute distance calculation."""
        pivot = Pivot(level=50000.0)
        assert pivot.distance_to(50100.0) == 100.0
        assert pivot.distance_to(49900.0) == 100.0
        assert pivot.distance_to(50000.0) == 0.0
    
    def test_percent_distance_to(self):
        """Test percentage distance calculation."""
        pivot = Pivot(level=50000.0)
        assert pivot.percent_distance_to(51000.0) == 2.0  # 2% above
        assert pivot.percent_distance_to(49000.0) == 2.0  # 2% below
        assert pivot.percent_distance_to(50000.0) == 0.0
    
    def test_percent_distance_zero_level(self):
        """Test percentage distance when level is 0."""
        pivot = Pivot(level=0.0)
        assert pivot.percent_distance_to(100.0) == float('inf')
        assert pivot.percent_distance_to(0.0) == 0.0


class TestPivotComparisonMethods:
    """Test price comparison methods."""
    
    def test_is_above(self):
        """Test is_above method."""
        pivot = Pivot(level=50000.0)
        assert pivot.is_above(49000.0) is True
        assert pivot.is_above(51000.0) is False
        assert pivot.is_above(50000.0) is False
    
    def test_is_above_with_buffer(self):
        """Test is_above with buffer."""
        pivot = Pivot(level=50000.0, pip_size=0.0001)
        # Buffer of 10 pips = 0.001
        # pivot (50000) is above (49999 + 0.001) = 49999.001? Yes, 50000 > 49999.001
        assert pivot.is_above(49999.0, buffer_pips=10) is True
        # pivot (50000) is above (49999.5 + 0.001) = 49999.501? Yes, 50000 > 49999.501
        assert pivot.is_above(49999.5, buffer_pips=10) is True
        # pivot (50000) is above (49999.9 + 0.001) = 49999.901? Yes
        assert pivot.is_above(49999.9, buffer_pips=10) is True
        # pivot (50000) is above (50000.0 + 0.001) = 50000.001? No
        assert pivot.is_above(50000.0, buffer_pips=10) is False
    
    def test_is_below(self):
        """Test is_below method."""
        pivot = Pivot(level=50000.0)
        assert pivot.is_below(51000.0) is True
        assert pivot.is_below(49000.0) is False
        assert pivot.is_below(50000.0) is False
    
    def test_is_below_with_buffer(self):
        """Test is_below with buffer."""
        pivot = Pivot(level=50000.0, pip_size=0.0001)
        # Buffer of 10 pips = 0.001
        # pivot (50000) is below (50001.0 - 0.001) = 50000.999? Yes, 50000 < 50000.999
        assert pivot.is_below(50001.0, buffer_pips=10) is True
        # pivot (50000) is below (50000.5 - 0.001) = 50000.499? Yes, 50000 < 50000.499
        assert pivot.is_below(50000.5, buffer_pips=10) is True
        # pivot (50000) is below (50000.1 - 0.001) = 50000.099? Yes
        assert pivot.is_below(50000.1, buffer_pips=10) is True
        # pivot (50000) is below (50000.0 - 0.001) = 49999.999? No
        assert pivot.is_below(50000.0, buffer_pips=10) is False
    
    def test_is_near_with_pips(self):
        """Test is_near using pips."""
        pivot = Pivot(level=50000.0, pip_size=0.0001)
        # 10 pips = 0.001
        assert pivot.is_near(50000.0005, pips=10) is True
        assert pivot.is_near(50000.002, pips=10) is False
    
    def test_is_near_with_percentage(self):
        """Test is_near using percentage."""
        pivot = Pivot(level=50000.0)
        # 1% = 500
        assert pivot.is_near(50050.0, pips=1, use_percentage=True) is True
        assert pivot.is_near(50600.0, pips=1, use_percentage=True) is False


class TestPivotTimeMethods:
    """Test time-related methods."""
    
    def test_is_recent(self):
        """Test is_recent method."""
        recent_time = datetime.now(timezone.utc) - timedelta(days=5)
        old_time = datetime.now(timezone.utc) - timedelta(days=60)
        
        recent_pivot = Pivot(level=50000.0, bar_time=recent_time)
        old_pivot = Pivot(level=50000.0, bar_time=old_time)
        
        assert recent_pivot.is_recent() is True
        assert old_pivot.is_recent() is False
    
    def test_is_recent_custom_max_age(self):
        """Test is_recent with custom max_age."""
        test_time = datetime.now(timezone.utc) - timedelta(days=10)
        pivot = Pivot(level=50000.0, bar_time=test_time)
        
        assert pivot.is_recent(max_age=timedelta(days=15)) is True
        assert pivot.is_recent(max_age=timedelta(days=5)) is False
    
    def test_is_recent_none_time(self):
        """Test is_recent when bar_time is None."""
        pivot = Pivot(level=50000.0, bar_time=None)
        # Should set time in __post_init__, but test edge case
        pivot.bar_time = None
        assert pivot.is_recent() is False


class TestPivotMerge:
    """Test pivot merging functionality."""
    
    def test_merge_with_basic(self):
        """Test basic merge functionality."""
        pivot1 = Pivot(level=50000.0, bar_time=datetime(2024, 1, 1, tzinfo=timezone.utc), strength=3)
        pivot2 = Pivot(level=50100.0, bar_time=datetime(2024, 1, 2, tzinfo=timezone.utc), strength=2)
        
        merged = pivot1.merge_with(pivot2)
        assert merged.level == 50050.0  # Average
        assert merged.bar_time == datetime(2024, 1, 2, tzinfo=timezone.utc)  # Newer time
        assert merged.strength == 5  # Sum when strengthen=True
        assert merged.pivot_type == 'other'  # From pivot1 (higher strength)
    
    def test_merge_with_strengthen_false(self):
        """Test merge without strengthening."""
        pivot1 = Pivot(level=50000.0, strength=3)
        pivot2 = Pivot(level=50100.0, strength=2)
        
        merged = pivot1.merge_with(pivot2, strengthen=False)
        assert merged.strength == 3  # Max, not sum
    
    def test_merge_type_selection(self):
        """Test that merge selects type from stronger pivot."""
        pivot1 = Pivot(level=50000.0, pivot_type='support', strength=2)
        pivot2 = Pivot(level=50100.0, pivot_type='resistance', strength=5)
        
        merged = pivot1.merge_with(pivot2)
        assert merged.pivot_type == 'resistance'  # From pivot2 (higher strength)
    
    def test_merge_pip_size(self):
        """Test that merge averages pip_size."""
        pivot1 = Pivot(level=50000.0, pip_size=0.0001)
        pivot2 = Pivot(level=50100.0, pip_size=0.0002)
        
        merged = pivot1.merge_with(pivot2)
        # Use pytest.approx for floating point comparison
        assert merged.pip_size == pytest.approx(0.00015, rel=1e-9)  # Average
    
    def test_merge_with_invalid_type(self):
        """Test that merge raises TypeError for invalid type."""
        pivot = Pivot(level=50000.0)
        
        with pytest.raises(TypeError, match="Expected Pivot instance"):
            pivot.merge_with("not a pivot")
        
        with pytest.raises(TypeError, match="Expected Pivot instance"):
            pivot.merge_with(None)


class TestPivotEqualityAndComparison:
    """Test equality and comparison operators."""
    
    def test_equality_same_level_and_type(self):
        """Test __eq__ with same level and type."""
        pivot1 = Pivot(level=50000.0, pivot_type='support')
        pivot2 = Pivot(level=50000.0, pivot_type='support')
        assert pivot1 == pivot2
    
    def test_equality_different_level(self):
        """Test __eq__ with different level."""
        pivot1 = Pivot(level=50000.0, pivot_type='support')
        pivot2 = Pivot(level=50100.0, pivot_type='support')
        assert pivot1 != pivot2
    
    def test_equality_different_type(self):
        """Test __eq__ with different type."""
        pivot1 = Pivot(level=50000.0, pivot_type='support')
        pivot2 = Pivot(level=50000.0, pivot_type='resistance')
        assert pivot1 != pivot2
    
    def test_equality_with_non_pivot(self):
        """Test __eq__ with non-Pivot object."""
        pivot = Pivot(level=50000.0)
        assert pivot != "not a pivot"
        assert pivot != None
        assert pivot != 50000.0
    
    def test_less_than_comparison(self):
        """Test __lt__ for sorting."""
        pivot1 = Pivot(level=50000.0)
        pivot2 = Pivot(level=51000.0)
        
        assert pivot1 < pivot2
        assert not (pivot2 < pivot1)
    
    def test_less_than_with_non_pivot(self):
        """Test __lt__ with non-Pivot object."""
        pivot = Pivot(level=50000.0)
        # Python 3.13 may raise TypeError instead of returning NotImplemented
        try:
            result = pivot < "not a pivot"
            assert result is NotImplemented
        except TypeError:
            # This is also acceptable behavior
            pass


class TestPivotEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_level(self):
        """Test pivot with zero level."""
        pivot = Pivot(level=0.0)
        assert pivot.distance_to(100.0) == 100.0
        assert pivot.percent_distance_to(100.0) == float('inf')
    
    def test_very_large_level(self):
        """Test pivot with very large level."""
        pivot = Pivot(level=1e10)
        assert pivot.distance_to(1e10 + 1000) == 1000.0
    
    def test_very_small_pip_size(self):
        """Test pivot with very small pip_size."""
        pivot = Pivot(level=50000.0, pip_size=1e-8)
        # 10 pips = 10 * 1e-8 = 1e-7
        # Use slightly smaller value to account for floating point precision
        # distance = 0.5e-7, threshold = 1e-7, so should be True (within range)
        assert pivot.is_near(50000.0 + 0.5e-7, pips=10) is True
        # Use value slightly less than threshold to avoid floating point precision issues
        assert pivot.is_near(50000.0 + 0.99e-7, pips=10) is True
        # distance = 2e-7, threshold = 1e-7, so should be False (outside range)
        assert pivot.is_near(50000.0 + 2e-7, pips=10) is False
    
    def test_different_pivot_types(self):
        """Test all pivot types."""
        types = ['support', 'resistance', 'swing_high', 'swing_low', 'other']
        for pivot_type in types:
            pivot = Pivot(level=50000.0, pivot_type=pivot_type)
            assert pivot.pivot_type == pivot_type
    
    def test_merge_same_pivot(self):
        """Test merging pivot with itself."""
        pivot = Pivot(level=50000.0, strength=3)
        merged = pivot.merge_with(pivot)
        assert merged.level == 50000.0
        assert merged.strength == 6  # Doubled when strengthen=True


class TestPivotIntegration:
    """Integration tests for Pivot class."""
    
    def test_full_workflow(self):
        """Test complete workflow with multiple operations."""
        # Create pivot with recent time
        recent_time = datetime.now(timezone.utc) - timedelta(days=5)
        pivot = Pivot(
            level=50000.0,
            bar_time=recent_time,
            pivot_type='support',
            strength=5
        )
        
        # Check distance
        assert pivot.distance_to(50100.0) == 100.0
        assert pivot.percent_distance_to(50100.0) == 0.2
        
        # Check comparisons
        assert pivot.is_above(49000.0) is True
        assert pivot.is_below(51000.0) is True
        # 100 pips with pip_size=0.0001 = 0.01, distance to 50010.0 = 10.0, so False
        assert pivot.is_near(50010.0, pips=100) is False
        # But with percentage: 50010 is 0.02% away, so within 1%
        assert pivot.is_near(50010.0, pips=1, use_percentage=True) is True
        
        # Check time (should be recent)
        assert pivot.is_recent() is True
        
        # Merge with another pivot
        pivot2 = Pivot(level=50050.0, strength=3)
        merged = pivot.merge_with(pivot2)
        assert merged.level == 50025.0
        assert merged.strength == 8
    
    def test_sorting_pivots(self):
        """Test sorting list of pivots."""
        pivots = [
            Pivot(level=51000.0),
            Pivot(level=49000.0),
            Pivot(level=50000.0),
        ]
        sorted_pivots = sorted(pivots)
        assert sorted_pivots[0].level == 49000.0
        assert sorted_pivots[1].level == 50000.0
        assert sorted_pivots[2].level == 51000.0
    
    def test_pivot_in_set(self):
        """Test using pivot in set (requires __eq__ and __hash__)."""
        # Note: dataclass automatically provides __hash__ if frozen=True
        # But Pivot is not frozen, so we can't use it in sets directly
        # This test verifies equality works correctly
        pivot1 = Pivot(level=50000.0, pivot_type='support')
        pivot2 = Pivot(level=50000.0, pivot_type='support')
        pivot3 = Pivot(level=50000.0, pivot_type='resistance')
        
        assert pivot1 == pivot2
        assert pivot1 != pivot3

