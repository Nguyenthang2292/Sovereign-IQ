"""
Tests for CNN feature mixin.
"""
import pytest

from modules.lstm.models.trainer.cnn_mixin import (
    CNNFeatureMixin,
    CNN_MIN_BATCH_SIZE,
    CNN_BATCH_SIZE_REDUCTION_FACTOR,
    CNN_MIN_ADDITIONAL_SAMPLES,
)


@pytest.fixture
def mixin():
    """Create CNNFeatureMixin instance for testing."""
    return CNNFeatureMixin()


class TestCNNFeatureMixin:
    """Test suite for CNNFeatureMixin class."""
    
    def test_mixin_initialization(self, mixin):
        """Test that mixin can be instantiated."""
        assert isinstance(mixin, CNNFeatureMixin)
    
    @pytest.mark.parametrize("original_batch,expected", [
        (64, max(CNN_MIN_BATCH_SIZE, 64 // CNN_BATCH_SIZE_REDUCTION_FACTOR)),
        (8, CNN_MIN_BATCH_SIZE),  # Small batch should be at least minimum
        (1, CNN_MIN_BATCH_SIZE),  # Minimum batch size
        (CNN_BATCH_SIZE_REDUCTION_FACTOR * CNN_MIN_BATCH_SIZE, CNN_MIN_BATCH_SIZE),  # Exact reduction
    ])
    def test_adjust_batch_size(self, mixin, original_batch, expected):
        """Test batch size adjustment for various input sizes."""
        adjusted = mixin._adjust_batch_size_for_cnn(original_batch)
        assert adjusted == expected
        assert adjusted >= CNN_MIN_BATCH_SIZE
    
    @pytest.mark.parametrize("data_length,look_back,should_raise", [
        # Sufficient data cases
        (20 + CNN_MIN_ADDITIONAL_SAMPLES + 50, 20, False),
        (20 + CNN_MIN_ADDITIONAL_SAMPLES, 20, False),  # Exact minimum
        # Insufficient data cases
        (20 + CNN_MIN_ADDITIONAL_SAMPLES - 1, 20, True),
        (10, 20, True),  # Very short data
        (0, 20, True),  # Zero length
    ])
    def test_validate_cnn_data_requirements(self, mixin, data_length, look_back, should_raise):
        """Test validation with various data lengths."""
        if should_raise:
            with pytest.raises(ValueError, match="Insufficient data for CNN model"):
                mixin._validate_cnn_data_requirements(data_length, look_back)
        else:
            # Should not raise
            mixin._validate_cnn_data_requirements(data_length, look_back)
    
    @pytest.mark.parametrize("look_back", [10, 20, 30, 50])
    def test_validate_cnn_data_requirements_different_look_back(self, mixin, look_back):
        """Test validation with different look_back values."""
        min_required = look_back + CNN_MIN_ADDITIONAL_SAMPLES
        
        # Sufficient
        mixin._validate_cnn_data_requirements(min_required, look_back)
        
        # Insufficient
        with pytest.raises(ValueError, match="Insufficient data for CNN model"):
            mixin._validate_cnn_data_requirements(min_required - 1, look_back)

