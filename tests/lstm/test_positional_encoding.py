import warnings

import pytest
import torch

from modules.lstm.core.positional_encoding import PositionalEncoding

"""
Tests for PositionalEncoding module.
"""


class TestPositionalEncoding:
    """Test suite for PositionalEncoding class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        pe = PositionalEncoding(d_model=64, max_seq_length=100)
        assert pe.pe.shape == (1, 100, 64)
        assert not pe._length_warning_emitted

    def test_init_invalid_d_model(self):
        """Test initialization with invalid d_model."""
        with pytest.raises(ValueError, match="d_model must be positive"):
            PositionalEncoding(d_model=0)

        with pytest.raises(ValueError, match="d_model must be positive"):
            PositionalEncoding(d_model=-1)

    def test_init_invalid_max_seq_length(self):
        """Test initialization with invalid max_seq_length."""
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            PositionalEncoding(d_model=64, max_seq_length=0)

        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            PositionalEncoding(d_model=64, max_seq_length=-1)

    def test_forward_valid_input(self):
        """Test forward pass with valid input."""
        pe = PositionalEncoding(d_model=64, max_seq_length=100)
        x = torch.randn(2, 50, 64)  # batch_size=2, seq_len=50, d_model=64

        output = pe(x)
        assert output.shape == (2, 50, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_invalid_dimensions(self):
        """Test forward pass with invalid input dimensions."""
        pe = PositionalEncoding(d_model=64, max_seq_length=100)

        # 2D input instead of 3D
        with pytest.raises(ValueError, match="Input must be 3D tensor"):
            pe(torch.randn(50, 64))

        # 4D input
        with pytest.raises(ValueError, match="Input must be 3D tensor"):
            pe(torch.randn(2, 50, 64, 1))

    def test_forward_d_model_mismatch(self):
        """Test forward pass with d_model mismatch."""
        pe = PositionalEncoding(d_model=64, max_seq_length=100)
        x = torch.randn(2, 50, 32)  # Wrong d_model

        with pytest.raises(ValueError, match="Input d_model.*doesn't match"):
            pe(x)

    def test_forward_sequence_shorter_than_max(self):
        """Test forward pass when sequence length is shorter than max_seq_length."""
        pe = PositionalEncoding(d_model=64, max_seq_length=100)
        x = torch.randn(2, 50, 64)

        output = pe(x)
        assert output.shape == (2, 50, 64)
        # Should use pre-computed encodings
        assert not pe._length_warning_emitted

    def test_forward_sequence_longer_than_max(self):
        """Test forward pass when sequence length exceeds max_seq_length."""
        pe = PositionalEncoding(d_model=64, max_seq_length=50)
        x = torch.randn(2, 100, 64)  # seq_len > max_seq_length

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = pe(x)

            # Should emit warning once
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "exceeds max_seq_length" in str(w[0].message)
            assert pe._length_warning_emitted

        assert output.shape == (2, 100, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_warning_only_once(self):
        """Test that warning is only emitted once per instance."""
        pe = PositionalEncoding(d_model=64, max_seq_length=50)
        x = torch.randn(2, 100, 64)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # First forward pass
            pe(x)
            assert len(w) == 1
            assert pe._length_warning_emitted

            # Second forward pass - should not emit warning again
            pe(x)
            assert len(w) == 1  # Still only one warning

    def test_forward_device_transfer(self):
        """Test forward pass with device transfer."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        pe = PositionalEncoding(d_model=64, max_seq_length=100)
        pe = pe.cuda()
        x = torch.randn(2, 50, 64).cuda()

        output = pe(x)
        assert output.device.type == "cuda"
        assert output.shape == (2, 50, 64)

    def test_forward_device_mismatch(self):
        """Test forward pass when input is on different device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        pe = PositionalEncoding(d_model=64, max_seq_length=100)
        # pe is on CPU, x is on CUDA
        x = torch.randn(2, 50, 64).cuda()

        output = pe(x)
        assert output.device.type == "cuda"
        assert output.shape == (2, 50, 64)

    def test_forward_odd_d_model(self):
        """Test forward pass with odd d_model to ensure positional encoding handles it correctly."""
        # Test with odd d_model (65) - this tests the internal logic for handling odd dimensions
        pe = PositionalEncoding(d_model=65, max_seq_length=100)
        x = torch.randn(2, 50, 65)

        output = pe(x)
        assert output.shape == (2, 50, 65)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Verify positional encoding is added correctly
        # The output should be different from input (positional encoding added)
        assert not torch.allclose(output, x, atol=1e-6)

    def test_forward_extended_sequence_encoding(self):
        """Test that positional encoding is correctly computed for sequences longer than max_seq_length."""
        # This tests the on-the-fly encoding creation (start_pos > 0 case)
        pe = PositionalEncoding(d_model=64, max_seq_length=50)
        x = torch.randn(2, 100, 64)  # seq_len > max_seq_length

        # Suppress expected warning about sequence length exceeding max_seq_length
        # This is intentional to test on-the-fly encoding behavior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            output = pe(x)

        assert output.shape == (2, 100, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_exact_max_length(self):
        """Test forward pass with sequence length exactly equal to max_seq_length."""
        pe = PositionalEncoding(d_model=64, max_seq_length=100)
        x = torch.randn(2, 100, 64)

        output = pe(x)
        assert output.shape == (2, 100, 64)
        assert not pe._length_warning_emitted

    def test_forward_batch_consistency(self):
        """Test that forward pass produces consistent results for same input."""
        pe = PositionalEncoding(d_model=64, max_seq_length=100)
        torch.manual_seed(42)
        x1 = torch.randn(2, 50, 64)
        torch.manual_seed(42)
        x2 = torch.randn(2, 50, 64)

        # Verify inputs are identical
        assert torch.equal(x1, x2)

        output1 = pe(x1)
        output2 = pe(x2)

        # Results should be identical (deterministic)
        assert torch.allclose(output1, output2)

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        pe = PositionalEncoding(d_model=64, max_seq_length=100)

        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 50, 64)
            output = pe(x)
            assert output.shape == (batch_size, 50, 64)

    def test_forward_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        # Set max_seq_length to cover all test sequence lengths to avoid warnings
        pe = PositionalEncoding(d_model=64, max_seq_length=200)

        for seq_len in [10, 50, 100, 150, 200]:
            x = torch.randn(2, seq_len, 64)
            output = pe(x)
            assert output.shape == (2, seq_len, 64)
