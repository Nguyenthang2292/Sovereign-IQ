
import pytest
import torch

from modules.lstm.core.feed_forward import FeedForward
from modules.lstm.core.feed_forward import FeedForward

"""
Tests for FeedForward module.
"""




class TestFeedForward:
    """Test suite for FeedForward class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        ff = FeedForward(d_model=64, d_ff=256, dropout=0.1)
        assert ff.linear1.in_features == 64
        assert ff.linear1.out_features == 256
        assert ff.linear2.in_features == 256
        assert ff.linear2.out_features == 64

    def test_forward_basic(self):
        """Test basic forward pass."""
        ff = FeedForward(d_model=64, d_ff=256, dropout=0.1)
        x = torch.randn(2, 10, 64)  # batch_size=2, seq_len=10, d_model=64

        output = ff(x)
        assert output.shape == (2, 10, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_forward_different_batch_sizes(self, batch_size):
        """Test forward pass with different batch sizes."""
        ff = FeedForward(d_model=64, d_ff=256, dropout=0.1)
        x = torch.randn(batch_size, 10, 64)
        output = ff(x)
        assert output.shape == (batch_size, 10, 64)

    @pytest.mark.parametrize("seq_len", [1, 10, 50, 100])
    def test_forward_different_sequence_lengths(self, seq_len):
        """Test forward pass with different sequence lengths."""
        ff = FeedForward(d_model=64, d_ff=256, dropout=0.1)
        x = torch.randn(2, seq_len, 64)
        output = ff(x)
        assert output.shape == (2, seq_len, 64)

    @pytest.mark.parametrize("d_model", [32, 64, 128, 256])
    def test_forward_different_d_models(self, d_model):
        """Test forward pass with different d_model values."""
        ff = FeedForward(d_model=d_model, d_ff=d_model * 2, dropout=0.1)
        x = torch.randn(2, 10, d_model)
        output = ff(x)
        assert output.shape == (2, 10, d_model)

    def test_forward_dropout_training(self):
        """Test that dropout is applied during training."""
        ff = FeedForward(d_model=64, d_ff=256, dropout=0.5)
        ff.train()

        x = torch.randn(2, 10, 64)
        output1 = ff(x)
        output2 = ff(x)

        # With dropout, outputs should differ (with high probability)
        assert not torch.allclose(output1, output2, atol=1e-6)

    def test_forward_dropout_eval(self):
        """Test that dropout is not applied during eval."""
        ff = FeedForward(d_model=64, d_ff=256, dropout=0.5)
        ff.eval()

        x = torch.randn(2, 10, 64)
        output1 = ff(x)
        output2 = ff(x)

        # Without dropout, outputs should be identical
        assert torch.allclose(output1, output2)

    def test_forward_no_dropout(self):
        """Test forward pass with dropout=0."""
        ff = FeedForward(d_model=64, d_ff=256, dropout=0.0)
        ff.train()

        x = torch.randn(2, 10, 64)
        output1 = ff(x)
        output2 = ff(x)

        # Without dropout, outputs should be identical
        assert torch.allclose(output1, output2)

    def test_forward_gelu_activation(self):
        """Test that GELU activation is used."""
        ff = FeedForward(d_model=64, d_ff=256, dropout=0.1)

        # Check that the activation is GELU
        assert isinstance(ff.activation, torch.nn.GELU)

        # Also verify GELU behavior: negative inputs should produce non-zero outputs
        x = torch.full((2, 10, 64), -1.0)
        output = ff(x)
        # GELU(-1.0) ≈ -0.159, so output should have non-zero values
        assert (output != 0).any(), "GELU should produce non-zero outputs for negative inputs"

    def test_forward_identity_input(self):
        """Test forward pass with identity-like input."""
        ff = FeedForward(d_model=64, d_ff=256, dropout=0.1)
        x = torch.zeros(2, 10, 64)

        output = ff(x)
        assert output.shape == (2, 10, 64)
        # Output may not be zero due to bias terms
        assert not torch.isnan(output).any()

    def test_forward_large_input(self):
        """Test forward pass with large input values."""
        ff = FeedForward(d_model=64, d_ff=256, dropout=0.1)
        x = torch.randn(2, 10, 64) * 10  # Large values

        output = ff(x)
        assert output.shape == (2, 10, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_gradient_flow(self):
        """Test that gradients can flow through the network."""
        ff = FeedForward(d_model=64, d_ff=256, dropout=0.1)
        x = torch.randn(2, 10, 64, requires_grad=True)

        output = ff(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
