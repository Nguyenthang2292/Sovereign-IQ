import pytest
import torch

from modules.lstm.core.multi_head_attention import MultiHeadAttention

"""
Tests for MultiHeadAttention module.
"""


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)
        assert attn.d_model == 64
        assert attn.num_heads == 8
        assert attn.d_k == 8  # 64 // 8

    def test_init_invalid_num_heads(self):
        """Test initialization with invalid number of heads."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=64, num_heads=7)  # 64 % 7 != 0

    def test_forward_basic(self):
        """Test basic forward pass."""
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)
        batch_size = 2
        seq_len = 10

        query = torch.randn(batch_size, seq_len, 64)
        key = torch.randn(batch_size, seq_len, 64)
        value = torch.randn(batch_size, seq_len, 64)

        output = attn(query, key, value)
        assert output.shape == (batch_size, seq_len, 64)

    def test_forward_different_sequence_lengths(self):
        """Test forward pass with different query and key/value sequence lengths."""
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)

        query = torch.randn(2, 10, 64)
        key = torch.randn(2, 5, 64)
        value = torch.randn(2, 5, 64)

        output = attn(query, key, value)
        assert output.shape == (2, 10, 64)  # Output follows query length

    def test_forward_key_value_mismatch(self):
        """Test forward pass with key and value sequence length mismatch."""
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)

        query = torch.randn(2, 10, 64)
        key = torch.randn(2, 5, 64)
        value = torch.randn(2, 6, 64)  # Different from key

        with pytest.raises(AssertionError, match="Key and value must have the same sequence length"):
            attn(query, key, value)

    def test_forward_batch_size_mismatch_query_key(self):
        """Test forward pass with batch size mismatch between query and key."""
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)

        query = torch.randn(2, 10, 64)
        key = torch.randn(3, 10, 64)  # Different batch size
        value = torch.randn(2, 10, 64)

        # PyTorch will raise error during tensor operations with mismatched batch sizes
        with pytest.raises((RuntimeError, ValueError)):
            attn(query, key, value)

    def test_forward_batch_size_mismatch_query_value(self):
        """Test forward pass with batch size mismatch between query and value."""
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)

        query = torch.randn(2, 10, 64)
        key = torch.randn(2, 10, 64)
        value = torch.randn(4, 10, 64)  # Different batch size

        # PyTorch will raise error during tensor operations with mismatched batch sizes
        with pytest.raises((RuntimeError, ValueError)):
            attn(query, key, value)

    def test_forward_batch_size_mismatch_key_value(self):
        """Test forward pass with batch size mismatch between key and value."""
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)

        query = torch.randn(2, 10, 64)
        key = torch.randn(3, 10, 64)  # Different batch size
        value = torch.randn(5, 10, 64)  # Different batch size from both

        # PyTorch will raise error during tensor operations with mismatched batch sizes
        with pytest.raises((RuntimeError, ValueError)):
            attn(query, key, value)

    def test_forward_with_mask(self):
        """Test forward pass with attention mask."""
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)
        batch_size = 2
        seq_len = 10

        query = torch.randn(batch_size, seq_len, 64)
        key = torch.randn(batch_size, seq_len, 64)
        value = torch.randn(batch_size, seq_len, 64)

        # Create mask (1 for valid positions, 0 for masked)
        # Mask shape should be (batch_size, seq_len_q, seq_len_k) which will be broadcast
        # to (batch_size, num_heads, seq_len_q, seq_len_k) in attention
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, 5:] = 0  # Mask last 5 positions

        output = attn(query, key, value, mask=mask)
        assert output.shape == (batch_size, seq_len, 64)

    @pytest.mark.parametrize("d_model", [32, 64, 128, 256])
    def test_forward_different_d_models(self, d_model):
        """Test forward pass with different d_model values."""
        num_heads = 8
        # All d_model values in the list are divisible by num_heads
        attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        query = torch.randn(2, 10, d_model)
        key = torch.randn(2, 10, d_model)
        value = torch.randn(2, 10, d_model)

        output = attn(query, key, value)
        assert output.shape == (2, 10, d_model)

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    def test_forward_different_num_heads(self, num_heads):
        """Test forward pass with different number of heads."""
        d_model = 64
        # All num_heads values in the list divide d_model evenly
        attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        query = torch.randn(2, 10, d_model)
        key = torch.randn(2, 10, d_model)
        value = torch.randn(2, 10, d_model)

        output = attn(query, key, value)
        assert output.shape == (2, 10, d_model)

    def test_forward_dropout(self):
        """Test that dropout is applied during training."""
        torch.manual_seed(42)
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.5)
        attn.train()

        query = torch.randn(2, 10, 64)
        key = torch.randn(2, 10, 64)
        value = torch.randn(2, 10, 64)

        output1 = attn(query, key, value)
        output2 = attn(query, key, value)

        # With dropout, outputs should differ (with high probability)
        assert not torch.allclose(output1, output2, atol=1e-6)

    def test_forward_eval_mode(self):
        """Test forward pass in eval mode (no dropout)."""
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.5)
        attn.eval()

        query = torch.randn(2, 10, 64)
        key = torch.randn(2, 10, 64)
        value = torch.randn(2, 10, 64)

        output1 = attn(query, key, value)
        output2 = attn(query, key, value)

        # Without dropout, outputs should be identical
        assert torch.allclose(output1, output2)

    def test_attention_method(self):
        """Test the attention method directly."""
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)
        batch_size = 2
        seq_len_q = 10
        seq_len_k = 5
        d_k = 8

        query = torch.randn(batch_size, 8, seq_len_q, d_k)  # (batch, heads, seq_q, d_k)
        key = torch.randn(batch_size, 8, seq_len_k, d_k)
        value = torch.randn(batch_size, 8, seq_len_k, d_k)

        output = attn.attention(query, key, value, mask=None, dropout=attn.dropout)
        assert output.shape == (batch_size, 8, seq_len_q, d_k)

    def test_attention_with_mask(self):
        """Test attention method with mask."""
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)
        batch_size = 2
        seq_len_q = 10
        seq_len_k = 10
        d_k = 8

        query = torch.randn(batch_size, 8, seq_len_q, d_k)
        key = torch.randn(batch_size, 8, seq_len_k, d_k)
        value = torch.randn(batch_size, 8, seq_len_k, d_k)
        mask = torch.ones(batch_size, seq_len_q, seq_len_k)
        mask[:, :, 5:] = 0

        output = attn.attention(query, key, value, mask=mask, dropout=attn.dropout)
        assert output.shape == (batch_size, 8, seq_len_q, d_k)

    def test_forward_large_batch(self):
        """Test forward pass with large batch size."""
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)
        query = torch.randn(32, 10, 64)
        key = torch.randn(32, 10, 64)
        value = torch.randn(32, 10, 64)

        output = attn(query, key, value)
        assert output.shape == (32, 10, 64)

    def test_forward_long_sequence(self):
        """Test forward pass with long sequence."""
        attn = MultiHeadAttention(d_model=64, num_heads=8, dropout=0.1)
        query = torch.randn(2, 100, 64)
        key = torch.randn(2, 100, 64)
        value = torch.randn(2, 100, 64)

        output = attn(query, key, value)
        assert output.shape == (2, 100, 64)
