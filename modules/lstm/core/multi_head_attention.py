"""
multi_head_attention.py

This module implements the Multi-Head Attention mechanism, designed to work with the outputs
of LSTM networks. The main purpose is to allow the model to jointly attend to information
from different representation subspaces at different positions, as popularized by the Transformer
architecture.

Classes:
    MultiHeadAttention:
        PyTorch nn.Module implementing multi-head scaled dot-product attention for use with LSTM outputs.
        Includes learned linear projections for queries, keys, values, and outputs, as well as dropout and
        masking support.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Attention Mechanism Components
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism for LSTM outputs
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k**0.5

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)

        assert seq_len_k == seq_len_v, (
            f"Key and value must have the same sequence length, got {seq_len_k} and {seq_len_v}"
        )

        # Linear transformations and split into num_heads
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        attention = self.attention(Q, K, V, mask, self.dropout)

        # Concatenate heads and put through final linear layer
        # Output shape will be (batch_size, num_heads, seq_len_q, d_k) after attention
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.W_o(attention)

        return output

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[nn.Dropout] = None,
    ) -> torch.Tensor:
        """
        Scaled dot-product attention

        Args:
            mask: Optional mask tensor where 0 indicates positions to mask out.
                  Expected shape: (batch_size, seq_len_q, seq_len_k) or
                                 (batch_size, 1, seq_len_q, seq_len_k)
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        if mask is not None:
            # Handle mask shape: mask can be (batch_size, seq_len_q, seq_len_k)
            # or (batch_size, 1, seq_len_q, seq_len_k), but scores is (batch_size, num_heads, seq_len_q, seq_len_k)
            # We need to add a dimension for num_heads if mask doesn't have it
            if mask.dim() == 3:
                # (batch_size, seq_len_q, seq_len_k) -> (batch_size, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)

        if dropout is not None:
            attention_weights = dropout(attention_weights)

        return torch.matmul(attention_weights, value)
