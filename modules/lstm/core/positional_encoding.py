"""
This module provides the PositionalEncoding class for injecting positional information into sequence data.

The sinusoidal positional encoding implemented here is commonly used in sequence models such as
Transformers, as described in "Attention Is All You Need" (Vaswani et al., 2017).
It allows the model to make use of the order of sequence elements.

Classes:
    PositionalEncoding (nn.Module): Adds sinusoidal positional encodings to the input tensor.

Typical Usage:
    pe = PositionalEncoding(d_model=512)
    x = torch.zeros(batch_size, seq_len, 512)
    x_pe = pe(x)

Args:
    d_model (int): Dimension of the embedding/feature space.
    max_seq_length (int, optional): Maximum length to precompute positional encodings for. Default: 5000.
"""

import math
import warnings

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence data using sinusoidal encoding.

    This module implements the standard Transformer positional encoding as described
    in "Attention Is All You Need" (Vaswani et al., 2017).

    Performance Note:
        When input sequences are longer than max_seq_length, additional positional
        encodings are computed on-the-fly and concatenated. This avoids thread-safety
        issues in distributed training (DDP) and multi-worker DataLoaders, but incurs
        a performance cost due to repeated tensor allocations.

        If your model frequently processes sequences longer than max_seq_length, consider
        increasing max_seq_length during initialization to avoid repeated allocations.
        This trades memory for performance.

    Args:
        d_model: Dimension of the model embeddings
        max_seq_length: Maximum sequence length to pre-compute encodings for (default: 5000)
    """

    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive, got {0}".format(d_model))
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive, got {0}".format(max_seq_length))

        pe = self._create_positional_encoding(start_pos=0, length=max_seq_length, d_model=d_model, device=None)

        # Store as buffer without extra dimensions
        self.register_buffer("pe", pe.unsqueeze(0))

        # Flag to track if length warning has been emitted (to avoid log spam)
        self._length_warning_emitted = False

    def _create_positional_encoding(self, start_pos, length, d_model, device=None):
        """
        Helper method to create positional encoding tensor.

        Args:
            start_pos: Starting position index
            length: Number of positions to create
            d_model: Dimension of the model
            device: Device to create tensor on (None defaults to CPU)

        Returns:
            Positional encoding tensor of shape (length, d_model)
        """
        # PyTorch accepts device=None and defaults to CPU, eliminating the need for branching
        pe = torch.zeros(length, d_model, device=device)
        position = torch.arange(start_pos, start_pos + length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("Input must be 3D tensor [batch, seq_len, d_model], got shape {0}".format(x.shape))
        if x.size(2) != self.pe.size(2):
            raise ValueError("Input d_model {0} doesn't match expected d_model {1}".format(x.size(2), self.pe.size(2)))

        seq_len = x.size(1)
        device = x.device
        stored_seq_len = self.pe.size(1)
        d_model = self.pe.size(2)

        # Optimize device transfer: only transfer if buffer is on different device
        # Registered buffers should automatically move with model.to(device), but we
        # check here to avoid unnecessary transfers in case of device mismatches
        pe_on_device = self.pe if self.pe.device == device else self.pe.to(device)

        if seq_len > stored_seq_len:
            # Compute extra required positional encodings on-the-fly
            # Performance note: This allocates new tensors on every forward pass when
            # seq_len > stored_seq_len. To avoid this, increase max_seq_length during
            # initialization if your model frequently processes longer sequences.
            extra_len = seq_len - stored_seq_len
            extra_pe = self._create_positional_encoding(
                start_pos=stored_seq_len, length=extra_len, d_model=d_model, device=device
            )
            # Concatenate with existing encoding for this device.
            # Do NOT update buffer: modifying registered buffers is not thread-safe and
            # can cause race conditions in distributed training (DDP) and multi-worker DataLoaders.
            full_pe = torch.cat([pe_on_device.squeeze(0), extra_pe], dim=0).unsqueeze(0)

            # Warn about performance impact when sequence length exceeds max_seq_length
            # Only warn once per instance to avoid log spam during training with variable-length sequences
            if not self._length_warning_emitted:
                warnings.warn(
                    f"PositionalEncoding: input sequence length ({seq_len}) exceeds "
                    f"max_seq_length ({stored_seq_len}). Computing encodings on-the-fly, "
                    f"which may impact performance. Consider increasing max_seq_length "
                    f"during initialization if this occurs frequently.",
                    UserWarning,
                    stacklevel=2,
                )
                self._length_warning_emitted = True
        else:
            full_pe = pe_on_device[:, :seq_len, :]

        return x + full_pe
