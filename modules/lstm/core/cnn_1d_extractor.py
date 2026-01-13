"""
cnn_1d_extractor.py

This module implements a 1D Convolutional Neural Network (CNN) feature extractor for time series
or sequence data.
The main class, CNN1DExtractor, provides multi-scale 1D convolutional layers for extracting
hierarchical temporal features.
It supports configurable input channels, variable kernel sizes for multi-scale receptive fields,
and dropout for regularization.

Typical usage example:

    extractor = CNN1DExtractor(input_channels=16, cnn_features=64, kernel_sizes=[3, 5, 7], dropout=0.3)
    output = extractor(input_tensor)

"""

from typing import List

import torch
import torch.nn as nn

from modules.common.ui.logging import log_model


class CNN1DExtractor(nn.Module):
    """
    1D CNN feature extractor for time series data.

    Args:
        input_channels: Number of input channels
        cnn_features: Number of CNN features to extract (default: 64)
        kernel_sizes: List of kernel sizes for multi-scale convolution (default: [3, 5, 7])
        dropout: Dropout probability (default: 0.3)
    """

    def __init__(
        self, input_channels: int, cnn_features: int = 64, kernel_sizes: List[int] = None, dropout: float = 0.3
    ) -> None:
        super().__init__()

        if not isinstance(input_channels, int) or input_channels <= 0:
            raise ValueError("input_channels must be positive, got {0}".format(input_channels))

        if not isinstance(cnn_features, int) or cnn_features <= 0:
            raise ValueError("cnn_features must be positive, got {0}".format(cnn_features))

        if not (0 <= dropout <= 1):
            raise ValueError("dropout must be in [0, 1], got {0}".format(dropout))

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        if not kernel_sizes:
            raise ValueError("kernel_sizes cannot be an empty list")
        self.input_channels = input_channels
        self.cnn_features = cnn_features

        num_scales = len(kernel_sizes)
        base = cnn_features // num_scales
        remainder = cnn_features % num_scales
        out_channels = []
        for i in range(num_scales):
            ch = base + (1 if i < remainder else 0)
            out_channels.append(ch)

        self.conv_layers = nn.ModuleList()
        for idx, kernel_size in enumerate(kernel_sizes):
            conv_block = nn.Sequential(
                nn.Conv1d(input_channels, out_channels[idx], kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels[idx]),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.conv_layers.append(conv_block)

        self.conv_refine = nn.Sequential(
            nn.Conv1d(cnn_features, cnn_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_features, cnn_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        log_model(
            "CNN1D Extractor initialized with {0} input channels, {1} features".format(input_channels, cnn_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN extractor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, features)

        Returns:
            Extracted features of shape (batch_size, seq_len, cnn_features)
        """
        x = x.transpose(1, 2)

        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x)
            conv_outputs.append(conv_out)

        x = torch.cat(conv_outputs, dim=1)
        x = self.conv_refine(x)
        x = x.transpose(1, 2)

        return x
