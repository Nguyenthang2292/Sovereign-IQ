
import torch
import torch.nn as nn



class FeedForward(nn.Module):
    """
    Generic feed-forward/position-wise feed-forward layer usable across architectures.

    Applies two linear transformations with ReLU activation and dropout:
    FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        d_model: Input and output dimension
        d_ff: Hidden dimension of the feed-forward layer
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
