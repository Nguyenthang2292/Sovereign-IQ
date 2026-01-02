from typing import Literal, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance in classification tasks.

    Focal Loss addresses class imbalance by down-weighting easy examples and focusing
    on hard negatives. The loss is computed as: FL(pt) = -α(1-pt)^γ * log(pt)

    Args:
        alpha: Weighting factor(s) for classes. Can be either:
            - float: Scalar value applied uniformly to all classes (default: 0.25).
            - torch.Tensor: 1D tensor of shape (C,) for per-class weights, where C is the number of classes.
                The length of the alpha tensor MUST match the number of classes in your classification task.
               A ValueError will be raised if targets contain indices outside [0, num_classes-1].
        gamma: Focusing parameter to down-weight easy examples (default: 2.0).
        reduction: Specifies the reduction to apply to the output: 'mean', 'sum', or 'none' (default: 'mean').
    """
    
    def __init__(
        self, 
        alpha: Union[float, int, torch.Tensor] = 0.25,
        gamma: float = 2.0, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ) -> None:
        super().__init__()
        
        if isinstance(alpha, (float, int)):
            self.alpha = float(alpha)
        elif isinstance(alpha, torch.Tensor):
            if alpha.dim() != 1:
                raise ValueError(f"alpha tensor must be 1D, got shape {alpha.shape}")
            if alpha.shape[0] == 0:
                raise ValueError("alpha tensor must contain at least one element")
            # Note: Validation against number of classes is done in forward() method
            # where we have access to the actual input data
            self.register_buffer('alpha', alpha)
        else:
            raise TypeError(f"alpha must be float, int, or torch.Tensor, got {type(alpha)}")        
        
        self.gamma = gamma
        
        # Validate reduction mode
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: '{reduction}'. Must be one of 'mean', 'sum', or 'none'")
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits of shape (N, C) where N is batch size, C is number of classes
            targets: Ground truth labels of shape (N,) with class indices
            
        Returns:
            Computed focal loss tensor
        """
        num_classes = inputs.shape[1]
        
        # Validate alpha tensor size matches number of classes
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.shape[0] != num_classes:
                raise ValueError(
                    f"alpha tensor must have same number of elements as number of classes, "
                    f"got {self.alpha.shape[0]} and {num_classes}"
                )
            # Check for out-of-range target indices
            if targets.numel() > 0 and (targets.max() >= num_classes or targets.min() < 0):
                raise ValueError(
                    f"Target indices must be in range [0, {num_classes-1}], "
                    f"got range [{targets.min().item()}, {targets.max().item()}]"
                )        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply per-class alpha weights if alpha is a tensor, otherwise use scalar
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

