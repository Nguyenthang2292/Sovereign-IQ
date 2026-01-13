import pytest
import torch

from modules.lstm.core.focal_loss import FocalLoss

"""
Tests for FocalLoss module.
"""


class TestFocalLoss:
    """Test suite for FocalLoss class."""

    def test_init_with_float_alpha(self):
        """Test initialization with float alpha."""
        loss = FocalLoss(alpha=0.25, gamma=2.0)
        assert isinstance(loss.alpha, float)
        assert loss.alpha == 0.25
        assert loss.gamma == 2.0
        assert loss.reduction == "mean"

    def test_init_with_int_alpha(self):
        """Test initialization with int alpha."""
        loss = FocalLoss(alpha=1, gamma=2.0)
        assert isinstance(loss.alpha, float)
        assert loss.alpha == 1.0

    def test_init_with_tensor_alpha(self):
        """Test initialization with tensor alpha."""
        alpha_tensor = torch.tensor([0.25, 0.5, 0.25])
        loss = FocalLoss(alpha=alpha_tensor, gamma=2.0)
        assert isinstance(loss.alpha, torch.Tensor)
        assert torch.equal(loss.alpha, alpha_tensor)

    def test_init_invalid_alpha_type(self):
        """Test initialization with invalid alpha type."""
        with pytest.raises(TypeError, match="alpha must be float, int, or torch.Tensor"):
            FocalLoss(alpha="invalid")

    def test_init_invalid_tensor_dim(self):
        """Test initialization with invalid tensor dimension."""
        alpha_2d = torch.tensor([[0.25, 0.5], [0.25, 0.5]])
        with pytest.raises(ValueError, match="alpha tensor must be 1D"):
            FocalLoss(alpha=alpha_2d)

    def test_init_empty_tensor(self):
        """Test initialization with empty tensor."""
        alpha_empty = torch.tensor([])
        with pytest.raises(ValueError, match="alpha tensor must contain at least one element"):
            FocalLoss(alpha=alpha_empty)

    def test_forward_with_float_alpha(self):
        """Test forward pass with float alpha."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        inputs = torch.randn(4, 3)  # batch_size=4, num_classes=3
        targets = torch.randint(0, 3, (4,))

        loss = loss_fn(inputs, targets)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0

    def test_forward_with_tensor_alpha(self):
        """Test forward pass with tensor alpha."""
        alpha_tensor = torch.tensor([0.25, 0.5, 0.25])
        loss_fn = FocalLoss(alpha=alpha_tensor, gamma=2.0)
        inputs = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        loss = loss_fn(inputs, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_tensor_alpha_size_mismatch(self):
        """Test forward pass with tensor alpha size mismatch."""
        alpha_tensor = torch.tensor([0.25, 0.5])  # 2 classes
        loss_fn = FocalLoss(alpha=alpha_tensor, gamma=2.0)
        inputs = torch.randn(4, 3)  # 3 classes

        with pytest.raises(ValueError, match="alpha tensor must have same number of elements"):
            loss_fn(inputs, torch.randint(0, 3, (4,)))

    def test_forward_target_out_of_range(self):
        """Test forward pass with out-of-range target indices."""
        alpha_tensor = torch.tensor([0.25, 0.5, 0.25])
        loss_fn = FocalLoss(alpha=alpha_tensor, gamma=2.0)
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 3])  # 3 is out of range

        with pytest.raises(ValueError, match="Target indices must be in range"):
            loss_fn(inputs, targets)

    def test_forward_target_negative(self):
        """Test forward pass with negative target indices."""
        alpha_tensor = torch.tensor([0.25, 0.5, 0.25])
        loss_fn = FocalLoss(alpha=alpha_tensor, gamma=2.0)
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, -1])  # -1 is out of range

        with pytest.raises(ValueError, match="Target indices must be in range"):
            loss_fn(inputs, targets)

    def test_forward_reduction_mean(self):
        """Test forward pass with mean reduction."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        inputs = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        loss = loss_fn(inputs, targets)
        assert loss.dim() == 0

    def test_forward_reduction_sum(self):
        """Test forward pass with sum reduction."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction="sum")
        inputs = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        loss = loss_fn(inputs, targets)
        assert loss.dim() == 0
        # Sum should be larger than mean for same inputs
        loss_mean = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")(inputs, targets)
        assert loss.item() > loss_mean.item()

    def test_forward_reduction_none(self):
        """Test forward pass with none reduction."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        inputs = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        loss = loss_fn(inputs, targets)
        assert loss.shape == (4,)  # Per-sample losses

    def test_init_invalid_reduction(self):
        """Test initialization with invalid reduction mode."""
        with pytest.raises(ValueError, match="Invalid reduction mode"):
            FocalLoss(alpha=0.25, gamma=2.0, reduction="invalid")

    def test_forward_different_gamma(self):
        """Test forward pass with different gamma values."""
        inputs = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        loss_gamma_0 = FocalLoss(alpha=0.25, gamma=0.0)(inputs, targets)
        loss_gamma_2 = FocalLoss(alpha=0.25, gamma=2.0)(inputs, targets)

        # Higher gamma should give different loss values
        assert not torch.allclose(loss_gamma_0, loss_gamma_2)

    def test_forward_perfect_predictions(self):
        """Test forward pass with perfect predictions."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        # Create inputs where model is very confident and correct
        inputs = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        targets = torch.tensor([0, 1, 2])

        loss = loss_fn(inputs, targets)
        # Loss should be very small for perfect predictions
        assert loss.item() < 0.1

    def test_forward_wrong_predictions(self):
        """Test forward pass with wrong predictions."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        # Create inputs where model is very confident but wrong
        inputs = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        targets = torch.tensor([1, 2, 0])  # All wrong

        loss = loss_fn(inputs, targets)
        # Loss should be high for wrong predictions
        assert loss.item() > 1.0

    def test_forward_empty_batch(self):
        """Test forward pass with empty batch."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        inputs = torch.randn(0, 3)
        targets = torch.tensor([], dtype=torch.long)

        # Should handle empty batch gracefully
        loss = loss_fn(inputs, targets)
        assert loss.dim() == 0

    def test_forward_large_batch(self):
        """Test forward pass with large batch size."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        inputs = torch.randn(128, 3)
        targets = torch.randint(0, 3, (128,))

        loss = loss_fn(inputs, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_many_classes(self):
        """Test forward pass with many classes."""
        num_classes = 10
        alpha_tensor = torch.ones(num_classes) / num_classes
        loss_fn = FocalLoss(alpha=alpha_tensor, gamma=2.0)
        inputs = torch.randn(4, num_classes)
        targets = torch.randint(0, num_classes, (4,))

        loss = loss_fn(inputs, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0
