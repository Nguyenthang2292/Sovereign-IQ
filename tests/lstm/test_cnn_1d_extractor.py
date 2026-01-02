"""
Tests for CNN1DExtractor module.
"""
import pytest
import torch

from modules.lstm.core.cnn_1d_extractor import CNN1DExtractor


class TestCNN1DExtractor:
    """Test suite for CNN1DExtractor class."""
    
    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        extractor = CNN1DExtractor(input_channels=10, cnn_features=64, dropout=0.3)
        assert extractor.input_channels == 10
        assert extractor.cnn_features == 64
        assert len(extractor.conv_layers) == 3  # Default kernel_sizes=[3, 5, 7]
    
    def test_init_custom_kernel_sizes(self):
        """Test initialization with custom kernel sizes."""
        extractor = CNN1DExtractor(
            input_channels=10, 
            cnn_features=64, 
            kernel_sizes=[3, 5],
            dropout=0.3
        )
        assert len(extractor.conv_layers) == 2
    
    def test_init_invalid_input_channels(self):
        """Test initialization with invalid input_channels."""
        with pytest.raises(ValueError, match="input_channels must be positive"):
            CNN1DExtractor(input_channels=0, cnn_features=64)
        
        with pytest.raises(ValueError, match="input_channels must be positive"):
            CNN1DExtractor(input_channels=-1, cnn_features=64)
    
    def test_init_invalid_cnn_features(self):
        """Test initialization with invalid cnn_features."""
        with pytest.raises(ValueError, match="cnn_features must be positive"):
            CNN1DExtractor(input_channels=10, cnn_features=0)
        
        with pytest.raises(ValueError, match="cnn_features must be positive"):
            CNN1DExtractor(input_channels=10, cnn_features=-1)
    
    def test_init_invalid_dropout(self):
        """Test initialization with invalid dropout."""
        with pytest.raises(ValueError, match="dropout must be in"):
            CNN1DExtractor(input_channels=10, cnn_features=64, dropout=1.5)
        
        with pytest.raises(ValueError, match="dropout must be in"):
            CNN1DExtractor(input_channels=10, cnn_features=64, dropout=-0.1)
    
    def test_init_empty_kernel_sizes(self):
        """Test initialization with empty kernel_sizes."""
        with pytest.raises(ValueError, match="kernel_sizes cannot be an empty list"):
            CNN1DExtractor(input_channels=10, cnn_features=64, kernel_sizes=[])
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        extractor = CNN1DExtractor(input_channels=10, cnn_features=64, dropout=0.3)
        x = torch.randn(2, 50, 10)  # batch_size=2, seq_len=50, features=10
        
        output = extractor(x)
        assert output.shape == (2, 50, 64)  # Output features = cnn_features
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_forward_different_batch_sizes(self, batch_size):
        """Test forward pass with different batch sizes."""
        extractor = CNN1DExtractor(input_channels=10, cnn_features=64, dropout=0.3)
        x = torch.randn(batch_size, 50, 10)
        output = extractor(x)
        assert output.shape == (batch_size, 50, 64)
    
    @pytest.mark.parametrize("seq_len", [10, 50, 100, 200])
    def test_forward_different_sequence_lengths(self, seq_len):
        """Test forward pass with different sequence lengths."""
        extractor = CNN1DExtractor(input_channels=10, cnn_features=64, dropout=0.3)
        x = torch.randn(2, seq_len, 10)
        output = extractor(x)
        assert output.shape == (2, seq_len, 64)
    
    @pytest.mark.parametrize("input_channels", [5, 10, 20, 50])
    def test_forward_different_input_channels(self, input_channels):
        """Test forward pass with different input channel sizes."""
        extractor = CNN1DExtractor(input_channels=input_channels, cnn_features=64, dropout=0.3)
        x = torch.randn(2, 50, input_channels)
        output = extractor(x)
        assert output.shape == (2, 50, 64)
    
    @pytest.mark.parametrize("cnn_features", [32, 64, 128, 256])
    def test_forward_different_cnn_features(self, cnn_features):
        """Test forward pass with different cnn_features."""
        extractor = CNN1DExtractor(input_channels=10, cnn_features=cnn_features, dropout=0.3)
        x = torch.randn(2, 50, 10)
        output = extractor(x)
        assert output.shape == (2, 50, cnn_features)
    
    def test_forward_dropout_training(self):
        """Test that dropout is applied during training."""
        extractor = CNN1DExtractor(input_channels=10, cnn_features=64, dropout=0.5)
        extractor.train()
        
        x = torch.randn(2, 50, 10)
        output1 = extractor(x)
        output2 = extractor(x)
        
        # With dropout, outputs should differ (with high probability)
        assert not torch.allclose(output1, output2, atol=1e-6)
    
    def test_forward_dropout_eval(self):
        """Test that dropout is not applied during eval."""
        extractor = CNN1DExtractor(input_channels=10, cnn_features=64, dropout=0.5)
        extractor.eval()
        
        x = torch.randn(2, 50, 10)
        output1 = extractor(x)
        output2 = extractor(x)
        
        # Without dropout, outputs should be identical
        assert torch.allclose(output1, output2)
    
    def test_forward_single_kernel_size(self):
        """Test forward pass with single kernel size."""
        extractor = CNN1DExtractor(
            input_channels=10, 
            cnn_features=64, 
            kernel_sizes=[3],
            dropout=0.3
        )
        x = torch.randn(2, 50, 10)
        output = extractor(x)
        assert output.shape == (2, 50, 64)
    
    def test_forward_many_kernel_sizes(self):
        """Test forward pass with many kernel sizes."""
        extractor = CNN1DExtractor(
            input_channels=10, 
            cnn_features=64, 
            kernel_sizes=[3, 5, 7, 9, 11],
            dropout=0.3
        )
        x = torch.randn(2, 50, 10)
        output = extractor(x)
        assert output.shape == (2, 50, 64)
    
    def test_forward_preserves_sequence_length(self):
        """Test that forward pass preserves sequence length."""
        extractor = CNN1DExtractor(input_channels=10, cnn_features=64, dropout=0.3)
        seq_len = 100
        x = torch.randn(2, seq_len, 10)
        output = extractor(x)
        assert output.shape[1] == seq_len
    
    def test_forward_gradient_flow(self):
        """Test that gradients can flow through the network."""
        extractor = CNN1DExtractor(input_channels=10, cnn_features=64, dropout=0.3)
        x = torch.randn(2, 50, 10, requires_grad=True)
        
        output = extractor(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_has_batch_norm_layers(self):
        """Test that CNN1DExtractor includes at least one batch normalization layer."""
        extractor = CNN1DExtractor(input_channels=10, cnn_features=64, dropout=0.3)
        batch_norm_found = any(
            isinstance(m, torch.nn.BatchNorm1d)
            for m in extractor.modules()
        )
        assert batch_norm_found, "No BatchNorm1d layer found in CNN1DExtractor"

