from unittest.mock import patch

import pytest
import torch

"""
Tests for batch size calculation utilities.
"""

from config.lstm import (
    CPU_BATCH_DIVISOR_CNN_LSTM,
    CPU_BATCH_DIVISOR_LSTM_ATTENTION,
    CPU_BATCH_SIZE,
    CPU_MIN_BATCH_CNN_LSTM,
    CPU_MIN_BATCH_LSTM_ATTENTION,
    FALLBACK_BATCH_DIVISOR_CNN_LSTM,
    FALLBACK_BATCH_DIVISOR_LSTM_ATTENTION,
    FALLBACK_MIN_BATCH_CNN_LSTM,
    FALLBACK_MIN_BATCH_LSTM_ATTENTION,
    GPU_BATCH_SIZE,
    MAX_BATCH_SIZE_CNN_LSTM,
    MAX_BATCH_SIZE_LSTM,
    MAX_BATCH_SIZE_LSTM_ATTENTION,
    MIN_BATCH_SIZE,
    TRAINING_MEMORY_MULTIPLIER,
)
from modules.lstm.utils.batch_size import _estimate_memory_per_sample, get_optimal_batch_size


class TestEstimateMemoryPerSample:
    """Test suite for _estimate_memory_per_sample function."""

    def test_estimate_memory_valid_inputs(self):
        """Test memory estimation with valid inputs."""
        input_size = 50
        sequence_length = 60
        model_type = "lstm"
        is_training = True

        memory = _estimate_memory_per_sample(input_size, sequence_length, model_type, is_training)

        # Base memory = (50 * 60 * 4) / 1024^2 = 0.0114 MB
        # With complexity_mult=1.0 and training_mult=3.0 = 0.0343 MB
        assert memory > 0
        assert isinstance(memory, float)

    def test_estimate_memory_inference_mode(self):
        """Test memory estimation in inference mode (lower than training)."""
        input_size = 50
        sequence_length = 60
        model_type = "lstm"

        training_memory = _estimate_memory_per_sample(input_size, sequence_length, model_type, True)
        inference_memory = _estimate_memory_per_sample(input_size, sequence_length, model_type, False)

        # Training should use more memory
        assert training_memory > inference_memory
        assert training_memory / inference_memory == pytest.approx(TRAINING_MEMORY_MULTIPLIER, rel=0.01)

    def test_estimate_memory_different_model_types(self):
        """Test memory estimation for different model types."""
        input_size = 50
        sequence_length = 60
        is_training = True

        lstm_memory = _estimate_memory_per_sample(input_size, sequence_length, "lstm", is_training)
        attention_memory = _estimate_memory_per_sample(input_size, sequence_length, "lstm_attention", is_training)
        cnn_lstm_memory = _estimate_memory_per_sample(input_size, sequence_length, "cnn_lstm", is_training)

        # CNN-LSTM should use most memory, then attention, then basic LSTM
        assert cnn_lstm_memory > attention_memory
        assert attention_memory > lstm_memory

    def test_estimate_memory_invalid_input_size(self):
        """Test memory estimation with invalid input_size."""
        memory = _estimate_memory_per_sample(0, 60, "lstm", True)
        assert memory == 0.0

        memory = _estimate_memory_per_sample(-1, 60, "lstm", True)
        assert memory == 0.0

    def test_estimate_memory_invalid_sequence_length(self):
        """Test memory estimation with invalid sequence_length."""
        memory = _estimate_memory_per_sample(50, 0, "lstm", True)
        assert memory == 0.0

        memory = _estimate_memory_per_sample(50, -1, "lstm", True)
        assert memory == 0.0

    def test_estimate_memory_unknown_model_type(self):
        """Test memory estimation with unknown model type (uses default multiplier)."""
        memory = _estimate_memory_per_sample(50, 60, "unknown_model", True)
        # Should use default complexity_mult=1.0
        assert memory > 0


class TestGetOptimalBatchSizeCPU:
    """Test suite for get_optimal_batch_size on CPU."""

    def test_cpu_basic_lstm(self):
        """Test batch size calculation for basic LSTM on CPU."""
        device = torch.device("cpu")
        batch_size = get_optimal_batch_size(
            device, input_size=50, sequence_length=60, model_type="lstm", is_training=True
        )

        assert batch_size >= MIN_BATCH_SIZE
        assert batch_size <= CPU_BATCH_SIZE
        assert isinstance(batch_size, int)

    def test_cpu_lstm_attention(self):
        """Test batch size calculation for LSTM-Attention on CPU."""
        device = torch.device("cpu")
        batch_size = get_optimal_batch_size(
            device, input_size=50, sequence_length=60, model_type="lstm_attention", is_training=True
        )

        assert batch_size >= CPU_MIN_BATCH_LSTM_ATTENTION
        max_expected = max(CPU_MIN_BATCH_LSTM_ATTENTION, CPU_BATCH_SIZE // CPU_BATCH_DIVISOR_LSTM_ATTENTION)
        assert batch_size <= max_expected

    def test_cpu_cnn_lstm(self):
        """Test batch size calculation for CNN-LSTM on CPU."""
        device = torch.device("cpu")
        batch_size = get_optimal_batch_size(
            device, input_size=50, sequence_length=60, model_type="cnn_lstm", is_training=True
        )

        assert batch_size >= CPU_MIN_BATCH_CNN_LSTM
        max_expected = max(CPU_MIN_BATCH_CNN_LSTM, CPU_BATCH_SIZE // CPU_BATCH_DIVISOR_CNN_LSTM)
        assert batch_size <= max_expected

    def test_cpu_inference_mode(self):
        """Test batch size calculation in inference mode (should allow larger batches)."""
        device = torch.device("cpu")
        training_batch = get_optimal_batch_size(
            device, input_size=50, sequence_length=60, model_type="lstm", is_training=True
        )
        inference_batch = get_optimal_batch_size(
            device, input_size=50, sequence_length=60, model_type="lstm", is_training=False
        )

        # Inference should allow larger or equal batch size
        assert inference_batch >= training_batch

    def test_cpu_large_input(self):
        """Test batch size calculation with large input dimensions."""
        device = torch.device("cpu")
        batch_size = get_optimal_batch_size(
            device, input_size=500, sequence_length=200, model_type="lstm", is_training=True
        )

        # Should still respect minimum batch size
        assert batch_size >= MIN_BATCH_SIZE

    def test_cpu_invalid_dimensions(self):
        """Test batch size calculation with invalid dimensions."""
        device = torch.device("cpu")
        # Should handle gracefully and return default
        batch_size = get_optimal_batch_size(
            device, input_size=0, sequence_length=60, model_type="lstm", is_training=True
        )
        assert batch_size >= MIN_BATCH_SIZE


class TestGetOptimalBatchSizeGPU:
    """Test suite for get_optimal_batch_size on GPU."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_basic_lstm(self):
        """Test batch size calculation for basic LSTM on GPU."""
        device = torch.device("cuda")
        batch_size = get_optimal_batch_size(
            device, input_size=50, sequence_length=60, model_type="lstm", is_training=True
        )

        assert batch_size >= MIN_BATCH_SIZE
        assert batch_size <= MAX_BATCH_SIZE_LSTM
        assert isinstance(batch_size, int)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_lstm_attention(self):
        """Test batch size calculation for LSTM-Attention on GPU."""
        device = torch.device("cuda")
        batch_size = get_optimal_batch_size(
            device, input_size=50, sequence_length=60, model_type="lstm_attention", is_training=True
        )

        assert batch_size >= MIN_BATCH_SIZE
        assert batch_size <= MAX_BATCH_SIZE_LSTM_ATTENTION

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_cnn_lstm(self):
        """Test batch size calculation for CNN-LSTM on GPU."""
        device = torch.device("cuda")
        batch_size = get_optimal_batch_size(
            device, input_size=50, sequence_length=60, model_type="cnn_lstm", is_training=True
        )

        assert batch_size >= MIN_BATCH_SIZE
        assert batch_size <= MAX_BATCH_SIZE_CNN_LSTM

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_inference_mode(self):
        """Test batch size calculation in inference mode on GPU."""
        device = torch.device("cuda")
        training_batch = get_optimal_batch_size(
            device, input_size=50, sequence_length=60, model_type="lstm", is_training=True
        )
        inference_batch = get_optimal_batch_size(
            device, input_size=50, sequence_length=60, model_type="lstm", is_training=False
        )

        # Inference should allow larger or equal batch size
        assert inference_batch >= training_batch

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_with_specific_device_index(self):
        """Test batch size calculation with specific CUDA device index."""
        device_count = torch.cuda.device_count()
        if device_count <= 1:
            pytest.skip("Multiple GPUs not available (requires 2+ GPUs to test cuda:1)")

        # Test with cuda:1 when multiple GPUs are available
        device = torch.device("cuda:1")
        batch_size = get_optimal_batch_size(
            device, input_size=50, sequence_length=60, model_type="lstm", is_training=True
        )
        assert batch_size >= MIN_BATCH_SIZE

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_low_memory(self):
        """Test batch size calculation with low GPU memory."""
        device = torch.device("cuda")
        # Mock low memory scenario
        with patch("torch.cuda.mem_get_info", return_value=(1024**3, 2 * 1024**3)):  # 1GB free, 2GB total
            batch_size = get_optimal_batch_size(
                device, input_size=50, sequence_length=60, model_type="lstm", is_training=True
            )
            assert batch_size >= MIN_BATCH_SIZE


class TestGetOptimalBatchSizeEdgeCases:
    """Test suite for edge cases in get_optimal_batch_size."""

    def test_unsupported_device_type(self):
        """Test batch size calculation with unsupported device type."""
        device = torch.device("mps")  # Metal Performance Shaders (Mac)
        batch_size = get_optimal_batch_size(
            device, input_size=50, sequence_length=60, model_type="lstm", is_training=True
        )
        # Should fallback to GPU_BATCH_SIZE
        assert batch_size == GPU_BATCH_SIZE

    def test_exception_handling(self):
        """Test exception handling in batch size calculation."""
        device = torch.device("cuda")

        # Mock an exception during GPU memory query
        with patch("torch.cuda.mem_get_info", side_effect=Exception("GPU error")):
            batch_size = get_optimal_batch_size(
                device, input_size=50, sequence_length=60, model_type="lstm", is_training=True
            )
            # Should fallback to model-specific default
            assert batch_size >= MIN_BATCH_SIZE

    def test_fallback_batch_sizes(self):
        """Test fallback batch sizes for different model types."""
        device = torch.device("cuda")

        with patch("torch.cuda.mem_get_info", side_effect=Exception("GPU error")):
            lstm_batch = get_optimal_batch_size(
                device, input_size=50, sequence_length=60, model_type="lstm", is_training=True
            )
            attention_batch = get_optimal_batch_size(
                device, input_size=50, sequence_length=60, model_type="lstm_attention", is_training=True
            )
            cnn_batch = get_optimal_batch_size(
                device, input_size=50, sequence_length=60, model_type="cnn_lstm", is_training=True
            )

            # Check fallback values
            assert lstm_batch == GPU_BATCH_SIZE
            assert attention_batch == max(
                FALLBACK_MIN_BATCH_LSTM_ATTENTION, GPU_BATCH_SIZE // FALLBACK_BATCH_DIVISOR_LSTM_ATTENTION
            )
            assert cnn_batch == max(FALLBACK_MIN_BATCH_CNN_LSTM, GPU_BATCH_SIZE // FALLBACK_BATCH_DIVISOR_CNN_LSTM)

    def test_unknown_model_type_fallback(self):
        """Test fallback for unknown model type."""
        device = torch.device("cuda")

        with patch("torch.cuda.mem_get_info", side_effect=Exception("GPU error")):
            batch_size = get_optimal_batch_size(
                device, input_size=50, sequence_length=60, model_type="unknown", is_training=True
            )
            # Should fallback to GPU_BATCH_SIZE
            assert batch_size == GPU_BATCH_SIZE

    def test_zero_memory_per_sample(self):
        """Test handling when memory per sample is zero."""
        device = torch.device("cpu")
        # With zero input_size, memory_per_sample will be 0
        batch_size = get_optimal_batch_size(
            device, input_size=0, sequence_length=60, model_type="lstm", is_training=True
        )
        # Should use default CPU_BATCH_SIZE
        assert batch_size == CPU_BATCH_SIZE

    def test_very_large_dimensions(self):
        """Test batch size calculation with very large dimensions."""
        device = torch.device("cpu")
        batch_size = get_optimal_batch_size(
            device, input_size=10000, sequence_length=1000, model_type="lstm", is_training=True
        )
        # Should still respect minimum batch size
        assert batch_size >= MIN_BATCH_SIZE
