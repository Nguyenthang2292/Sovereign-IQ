
import torch

from config.lstm import (
from config.lstm import (


    COMPLEXITY_MULTIPLIER,
    CPU_BATCH_DIVISOR_CNN_LSTM,
    CPU_BATCH_DIVISOR_LSTM_ATTENTION,
    CPU_BATCH_SIZE,
    CPU_MAX_USABLE_MEMORY_MB,
    CPU_MIN_BATCH_CNN_LSTM,
    CPU_MIN_BATCH_LSTM_ATTENTION,
    FALLBACK_BATCH_DIVISOR_CNN_LSTM,
    FALLBACK_BATCH_DIVISOR_LSTM_ATTENTION,
    FALLBACK_MIN_BATCH_CNN_LSTM,
    FALLBACK_MIN_BATCH_LSTM_ATTENTION,
    GPU_BATCH_SIZE,
    GPU_MEMORY_USAGE_RATIO,
    INFERENCE_MEMORY_MULTIPLIER,
    MAX_BATCH_SIZE_CNN_LSTM,
    MAX_BATCH_SIZE_LSTM,
    MAX_BATCH_SIZE_LSTM_ATTENTION,
    MIN_BATCH_SIZE,
    TRAINING_MEMORY_MULTIPLIER,
)
from modules.common.ui.logging import log_debug, log_model, log_warn


def _estimate_memory_per_sample(input_size, sequence_length, model_type, is_training):
    """
    Estimate memory per sample in MB.

    NOTE: This only accounts for input tensor size. Actual memory usage includes:
    - Model parameters, gradients, optimizer states (training only)
    - LSTM hidden/cell states, intermediate activations
    - These are approximated via complexity_multiplier and training_multiplier

    Args:
        input_size: Number of input features
        sequence_length: LSTM sequence length
        model_type: Type of model ('lstm', 'lstm_attention', 'cnn_lstm')
        is_training: Whether this is for training

    Returns:
        float: Estimated memory per sample in MB
    """
    # Validate inputs
    if input_size <= 0 or sequence_length <= 0:
        log_warn(f"Invalid input dimensions: input_size={input_size}, sequence_length={sequence_length}")
        return 0.0

    # Base memory calculation (input tensor size, float32 = 4 bytes)
    base_memory_mb = (input_size * sequence_length * 4) / 1024**2

    # Apply model complexity multiplier
    complexity_mult = COMPLEXITY_MULTIPLIER.get(model_type, 1.0)
    memory_per_sample_mb = base_memory_mb * complexity_mult

    # Apply training multiplier if in training mode
    training_mult = TRAINING_MEMORY_MULTIPLIER if is_training else INFERENCE_MEMORY_MULTIPLIER
    memory_per_sample_mb *= training_mult

    return memory_per_sample_mb


def get_optimal_batch_size(device, input_size, sequence_length, model_type="lstm", is_training=True):
    """
    Dynamically determine optimal batch size based on GPU memory and model complexity.

    **Important Limitations:**
    This function provides a rough estimation based on input tensor size and model complexity
    multipliers. It does NOT account for:
    - Model parameters (weights, biases)
    - Gradients during training
    - Optimizer states (e.g., Adam maintains 2x parameters)
    - LSTM hidden/cell states
    - Intermediate activations in forward/backward passes

    The estimation may significantly underestimate actual memory requirements, especially for
    larger models or during training. For accurate batch sizing, consider profiling actual
    memory usage with representative models.

    Args:
        device: torch device
        input_size: Number of input features
        sequence_length: LSTM sequence length
        model_type: Type of model ('lstm', 'lstm_attention', 'cnn_lstm')
        is_training: Whether this is for training (default: True). Training requires
            approximately 3x more memory than inference due to gradients and optimizer states.

    Returns:
        int: Optimal batch size (estimated, may need adjustment based on actual usage)
    """
    try:
        if device.type == "cpu":
            # Estimate memory per sample using shared helper function
            memory_per_sample_mb = _estimate_memory_per_sample(input_size, sequence_length, model_type, is_training)

            # Assume an arbitrary "safe" cap for user machines (2GB for batch, not all RAM)
            max_usable_memory_mb = CPU_MAX_USABLE_MEMORY_MB
            min_batch = MIN_BATCH_SIZE

            # Try to select batch size so memory_per_sample_mb * batch <= max_usable_memory_mb
            if memory_per_sample_mb > 0:
                calc_batch = int(max_usable_memory_mb / memory_per_sample_mb)
            else:
                calc_batch = CPU_BATCH_SIZE

            if model_type == "cnn_lstm":
                max_batch = max(CPU_MIN_BATCH_CNN_LSTM, CPU_BATCH_SIZE // CPU_BATCH_DIVISOR_CNN_LSTM)
            elif model_type == "lstm_attention":
                max_batch = max(CPU_MIN_BATCH_LSTM_ATTENTION, CPU_BATCH_SIZE // CPU_BATCH_DIVISOR_LSTM_ATTENTION)
            else:
                max_batch = CPU_BATCH_SIZE

            optimal_batch = max(min_batch, min(calc_batch, max_batch))
            mode_str = "training" if is_training else "inference"
            log_model(
                f"CPU: Model: {model_type}, Mode: {mode_str}, Estimated per-sample MB: {memory_per_sample_mb:.4f}, Optimal batch: {optimal_batch}"
            )
            log_warn(
                "Memory estimation is approximate and may underestimate actual requirements. "
                "Monitor actual memory usage and adjust batch size if OOM errors occur."
            )
            return optimal_batch

        elif device.type == "cuda":
            # Get GPU memory info
            device_index = device.index if device.index is not None else 0
            free_memory, total_memory = torch.cuda.mem_get_info(device_index)
            gpu_memory_gb = free_memory / 1024**3  # Use available memory instead of total

            # Estimate memory per sample using shared helper function
            memory_per_sample_mb = _estimate_memory_per_sample(input_size, sequence_length, model_type, is_training)

            # Calculate optimal batch size based on available memory and actual memory requirements
            usable_memory_gb = gpu_memory_gb * GPU_MEMORY_USAGE_RATIO
            min_batch = MIN_BATCH_SIZE

            # Compute optimal batch based on available memory and per-sample requirements
            # Add zero-check to prevent division by zero
            if memory_per_sample_mb > 0:
                calc_batch = int((usable_memory_gb * 1024) / memory_per_sample_mb)
            else:
                log_warn(
                    f"Memory per sample is zero (input_size={input_size}, sequence_length={sequence_length}), using default GPU_BATCH_SIZE"
                )
                calc_batch = GPU_BATCH_SIZE

            # Apply reasonable bounds based on model type
            if model_type == "cnn_lstm":
                max_batch = MAX_BATCH_SIZE_CNN_LSTM
            elif model_type == "lstm_attention":
                max_batch = MAX_BATCH_SIZE_LSTM_ATTENTION
            else:
                max_batch = MAX_BATCH_SIZE_LSTM

            optimal_batch = max(min_batch, min(calc_batch, max_batch))

            # Safety check: ensure batch size doesn't exceed memory limits
            estimated_memory_gb = (optimal_batch * memory_per_sample_mb) / 1024
            if estimated_memory_gb > usable_memory_gb:
                # Reduce optimal_batch so it's within usable memory (min allowed: MIN_BATCH_SIZE)
                max_possible_batch = (
                    int((usable_memory_gb * 1024) // memory_per_sample_mb)
                    if memory_per_sample_mb > 0
                    else MIN_BATCH_SIZE
                )
                optimal_batch = max(MIN_BATCH_SIZE, min(optimal_batch, max_possible_batch))
                estimated_memory_gb = (optimal_batch * memory_per_sample_mb) / 1024
                if estimated_memory_gb > usable_memory_gb:
                    log_warn(
                        f"Even the reduced batch size ({optimal_batch}) may exceed available memory ({usable_memory_gb:.2f}GB)"
                    )

            mode_str = "training" if is_training else "inference"
            log_model(
                "GPU Memory: {0:.1f}GB, Model: {1}, Mode: {2}, Optimal batch size: {3}".format(
                    gpu_memory_gb, model_type, mode_str, optimal_batch
                )
            )
            log_debug(
                "Estimated memory usage: {0:.2f}GB ({1:.1f}%)".format(
                    estimated_memory_gb, (estimated_memory_gb / gpu_memory_gb) * 100
                )
            )
            log_warn(
                "Memory estimation is approximate and may underestimate actual requirements. "
                "Monitor actual memory usage and adjust batch size if OOM errors occur."
            )

            return optimal_batch
        else:
            # Fallback for unsupported device types
            log_warn(f"Unsupported device type: {device.type}, using default batch size")
            return GPU_BATCH_SIZE

    except Exception as e:
        log_warn("Could not determine optimal batch size: {0}".format(e))

        # Fallback batch sizes based on model type
        fallback_sizes = {
            "cnn_lstm": max(FALLBACK_MIN_BATCH_CNN_LSTM, GPU_BATCH_SIZE // FALLBACK_BATCH_DIVISOR_CNN_LSTM),
            "lstm_attention": max(
                FALLBACK_MIN_BATCH_LSTM_ATTENTION, GPU_BATCH_SIZE // FALLBACK_BATCH_DIVISOR_LSTM_ATTENTION
            ),
            "lstm": GPU_BATCH_SIZE,
        }
        return fallback_sizes.get(model_type, GPU_BATCH_SIZE)
