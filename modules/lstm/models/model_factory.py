"""
Model factory for creating CNN-LSTM-Attention models.
"""

from typing import Literal, Union

from config.lstm import WINDOW_SIZE_LSTM
from modules.common.ui.logging import log_model
from modules.lstm.models.lstm_models import CNNLSTMAttentionModel, LSTMAttentionModel, LSTMModel

# Allowed parameter sets for each model type
ATTENTION_ALLOWED_KEYS = {
    "num_heads",
    "dropout",
    "hidden_size",
    "num_layers",
    "num_classes",
    "use_positional_encoding",
    "output_mode",
}
ENCODER_ALLOWED_KEYS = {"dropout", "hidden_size", "num_layers", "num_classes", "output_mode"}
CNN_ALLOWED_KEYS = {
    "cnn_features",
    "lstm_hidden",
    "num_layers",
    "num_classes",
    "num_heads",
    "dropout",
    "use_positional_encoding",
    "output_mode",
}


def create_cnn_lstm_attention_model(
    input_size: int,
    use_attention: bool = True,
    use_cnn: bool = False,
    look_back: int = WINDOW_SIZE_LSTM,
    output_mode: Literal["classification", "regression"] = "classification",
    **kwargs,
) -> Union[LSTMModel, LSTMAttentionModel, CNNLSTMAttentionModel]:
    """
    Create CNN-LSTM-Attention model based on configuration.

    Args:
        input_size: Number of input features
        use_attention: Whether to use attention mechanism
        use_cnn: Whether to use CNN layers
        look_back: Sequence length for time series
        output_mode: 'classification' or 'regression'
        **kwargs: Additional model parameters

    Returns:
        Neural network model configured according to parameters
    """
    if not isinstance(input_size, int) or input_size <= 0:
        raise ValueError(f"input_size must be positive integer, got {input_size}")
    if not isinstance(look_back, int) or look_back <= 0:
        raise ValueError(f"look_back must be positive integer, got {look_back}")
    # Runtime validation: Type hints enforce at static type-checking time,
    # but this public API may be called from dynamic code or without type checking.
    # This defensive check ensures runtime safety regardless of type checking.
    if output_mode not in ["classification", "regression"]:
        raise ValueError(f"output_mode must be 'classification' or 'regression', got {output_mode}")

    # Set default num_classes based on output_mode if not provided in kwargs
    if "num_classes" not in kwargs:
        kwargs["num_classes"] = 1 if output_mode == "regression" else 3

    if use_cnn:
        log_model(f"Creating CNN-LSTM-Attention model with {output_mode} mode")

        # Validate CNN model kwargs
        unknown_keys = set(kwargs.keys()) - CNN_ALLOWED_KEYS
        if unknown_keys:
            raise ValueError(
                f"Unknown parameters for CNN-LSTM-Attention model: {sorted(unknown_keys)}. "
                f"Allowed parameters: {sorted(CNN_ALLOWED_KEYS)}"
            )

        model = CNNLSTMAttentionModel(
            input_size=input_size,
            look_back=look_back,
            output_mode=output_mode,  # Already validated above
            use_attention=use_attention,
            **kwargs,
        )
        log_model(f"Created CNN-LSTM-Attention model with {output_mode} mode")
        return model

    if use_attention:
        log_model(f"Creating LSTM model with Multi-Head Attention (output_mode: {output_mode})")

        # Validate attention model kwargs
        unknown_keys = set(kwargs.keys()) - ATTENTION_ALLOWED_KEYS
        if unknown_keys:
            raise ValueError(
                f"Unknown parameters for LSTM-Attention model: {sorted(unknown_keys)}. "
                f"Allowed parameters: {sorted(ATTENTION_ALLOWED_KEYS)}"
            )

        return LSTMAttentionModel(input_size=input_size, output_mode=output_mode, **kwargs)

    log_model(f"Creating standard LSTM model (output_mode: {output_mode})")

    # Validate encoder (LSTM) model kwargs
    unknown_keys = set(kwargs.keys()) - ENCODER_ALLOWED_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown parameters for LSTM encoder model: {sorted(unknown_keys)}. "
            f"Allowed parameters: {sorted(ENCODER_ALLOWED_KEYS)}"
        )

    return LSTMModel(input_size=input_size, output_mode=output_mode, **kwargs)
