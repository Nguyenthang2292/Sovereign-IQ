"""
LSTM Model classes and factories.

This module contains model-related classes and factories for LSTM models:
- LSTMTrainer: Unified trainer class for all LSTM variants
- create_cnn_lstm_attention_model: Factory function for creating LSTM model architectures
- Model utilities: Functions for loading models and generating signals

Migration Notes:
----------------
This module has been refactored to remove several previously exported APIs:
- `CNNLSTMAttentionTrainer`
- `train_lstm_attention_model`
- `train_and_save_global_lstm_attention_model`

If you were using any of these, please migrate to using `LSTMTrainer` and the new unified training interface.
Check the documentation or MIGRATION.md for details on updating your import and usage patterns.
"""

from modules.lstm.models.model_factory import create_cnn_lstm_attention_model
from modules.lstm.models.model_utils import (
    get_latest_lstm_attention_signal,
    get_latest_signal,
    # Backward compatibility aliases
    load_lstm_attention_model,
    load_lstm_model,
    load_model_and_scaler,
)
from modules.lstm.models.unified_trainer import LSTMTrainer


def _removed_api(name: str):
    """
    Factory for raising informative errors for removed APIs.

    Args:
        name: Name of the removed API

    Returns:
        Callable that raises ImportError with migration guidance
    """

    def _raise(*args, **kwargs):
        raise ImportError(
            f"'{name}' has been removed. Please migrate to 'LSTMTrainer'. See MIGRATION_GUIDE.md for details."
        )

    return _raise


# Deprecated stubs for migration guidance
# These allow imports to succeed but raise informative errors when called
CNNLSTMAttentionTrainer = _removed_api("CNNLSTMAttentionTrainer")
train_lstm_attention_model = _removed_api("train_lstm_attention_model")
train_and_save_global_lstm_attention_model = _removed_api("train_and_save_global_lstm_attention_model")

__all__ = [
    "LSTMTrainer",
    "create_cnn_lstm_attention_model",
    "load_lstm_model",
    "load_model_and_scaler",
    "get_latest_signal",
    # Backward compatibility aliases
    "load_lstm_attention_model",
    "get_latest_lstm_attention_signal",
    # Deprecated stubs (for migration guidance)
    "CNNLSTMAttentionTrainer",
    "train_lstm_attention_model",
    "train_and_save_global_lstm_attention_model",
]
