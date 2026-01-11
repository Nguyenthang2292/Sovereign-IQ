
from modules.common.ui.logging import log_system

"""
CNN feature mixin for LSTM trainers.
"""


# CNN models require more memory, so we reduce batch size
CNN_MIN_BATCH_SIZE = 4
CNN_BATCH_SIZE_REDUCTION_FACTOR = 8
CNN_MIN_ADDITIONAL_SAMPLES = 100  # Minimum samples needed beyond look_back for CNN training


class CNNFeatureMixin:
    """
    Mixin class providing CNN-specific functionality for LSTM trainers.
    """

    def _adjust_batch_size_for_cnn(self, batch_size: int) -> int:
        """
        Adjust batch size for CNN models (which require more memory).

        Args:
            batch_size: Original batch size

        Returns:
            Adjusted batch size for CNN models
        """
        adjusted = max(CNN_MIN_BATCH_SIZE, batch_size // CNN_BATCH_SIZE_REDUCTION_FACTOR)
        log_system(f"CNN optimized batch size: {adjusted}")
        return adjusted

    def _validate_cnn_data_requirements(self, data_length: int, look_back: int) -> None:
        """
        Validate that data meets CNN requirements.

        Args:
            data_length: Length of input data
            look_back: Sequence length

        Raises:
            ValueError: If data is insufficient for CNN models
        """
        min_required = look_back + CNN_MIN_ADDITIONAL_SAMPLES
        if data_length < min_required:
            raise ValueError(f"Insufficient data for CNN model: {data_length} rows, need at least {min_required}")
