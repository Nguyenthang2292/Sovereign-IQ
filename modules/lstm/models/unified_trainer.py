"""Unified LSTM Trainer supporting all model variants.

This trainer can handle:
- LSTM (use_cnn=False, use_attention=False)
- LSTM-Attention (use_cnn=False, use_attention=True)
- CNN-LSTM (use_cnn=True, use_attention=False)
- CNN-LSTM-Attention (use_cnn=True, use_attention=True)
"""

import time
import traceback
from typing import Optional, Tuple

import pandas as pd
import torch.nn as nn

from config.lstm import DEFAULT_EPOCHS, GPU_MODEL_CONFIG, WINDOW_SIZE_LSTM
from modules.common.ui.logging import log_error, log_model, log_success
from modules.lstm.models.model_factory import create_cnn_lstm_attention_model
from modules.lstm.models.trainer.attention_mixin import AttentionFeatureMixin
from modules.lstm.models.trainer.base_trainer import BaseLSTMTrainer
from modules.lstm.models.trainer.cnn_mixin import CNNFeatureMixin


class LSTMTrainer(BaseLSTMTrainer, CNNFeatureMixin, AttentionFeatureMixin):
    """
    Unified trainer for all LSTM model variants.

    Inherits common training logic from BaseLSTMTrainer and specialized
    functionality from CNN and Attention mixins.
    """

    def __init__(
        self,
        use_cnn: bool = False,
        use_attention: bool = False,
        look_back: int = WINDOW_SIZE_LSTM,
        output_mode: str = "classification",
        attention_heads: int = GPU_MODEL_CONFIG["nhead"],
        use_early_stopping: bool = True,
        early_stopping_patience: int = 10,
        cnn_features: int = 64,
        lstm_hidden: int = 32,
        dropout: float = 0.2,
        use_kalman_filter: bool = False,
        kalman_params: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize unified trainer.

        Args:
            use_cnn: Enable CNN feature extraction
            use_attention: Enable Multi-Head Attention
            look_back: Sequence length
            output_mode: 'classification' or 'regression'
            attention_heads: Number of attention heads
            use_early_stopping: Enable early stopping
            early_stopping_patience: Patience for early stopping
            cnn_features: Number of CNN filters
            lstm_hidden: LSTM hidden state size
            dropout: Dropout rate
            use_kalman_filter: Enable Kalman Filter preprocessing
            kalman_params: Optional Kalman Filter parameters
            **kwargs: Additional parameters passed to BaseLSTMTrainer
        """
        # Initialize base class
        super().__init__(
            look_back=look_back,
            output_mode=output_mode,
            use_early_stopping=use_early_stopping,
            use_kalman_filter=use_kalman_filter,
            kalman_params=kalman_params,
            **kwargs,
        )

        self.config = kwargs
        self.use_cnn = use_cnn
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.early_stopping_patience = early_stopping_patience
        self.cnn_features = cnn_features
        self.lstm_hidden = lstm_hidden
        self.dropout = dropout

    def _get_model_type_name(self) -> str:
        """
        Get model type name for logging and identification.

        Returns:
            String representing the model architecture
        """
        parts = []
        if self.use_cnn:
            parts.append("CNN")
        parts.append("LSTM")
        if self.use_attention:
            parts.append("Attention")
        return "-".join(parts)

    def create_model(self, input_size: int):
        """
        Create model instance based on trainer configuration.

        Args:
            input_size: Number of input features

        Returns:
            Initialized model instance
        """
        # Define model parameters
        model_params = {
            "num_layers": self.config.get("num_layers", 2),
            "dropout": self.dropout,
        }

        if self.use_cnn:
            model_params.update(
                {
                    "cnn_features": self.cnn_features,
                    "lstm_hidden": self.lstm_hidden,
                    "num_heads": self.attention_heads,
                    "use_positional_encoding": self.config.get("use_positional_encoding", True),
                }
            )
        elif self.use_attention:
            model_params.update(
                {
                    "num_heads": self.attention_heads,
                    "hidden_size": self.lstm_hidden,
                    "use_positional_encoding": self.config.get("use_positional_encoding", True),
                }
            )
        else:
            model_params.update(
                {
                    "hidden_size": self.lstm_hidden,
                }
            )

        self.model = create_cnn_lstm_attention_model(
            input_size=input_size,
            use_attention=self.use_attention,
            use_cnn=self.use_cnn,
            look_back=self.look_back,
            output_mode=self.output_mode,  # type: ignore
            **model_params,
        )
        return self.model

    def _validate_and_preprocess_data(self, df_input: pd.DataFrame):
        """Override to add CNN-specific validation."""
        if self.use_cnn:
            # First perform basic validation
            if df_input.empty:
                raise ValueError(f"Insufficient data: {len(df_input)} rows, need at least {self.look_back + 50}")
            # Then perform CNN-specific validation
            try:
                self._validate_cnn_data_requirements(len(df_input), self.look_back)
            except ValueError as e:
                # Add CNN to error message as expected by some tests
                raise ValueError(f"{str(e)}")

        return super()._validate_and_preprocess_data(df_input)

    def train(
        self,
        df_input: pd.DataFrame,
        epochs: int = DEFAULT_EPOCHS,
        save_model: bool = True,
        model_filename: Optional[str] = None,
    ) -> Tuple[Optional[nn.Module], object, Optional[str]]:
        """
        Complete training pipeline.

        Args:
            df_input: Market data
            epochs: Max training epochs
            save_model: Whether to save the best model
            model_filename: Custom filename for saved model

        Returns:
            Tuple of (best_model, threshold_optimizer, model_path)
        """
        try:
            # 1. Preprocess data
            X, y = self._validate_and_preprocess_data(df_input)

            # 2. Prepare tensors
            X_train, X_val, X_test, y_train, y_val, y_test, num_classes, test_indices = self._prepare_tensors(X, y)
            input_size = X_train.shape[2]

            # 3. Create model
            model = self.create_model(input_size).to(self.device)

            # 4. Get batch size
            batch_size = self._get_batch_size(input_size)
            if self.use_cnn:
                batch_size = self._adjust_batch_size_for_cnn(batch_size)

            # 5. Setup components
            optimizer, scheduler, train_loader, val_loader, criterion, scaler_amp = self._setup_training_components(
                input_size, X_train, y_train, X_val, y_val, model, batch_size
            )

            # 6. Training loop
            best_val_loss = float("inf")
            patience_counter = 0
            training_history = {"train_loss": [], "val_loss": []}

            model_type = self._get_model_type_name()
            log_model(f"Training {model_type} for {epochs} epochs - Device: {self.device}")

            start_time = time.time()

            for epoch in range(epochs):
                train_loss, _, _ = self._train_epoch(model, train_loader, optimizer, criterion, scaler_amp)
                val_loss, _, _ = self._validate_epoch(model, val_loader, criterion)

                training_history["train_loss"].append(train_loss)
                training_history["val_loss"].append(val_loss)

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state in memory
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1

                if self.use_early_stopping and patience_counter >= self.early_stopping_patience:
                    log_model(f"Early stopping at epoch {epoch + 1}")
                    break

            # Load best model back
            model.load_state_dict(best_model_state)

            elapsed_time = time.time() - start_time
            log_success(f"{model_type} training completed in {elapsed_time:.2f}s")

            # 7. Post-training evaluation and threshold optimization
            self._evaluate_model(model, X_test, y_test, df_input)

            # 8. Save model
            model_path = None
            if save_model:
                model_path = self._save_model(
                    model,
                    model_filename,
                    input_size,
                    num_classes,
                    X_train,
                    X_val,
                    X_test,
                    optimizer,
                    best_val_loss,
                    epoch,
                    training_history,
                    self.use_cnn,
                    self.use_attention,
                    self.attention_heads,
                )

            return model, self.threshold_optimizer, model_path

        except Exception as e:
            log_error(f"Error during unified model training: {e}")
            log_error(f"Traceback: {traceback.format_exc()}")
            return None, self.threshold_optimizer, None
