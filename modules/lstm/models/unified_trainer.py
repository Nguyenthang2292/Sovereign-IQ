"""
Unified LSTM Trainer supporting all model variants.

This trainer can handle:
- LSTM (use_cnn=False, use_attention=False)
- LSTM-Attention (use_cnn=False, use_attention=True)
- CNN-LSTM (use_cnn=True, use_attention=False)
- CNN-LSTM-Attention (use_cnn=True, use_attention=True)
"""
import copy
import numpy as np
import pandas as pd
import time
import traceback
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

from modules.common.ui.logging import (
    log_error,
    log_model,
    log_analysis,
    log_success,
)
from config.lstm import (
    DEFAULT_EPOCHS,
    GPU_MODEL_CONFIG,
    WINDOW_SIZE_LSTM,
)

from modules.lstm.models.trainer.base_trainer import BaseLSTMTrainer
from modules.lstm.models.trainer.cnn_mixin import CNNFeatureMixin
from modules.lstm.models.model_factory import create_cnn_lstm_attention_model
from modules.lstm.core.threshold_optimizer import GridSearchThresholdOptimizer


class LSTMTrainer(BaseLSTMTrainer, CNNFeatureMixin):
    """
    Unified LSTM Trainer supporting all variants.
    
    This class provides a single interface for training all LSTM model variants
    based on the use_cnn and use_attention flags.
    
    Example:
        # LSTM only
        trainer = LSTMTrainer(use_cnn=False, use_attention=False)
        
        # LSTM with Attention
        trainer = LSTMTrainer(use_cnn=False, use_attention=True)
        
        # CNN-LSTM
        trainer = LSTMTrainer(use_cnn=True, use_attention=False)
        
        # CNN-LSTM-Attention
        trainer = LSTMTrainer(use_cnn=True, use_attention=True)
        
        model, threshold_optimizer, model_path = trainer.train(
            df_input=df,
            epochs=50,
            save_model=True
        )
    """
    
    def __init__(
        self,
        use_cnn: bool = False,
        use_attention: bool = False,
        look_back: int = WINDOW_SIZE_LSTM,
        output_mode: str = 'classification',
        attention_heads: int = GPU_MODEL_CONFIG['nhead'],
        use_early_stopping: bool = True,
        early_stopping_patience: int = 10,
        cnn_features: int = 64,
        lstm_hidden: int = 32,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        adam_eps: float = 1e-8,
        scheduler_T_0: int = 10,
        scheduler_T_mult: int = 2,
        scheduler_eta_min: float = 1e-6,
    ):
        """
        Initialize unified LSTM trainer.
        
        Args:
            use_cnn: Whether to use CNN layers
            use_attention: Whether to use attention mechanism
            look_back: Sequence length for time series
            output_mode: 'classification' or 'regression'
            attention_heads: Number of attention heads (if use_attention=True)
            use_early_stopping: Enable early stopping
            early_stopping_patience: Number of epochs to wait before early stopping (default: 10)
            cnn_features: Number of CNN feature maps (default: 64)
            lstm_hidden: LSTM hidden layer size (default: 32)
            dropout: Dropout rate (default: 0.3)
            learning_rate: Learning rate for AdamW optimizer (default: 0.001)
            weight_decay: Weight decay for AdamW optimizer (default: 0.01)
            adam_eps: Epsilon value for AdamW optimizer (default: 1e-8)
            scheduler_T_0: Initial period for CosineAnnealingWarmRestarts (default: 10)
            scheduler_T_mult: Multiplicative factor for period (default: 2)
            scheduler_eta_min: Minimum learning rate (default: 1e-6)
        """
        BaseLSTMTrainer.__init__(
            self,
            look_back=look_back,
            output_mode=output_mode,
            use_early_stopping=use_early_stopping,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_eps=adam_eps,
            scheduler_T_0=scheduler_T_0,
            scheduler_T_mult=scheduler_T_mult,
            scheduler_eta_min=scheduler_eta_min,
        )
        
        self.use_cnn = use_cnn
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.early_stopping_patience = early_stopping_patience
        self.cnn_features = cnn_features
        self.lstm_hidden = lstm_hidden
        self.dropout = dropout
    
    def create_model(self, input_size: int, **kwargs) -> nn.Module:
        """
        Create LSTM model based on configuration.
        
        Args:
            input_size: Number of input features
            **kwargs: Additional model parameters
            
        Returns:
            Neural network model
        """
        num_classes = 3 if self.output_mode == 'classification' else 1
        
        # Build kwargs based on model type
        model_kwargs = {'num_classes': num_classes, 'dropout': self.dropout}
        
        if self.use_cnn:
            # CNN models use cnn_features and lstm_hidden
            model_kwargs['cnn_features'] = self.cnn_features
            model_kwargs['lstm_hidden'] = self.lstm_hidden
            if self.use_attention:
                model_kwargs['num_heads'] = self.attention_heads
        elif self.use_attention:
            # LSTM-Attention uses hidden_size and num_heads
            model_kwargs['hidden_size'] = self.lstm_hidden  # Map lstm_hidden to hidden_size
            model_kwargs['num_heads'] = self.attention_heads
        else:
            # Standard LSTM uses hidden_size
            model_kwargs['hidden_size'] = self.lstm_hidden  # Map lstm_hidden to hidden_size
        
        # Merge with any additional kwargs
        model_kwargs.update(kwargs)
        
        model = create_cnn_lstm_attention_model(
            input_size=input_size,
            use_attention=self.use_attention,
            use_cnn=self.use_cnn,
            look_back=self.look_back,
            output_mode=self.output_mode,
            **model_kwargs
        ).to(self.device)
        
        self.model = model
        model_type = self._get_model_type_name()
        log_model(f"{model_type} model created - Input: {input_size}, Look back: {self.look_back}, Classes: {num_classes}")
        return model
    
    def _get_model_type_name(self) -> str:
        """Get human-readable model type name."""
        if self.use_cnn:
            if self.use_attention:
                return "CNN-LSTM-Attention"
            else:
                return "CNN-LSTM"
        else:
            if self.use_attention:
                return "LSTM-Attention"
            else:
                return "LSTM"
    
    def _validate_and_preprocess_data(self, df_input: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and preprocess input data.
        
        Args:
            df_input: Input DataFrame with price data
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        # CNN-specific validation
        if self.use_cnn:
            self._validate_cnn_data_requirements(len(df_input), self.look_back)
        
        # Call base class preprocessing
        return super()._validate_and_preprocess_data(df_input)
    
    def _setup_training_components(
        self, 
        input_size: int, 
        X_train: torch.Tensor, 
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler, DataLoader, DataLoader, nn.Module, Optional[GradScaler]]:
        """
        Setup model, optimizer, scheduler, and data loaders.
        
        Args:
            input_size: Number of input features
            X_train, y_train: Training tensors
            X_val, y_val: Validation tensors
            
        Returns:
            Tuple of (model, optimizer, scheduler, train_loader, val_loader, criterion, scaler_amp)
        """
        # Create model
        try:
            model = self.create_model(input_size)
        except Exception as model_error:
            log_error(f"Model creation failed: {model_error}")
            raise ValueError(f"Cannot create {self._get_model_type_name()} model: {model_error}")
        
        # Get batch size (adjust for CNN if needed)
        batch_size = self._get_batch_size(input_size)
        if self.use_cnn:
            batch_size = self._adjust_batch_size_for_cnn(batch_size)
        
        # Setup optimizer, scheduler, and data loaders
        optimizer, scheduler, train_loader, val_loader, criterion, scaler_amp = super()._setup_training_components(
            input_size, X_train, y_train, X_val, y_val, model, batch_size
        )
        
        return model, optimizer, scheduler, train_loader, val_loader, criterion, scaler_amp
    
    def train(
        self,
        df_input: pd.DataFrame,
        epochs: int = DEFAULT_EPOCHS,
        save_model: bool = True,
        model_filename: Optional[str] = None,
    ) -> Tuple[Optional[nn.Module], GridSearchThresholdOptimizer, Optional[str]]:
        """
        Train LSTM model.
        
        Args:
            df_input: Input DataFrame with price data
            epochs: Number of training epochs
            save_model: Whether to save the trained model
            model_filename: Optional custom model filename
            
        Returns:
            Tuple of (trained_model, threshold_optimizer, model_path_string)
        """
        start_time = time.time()
        best_model_state = None
        
        try:
            # Validate and preprocess data
            X, y = self._validate_and_preprocess_data(df_input)
            
            # Prepare tensors
            X_train, X_val, X_test, y_train, y_val, y_test, num_classes, test_indices = self._prepare_tensors(X, y)
            self.test_indices = test_indices  # Store test indices for price alignment
            
            # Setup training components
            input_size = len(self.feature_names)
            model, optimizer, scheduler, train_loader, val_loader, criterion, scaler_amp = self._setup_training_components(
                input_size, X_train, y_train, X_val, y_val
            )
            
            # Training variables
            best_val_loss = float('inf')
            patience_counter = 0
            patience = self.early_stopping_patience
            self.training_history = {'train_loss': [], 'val_loss': []}
            
            model_type = self._get_model_type_name()
            log_model(f"Training {model_type} for {epochs} epochs - Batch size: {train_loader.batch_size}, Mixed precision: {self.use_mixed_precision}")
            
            # Training loop
            epoch = 0
            try:
                for epoch in range(epochs):
                    # Training phase
                    train_loss, train_correct, train_total = self._train_epoch(
                        model, train_loader, optimizer, criterion, scaler_amp
                    )
                    
                    # Validation phase
                    val_loss, val_correct, val_total = self._validate_epoch(model, val_loader, criterion)
                    
                    # Calculate metrics and update
                    train_loss /= len(train_loader)
                    val_loss /= len(val_loader)
                    self.training_history['train_loss'].append(train_loss)
                    self.training_history['val_loss'].append(val_loss)
                    scheduler.step()
                    
                    # Logging
                    if self.output_mode == 'classification':
                        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
                        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
                        log_analysis(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                                     f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {optimizer.param_groups[0]["lr"]:.6f}')
                    else:
                        log_analysis(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                                     f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
                    
                    # Early stopping
                    if self.use_early_stopping:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_model_state = copy.deepcopy(model.state_dict())
                            log_model(f"New best validation loss: {val_loss:.4f}")
                        else:
                            patience_counter += 1
                            
                        if patience_counter >= patience:
                            log_model(f"Early stopping triggered after {epoch+1} epochs (patience: {patience})")
                            if best_model_state is not None:
                                model.load_state_dict(best_model_state)
                                log_model("Restored best model state")
                            break
            
            except Exception as training_error:
                log_error(f"Training failed at epoch {epoch + 1}: {training_error}")
                log_error(f"Traceback: {traceback.format_exc()}")
                # Restore best model state if available, otherwise abort
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    log_model("Restored best model state before failure")
                else:
                    log_error("No valid model state to restore - aborting")
                    return None, GridSearchThresholdOptimizer(), None
            
            # Evaluate model
            self._evaluate_model(model, X_test, y_test, df_input)
            
            # Save model
            model_path_str = None
            if save_model and model is not None:
                model_path_str = self._save_model(
                    model, model_filename, input_size, num_classes,
                    X_train, X_val, X_test, optimizer, best_val_loss,
                    epoch, self.training_history,
                    self.use_cnn, self.use_attention, self.attention_heads
                )
                if model_path_str:
                    elapsed_time = time.time() - start_time
                    log_success(f"Training completed in {elapsed_time:.2f}s")
            else:
                if model is not None:
                    elapsed_time = time.time() - start_time
                    log_analysis(f"{model_type} model training completed in {elapsed_time:.2f}s (model not saved)")
            
            return model, self.threshold_optimizer, model_path_str
            
        except Exception as e:
            log_error(f"Error during {self._get_model_type_name()} model training: {e}")
            log_error(f"Traceback: {traceback.format_exc()}")
            return None, GridSearchThresholdOptimizer(), None

