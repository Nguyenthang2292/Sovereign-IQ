"""
Base trainer class with common training logic for all LSTM variants.
"""
import copy
import numpy as np
import os
import pandas as pd
import time
import traceback
from datetime import datetime
from typing import Optional, Tuple

from modules.common.ui.logging import (
    log_success, 
    log_error, 
    log_warn, 
    log_debug,
    log_model, 
    log_analysis, 
    log_system
)

# Environment setup for PyTorch from config
from modules.common.utils.system import get_pytorch_env
os.environ.update(get_pytorch_env())


def _import_pytorch_with_check():
    """
    Dynamically import PyTorch modules and perform optional CUDA detection at runtime.

    Raises:
        ImportError: If PyTorch or required submodules are not installed.
    Returns:
        dict: Imported torch modules for use.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from torch.amp.autocast_mode import autocast
        from torch.amp.grad_scaler import GradScaler
        return {
            'torch': torch,
            'nn': nn,
            'optim': optim,
            'DataLoader': DataLoader,
            'TensorDataset': TensorDataset,
            'autocast': autocast,
            'GradScaler': GradScaler,
        }
    except ImportError as e:
        log_error(f"Failed to import PyTorch or required modules: {e}")
        log_error("PyTorch is required: pip install torch torchvision torchaudio")
        raise

# Import PyTorch modules using the safe import function
_pytorch_modules = _import_pytorch_with_check()
torch = _pytorch_modules['torch']
nn = _pytorch_modules['nn']
optim = _pytorch_modules['optim']
DataLoader = _pytorch_modules['DataLoader']
TensorDataset = _pytorch_modules['TensorDataset']
autocast = _pytorch_modules['autocast']
GradScaler = _pytorch_modules['GradScaler']

from config.lstm import (
    DEFAULT_EPOCHS,
    MODELS_DIR,
    TRAIN_TEST_SPLIT,
    VALIDATION_SPLIT,
    WINDOW_SIZE_LSTM,
)
from config.model_features import MODEL_FEATURES

from modules.common.utils.system import detect_pytorch_gpu_availability, configure_gpu_memory
from modules.lstm.core.focal_loss import FocalLoss
from modules.lstm.core.threshold_optimizer import GridSearchThresholdOptimizer
from modules.lstm.utils.batch_size import get_optimal_batch_size
from modules.lstm.utils.preprocessing import preprocess_cnn_lstm_data
from modules.lstm.utils.data_utils import split_train_test_data


class BaseLSTMTrainer:
    """
    Base trainer class containing common training logic for all LSTM variants.
    
    This class provides the foundation for training LSTM models regardless of
    whether they use CNN or attention mechanisms.
    """
    
    def __init__(
        self,
        look_back: int = WINDOW_SIZE_LSTM,
        output_mode: str = 'classification',
        use_early_stopping: bool = True,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        adam_eps: float = 1e-8,
        scheduler_T_0: int = 10,
        scheduler_T_mult: int = 2,
        scheduler_eta_min: float = 1e-6,
    ):
        """
        Initialize base trainer.
        
        Args:
            look_back: Sequence length for time series
            output_mode: 'classification' or 'regression'
            use_early_stopping: Enable early stopping
            learning_rate: Learning rate for AdamW optimizer (default: 0.001)
            weight_decay: Weight decay for AdamW optimizer (default: 0.01)
            adam_eps: Epsilon value for AdamW optimizer (default: 1e-8)
            scheduler_T_0: Initial period for CosineAnnealingWarmRestarts (default: 10)
            scheduler_T_mult: Multiplicative factor for period (default: 2)
            scheduler_eta_min: Minimum learning rate (default: 1e-6)
        """
        self.look_back = look_back
        self.output_mode = output_mode
        self.use_early_stopping = use_early_stopping
        
        # Training hyperparameters (can be overridden before calling train)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.scheduler_T_0 = scheduler_T_0
        self.scheduler_T_mult = scheduler_T_mult
        self.scheduler_eta_min = scheduler_eta_min
        
        # State variables (initialized during training)
        self.device = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.threshold_optimizer = GridSearchThresholdOptimizer()
        self.training_history = None
        self.gpu_available = False
        self.use_mixed_precision = False
        self.test_indices = None  # Store test indices for accurate price alignment
        
        # Setup GPU
        self._setup_device()
    
    def _setup_device(self) -> None:
        """Setup GPU device and mixed precision."""
        self.gpu_available = detect_pytorch_gpu_availability()
        self.device = torch.device('cuda:0' if self.gpu_available and configure_gpu_memory() else 'cpu')
        self.use_mixed_precision = (
            self.device.type == 'cuda'
            and torch.cuda.get_device_capability(0)[0] >= 7
        )
        if self.use_mixed_precision:
            log_system("Using mixed precision training")
    
    def _validate_and_preprocess_data(self, df_input: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and preprocess input data.
        
        Note: In fallback scenarios with insufficient data, this method may modify
        self.look_back to a smaller value to accommodate the available data. This change
        is permanent and affects subsequent operations (batch size calculation, model
        architecture, and evaluation alignment). A CRITICAL level log message is emitted
        when this occurs.
        
        Args:
            df_input: Input DataFrame with price data
            
        Returns:
            Tuple of (X_sequences, y_targets)
        """
        # Data validation
        min_required = self.look_back + 50
        if df_input.empty or len(df_input) < min_required:
            raise ValueError(f"Insufficient data: {len(df_input)} rows, need at least {min_required}")
        
        log_model(f"Starting LSTM pipeline - Look back: {self.look_back}, Mode: {self.output_mode}")
        
        # Preprocess data
        X, y, self.scaler, self.feature_names = preprocess_cnn_lstm_data(
            df_input, look_back=self.look_back, output_mode=self.output_mode, scaler_type='minmax'
        )
        
        if len(X) == 0:
            log_error(f"Preprocessing failed - Input shape: {df_input.shape}")
            basic_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df_input.columns]
            log_error(f"Available columns: {list(df_input.columns)}")
            log_error(f"Required features: {MODEL_FEATURES}")
            log_error(f"Feature names returned: {self.feature_names}")
            
            if not basic_cols:
                log_error("No basic OHLC columns found in input data")
                if 'close' in df_input.columns:
                    log_warn("Attempting to create minimal feature set...")
                    minimal_df = df_input.copy()
                    close_col = 'close'
                    
                    # Create basic features from close price only
                    minimal_df['returns'] = minimal_df[close_col].pct_change()
                    minimal_df['sma_5'] = minimal_df[close_col].rolling(5).mean()
                    minimal_df['sma_20'] = minimal_df[close_col].rolling(20).mean()
                    minimal_df['volatility'] = minimal_df['returns'].rolling(10).std()
                    
                    # Retry preprocessing with minimal features
                    log_warn("Retrying preprocessing with minimal feature set...")
                    original_look_back = self.look_back
                    fallback_look_back = max(5, self.look_back // 2)
                    X, y, self.scaler, self.feature_names = preprocess_cnn_lstm_data(
                        minimal_df, look_back=fallback_look_back, output_mode=self.output_mode, scaler_type='minmax'
                    )
                    
                    if len(X) > 0:
                        # Store both original and effective look_back for traceability
                        self.original_look_back = original_look_back
                        self.look_back = fallback_look_back
                        log_model(
                            f"Reduced look_back from {self.original_look_back} to {self.look_back} "
                            f"due to insufficient data. This affects batch size, model architecture, and evaluation. "
                            f"Created {len(X)} sequences with minimal features."
                        )
                    else:
                        raise ValueError("Failed to create sequences even with minimal features. The data may be corrupted or insufficient.")
                else:
                    raise ValueError("No price data columns found in input data")
            else:
                raise ValueError(f"Data preprocessing failed - no valid sequences created. Available basic columns: {basic_cols}")
        
        return X, y
    
    def _prepare_tensors(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, np.ndarray]:
        """
        Split data and prepare tensors.
        
        Args:
            X: Input sequences array
            y: Target array
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, num_classes, test_indices)
        """
        # Split data with shuffle=False for time series to preserve temporal order
        # This ensures test set corresponds to the last portion of the data chronologically
        # Test indices are returned to enable accurate price alignment in evaluation
        X_train, X_val, X_test, y_train, y_val, y_test, test_indices = split_train_test_data(
            X, y, TRAIN_TEST_SPLIT, VALIDATION_SPLIT, shuffle=False, return_indices=True
        )
        
        if len(X_train) == 0:
            raise ValueError("Insufficient data after train/test split")
        
        # Convert to tensors
        X_train, X_val, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_val), torch.FloatTensor(X_test)
        
        if self.output_mode == 'classification':
            # Validate expected label range (-1, 0, 1) before shifting to (0, 1, 2)
            # Check all splits to ensure consistency
            for split_name, split_labels in [('train', y_train), ('validation', y_val), ('test', y_test)]:
                unique_labels = np.unique(split_labels)
                if not np.all(np.isin(unique_labels, [-1, 0, 1])):
                    error_msg = f"Unexpected label values in {split_name} set: {unique_labels}. Expected -1, 0, 1."
                    log_error(error_msg)
                    raise ValueError(error_msg)
            y_train = torch.LongTensor(y_train + 1)
            y_val = torch.LongTensor(y_val + 1)
            y_test = torch.LongTensor(y_test + 1)
            num_classes = 3
        else:
            y_train = torch.FloatTensor(y_train).unsqueeze(1)
            y_val = torch.FloatTensor(y_val).unsqueeze(1)
            y_test = torch.FloatTensor(y_test).unsqueeze(1)
            num_classes = 1
        
        # Store test indices for accurate price alignment in evaluation
        self.test_indices = test_indices
        
        return X_train, X_val, X_test, y_train, y_val, y_test, num_classes, test_indices
    
    def _get_batch_size(self, input_size: int) -> int:
        """
        Get optimal batch size for training.
        
        Args:
            input_size: Number of input features
            
        Returns:
            Optimal batch size
        """
        return get_optimal_batch_size(self.device, input_size, self.look_back)
    
    def _setup_training_components(
        self, 
        input_size: int, 
        X_train: torch.Tensor, 
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        model: nn.Module,
        batch_size: int
    ) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler, DataLoader, DataLoader, nn.Module, Optional[GradScaler]]:
        """
        Setup optimizer, scheduler, and data loaders.
        
        Args:
            input_size: Number of input features
            X_train, y_train: Training tensors
            X_val, y_val: Validation tensors
            model: Created model instance
            batch_size: Batch size to use
            
        Returns:
            Tuple of (optimizer, scheduler, train_loader, val_loader, criterion, scaler_amp)
        """
        # Training setup with configurable hyperparameters
        criterion = FocalLoss(alpha=0.25, gamma=2.0) if self.output_mode == 'classification' else nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.adam_eps
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.scheduler_T_0,
            T_mult=self.scheduler_T_mult,
            eta_min=self.scheduler_eta_min
        )
        
        # Data loaders
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=batch_size, 
            shuffle=True, pin_memory=self.gpu_available, num_workers=0, drop_last=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=batch_size,
            shuffle=False, pin_memory=self.gpu_available, num_workers=0
        )
        
        scaler_amp = None
        if self.use_mixed_precision:
            scaler_amp = GradScaler('cuda')
        
        return optimizer, scheduler, train_loader, val_loader, criterion, scaler_amp
    
    def _train_epoch(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scaler_amp: Optional[GradScaler]
    ) -> Tuple[float, int, int]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (train_loss, train_correct, train_total)
        """
        model.train()
        train_loss = 0.0
        train_correct = train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            
            if self.use_mixed_precision and scaler_amp is not None:
                with autocast('cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            
            if self.output_mode == 'classification':
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
        
        return train_loss, train_correct, train_total
    
    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, int, int]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (val_loss, val_correct, val_total)
        """
        model.eval()
        val_loss = 0.0
        val_correct = val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with autocast('cuda'):
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                
                if self.output_mode == 'classification':
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
        
        return val_loss, val_correct, val_total
    
    def _evaluate_model(
        self,
        model: nn.Module,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        df_input: pd.DataFrame
    ) -> None:
        """
        Evaluate model and optimize threshold.
        
        Args:
            model: Trained model
            X_test: Test input tensors
            y_test: Test target tensors
            df_input: Original input DataFrame for price data
        """
        log_model("Starting model evaluation...")
        
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            test_predictions = model(X_test).cpu().numpy()
        
        if self.output_mode == 'regression':
            test_returns = y_test.squeeze().cpu().numpy()
            close_col = 'close'

            # Align price array with test_returns using stored test_indices
            if self.test_indices is None:
                raise ValueError(
                    "test_indices not available. This should not happen if data split was performed correctly. "
                    "Ensure _prepare_tensors() was called before evaluation."
                )
            
            if len(self.test_indices) != len(test_returns):
                raise ValueError(
                    f"Length mismatch: test_indices={len(self.test_indices)}, "
                    f"test_returns={len(test_returns)}. This indicates a data processing error."
                )
            
            # Extract prices at positions corresponding to test_indices
            # Sequence at index i in X spans df indices [i:i+look_back], 
            # and target corresponds to return from i+look_back-1 to i+look_back
            # So we need price at index i+look_back (future price for the return)
            price_indices = self.test_indices + self.look_back
            
            # Clip indices to valid range to handle edge cases where preprocessing
            # may have dropped some rows or sequences extend to the end of data
            max_valid_index = len(df_input) - 1
            if price_indices.max() > max_valid_index:
                invalid_count = np.sum(price_indices > max_valid_index)
                log_warn(
                    f"{invalid_count} test price indices exceed df_input bounds "
                    f"(max valid: {max_valid_index}, got max: {price_indices.max()}). "
                    f"Clipping to valid range. This may occur if preprocessing dropped rows."
                )
                price_indices = np.clip(price_indices, 0, max_valid_index)
            
            prices = df_input[close_col].values[price_indices]
            
            if len(prices) != len(test_returns):
                raise ValueError(
                    f"Length mismatch after index alignment: prices={len(prices)}, "
                    f"test_returns={len(test_returns)}. This may occur if preprocessing and evaluation "
                    f"use different data or look_back values."
                )

            best_threshold, best_sharpe = self.threshold_optimizer.optimize_regression_threshold(
                test_predictions.flatten(), test_returns
            )
            log_model(f"Regression optimization - Threshold: {best_threshold or 0.02:.4f}, "
                        f"Sharpe: {best_sharpe if best_sharpe is not None else 0.0:.4f}")
        else:
            test_returns = y_test.squeeze().cpu().numpy() - 1
            best_confidence, best_sharpe = self.threshold_optimizer.optimize_classification_threshold(
                test_predictions, test_returns
            )
            log_model(f"Classification optimization - Confidence: {best_confidence or 0.7:.2f}, "
                        f"Sharpe: {best_sharpe if best_sharpe is not None else 0.0:.4f}")
    
    def _save_model(
        self,
        model: nn.Module,
        model_filename: Optional[str],
        input_size: int,
        num_classes: int,
        X_train: torch.Tensor,
        X_val: torch.Tensor,
        X_test: torch.Tensor,
        optimizer: optim.Optimizer,
        best_val_loss: float,
        epoch: int,
        training_history: dict,
        use_cnn: bool,
        use_attention: bool,
        attention_heads: int
    ) -> Optional[str]:
        """
        Save trained model.
        
        Args:
            use_cnn: Whether model uses CNN
            use_attention: Whether model uses attention
            attention_heads: Number of attention heads
            
        Returns:
            Model path string or None if saving failed
        """
        try:
            # Auto-generate filename with timestamp if not provided
            if model_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                model_type = "cnn_lstm_attention" if use_cnn else ("lstm_attention" if use_attention else "lstm")
                model_filename = f"{model_type}_{self.output_mode}_model_{timestamp}.pth"
            
            model_path = MODELS_DIR / model_filename
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
            save_dict = {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size': input_size,
                    'look_back': self.look_back,
                    'original_look_back': getattr(self, 'original_look_back', None),
                    'output_mode': self.output_mode,
                    'use_cnn': use_cnn,
                    'use_attention': use_attention,
                    'attention_heads': attention_heads,
                    'num_classes': num_classes
                },
                'training_info': {
                    'epochs_trained': epoch + 1,
                    'best_val_loss': best_val_loss,
                    'final_lr': optimizer.param_groups[0]['lr']
                },
                'data_info': {
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test)
                },
                'optimization_results': {
                    'optimal_threshold': self.threshold_optimizer.best_threshold,
                    'best_sharpe': self.threshold_optimizer.best_sharpe
                },
                'training_history': training_history
            }
            
            torch.save(save_dict, model_path)
            log_success(f"LSTM model saved to {model_path}")
            return str(model_path)
        except Exception as e:
            log_error(f"Failed to save model: {e}")
            return None

