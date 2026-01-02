"""
Data utility functions for LSTM models.
"""

import numpy as np
from typing import Tuple, Optional, Union
from sklearn.model_selection import train_test_split

from modules.common.ui.logging import log_model
from config.lstm import TRAIN_TEST_SPLIT, VALIDATION_SPLIT


def split_train_test_data(
    X: np.ndarray, 
    y: np.ndarray, 
    train_ratio: float = TRAIN_TEST_SPLIT, 
    validation_ratio: float = VALIDATION_SPLIT,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    return_indices: bool = False
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Split data into train/validation/test sets with data validation.
    
    Args:
        X: Input sequences array 
        y: Target array
        train_ratio: Ratio for training set
        validation_ratio: Ratio for validation set
        shuffle: Whether to shuffle the data before splitting.
                 For time series data, set shuffle=False to preserve temporal order.
        random_state: Random state for reproducibility
        return_indices: If True, also return indices for test set to enable accurate price alignment
        
    Returns:
        If return_indices=False: X_train, X_val, X_test, y_train, y_val, y_test
        If return_indices=True: X_train, X_val, X_test, y_train, y_val, y_test, test_indices
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays")
    
    n_samples = len(X)
    if n_samples != len(y):
        raise ValueError(f"X and y length mismatch: X={n_samples}, y={len(y)}")
    
    if n_samples < 10:
        raise ValueError(f"Insufficient data: {n_samples} samples, need at least 10")
    
    if not (0 < train_ratio < 1) or not (0 < validation_ratio < 1):
        raise ValueError("Ratios must be between 0 and 1")
    
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Sum of ratios must be less than 1")
    
    test_ratio = 1 - train_ratio - validation_ratio
    if test_ratio <= 0:
        raise ValueError(f"Test ratio must be > 0, got {test_ratio}")
    
    # Create indices array for tracking
    indices = np.arange(n_samples)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test, _, test_indices = train_test_split(
        X, y, indices,
        test_size=test_ratio,
        shuffle=shuffle,
        random_state=random_state
    )
    
    # Second split: split remaining data into train and validation
    # test_size for this split is validation_ratio relative to the remaining data
    val_size_relative = validation_ratio / (train_ratio + validation_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_relative,
        shuffle=shuffle,
        random_state=random_state
    )
    
    log_model(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    if shuffle and return_indices:
        log_model("WARNING: shuffle=True detected. For time series data, ensure test_indices are used for price alignment.")
    
    if return_indices:
        return X_train, X_val, X_test, y_train, y_val, y_test, test_indices
    else:
        return X_train, X_val, X_test, y_train, y_val, y_test

