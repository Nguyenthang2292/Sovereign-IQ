"""
Zero-Copy Utilities for High-Performance Data Handling.

This module provides tools to verify and enforce zero-copy operations
on NumPy arrays and Pandas Series/DataFrames.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


def ensure_zero_copy(arr: np.ndarray, base: np.ndarray) -> bool:
    """
    Verify that 'arr' shares memory with 'base'.

    Args:
        arr: The derived array (slice/view).
        base: The original base array.

    Returns:
        True if they share memory, False otherwise.
    """
    return np.shares_memory(arr, base)


def assert_zero_copy(arr: np.ndarray, base: np.ndarray, name: str = "array") -> None:
    """
    Raise ValueError if 'arr' does not share memory with 'base'.

    Args:
        arr: The derived array.
        base: The original array.
        name: Name for error message.
    """
    if not np.shares_memory(arr, base):
        raise ValueError(f"Zero-copy violation: '{name}' does not share memory with base array")


def is_contiguous(arr: Union[np.ndarray, pd.Series]) -> bool:
    """
    Check if the array is C-contiguous (essential for efficient SIMD/CuPy transfer).
    """
    if isinstance(arr, pd.Series):
        values = arr.values
    else:
        values = arr

    return values.flags["C_CONTIGUOUS"]


def make_readonly(obj: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
    """
    Make an object read-only to prevent accidental modification (and CoW).

    Args:
        obj: Array or DataFrame/Series.

    Returns:
        The same object, marked read-only.
    """
    if isinstance(obj, np.ndarray):
        obj.flags.writeable = False
        return obj
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        # Pandas doesn't have a simple 'readonly' flag effectively exposed for all ops,
        # but we can set underlying numpy arrays to readonly.
        if isinstance(obj, pd.Series):
            obj.values.flags.writeable = False
        else:
            # DataFrame
            for col in obj.columns:
                obj[col].values.flags.writeable = False
        return obj
    return obj
