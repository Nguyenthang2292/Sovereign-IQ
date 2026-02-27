"""Memory-mapped array utilities for large feature matrices."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def dataframe_to_memmap(
    df: pd.DataFrame,
    file_path: Path,
    columns: Iterable[str] | None = None,
    dtype: np.dtype | type = np.float32,
) -> tuple[np.memmap, list[str]]:
    """Write DataFrame values to a memory-mapped file and return the mapped array.

    Args:
        df: Source DataFrame.
        file_path: Output path for the memory-mapped binary file.
        columns: Optional subset of columns to export.
        dtype: Target dtype.

    Returns:
        Tuple of `(memmap_array, used_columns)`.
    """
    used_columns = list(columns) if columns is not None else list(df.columns)
    if not used_columns:
        raise ValueError("No columns provided for memory-mapped export")

    values = df[used_columns].to_numpy(dtype=dtype, copy=False)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    mapped = np.memmap(file_path, mode="w+", dtype=dtype, shape=values.shape)
    mapped[:] = values
    mapped.flush()

    readonly = np.memmap(file_path, mode="r", dtype=dtype, shape=values.shape)
    return readonly, used_columns


def load_memmap(file_path: Path, shape: tuple[int, int], dtype: np.dtype | type = np.float32) -> np.memmap:
    """Load a previously-created memory-mapped matrix in read-only mode."""
    return np.memmap(file_path, mode="r", dtype=dtype, shape=shape)
