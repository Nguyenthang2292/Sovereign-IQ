"""Memory-mapped array utilities for large datasets."""

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from modules.common.utils import log_error, log_info, log_warn
except ImportError:

    def log_info(message: str) -> None:
        print(f"[INFO] {message}")

    def log_error(message: str) -> None:
        print(f"[ERROR] {message}")

    def log_warn(message: str) -> None:
        print(f"[WARN] {message}")


@dataclass
class MemmapDescriptor:
    """Descriptor for a memory-mapped file with metadata."""

    mmap_path: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    columns: List[str]
    index_name: Optional[str]
    timestamp: Optional[float]
    original_path: str


class MemoryMappedDataManager:
    """Manager for memory-mapped data files."""

    def __init__(self, cache_dir: str = ".cache/mmap"):
        """Initialize the memory-mapped data manager.

        Args:
            cache_dir: Directory to store memory-mapped files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        log_info(f"Memory-mapped cache directory: {self.cache_dir}")

    def _generate_cache_key(self, csv_path: str, symbol_column: str, price_column: str) -> str:
        """Generate a cache key for the CSV file.

        Args:
            csv_path: Path to the CSV file
            symbol_column: Name of symbol column
            price_column: Name of price column

        Returns:
            Cache key string
        """
        import hashlib

        # Use file path, modification time, and column info to create unique key
        stat = os.stat(csv_path)
        key_str = f"{csv_path}|{stat.st_mtime}|{symbol_column}|{price_column}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def create_memory_mapped_from_csv(
        self,
        csv_path: str,
        symbol_column: str = "symbol",
        price_column: str = "close",
        columns_to_map: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> MemmapDescriptor:
        """Create a memory-mapped file from CSV data.

        Args:
            csv_path: Path to the CSV file
            symbol_column: Name of symbol column
            price_column: Name of price column
            columns_to_map: List of columns to memory-map (None = all numeric columns)
            overwrite: Overwrite existing memmap file

        Returns:
            MemmapDescriptor with metadata

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Generate cache key and paths
        cache_key = self._generate_cache_key(str(csv_path), symbol_column, price_column)
        mmap_filename = f"{cache_key}.mmap"
        metadata_filename = f"{cache_key}.metadata"
        mmap_path = self.cache_dir / mmap_filename
        metadata_path = self.cache_dir / metadata_filename

        # Check if already exists
        if mmap_path.exists() and metadata_path.exists() and not overwrite:
            log_info(f"Memory-mapped file already exists: {mmap_path}")
            return self.load_descriptor(metadata_path)

        log_info(f"Creating memory-mapped file from {csv_path}")

        # Read CSV with pandas
        df = pd.read_csv(csv_path)

        # Validate required columns
        if symbol_column not in df.columns:
            raise ValueError(f"Symbol column '{symbol_column}' not found in CSV")
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found in CSV")

        # Determine columns to map
        if columns_to_map is None:
            # Map all numeric columns
            columns_to_map = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Validate columns exist
            for col in columns_to_map:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in CSV")

        log_info(f"Mapping columns: {columns_to_map}")

        # Create memory-mapped array
        # Use a consistent dtype (float64 for all numeric columns)
        dtype_list = [(col, np.float64) for col in columns_to_map]
        dtype = np.dtype(dtype_list)

        # Extract numeric data as structured array
        mmap_data = np.empty(len(df), dtype=dtype)
        for i, col in enumerate(columns_to_map):
            mmap_data[col] = df[col].values

        # Write to memory-mapped file
        mmap_array = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=mmap_data.shape)
        mmap_array[:] = mmap_data[:]
        mmap_array.flush()

        # Create descriptor
        descriptor = MemmapDescriptor(
            mmap_path=str(mmap_path),
            shape=mmap_data.shape,
            dtype=dtype,
            columns=columns_to_map,
            index_name=df.index.name,
            timestamp=None,
            original_path=str(csv_path),
        )

        # Save metadata
        self._save_descriptor(metadata_path, descriptor)

        log_info(f"Created memory-mapped file: {mmap_path} (shape={descriptor.shape})")

        return descriptor

    def _save_descriptor(self, metadata_path: Path, descriptor: MemmapDescriptor) -> None:
        """Save descriptor metadata to file.

        Args:
            metadata_path: Path to save metadata
            descriptor: MemmapDescriptor to save
        """
        with open(metadata_path, "wb") as f:
            pickle.dump(descriptor, f)

    def load_descriptor(self, metadata_path: Path) -> MemmapDescriptor:
        """Load descriptor metadata from file.

        Args:
            metadata_path: Path to metadata file

        Returns:
            MemmapDescriptor

        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "rb") as f:
            descriptor = pickle.load(f)

        return descriptor

    def load_from_descriptor(self, descriptor: MemmapDescriptor) -> np.memmap:
        """Load memory-mapped array from descriptor.

        Args:
            descriptor: MemmapDescriptor

        Returns:
            Memory-mapped numpy array
        """
        mmap_array = np.memmap(descriptor.mmap_path, dtype=descriptor.dtype, mode="r", shape=descriptor.shape)
        return mmap_array

    def load_from_csv_path(
        self,
        csv_path: str,
        symbol_column: str = "symbol",
        price_column: str = "close",
    ) -> Tuple[Optional[MemmapDescriptor], Optional[np.memmap]]:
        """Load or create memory-mapped data for a CSV file.

        Args:
            csv_path: Path to the CSV file
            symbol_column: Name of symbol column
            price_column: Name of price column

        Returns:
            Tuple of (MemmapDescriptor, np.memmap) or (None, None) if failed
        """
        try:
            # Generate cache key and paths
            cache_key = self._generate_cache_key(csv_path, symbol_column, price_column)
            metadata_filename = f"{cache_key}.metadata"
            metadata_path = self.cache_dir / metadata_filename

            # Try to load existing descriptor
            if metadata_path.exists():
                descriptor = self.load_descriptor(metadata_path)
                log_info(f"Loaded existing memory-mapped descriptor for {csv_path}")
            else:
                # Create new memory-mapped file
                descriptor = self.create_memory_mapped_from_csv(csv_path, symbol_column, price_column)

            # Load memory-mapped array
            mmap_array = self.load_from_descriptor(descriptor)

            return descriptor, mmap_array

        except Exception as e:
            log_error(f"Failed to load memory-mapped data: {e}")
            return None, None

    def cleanup(self, older_than_days: int = 7) -> int:
        """Clean up old memory-mapped files.

        Args:
            older_than_days: Remove files older than this many days

        Returns:
            Number of files removed
        """
        import time

        current_time = time.time()
        cutoff_time = current_time - (older_than_days * 24 * 60 * 60)
        files_removed = 0

        for file_path in self.cache_dir.glob("*"):
            if file_path.is_file():
                file_time = file_path.stat().st_mtime
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                        files_removed += 1
                        log_info(f"Removed old cache file: {file_path}")
                    except Exception as e:
                        log_error(f"Failed to remove {file_path}: {e}")

        log_info(f"Cleaned up {files_removed} old files")
        return files_removed


# Global instance
_manager: Optional[MemoryMappedDataManager] = None


def get_manager(cache_dir: str = ".cache/mmap") -> MemoryMappedDataManager:
    """Get global MemoryMappedDataManager instance."""
    global _manager
    if _manager is None:
        _manager = MemoryMappedDataManager(cache_dir=cache_dir)
    return _manager


def create_memory_mapped_from_csv(
    csv_path: str,
    symbol_column: str = "symbol",
    price_column: str = "close",
    columns_to_map: Optional[List[str]] = None,
    cache_dir: str = ".cache/mmap",
    overwrite: bool = False,
) -> MemmapDescriptor:
    """Create a memory-mapped file from CSV data.

    Args:
        csv_path: Path to the CSV file
        symbol_column: Name of symbol column
        price_column: Name of price column
        columns_to_map: List of columns to memory-map (None = all numeric columns)
        cache_dir: Directory to store memory-mapped files
        overwrite: Overwrite existing memmap file

    Returns:
        MemmapDescriptor with metadata

    Example:
        >>> descriptor = create_memory_mapped_from_csv("data.csv")
        >>> print(descriptor.shape)
        (1000000,)
        >>> mmap_array = np.memmap(descriptor.mmap_path, dtype=descriptor.dtype, mode='r')
        >>> print(mmap_array['close'][:5])
    """
    manager = get_manager(cache_dir)
    return manager.create_memory_mapped_from_csv(csv_path, symbol_column, price_column, columns_to_map, overwrite)


def load_memory_mapped_from_csv(
    csv_path: str,
    symbol_column: str = "symbol",
    price_column: str = "close",
    cache_dir: str = ".cache/mmap",
) -> Tuple[Optional[MemmapDescriptor], Optional[np.memmap]]:
    """Load or create memory-mapped data for a CSV file.

    Args:
        csv_path: Path to the CSV file
        symbol_column: Name of symbol column
        price_column: Name of price column
        cache_dir: Directory to store memory-mapped files

    Returns:
        Tuple of (MemmapDescriptor, np.memmap) or (None, None) if failed
    """
    manager = get_manager(cache_dir)
    return manager.load_from_csv_path(csv_path, symbol_column, price_column)
