"""Data compression utilities using blosc."""

import os
import pickle
from typing import Any, Optional

try:
    import blosc

    BLOSC_AVAILABLE = True
except ImportError:
    BLOSC_AVAILABLE = False

try:
    from modules.common.utils import log_info, log_warn
except ImportError:

    def log_info(message: str) -> None:
        print(f"[INFO] {message}")

    def log_warn(message: str) -> None:
        print(f"[WARN] {message}")


def compress_pickle(
    obj: Any,
    compression_level: int = 5,
    algorithm: str = "blosclz",
    shuffle: bool = True,
) -> bytes:
    """Compress a Python object using blosc + pickle.

    Args:
        obj: Python object to compress
        compression_level: Compression level (0-9, higher = more compression)
        algorithm: Compression algorithm ('blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd')
        shuffle: Whether to use shuffle (recommended for speed)

    Returns:
        Compressed bytes

    Raises:
        RuntimeError: If blosc is not available
    """
    if not BLOSC_AVAILABLE:
        raise RuntimeError("blosc is not available. Install it with: pip install blosc")

    pickle_data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    compressed_data = blosc.compress(
        pickle_data,
        cname=algorithm,
        clevel=compression_level,
        shuffle=blosc.SHUFFLE if shuffle else blosc.NOSHUFFLE,
    )

    return compressed_data


def decompress_pickle(compressed_data: bytes) -> Any:
    """Decompress blosc-compressed pickle data.

    Args:
        compressed_data: Compressed bytes from compress_pickle()

    Returns:
        Original Python object

    Raises:
        RuntimeError: If blosc is not available
        ValueError: If decompression fails
    """
    if not BLOSC_AVAILABLE:
        raise RuntimeError("blosc is not available. Install it with: pip install blosc")

    try:
        pickle_data = blosc.decompress(compressed_data)
        obj = pickle.loads(pickle_data)
        return obj
    except Exception as e:
        raise ValueError(f"Failed to decompress data: {e}")


def compress_to_file(
    obj: Any,
    filepath: str,
    compression_level: int = 5,
    algorithm: str = "blosclz",
) -> int:
    """Compress an object and save to file.

    Args:
        obj: Python object to compress
        filepath: Path to save compressed data (adds .blosc extension if not present)
        compression_level: Compression level (0-9)
        algorithm: Compression algorithm

    Returns:
        Size of compressed file in bytes
    """
    if not filepath.endswith(".blosc"):
        filepath += ".blosc"

    compressed_data = compress_pickle(obj, compression_level, algorithm)

    with open(filepath, "wb") as f:
        f.write(compressed_data)

    file_size = len(compressed_data)
    log_info(f"Compressed and saved to {filepath} ({file_size:,} bytes)")

    return file_size


def decompress_from_file(filepath: str) -> Any:
    """Load and decompress a blosc-compressed file.

    Args:
        filepath: Path to compressed file

    Returns:
        Original Python object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If decompression fails
    """
    if not filepath.endswith(".blosc"):
        filepath += ".blosc"

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Compressed file not found: {filepath}")

    with open(filepath, "rb") as f:
        compressed_data = f.read()

    return decompress_pickle(compressed_data)


def get_compression_ratio(original_size: int, compressed_size: int) -> float:
    """Calculate compression ratio.

    Args:
        original_size: Size of uncompressed data in bytes
        compressed_size: Size of compressed data in bytes

    Returns:
        Compression ratio (original_size / compressed_size). Higher is better.
    """
    if compressed_size == 0:
        return 0.0

    return original_size / compressed_size


def is_compression_available() -> bool:
    """Check if blosc compression is available.

    Returns:
        True if blosc is installed and available
    """
    return BLOSC_AVAILABLE


def get_default_compression_level() -> int:
    """Get recommended default compression level.

    Returns:
        Default compression level (5 is a good balance of speed/size)
    """
    return 5


def get_supported_algorithms() -> list[str]:
    """Get list of supported compression algorithms.

    Returns:
        List of algorithm names supported by blosc
    """
    if not BLOSC_AVAILABLE:
        return []

    return ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]


def validate_compression_params(compression_level: int, algorithm: str) -> tuple[bool, Optional[str]]:
    """Validate compression parameters.

    Args:
        compression_level: Compression level to validate
        algorithm: Algorithm name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not BLOSC_AVAILABLE:
        return False, "blosc is not available"

    valid_algorithms = get_supported_algorithms()

    if algorithm not in valid_algorithms:
        return False, f"Invalid algorithm '{algorithm}'. Supported: {valid_algorithms}"

    if not 0 <= compression_level <= 9:
        return False, f"Compression level must be 0-9, got {compression_level}"

    return True, None
