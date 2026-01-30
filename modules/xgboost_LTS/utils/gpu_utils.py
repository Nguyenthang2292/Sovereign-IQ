"""
GPU utilities for XGBoost module.

Provides cached GPU detection to avoid repeated subprocess calls.
"""

import functools
import subprocess
from typing import Optional


@functools.lru_cache(maxsize=1)
def detect_cuda_available() -> bool:
    """
    Detect if CUDA GPU is available for XGBoost.

    Uses nvidia-smi to check for GPU availability.
    Result is cached after first call.

    Returns:
        True if CUDA GPU is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, Exception):
        return False


@functools.lru_cache(maxsize=1)
def get_gpu_info() -> Optional[str]:
    """
    Get GPU name if available.

    Returns:
        GPU name string or None if not available.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None
