"""
GPU Multi-Stream Management for Adaptive Trend LTS

This module provides a manager to handle multiple CUDA streams via CuPy,
allowing concurrent execution of kernels and overlapping data transfers.
"""

from typing import Optional

from modules.common.ui.logging import log_info, log_warn

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


class GPUStreamManager:
    """
    Manager for CUDA streams to enable parallelism on the GPU.
    """

    def __init__(self, num_streams: int = 1):
        self.num_streams = num_streams
        self.streams = []

        if _HAS_CUPY:
            try:
                self.streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
                log_info(f"GPUStreamManager: Initialized with {num_streams} concurrent streams.")
            except Exception as e:
                log_warn(f"GPUStreamManager: Failed to initialize streams: {e}. Falling back to default stream.")
                self.num_streams = 0
        else:
            log_warn("GPUStreamManager: CuPy not available. Multi-stream disabled.")
            self.num_streams = 0

        self._current_idx = 0

    def get_stream(self) -> Optional["cp.cuda.Stream"]:
        """Get the next available stream in a round-robin fashion."""
        if not self.streams:
            return None

        stream = self.streams[self._current_idx]
        self._current_idx = (self._current_idx + 1) % len(self.streams)
        return stream

    def synchronize_all(self):
        """Synchronize all streams."""
        if _HAS_CUPY:
            for stream in self.streams:
                stream.synchronize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.synchronize_all()


# Global default manager
_default_stream_manager = None


def get_gpu_stream_manager(num_streams: int = 1) -> GPUStreamManager:
    """Get or create a GPU stream manager."""
    global _default_stream_manager
    if _default_stream_manager is None or _default_stream_manager.num_streams != num_streams:
        _default_stream_manager = GPUStreamManager(num_streams)
    return _default_stream_manager
