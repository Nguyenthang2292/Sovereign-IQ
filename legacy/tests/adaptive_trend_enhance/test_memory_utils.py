"""
Tests for memory management utilities.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.common.system import cleanup_series, get_memory_manager, temp_series


def test_temp_series_decorator():
    """Verify that @temp_series triggers cleanup."""

    # We can't easily verify GC in a unit test without complex mocking,
    # but we can verify it doesn't break function execution.
    @temp_series
    def my_func(x):
        return x * 2

    assert my_func(10) == 20


def test_cleanup_series():
    """Verify cleanup_series utility."""
    # Small series - shouldn't trigger GC
    s_small = pd.Series(range(100))
    cleanup_series(s_small)

    # Large series - should trigger GC
    s_large = pd.Series(range(10000000))  # About 80MB
    # Triggering GC doesn't return anything, but we check for exceptions
    cleanup_series(s_large)


def test_memory_manager_integration():
    """Verify memory manager continues to work with new utilities."""
    manager = get_memory_manager()
    with manager.temp_series_scope():
        s = pd.Series(range(1000))
        assert len(s) == 1000

    # After scope, memory manager should still be functional
    assert manager.get_current_usage() is not None
