"""
This module configures pytest settings and provides fixtures for testing.
It suppresses certain warnings, handles pytest capture issues on Windows,
and defines reusable fixtures to streamline test setup.
"""

import sys
import warnings

import numpy as np
import pytest

# Suppress warnings from pytorch_forecasting about non-writable NumPy arrays
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")


# Workaround for pytest capture bug on Windows with Python 3.12+
# This prevents the "I/O operation on closed file" error
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configure pytest to avoid capture issues on Windows."""

    # Ensure capture is disabled if not already set
    if hasattr(config.option, "capture") and config.option.capture == "no":
        # Set up safe stderr handling
        if hasattr(sys, "stderr") and sys.stderr is not None:
            try:
                sys.stderr.flush()
            except (ValueError, OSError):
                pass


@pytest.fixture
def config_factory():
    """Factory fixture for creating Config instances for testing."""

    def _create_config(**kwargs):
        """Create a Config instance with specified attributes."""
        from types import SimpleNamespace

        config = SimpleNamespace()
        # Set default values for all possible attributes
        config.timeframe = kwargs.get("timeframe", None)
        config.no_menu = kwargs.get("no_menu", False)
        config.enable_spc = kwargs.get("enable_spc", False)
        config.spc_k = kwargs.get("spc_k", None)
        config.enable_xgboost = kwargs.get("enable_xgboost", False)
        config.enable_hmm = kwargs.get("enable_hmm", False)
        config.enable_random_forest = kwargs.get("enable_random_forest", False)
        config.use_decision_matrix = kwargs.get("use_decision_matrix", None)
        config.spc_strategy = kwargs.get("spc_strategy", None)
        config.limit = kwargs.get("limit", None)
        config.max_workers = kwargs.get("max_workers", None)

        return config

    return _create_config


@pytest.fixture
def seeded_random():
    """
    Fixture to provide a seeded NumPy random generator.

    Returns a NumPy Generator instance seeded with 42 for reproducible
    random number generation in tests.

    Usage:
        def test_something(seeded_random):
            # Use the seeded generator
            data = seeded_random.standard_normal(100)
            # ... rest of test
    """
    rng = np.random.default_rng(42)
    return rng
