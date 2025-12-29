import warnings
import pytest
import sys

# Suppress warnings from pytorch_forecasting about non-writable NumPy arrays
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")

# Workaround for pytest capture bug on Windows with Python 3.12+
# This prevents the "I/O operation on closed file" error
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configure pytest to avoid capture issues on Windows."""
    
    # Ensure capture is disabled if not already set
    if hasattr(config.option, 'capture') and config.option.capture == 'no':
        # Set up safe stderr handling
        if hasattr(sys, 'stderr') and sys.stderr is not None:
            try:
                sys.stderr.flush()
            except (ValueError, OSError):
                pass


@pytest.fixture
def config_factory():
    """Factory fixture for creating Config instances for testing."""
    def _create_config(**kwargs):
        """Create a Config instance with specified attributes."""
        class Config:
            pass
        
        config = Config()
        # Set default values for all possible attributes
        config.timeframe = kwargs.get('timeframe', None)
        config.no_menu = kwargs.get('no_menu', False)
        config.enable_spc = kwargs.get('enable_spc', False)
        config.spc_k = kwargs.get('spc_k', None)
        config.enable_xgboost = kwargs.get('enable_xgboost', False)
        config.enable_hmm = kwargs.get('enable_hmm', False)
        config.enable_random_forest = kwargs.get('enable_random_forest', False)
        config.use_decision_matrix = kwargs.get('use_decision_matrix', None)
        config.spc_strategy = kwargs.get('spc_strategy', None)
        config.limit = kwargs.get('limit', None)
        config.max_workers = kwargs.get('max_workers', None)
        
        return config
    
    return _create_config

