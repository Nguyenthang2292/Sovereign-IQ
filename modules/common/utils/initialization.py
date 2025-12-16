"""
Component initialization utilities.
"""

from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.common.core.exchange_manager import ExchangeManager
    from modules.common.core.data_fetcher import DataFetcher


def initialize_components() -> Tuple['ExchangeManager', 'DataFetcher']:
    """
    Initialize ExchangeManager and DataFetcher components.
    
    This function creates and returns the core components needed for data fetching
    across different main entry points.
    
    Returns:
        Tuple containing (ExchangeManager, DataFetcher) instances
    """
    # Import here to avoid circular import
    from modules.common.core.exchange_manager import ExchangeManager
    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.ui.logging import log_progress
    
    log_progress("Initializing components...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    return exchange_manager, data_fetcher

