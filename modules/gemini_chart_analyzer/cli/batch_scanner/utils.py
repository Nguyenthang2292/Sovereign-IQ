"""
Utilities for batch scanner CLI.
"""

from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.ui.logging import log_info


def init_components():
    """Initialize ExchangeManager and DataFetcher."""
    log_info("Initializing ExchangeManager and DataFetcher...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    return exchange_manager, data_fetcher
