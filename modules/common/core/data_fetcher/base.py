"""Base DataFetcher infrastructure with core functionality."""

from typing import Dict, Optional, Tuple

import pandas as pd

from modules.common.core.exchange_manager import ExchangeManager


class DataFetcherBase:
    """Base class for DataFetcher with core infrastructure."""

    def __init__(self, exchange_manager: ExchangeManager, shutdown_event=None):
        """
        Initialize DataFetcher base.

        Args:
            exchange_manager: ExchangeManager instance for exchange operations
            shutdown_event: Optional threading.Event for graceful shutdown
        """
        self.exchange_manager = exchange_manager
        self.shutdown_event = shutdown_event
        self._ohlcv_dataframe_cache: Dict[Tuple[str, str, int], Tuple[pd.DataFrame, Optional[str]]] = {}
        self.market_prices: Dict[str, float] = {}

    def should_stop(self) -> bool:
        """
        Check if shutdown was requested.

        Returns:
            True if shutdown event is set, False otherwise
        """
        if self.shutdown_event:
            return self.shutdown_event.is_set()
        return False
