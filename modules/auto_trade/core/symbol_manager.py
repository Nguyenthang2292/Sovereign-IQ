"""
Symbol Manager Module

Responsible for:
- Loading available symbols from DataFetcher
- Filtering symbols based on configuration (whitelist, blacklist)
- Random sampling of symbols
- Caching symbol lists
"""

import random
from typing import List, Optional

from modules.common.core.data_fetcher import DataFetcher
from modules.common.ui.logging import log_info, log_warn


class SymbolManager:
    """Manages the list of trading pairs."""

    def __init__(
        self,
        data_fetcher: DataFetcher,
        whitelist: Optional[List[str]] = None,
        blacklist: Optional[List[str]] = None,
        max_symbols: int = 100,  # Default limit for top volume symbols
        random_seed: Optional[int] = None,  # For reproducible sampling in tests
    ):
        """
        Initialize SymbolManager.

        Args:
            data_fetcher: DataFetcher instance
            whitelist: Optional list of symbols to specifically include (only these will be traded)
            blacklist: Optional list of symbols to exclude
            max_symbols: Maximum number of symbols to fetch from exchange (sorted by volume, must be > 0)
            random_seed: Optional random seed for reproducible sampling (useful for testing)
        """
        # Input validation
        if max_symbols <= 0:
            raise ValueError(f"max_symbols must be positive, got {max_symbols}")
        if max_symbols > 10000:
            log_warn(f"max_symbols is very large ({max_symbols}), this may impact performance")

        self.data_fetcher = data_fetcher
        self.whitelist = set(whitelist) if whitelist else set()
        self.blacklist = set(blacklist) if blacklist else set()
        self.max_symbols = max_symbols
        self._cached_symbols: List[str] = []  # Volume-sorted (descending)
        self._random = random.Random(random_seed) if random_seed is not None else random

    def refresh_symbols(self) -> None:
        """Fetch fresh list of symbols from exchange."""
        log_info("Refreshing symbol list...")

        # Use DataFetcher's symbol discovery
        # This returns symbols sorted by volume descending
        # Note: blacklist already handled by exclude_symbols parameter
        filtered_symbols = self.data_fetcher.symbol_discovery.list_binance_futures_symbols(
            exclude_symbols=self.blacklist, max_candidates=self.max_symbols, progress_label="Refreshing Symbols"
        )

        # Whitelist logic: Only trade whitelisted symbols if whitelist is provided.
        # Preserve volume-sorted order from filtered_symbols.
        if self.whitelist:
            whitelist_active = [s for s in filtered_symbols if s in self.whitelist]

            # Check for missing whitelist symbols (not in top volume list)
            missing_whitelist = self.whitelist - set(whitelist_active)
            if missing_whitelist:
                log_warn(f"Whitelist symbols not in top {self.max_symbols} volume: {missing_whitelist}")

            if not whitelist_active:
                log_warn("No whitelist symbols found in active symbols. Consider increasing max_symbols.")

            self._cached_symbols = whitelist_active
        else:
            self._cached_symbols = filtered_symbols

        log_info(f"SymbolManager: Loaded {len(self._cached_symbols)} symbols.")

    def get_symbols(self, sample_percent: float = 100.0) -> List[str]:
        """
        Get a list of symbols, optionally sampled randomly.

        Args:
            sample_percent: Percentage of symbols to return (0.0 to 100.0)

        Returns:
            List of symbol strings.

        Raises:
            ValueError: If sample_percent is outside the 0-100 range
        """
        if not self._cached_symbols:
            self.refresh_symbols()

        if not self._cached_symbols:
            log_warn("SymbolManager: No symbols available.")
            return []

        # Validate sample_percent range
        if not 0.0 <= sample_percent <= 100.0:
            raise ValueError(f"sample_percent must be 0-100, got {sample_percent}")

        if sample_percent <= 0.0:
            return []

        if sample_percent >= 100.0:
            return self._cached_symbols.copy()

        # Calculate sample size with min constraint to avoid exceeding list length
        count = max(1, int(len(self._cached_symbols) * sample_percent / 100.0))
        count = min(count, len(self._cached_symbols))

        return self._random.sample(self._cached_symbols, count)

    def get_all_cached_symbols(self) -> List[str]:
        """Return all currently cached symbols."""
        return self._cached_symbols.copy()
