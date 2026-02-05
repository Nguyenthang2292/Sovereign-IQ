"""
Symbol Manager Module

Responsible for:
- Loading available symbols from DataFetcher
- Filtering symbols based on configuration (whitelist, blacklist)
- Random sampling of symbols
- Caching symbol lists

Example:
    >>> from modules.common.core.data_fetcher import DataFetcher
    >>> data_fetcher = DataFetcher()
    >>> manager = SymbolManager(
    ...     data_fetcher=data_fetcher,
    ...     whitelist=["BTC/USDT", "ETH/USDT"],
    ...     max_symbols=50
    ... )
    >>> symbols = manager.get_symbols(sample_percent=25.0)
"""

import random
from typing import List, Optional

from modules.common.core.data_fetcher import DataFetcher
from modules.common.ui.logging import log_info, log_warn


class SymbolManager:
    """Manages the list of trading pairs."""

    _REFRESH_PROGRESS_LABEL = "Refreshing Symbols"

    def __init__(
        self,
        data_fetcher: DataFetcher,
        whitelist: Optional[List[str]] = None,
        blacklist: Optional[List[str]] = None,
        max_symbols: int = 100,  # Default limit for top volume symbols
        random_seed: Optional[int] = None,  # For reproducible sampling in tests
        sample_percentage: float = 100.0,  # Percentage of symbols to sample
        sampling_strategy: str = "random",  # Sampling strategy: random, stratified, volume_weighted, etc.
    ):
        """
        Initialize SymbolManager.

        Args:
            data_fetcher: DataFetcher instance
            whitelist: Optional list of symbols to specifically include (only these will be traded)
            blacklist: Optional list of symbols to exclude
            max_symbols: Maximum number of symbols to fetch from exchange (sorted by volume, must be > 0)
            random_seed: Optional random seed for reproducible sampling (useful for testing)
            sample_percentage: Percentage of symbols to sample (0.0 to 100.0)
            sampling_strategy: Sampling strategy - random, stratified, volume_weighted, top_n_hybrid, systematic, liquidity_weighted
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
        self._random = random.Random(random_seed)  # Always use Random instance for consistency

        # Sampling configuration
        self.sample_percentage = sample_percentage
        self.sampling_strategy = sampling_strategy

    def refresh_symbols(self) -> None:
        """Fetch fresh list of symbols from exchange."""
        log_info("Refreshing symbol list...")

        # Use DataFetcher's symbol discovery
        # This returns symbols sorted by volume descending
        # Note: blacklist already handled by exclude_symbols parameter
        filtered_symbols = self.data_fetcher.list_binance_futures_symbols(
            exclude_symbols=self.blacklist,
            max_candidates=self.max_symbols,
            progress_label=self._REFRESH_PROGRESS_LABEL,
        )

        # Whitelist logic: Only trade whitelisted symbols if whitelist is provided.
        # Preserve volume-sorted order from filtered_symbols.
        if self.whitelist:
            whitelist_active = [s for s in filtered_symbols if s in self.whitelist]

            # Check for missing whitelist symbols (not in top volume list)
            missing_whitelist = self.whitelist - set(whitelist_active)
            if missing_whitelist:
                log_warn(
                    f"Whitelist symbols not in top {self.max_symbols} volume: {missing_whitelist}. "
                    f"Consider increasing max_symbols or removing these from whitelist."
                )

            if not whitelist_active:
                log_warn("No whitelist symbols found in active symbols. Consider increasing max_symbols.")

            self._cached_symbols = whitelist_active
        else:
            self._cached_symbols = filtered_symbols

        log_info(f"SymbolManager: Loaded {len(self._cached_symbols)} symbols.")

    def get_symbols(self, sample_percent: float = None) -> List[str]:
        """
        Get a list of symbols, optionally sampled using configured strategy.

        Args:
            sample_percent: Percentage of symbols to return (0.0 to 100.0).
                          If None, uses self.sample_percentage.

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

        # Use provided sample_percent or fall back to instance config
        if sample_percent is None:
            sample_percent = self.sample_percentage

        # Validate sample_percent range
        if not 0.0 <= sample_percent <= 100.0:
            raise ValueError(f"sample_percent must be 0-100, got {sample_percent}")

        if sample_percent <= 0.0:
            return []

        if sample_percent >= 100.0:
            return self._cached_symbols.copy()

        # Use configured sampling strategy
        if self.sampling_strategy and self.sampling_strategy != "random":
            try:
                from modules.auto_trade.core.scanner_sampling import sample_symbols

                return sample_symbols(
                    all_symbols=self._cached_symbols,
                    sample_percentage=sample_percent,
                    strategy=self.sampling_strategy,
                    data_fetcher=self.data_fetcher,
                )
            except Exception as e:
                log_warn(f"Sampling strategy '{self.sampling_strategy}' failed: {e}. Falling back to random.")
                # Fall through to random sampling

        # Default: random sampling
        count = max(1, round(len(self._cached_symbols) * sample_percent / 100.0))
        count = min(count, len(self._cached_symbols))
        return self._random.sample(self._cached_symbols, count)

    def get_all_cached_symbols(self) -> List[str]:
        """Return all currently cached symbols."""
        return self._cached_symbols.copy()
