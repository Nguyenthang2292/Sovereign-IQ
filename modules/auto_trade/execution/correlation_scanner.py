"""
Correlation Scanner Module

Scans for hedge candidates based on correlation analysis.
Finds the best hedge symbol for a given signal symbol.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, cast

import numpy as np
import pandas as pd

from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.quantitative_metrics.hedge_ratios.kalman_hedge_ratio import (
    calculate_kalman_hedge_ratio,
)
from modules.common.quantitative_metrics.hedge_ratios.ols_hedge_ratio import (
    calculate_ols_hedge_ratio,
)
from modules.common.quantitative_metrics.statistical_tests.correlation import (
    calculate_correlation,
)
from modules.common.ui.logging import log_error, log_info, log_warn


@dataclass
class HedgeCandidate:
    """Hedge candidate result."""

    symbol: str
    correlation: float
    hedge_ratio: float
    kalman_hedge_ratio: Optional[float]
    score: float
    regime: Literal["STAT_ARB", "MOMENTUM", "BLENDED"] = "STAT_ARB"


class CorrelationScanner:
    """
    Scans for hedge candidates based on correlation analysis.

    Features:
    - Correlation caching with TTL-based refresh
    - Filtering by minimum correlation threshold
    - Ranking by correlation strength
    - Reuses existing pairs_trading metrics
    """

    DEFAULT_MIN_CORRELATION = 0.65
    DEFAULT_LOOKBACK = 100
    DEFAULT_TIMEFRAME = "1h"
    DEFAULT_REFRESH_INTERVAL = 7200  # 2 hours in seconds

    def __init__(
        self,
        data_fetcher: Optional[DataFetcher] = None,
        min_correlation: float = DEFAULT_MIN_CORRELATION,
        lookback: int = DEFAULT_LOOKBACK,
        timeframe: str = DEFAULT_TIMEFRAME,
        refresh_interval: int = DEFAULT_REFRESH_INTERVAL,
        cache_ttl: Optional[int] = None,
    ):
        """
        Initialize CorrelationScanner.

        Args:
            data_fetcher: DataFetcher instance (will create if not provided)
            min_correlation: Minimum correlation threshold (0.50-0.90)
            lookback: Number of candles for correlation calculation
            timeframe: Timeframe for OHLCV data
            refresh_interval: Cache refresh interval in seconds
            cache_ttl: Deprecated, use refresh_interval
        """
        if not (0.50 <= min_correlation <= 0.90):
            raise ValueError(f"min_correlation must be between 0.50 and 0.90, got {min_correlation}")

        self.min_correlation = min_correlation
        self.lookback = lookback
        self.timeframe = timeframe
        self.refresh_interval = cache_ttl if cache_ttl is not None else refresh_interval

        self._data_fetcher = data_fetcher
        self._cache: Dict[str, Dict] = {}
        self._last_cache_refresh: Optional[float] = None

    def _get_data_fetcher(self) -> DataFetcher:
        """Get or create DataFetcher instance."""
        if self._data_fetcher is None:
            exchange_manager = ExchangeManager()
            self._data_fetcher = DataFetcher(exchange_manager=exchange_manager)
        return self._data_fetcher

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol formats like BTC-PERP/BTCPERP into BTC/USDT."""
        normalized = symbol.upper().strip()

        if normalized.endswith("-PERP"):
            return f"{normalized[:-5]}/USDT"

        if normalized.endswith("PERP"):
            base = normalized[:-4]
            if base.endswith("-"):
                base = base[:-1]
            return f"{base}/USDT"

        return normalized

    def _make_cache_key(self, symbol1: str, symbol2: str) -> str:
        """Generate cache key for a symbol pair."""
        return f"{self._normalize_symbol(symbol1)}:{self._normalize_symbol(symbol2)}"

    def _is_cache_entry_valid(self, cache_key: str) -> bool:
        """Check if a generic cache entry is still valid."""
        entry = self._cache.get(cache_key)
        if not entry:
            return False
        cached_time = entry.get("cached_at", 0)
        return (time.time() - cached_time) < self.refresh_interval

    def _is_cache_valid(self, symbol1: str, symbol2: str) -> bool:
        """Check if cache entry is still valid."""
        cache_key = self._make_cache_key(symbol1, symbol2)
        return self._is_cache_entry_valid(cache_key)

    def _should_refresh_full_cache(self) -> bool:
        """Check if full cache refresh is needed."""
        if self._last_cache_refresh is None:
            return True
        return (time.time() - self._last_cache_refresh) >= self.refresh_interval

    def calculate_correlation(self, symbol1: str, symbol2: str, lookback: Optional[int] = None) -> Optional[float]:
        """
        Calculate correlation between two symbols.

        Args:
            symbol1: First symbol (e.g., "BTC/USDT")
            symbol2: Second symbol (e.g., "ETH/USDT")
            lookback: Number of candles (uses default if not provided)

        Returns:
            Correlation coefficient (0-1), or None if calculation fails
        """
        symbol1 = self._normalize_symbol(symbol1)
        symbol2 = self._normalize_symbol(symbol2)
        cache_key = self._make_cache_key(symbol1, symbol2)

        if self._is_cache_valid(symbol1, symbol2):
            return self._cache[cache_key].get("correlation")

        try:
            df1 = self._get_data_fetcher().fetch_ohlcv(
                symbol=symbol1,
                timeframe=self.timeframe,
                limit=lookback or self.lookback,
            )
            df2 = self._get_data_fetcher().fetch_ohlcv(
                symbol=symbol2,
                timeframe=self.timeframe,
                limit=lookback or self.lookback,
            )

            if df1 is None or df1.empty or df2 is None or df2.empty:
                log_warn(f"[CorrelationScanner] Failed to fetch data for {symbol1} or {symbol2}")
                return None

            price1 = df1["close"]
            price2 = df2["close"]

            if not isinstance(price1, pd.Series) or not isinstance(price2, pd.Series):
                log_warn(f"[CorrelationScanner] Invalid price data type for {symbol1} or {symbol2}")
                return None

            correlation = calculate_correlation(
                price1,
                price2,
                min_points=min(20, (lookback or self.lookback) // 2),
            )

            if correlation is not None:
                self._cache[cache_key] = {
                    "correlation": correlation,
                    "cached_at": time.time(),
                }
                self._last_cache_refresh = time.time()

            return correlation

        except Exception as e:
            log_error(f"[CorrelationScanner] Error calculating correlation: {e}")
            return None

    def calculate_hedge_ratio(
        self,
        symbol1: str,
        symbol2: str,
        method: Literal["OLS", "KALMAN"] = "OLS",
        lookback: Optional[int] = None,
    ) -> Optional[float]:
        """
        Calculate hedge ratio between two symbols.

        Args:
            symbol1: First symbol (e.g., "BTC/USDT")
            symbol2: Second symbol (e.g., "ETH/USDT")
            method: Hedge ratio calculation method ("OLS" or "KALMAN")
            lookback: Number of candles (uses default if not provided)

        Returns:
            Hedge ratio, or None if calculation fails
        """
        symbol1 = self._normalize_symbol(symbol1)
        symbol2 = self._normalize_symbol(symbol2)
        cache_key = self._make_cache_key(symbol1, symbol2)
        cache_key_with_method = f"{cache_key}:{method}"

        if self._is_cache_entry_valid(cache_key_with_method):
            entry = self._cache.get(cache_key_with_method)
            if entry:
                return entry.get("hedge_ratio")

        try:
            df1 = self._get_data_fetcher().fetch_ohlcv(
                symbol=symbol1,
                timeframe=self.timeframe,
                limit=lookback or self.lookback,
            )
            df2 = self._get_data_fetcher().fetch_ohlcv(
                symbol=symbol2,
                timeframe=self.timeframe,
                limit=lookback or self.lookback,
            )

            if df1 is None or df1.empty or df2 is None or df2.empty:
                return None

            price1: pd.Series = cast(pd.Series, df1["close"])
            price2: pd.Series = cast(pd.Series, df2["close"])

            hedge_ratio: Optional[float] = None

            if method == "OLS":
                hedge_ratio = calculate_ols_hedge_ratio(price1, price2)  # type: ignore[arg-type]
            elif method == "KALMAN":
                hedge_ratio = calculate_kalman_hedge_ratio(price1, price2)  # type: ignore[arg-type]

            if hedge_ratio is not None and not np.isnan(hedge_ratio):
                if cache_key_with_method not in self._cache:
                    self._cache[cache_key_with_method] = {}
                self._cache[cache_key_with_method].update(
                    {
                        "hedge_ratio": hedge_ratio,
                        "cached_at": time.time(),
                    }
                )

            return hedge_ratio

        except Exception as e:
            log_error(f"[CorrelationScanner] Error calculating hedge ratio: {e}")
            return None

    def scan_hedge_candidates(
        self,
        signal_symbol: str,
        candidate_symbols: Optional[List[str]] = None,
        max_candidates: int = 10,
    ) -> List[HedgeCandidate]:
        """
        Scan for hedge candidates for a given signal symbol.

        Args:
            signal_symbol: The signal symbol to find hedges for
            candidate_symbols: List of candidate symbols to scan
                             (if None, uses default pool)
            max_candidates: Maximum number of candidates to return

        Returns:
            List of HedgeCandidate sorted by score (descending)
        """
        if candidate_symbols is None:
            candidate_symbols = self._get_default_symbol_pool(signal_symbol)

        candidates: List[HedgeCandidate] = []
        signal_symbol = self._normalize_symbol(signal_symbol)

        for candidate in candidate_symbols:
            candidate = self._normalize_symbol(candidate)

            if candidate == signal_symbol:
                continue

            try:
                correlation = self.calculate_correlation(signal_symbol, candidate)

                if correlation is None or correlation < self.min_correlation:
                    continue

                hedge_ratio = self.calculate_hedge_ratio(signal_symbol, candidate, method="OLS")
                kalman_hedge_ratio = self.calculate_hedge_ratio(signal_symbol, candidate, method="KALMAN")

                if hedge_ratio is None:
                    continue

                score = correlation * abs(hedge_ratio)

                candidates.append(
                    HedgeCandidate(
                        symbol=candidate,
                        correlation=correlation,
                        hedge_ratio=hedge_ratio,
                        kalman_hedge_ratio=kalman_hedge_ratio,
                        score=score,
                    )
                )

            except Exception as e:
                log_warn(f"[CorrelationScanner] Error scanning {candidate}: {e}")
                continue

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:max_candidates]

    def _get_default_symbol_pool(self, exclude_symbol: str) -> List[str]:
        """Get default symbol pool for scanning."""
        exclude_symbol = self._normalize_symbol(exclude_symbol)
        all_symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "DOGE/USDT",
            "AVAX/USDT",
            "DOT/USDT",
            "MATIC/USDT",
            "LINK/USDT",
            "LTC/USDT",
            "UNI/USDT",
            "ATOM/USDT",
            "XLM/USDT",
            "ETC/USDT",
            "FIL/USDT",
            "HBAR/USDT",
            "APT/USDT",
            "NEAR/USDT",
        ]
        return [s for s in all_symbols if s != exclude_symbol]

    def refresh_correlation_cache(self, symbols: Optional[List[str]] = None) -> None:
        """
        Force refresh correlation cache.

        Args:
            symbols: Specific symbols to refresh (if None, refreshes all cached)
        """
        log_info("[CorrelationScanner] Refreshing correlation cache")

        if symbols is None:
            self._cache.clear()
            self._last_cache_refresh = None
            return

        for symbol in symbols:
            keys_to_remove = [k for k in self._cache.keys() if symbol in k]
            for key in keys_to_remove:
                del self._cache[key]

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "last_refresh": datetime.fromtimestamp(self._last_cache_refresh, tz=timezone.utc).isoformat()
            if self._last_cache_refresh
            else None,
            "refresh_interval": self.refresh_interval,
        }

    def calculate_adx_for_regime(
        self, symbol1: str, symbol2: str, adx_low: float = 20, adx_high: float = 30
    ) -> Optional[Literal["STAT_ARB", "MOMENTUM", "BLENDED"]]:
        """
        Calculate ADX to determine trading regime.

        Args:
            symbol1: First symbol
            symbol2: Second symbol

        Returns:
            Regime based on ADX: STAT_ARB (<20), BLENDED (20-30), MOMENTUM (>=30)
        """
        try:
            if adx_low >= adx_high:
                log_warn(
                    f"[CorrelationScanner] Invalid ADX thresholds (low={adx_low}, high={adx_high}), using defaults."
                )
                adx_low, adx_high = 20, 30

            symbol1 = self._normalize_symbol(symbol1)
            symbol2 = self._normalize_symbol(symbol2)

            df1 = self._get_data_fetcher().fetch_ohlcv(symbol=symbol1, timeframe=self.timeframe, limit=self.lookback)
            df2 = self._get_data_fetcher().fetch_ohlcv(symbol=symbol2, timeframe=self.timeframe, limit=self.lookback)

            if df1 is None or df1.empty or df2 is None or df2.empty:
                return None

            high1: pd.Series = cast(pd.Series, df1["high"])
            low1: pd.Series = cast(pd.Series, df1["low"])
            close1: pd.Series = cast(pd.Series, df1["close"])
            high2: pd.Series = cast(pd.Series, df2["high"])
            low2: pd.Series = cast(pd.Series, df2["low"])
            close2: pd.Series = cast(pd.Series, df2["close"])

            atr1 = self._calculate_atr(high1, low1, close1)  # type: ignore[arg-type]
            atr2 = self._calculate_atr(high2, low2, close2)  # type: ignore[arg-type]

            avg_atr = (atr1 + atr2) / 2
            avg_price = (close1.iloc[-1] + close2.iloc[-1]) / 2  # type: ignore[index]

            adx_approx = (avg_atr / avg_price) * 100 * 14

            if adx_approx < adx_low:
                return "STAT_ARB"
            elif adx_approx < adx_high:
                return "BLENDED"
            else:
                return "MOMENTUM"

        except Exception as e:
            log_warn(f"[CorrelationScanner] Error calculating ADX: {e}")
            return None

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range."""
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]  # type: ignore[index]

        return float(atr) if not pd.isna(atr) else 0.0
