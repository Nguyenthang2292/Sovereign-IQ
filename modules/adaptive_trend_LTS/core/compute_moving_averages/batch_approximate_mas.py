"""Batch approximate moving averages scanner for multi-symbol processing.

This module provides BatchApproximateMAScanner class that efficiently
calculates approximate moving averages for multiple symbols in parallel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple, Literal

import pandas as pd

try:
    from modules.common.utils import log_debug, log_info, log_warn, log_error
except ImportError:

    def log_debug(msg: str) -> None:
        print(f"[DEBUG] {msg}")

    def log_info(msg: str) -> None:
        print(f"[INFO] {msg}")

    def log_warn(msg: str) -> None:
        print(f"[WARN] {msg}")

    def log_error(msg: str) -> None:
        print(f"[ERROR] {msg}")


from .approximate_mas import (
    fast_ema_approx,
    fast_hma_approx,
    fast_wma_approx,
    fast_dema_approx,
    fast_lsma_approx,
    fast_kama_approx,
)

from .adaptive_approximate_mas import get_adaptive_ma_approx


class BatchApproximateMAScanner:
    """Batch scanner for calculating approximate moving averages across multiple symbols.

    Supports all 6 MA types with fast approximate calculations.
    Optional adaptive tolerance mode for volatility-based precision adjustment.

    Usage:
        scanner = BatchApproximateMAScanner(use_adaptive=False, num_threads=4)
        scanner.add_symbol("BTCUSDT", btc_prices)
        scanner.add_symbol("ETHUSDT", eth_prices)
        emas = scanner.calculate_all("EMA", length=20)
        results = scanner.get_all_results()
    """

    def __init__(
        self,
        use_adaptive: bool = False,
        num_threads: int = 4,
        volatility_window: int = 20,
        base_tolerance: float = 0.05,
        volatility_factor: float = 1.0,
    ):
        """Initialize batch approximate MA scanner.

        Args:
            use_adaptive: Enable adaptive tolerance mode (default: False)
            num_threads: Number of threads for parallel processing (default: 4)
            volatility_window: Window for volatility calculation in adaptive mode (default: 20)
            base_tolerance: Base tolerance level for adaptive mode (default: 0.05)
            volatility_factor: Volatility multiplier for adaptive mode (default: 1.0)
        """
        self.use_adaptive = use_adaptive
        self.num_threads = num_threads
        self.volatility_window = volatility_window
        self.base_tolerance = base_tolerance
        self.volatility_factor = volatility_factor

        self.symbols: Dict[str, pd.Series] = {}
        self.results: Dict[str, Dict[Tuple[str, int], pd.Series]] = {}

    def add_symbol(self, symbol: str, prices: pd.Series) -> None:
        """Add a symbol with historical price data.

        Args:
            symbol: Symbol identifier (e.g., "BTCUSDT")
            prices: Historical price series
        """
        if symbol in self.symbols:
            log_warn(f"Symbol {symbol} already exists. Replacing.")
            del self.symbols[symbol]

        if len(prices) == 0:
            log_warn(f"Empty price series for symbol {symbol}")
            return

        self.symbols[symbol] = prices.copy()
        log_debug(f"Added symbol {symbol} with {len(prices)} prices")

    def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from the scanner.

        Args:
            symbol: Symbol identifier

        Returns:
            True if symbol was removed, False if not found
        """
        if symbol in self.symbols:
            del self.symbols[symbol]
            if symbol in self.results:
                del self.results[symbol]
            log_debug(f"Removed symbol {symbol}")
            return True
        log_warn(f"Symbol {symbol} not found")
        return False

    def _calculate_single_approx(self, symbol: str, ma_type: str, length: int) -> Optional[pd.Series]:
        """Calculate approximate MA for a single symbol.

        Args:
            symbol: Symbol identifier
            ma_type: MA type (EMA, HMA, WMA, DEMA, LSMA, KAMA)
            length: MA length

        Returns:
            Approximate MA series, or None if calculation fails
        """
        if symbol not in self.symbols:
            log_warn(f"Symbol {symbol} not found")
            return None

        prices = self.symbols[symbol]

        try:
            if self.use_adaptive:
                result = get_adaptive_ma_approx(
                    ma_type=ma_type,
                    prices=prices,
                    length=length,
                    volatility_window=self.volatility_window,
                    base_tolerance=self.base_tolerance,
                    volatility_factor=self.volatility_factor,
                )
            else:
                if ma_type == "EMA":
                    result = fast_ema_approx(prices, length)
                elif ma_type == "HMA":
                    result = fast_hma_approx(prices, length)
                elif ma_type == "WMA":
                    result = fast_wma_approx(prices, length)
                elif ma_type == "DEMA":
                    result = fast_dema_approx(prices, length)
                elif ma_type == "LSMA":
                    result = fast_lsma_approx(prices, length)
                elif ma_type == "KAMA":
                    result = fast_kama_approx(prices, length)
                else:
                    log_error(f"Unknown MA type: {ma_type}")
                    return None

            return result
        except Exception as e:
            log_error(f"Error calculating {ma_type}{length} for {symbol}: {e}")
            return None

    def calculate_all(self, ma_type: str, length: int, use_parallel: bool = True) -> Dict[str, pd.Series]:
        """Calculate approximate MA for all symbols.

        Args:
            ma_type: MA type (EMA, HMA, WMA, DEMA, LSMA, KAMA)
            length: MA length
            use_parallel: Use parallel processing (default: True)

        Returns:
            Dictionary mapping symbol to approximate MA series
        """
        results = {}

        if use_parallel and len(self.symbols) > 1:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self._calculate_single_approx, symbol, ma_type, length): symbol
                    for symbol in self.symbols
                }
                for future in futures:
                    symbol = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[symbol] = result
                            self._store_result(symbol, ma_type, length, result)
                    except Exception as e:
                        log_error(f"Error processing {symbol}: {e}")
        else:
            for symbol in self.symbols:
                result = self._calculate_single_approx(symbol, ma_type, length)
                if result is not None:
                    results[symbol] = result
                    self._store_result(symbol, ma_type, length, result)

        log_info(f"Calculated {ma_type}{length} for {len(results)}/{len(self.symbols)} symbols")
        return results

    def calculate_symbol(self, symbol: str, ma_type: str, length: int) -> Optional[pd.Series]:
        """Calculate approximate MA for a single symbol.

        Args:
            symbol: Symbol identifier
            ma_type: MA type (EMA, HMA, WMA, DEMA, LSMA, KAMA)
            length: MA length

        Returns:
            Approximate MA series, or None if symbol not found or calculation fails
        """
        result = self._calculate_single_approx(symbol, ma_type, length)
        if result is not None:
            self._store_result(symbol, ma_type, length, result)
        return result

    def _store_result(self, symbol: str, ma_type: str, length: int, result: pd.Series) -> None:
        """Store calculation result in results cache.

        Args:
            symbol: Symbol identifier
            ma_type: MA type
            length: MA length
            result: Approximate MA series
        """
        if symbol not in self.results:
            self.results[symbol] = {}
        self.results[symbol][(ma_type, length)] = result

    def get_all_results(self) -> Dict[str, Dict[Tuple[str, int], pd.Series]]:
        """Get all cached results.

        Returns:
            Dictionary mapping symbol to dictionary of (ma_type, length) -> MA series
        """
        return self.results

    def get_symbol_results(self, symbol: str) -> Optional[Dict[Tuple[str, int], pd.Series]]:
        """Get all cached results for a specific symbol.

        Args:
            symbol: Symbol identifier

        Returns:
            Dictionary of (ma_type, length) -> MA series, or None if symbol not found
        """
        return self.results.get(symbol)

    def get_symbol_result(self, symbol: str, ma_type: str, length: int) -> Optional[pd.Series]:
        """Get cached result for a specific symbol and MA type/length.

        Args:
            symbol: Symbol identifier
            ma_type: MA type
            length: MA length

        Returns:
            Approximate MA series, or None if not found
        """
        symbol_results = self.results.get(symbol)
        if symbol_results is None:
            return None
        return symbol_results.get((ma_type, length))

    def reset(self) -> None:
        """Reset all cached results."""
        self.results.clear()
        log_debug("Reset all cached results")

    def get_symbol_count(self) -> int:
        """Get the number of symbols in the scanner.

        Returns:
            Number of symbols
        """
        return len(self.symbols)

    def get_symbols(self) -> list[str]:
        """Get list of all symbols in the scanner.

        Returns:
            List of symbol identifiers
        """
        return list(self.symbols.keys())

    def calculate_set_of_mas(
        self,
        ma_type: str,
        base_length: int,
        robustness: str = "Medium",
        use_parallel: bool = True,
    ) -> Optional[Dict[str, Tuple[pd.Series, ...]]]:
        """Calculate a set of 9 MAs for all symbols.

        The set includes:
        1. The primary MA with the specified length.
        2. 4 MAs with lengths increasing from the primary length (MA1-MA4).
        3. 4 MAs with lengths decreasing from the primary length (MA_1-MA_4).

        Args:
            ma_type: MA type (EMA, HMA, WMA, DEMA, LSMA, KAMA)
            base_length: Base length for the moving average
            robustness: "Narrow", "Medium", or "Wide" (default: "Medium")
            use_parallel: Use parallel processing (default: True)

        Returns:
            Dictionary mapping symbol to tuple of 9 MAs, or None if calculation fails
        """
        try:
            from modules.adaptive_trend_enhance.utils import diflen

            L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen(base_length, robustness=robustness)
            ma_lengths = [base_length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]
            ma_names = ["MA", "MA1", "MA2", "MA3", "MA4", "MA_1", "MA_2", "MA_3", "MA_4"]

            results = {}

            if use_parallel and len(self.symbols) > 1:
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    futures = []
                    for symbol in self.symbols:
                        for length in ma_lengths:
                            future = executor.submit(self._calculate_single_approx, symbol, ma_type, length)
                            futures.append((future, symbol, length))

                    for future, symbol, length in futures:
                        try:
                            result = future.result()
                            if result is not None:
                                self._store_result(symbol, ma_type, length, result)
                        except Exception as e:
                            log_error(f"Error processing {symbol} for {ma_type}{length}: {e}")
            else:
                for symbol in self.symbols:
                    for length in ma_lengths:
                        result = self._calculate_single_approx(symbol, ma_type, length)
                        if result is not None:
                            self._store_result(symbol, ma_type, length, result)

            for symbol in self.symbols:
                mas = []
                symbol_results = self.results.get(symbol, {})
                all_valid = True
                for length in ma_lengths:
                    ma = symbol_results.get((ma_type, length))
                    if ma is None:
                        all_valid = False
                        break
                    mas.append(ma)

                if all_valid:
                    results[symbol] = tuple(mas)
                else:
                    log_warn(f"Incomplete MA set for {symbol}, skipping")

            log_info(f"Calculated MA sets for {len(results)}/{len(self.symbols)} symbols")
            return results if results else None

        except ImportError:
            log_error("Failed to import diflen utility")
            return None
        except Exception as e:
            log_error(f"Error calculating MA sets: {e}")
            return None


__all__ = ["BatchApproximateMAScanner"]
