"""
Risk calculator for portfolio risk metrics (PnL, Delta, Beta, VaR).

This module provides the PortfolioRiskCalculator class for calculating various
risk metrics including profit and loss (PnL), delta exposure, beta-weighted delta,
and Value at Risk (VaR) for cryptocurrency portfolios.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple, Any

try:
    from modules.common.models.position import Position
    from modules.common.utils import (
        log_warn,
        log_analysis,
        log_model,
        normalize_symbol,
    )
    from config import (
        DEFAULT_BETA_MIN_POINTS,
        DEFAULT_BETA_LIMIT,
        DEFAULT_BETA_TIMEFRAME,
        DEFAULT_VAR_CONFIDENCE,
        DEFAULT_VAR_LOOKBACK_DAYS,
        DEFAULT_VAR_MIN_HISTORY_DAYS,
        DEFAULT_VAR_MIN_PNL_SAMPLES,
        BENCHMARK_SYMBOL,
    )
except ImportError:
    Position = None
    log_warn = None
    log_analysis = None
    log_model = None
    normalize_symbol = None
    DEFAULT_BETA_MIN_POINTS = 50
    DEFAULT_BETA_LIMIT = 1000
    DEFAULT_BETA_TIMEFRAME = "1h"
    DEFAULT_VAR_CONFIDENCE = 0.95
    DEFAULT_VAR_LOOKBACK_DAYS = 90
    DEFAULT_VAR_MIN_HISTORY_DAYS = 20
    DEFAULT_VAR_MIN_PNL_SAMPLES = 10
    BENCHMARK_SYMBOL = "BTC/USDT"


class PortfolioRiskCalculator:
    """
    Calculates portfolio risk metrics including PnL, Delta, Beta, and VaR.

    This class provides methods to compute various risk metrics for a portfolio
    of cryptocurrency positions, including profit/loss calculations, delta exposure,
    beta-weighted delta, and historical simulation Value at Risk (VaR).

    Attributes:
        data_fetcher: Data fetcher instance for retrieving market data
        benchmark_symbol: Benchmark symbol for beta calculations (default: BTC/USDT)
        _beta_cache: Cache for computed beta values to avoid redundant calculations
        last_var_value: Last computed VaR value
        last_var_confidence: Confidence level used for last VaR calculation
    """

    def __init__(
        self, data_fetcher: Any, benchmark_symbol: str = BENCHMARK_SYMBOL
    ) -> None:
        """
        Initialize the PortfolioRiskCalculator.

        Args:
            data_fetcher: Data fetcher instance for retrieving OHLCV data
            benchmark_symbol: Benchmark symbol for beta calculations (default: BENCHMARK_SYMBOL)
        """
        self.data_fetcher = data_fetcher
        self.benchmark_symbol = benchmark_symbol
        self._beta_cache: Dict[str, float] = {}
        self.last_var_value: Optional[float] = None
        self.last_var_confidence: Optional[float] = None

    def calculate_stats(
        self, positions: List[Position], market_prices: Dict[str, float]
    ) -> Tuple[pd.DataFrame, float, float, float]:
        """
        Calculate PnL, simple delta, and beta-weighted delta for the portfolio.

        Args:
            positions: List of Position objects representing portfolio positions
            market_prices: Dictionary mapping symbol to current market price

        Returns:
            Tuple containing:
                - DataFrame with per-position statistics (Symbol, Direction, Entry, Current, Size, PnL, Delta, Beta, Beta Delta)
                - Total portfolio PnL in USDT
                - Total portfolio delta in USDT
                - Total portfolio beta-weighted delta in USDT

        Note:
            Positions without corresponding market prices are skipped.
            Beta-weighted delta is only calculated if beta is available for the symbol.
        """
        total_pnl = 0
        total_delta = 0
        total_beta_delta = 0

        results = []

        for p in positions:
            current_price = market_prices.get(p.symbol)
            if current_price is None:
                continue

            if p.direction == "LONG":
                pnl_pct = (current_price - p.entry_price) / p.entry_price
                delta = p.size_usdt
            else:
                pnl_pct = (p.entry_price - current_price) / p.entry_price
                delta = -p.size_usdt

            pnl_usdt = pnl_pct * p.size_usdt

            total_pnl += pnl_usdt
            total_delta += delta

            beta = self.calculate_beta(p.symbol)
            beta_delta = None
            if beta is not None:
                beta_delta = delta * beta
                total_beta_delta += beta_delta

            results.append(
                {
                    "Symbol": p.symbol,
                    "Direction": p.direction,
                    "Entry": p.entry_price,
                    "Current": current_price,
                    "Size": p.size_usdt,
                    "PnL": pnl_usdt,
                    "Delta": delta,
                    "Beta": beta,
                    "Beta Delta": beta_delta,
                }
            )

        return pd.DataFrame(results), total_pnl, total_delta, total_beta_delta

    def calculate_beta(
        self,
        symbol: str,
        benchmark_symbol: Optional[str] = None,
        min_points: int = DEFAULT_BETA_MIN_POINTS,
        limit: int = DEFAULT_BETA_LIMIT,
        timeframe: str = DEFAULT_BETA_TIMEFRAME,
    ) -> Optional[float]:
        """
        Calculate beta of a symbol versus a benchmark (default BTC/USDT).

        Beta measures the sensitivity of an asset's returns to benchmark returns.
        A beta of 1.0 means the asset moves in line with the benchmark.
        Beta > 1.0 indicates higher volatility than the benchmark.
        Beta < 1.0 indicates lower volatility than the benchmark.

        Args:
            symbol: Symbol to calculate beta for (e.g., 'ETH/USDT')
            benchmark_symbol: Benchmark symbol (default: self.benchmark_symbol)
            min_points: Minimum number of data points required for calculation
            limit: Maximum number of historical candles to fetch
            timeframe: Timeframe for historical data (e.g., '1h', '1d')

        Returns:
            Beta value (float) if calculation succeeds, None otherwise.

        Note:
            - Returns 1.0 if symbol equals benchmark symbol
            - Uses caching to avoid redundant calculations
            - Returns None if insufficient data or calculation fails
        """
        benchmark_symbol = benchmark_symbol or self.benchmark_symbol
        
        # Use normalize_symbol from utils if available, otherwise define fallback
        def normalize_symbol_fallback(user_input: str, quote: str = "USDT") -> str:
            """Fallback normalize_symbol function if import fails."""
            if not user_input:
                return f"BTC/{quote}"
            norm = user_input.strip().upper()
            if "/" in norm:
                return norm
            if norm.endswith(quote):
                return f"{norm[:-len(quote)]}/{quote}"
            return f"{norm}/{quote}"
        
        normalize_func = normalize_symbol if normalize_symbol is not None else normalize_symbol_fallback

        normalized_symbol = normalize_func(symbol)
        normalized_benchmark = normalize_func(benchmark_symbol)
        cache_key = f"{normalized_symbol}|{normalized_benchmark}|{timeframe}|{limit}"

        if normalized_symbol == normalized_benchmark:
            return 1.0

        if cache_key in self._beta_cache:
            return self._beta_cache[cache_key]

        asset_df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
            normalized_symbol, limit=limit, timeframe=timeframe
        )
        benchmark_df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
            normalized_benchmark, limit=limit, timeframe=timeframe
        )
        asset_series = self.data_fetcher.dataframe_to_close_series(asset_df)
        benchmark_series = self.data_fetcher.dataframe_to_close_series(benchmark_df)

        if asset_series is None or benchmark_series is None:
            return None

        df = pd.concat([asset_series, benchmark_series], axis=1, join="inner").dropna()
        if len(df) < min_points:
            return None

        returns = df.pct_change().dropna()
        if returns.empty:
            return None

        benchmark_var = returns.iloc[:, 1].var()
        if benchmark_var is None or benchmark_var <= 0:
            return None

        covariance = returns.iloc[:, 0].cov(returns.iloc[:, 1])
        if covariance is None:
            return None

        beta = covariance / benchmark_var
        if pd.isna(beta):
            return None

        self._beta_cache[cache_key] = beta
        return beta

    def calculate_portfolio_var(
        self,
        positions: List[Position],
        confidence: float = DEFAULT_VAR_CONFIDENCE,
        lookback_days: int = DEFAULT_VAR_LOOKBACK_DAYS,
    ) -> Optional[float]:
        """
        Calculate Historical Simulation Value at Risk (VaR) for the current portfolio.

        VaR represents the maximum potential loss (in USDT) that the portfolio could
        experience over a specified time period with a given confidence level, based
        on historical price movements.

        Args:
            positions: List of Position objects representing portfolio positions
            confidence: Confidence level for VaR calculation (default: 0.95 = 95%)
            lookback_days: Number of historical days to use for simulation (default: 90)

        Returns:
            VaR amount in USDT if calculation succeeds, None otherwise.

        Note:
            - Uses daily price data for historical simulation
            - Requires minimum history and PnL samples as defined in config
            - Stores result in self.last_var_value and self.last_var_confidence
            - Returns None if insufficient data or calculation fails
        """
        self.last_var_value = None
        self.last_var_confidence = None
        if not positions:
            if log_warn:
                log_warn("No positions available for VaR calculation.")
            return None

        confidence_pct = int(confidence * 100)
        if log_analysis:
            log_analysis(
                f"Calculating Historical VaR ({confidence_pct}% confidence, {lookback_days}d lookback)..."
            )

        price_history = {}
        fetch_limit = max(lookback_days * 2, lookback_days + 50)
        for pos in positions:
            df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                pos.symbol, limit=fetch_limit, timeframe="1d"
            )
            series = self.data_fetcher.dataframe_to_close_series(df)
            if series is not None:
                price_history[pos.symbol] = series

        if not price_history:
            if log_warn:
                log_warn("Unable to fetch historical data for VaR.")
            return None

        price_df = pd.DataFrame(price_history).dropna(how="all")
        if price_df.empty:
            if log_warn:
                log_warn("No overlapping history found for VaR.")
            return None

        if len(price_df) < lookback_days:
            if log_warn:
                log_warn(
                    f"Only {len(price_df)} daily points available (requested {lookback_days}). Using available history."
                )
        price_df = price_df.tail(lookback_days)

        if len(price_df) < DEFAULT_VAR_MIN_HISTORY_DAYS:
            if log_warn:
                log_warn(
                    f"Insufficient history (<{DEFAULT_VAR_MIN_HISTORY_DAYS} days) for reliable VaR."
                )
            return None

        returns_df = price_df.pct_change().dropna(how="all")
        if returns_df.empty:
            if log_warn:
                log_warn("Unable to compute returns for VaR.")
            return None

        # Vectorized PnL calculation
        # Aggregate net exposure per symbol (handles multiple positions per symbol)
        exposure_map = {}
        for pos in positions:
            sign = 1 if pos.direction == "LONG" else -1
            exposure_map[pos.symbol] = exposure_map.get(pos.symbol, 0.0) + (
                pos.size_usdt * sign
            )

        exposures = pd.Series(exposure_map)

        # Align returns with portfolio symbols
        common_symbols = returns_df.columns.intersection(exposures.index)

        if common_symbols.empty:
            if log_warn:
                log_warn("No overlapping symbols between portfolio and historical data.")
            return None

        # Filter returns to relevant symbols and remove days with no data
        portfolio_returns = returns_df[common_symbols].dropna(how="all")

        if portfolio_returns.empty:
            return None

        # Calculate daily PnL: sum(return * exposure)
        # fillna(0) treats missing individual symbol data as 0 return (no PnL change)
        daily_pnls = portfolio_returns.fillna(0).dot(exposures[common_symbols]).values

        if len(daily_pnls) < DEFAULT_VAR_MIN_PNL_SAMPLES:
            if log_warn:
                log_warn(
                    f"Not enough historical PnL samples for VaR (need at least {DEFAULT_VAR_MIN_PNL_SAMPLES})."
                )
            return None

        percentile = max(0, min(100, (1 - confidence) * 100))
        loss_percentile = np.percentile(daily_pnls, percentile)
        var_amount = max(0.0, -loss_percentile)

        if log_model:
            log_model(f"Historical VaR ({confidence_pct}%): {var_amount:.2f} USDT")
        self.last_var_value = var_amount
        self.last_var_confidence = confidence
        return var_amount
