
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas as pd

"""
Correlation analyzer for portfolio correlation calculations.

This module provides the PortfolioCorrelationAnalyzer class for calculating
various correlation metrics between portfolio positions and new symbols,
including weighted correlations and portfolio return correlations.
"""



if TYPE_CHECKING:
    from modules.common.models.position import Position

try:
    from config import (
        DEFAULT_CORRELATION_MIN_POINTS,
        DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS,
        HEDGE_CORRELATION_HIGH_THRESHOLD,
        HEDGE_CORRELATION_MEDIUM_THRESHOLD,
    )
    from modules.common.models.position import Position
    from modules.common.utils import (
        log_analysis,
        log_data,
        log_error,
        log_info,
        log_success,
        log_warn,
    )
except ImportError:
    Position = None
    log_warn = None
    log_error = None
    log_info = None
    log_analysis = None
    log_data = None
    log_success = None
    DEFAULT_CORRELATION_MIN_POINTS = 10
    DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS = 10
    HEDGE_CORRELATION_HIGH_THRESHOLD = 0.7
    HEDGE_CORRELATION_MEDIUM_THRESHOLD = 0.4


class PortfolioCorrelationAnalyzer:
    """
    Analyzes correlation between portfolio positions and new symbols.

    This class provides methods to calculate various correlation metrics:
    - Weighted internal correlation between portfolio positions
    - Weighted correlation between a new symbol and the portfolio
    - Portfolio return correlation with a new symbol

    Attributes:
        data_fetcher: Data fetcher instance for retrieving market data
        positions: List of current portfolio positions
        _series_cache: Cache for price series to avoid redundant fetches
    """

    def __init__(self, data_fetcher: Any, positions: List["Position"]) -> None:
        """
        Initialize the PortfolioCorrelationAnalyzer.

        Args:
            data_fetcher: Data fetcher instance for retrieving OHLCV data
            positions: List of current portfolio positions
        """
        self.data_fetcher = data_fetcher
        self.positions = positions
        self._series_cache: Dict[str, pd.Series] = {}

    def _fetch_symbol_series(self, symbol: str) -> Optional[pd.Series]:
        """
        Fetch price series for a symbol with caching.

        Args:
            symbol: Symbol to fetch price series for

        Returns:
            Price series (close prices) or None if fetch fails

        Note:
            Uses internal cache to avoid redundant API calls.
        """
        if symbol in self._series_cache:
            return self._series_cache[symbol]

        df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(symbol)
        series = self.data_fetcher.dataframe_to_close_series(df)
        if series is not None:
            self._series_cache[symbol] = series
        return series

    def _get_portfolio_series_dict(self) -> Dict[str, pd.Series]:
        """
        Get all price series for portfolio positions.

        Returns:
            Dictionary mapping symbol to price series
        """
        symbol_series = {}
        for pos in self.positions:
            if pos.symbol not in symbol_series:
                series = self._fetch_symbol_series(pos.symbol)
                if series is not None:
                    symbol_series[pos.symbol] = series
        return symbol_series

    def calculate_weighted_correlation(self, verbose: bool = True) -> Tuple[Optional[float], List[Dict[str, Any]]]:
        """
        Calculate weighted internal correlation of the current portfolio (between positions).

        This method calculates correlation between all pairs of positions in the portfolio,
        weighted by position sizes. Returns are adjusted for LONG/SHORT directions to
        reflect actual PnL correlation.

        Args:
            verbose: Whether to print detailed output

        Returns:
            Tuple containing:
                - weighted_correlation: Average weighted correlation between all position pairs (None if insufficient data)
                - position_correlations_list: List of correlation details for each pair
        """
        if len(self.positions) < 2:
            if verbose and log_warn:
                log_warn("Need at least 2 positions to calculate internal correlation.")
            return None, []

        if verbose and log_analysis:
            log_analysis("Portfolio Internal Correlation Analysis:")

        symbol_series = self._get_portfolio_series_dict()
        if len(symbol_series) < 2:
            if verbose and log_warn:
                log_warn("Insufficient data for internal correlation analysis.")
            return None, []

        # Create a DataFrame with all price series aligned
        price_df = pd.DataFrame(symbol_series).dropna(how="all")
        if len(price_df) < DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS:
            if verbose and log_warn:
                log_warn("Insufficient overlapping data for correlation analysis.")
            return None, []

        # Calculate returns
        returns_df = price_df.pct_change().dropna(how="all")
        if len(returns_df) < DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS:
            if verbose and log_warn:
                log_warn("Insufficient return data for correlation analysis.")
            return None, []

        # Adjust returns for SHORT positions
        adjusted_returns = returns_df.copy()
        position_map = {p.symbol: p for p in self.positions}

        for col in adjusted_returns.columns:
            pos = position_map.get(col)
            if pos and pos.direction == "SHORT":
                adjusted_returns[col] = -adjusted_returns[col]

        # Calculate correlation matrix (Vectorized O(1) operation relative to loop)
        corr_matrix = adjusted_returns.corr()

        correlations = []
        weights = []
        position_pairs = []

        # Extract pairwise correlations and weights
        symbols = list(adjusted_returns.columns)
        for i, symbol1 in enumerate(symbols):
            for j in range(i + 1, len(symbols)):
                symbol2 = symbols[j]

                # Check for minimum overlapping data points for this specific pair
                # This is important because dropna(how='all') might leave pairs with few common points
                pair_data = adjusted_returns[[symbol1, symbol2]].dropna()
                if len(pair_data) < DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS:
                    continue

                corr = corr_matrix.loc[symbol1, symbol2]
                if pd.isna(corr):
                    continue

                pos1 = position_map.get(symbol1)
                pos2 = position_map.get(symbol2)

                weight = ((pos1.size_usdt if pos1 else 0) + (pos2.size_usdt if pos2 else 0)) / 2

                correlations.append(corr)
                weights.append(weight)
                position_pairs.append(
                    {
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "direction1": pos1.direction if pos1 else "UNKNOWN",
                        "direction2": pos2.direction if pos2 else "UNKNOWN",
                        "correlation": corr,
                        "weight": weight,
                    }
                )

        if not correlations:
            if verbose and log_warn:
                log_warn("No valid correlation pairs found.")
            return None, []

        total_weight = sum(weights)
        weighted_corr = (
            sum(c * w for c, w in zip(correlations, weights)) / total_weight
            if total_weight > 0
            else sum(correlations) / len(correlations)
        )

        if verbose:
            if log_data:
                log_data("Position Pair Correlations (PnL-adjusted):")
            for pair in position_pairs:
                weight_pct = (pair["weight"] / total_weight * 100) if total_weight > 0 else 0
                direction1 = pair.get("direction1", "UNKNOWN")
                direction2 = pair.get("direction2", "UNKNOWN")
                if log_data:
                    log_data(
                        f"  {pair['symbol1']:12} ({direction1:5}) <-> "
                        f"{pair['symbol2']:12} ({direction2:5}) "
                        f"({pair['weight']:>8.2f} USDT, {weight_pct:>5.1f}%): "
                        f"{pair['correlation']:>6.4f}"
                    )

            if log_analysis:
                log_analysis("Weighted Internal Correlation:")
            if log_data:
                log_data(f"  Portfolio Internal: {weighted_corr:>6.4f}")

        return weighted_corr, position_pairs

    def analyze_correlation_with_new_symbol(
        self,
        new_symbol: str,
        new_position_size: float = 0.0,
        new_direction: str = "LONG",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze correlation impact of adding a new symbol to the portfolio.

        This method calculates correlation metrics before and after adding a new
        position, helping to assess the diversification impact.

        Args:
            new_symbol: Symbol to analyze
            new_position_size: Size of new position in USDT (for weighted calculation)
            new_direction: Direction of new position (LONG/SHORT)
            verbose: Whether to print detailed output

        Returns:
            Dictionary with before/after correlation metrics and impact analysis
        """
        result = {
            "before": {},
            "after": {},
            "impact": {},
        }

        # Calculate current portfolio internal correlation
        if verbose and log_analysis:
            log_analysis("=== Analyzing Correlation Impact of Adding New Symbol ===")

        internal_corr_before, _ = self.calculate_weighted_correlation(verbose=False)
        result["before"]["internal_correlation"] = internal_corr_before

        # Calculate correlation with new symbol
        weighted_corr, position_details = self.calculate_weighted_correlation_with_new_symbol(new_symbol, verbose=False)
        result["after"]["new_symbol_correlation"] = weighted_corr

        # Calculate portfolio return correlation with new symbol
        portfolio_return_corr, return_metadata = self.calculate_portfolio_return_correlation(new_symbol, verbose=False)
        result["after"]["portfolio_return_correlation"] = portfolio_return_corr

        # Simulate adding new position and recalculate internal correlation
        if new_position_size > 0:
            from modules.common.models.position import Position

            temp_positions = self.positions + [Position(new_symbol, new_direction, 0.0, new_position_size)]
            temp_analyzer = PortfolioCorrelationAnalyzer(self.data_fetcher, temp_positions)
            internal_corr_after, _ = temp_analyzer.calculate_weighted_correlation(verbose=False)
            result["after"]["internal_correlation"] = internal_corr_after

            # Calculate impact
            if internal_corr_before is not None and internal_corr_after is not None:
                correlation_change = internal_corr_after - internal_corr_before
                result["impact"]["correlation_change"] = correlation_change
                result["impact"]["diversification_improvement"] = abs(internal_corr_after) < abs(internal_corr_before)

        if verbose:
            if log_analysis:
                log_analysis("=== Summary ===")
            if internal_corr_before is not None and log_data:
                log_data(f"Current Portfolio Internal Correlation: {internal_corr_before:.4f}")
            if weighted_corr is not None and log_data:
                log_data(f"New Symbol vs Portfolio Correlation: {weighted_corr:.4f}")
            if portfolio_return_corr is not None and log_data:
                log_data(f"Portfolio Return vs New Symbol Correlation: {portfolio_return_corr:.4f}")
            if "internal_correlation" in result["after"] and log_data:
                log_data(f"Portfolio Internal Correlation After: {result['after']['internal_correlation']:.4f}")
            if "correlation_change" in result["impact"]:
                change = result["impact"]["correlation_change"]
                improvement = result["impact"]["diversification_improvement"]
                if log_data:
                    log_data(f"Correlation Change: {change:+.4f}")
                if improvement:
                    if log_success:
                        log_success(f"Diversification Improvement: {improvement}")
                else:
                    if log_warn:
                        log_warn(f"Diversification Improvement: {improvement}")

        return result

    def calculate_weighted_correlation_with_new_symbol(
        self, new_symbol: str, verbose: bool = True
    ) -> Tuple[Optional[float], List[Dict[str, Any]]]:
        """
        Calculate weighted correlation between a new symbol and the entire portfolio.

        This method calculates correlation between each position and the new symbol,
        weighted by position sizes. Returns are adjusted for LONG/SHORT directions.

        Args:
            new_symbol: Symbol to calculate correlation with
            verbose: Whether to print detailed output

        Returns:
            Tuple containing:
                - weighted_correlation: Weighted average correlation (None if insufficient data)
                - position_details: List of correlation details for each position
        """
        correlations = []
        weights = []
        position_details = []

        if verbose and log_analysis:
            log_analysis("Correlation Analysis (Weighted by Position Size):")

        new_df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(new_symbol)
        new_series = self.data_fetcher.dataframe_to_close_series(new_df)
        if new_series is None:
            if verbose and log_error:
                log_error(f"Could not fetch price history for {new_symbol}")
            return None, []

        for pos in self.positions:
            pos_series = self._fetch_symbol_series(pos.symbol)

            if pos_series is not None:
                df = pd.concat([pos_series, new_series], axis=1, join="inner")
                if len(df) < DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS:
                    continue

                # Calculate correlation on returns (pct_change) instead of prices
                # to avoid spurious correlation from non-stationary price series
                returns_df = df.pct_change().dropna()
                if len(returns_df) < DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS:
                    continue

                # Adjust returns based on position direction (LONG/SHORT)
                # SHORT positions have inverted returns for PnL correlation
                adjusted_returns = returns_df.copy()
                if pos.direction == "SHORT":
                    adjusted_returns.iloc[:, 0] = -adjusted_returns.iloc[:, 0]

                corr = adjusted_returns.iloc[:, 0].corr(adjusted_returns.iloc[:, 1])
                if pd.isna(corr):
                    continue

                weight = pos.size_usdt

                correlations.append(corr)
                weights.append(weight)

                position_details.append(
                    {
                        "symbol": pos.symbol,
                        "direction": pos.direction,
                        "size": pos.size_usdt,
                        "correlation": corr,
                        "weight": weight,
                    }
                )

        if not correlations:
            if verbose and log_warn:
                log_warn("Insufficient data for correlation analysis.")
            return None, []

        total_weight = sum(weights)
        weighted_corr = sum(c * w for c, w in zip(correlations, weights)) / total_weight

        if verbose:
            if log_data:
                log_data("Individual Correlations:")
            for detail in position_details:
                weight_pct = (detail["weight"] / total_weight) * 100
                if log_data:
                    log_data(
                        f"  {detail['symbol']:12} ({detail['direction']:5}, {detail['size']:>8.2f} USDT, {weight_pct:>5.1f}%): "
                        f"{detail['correlation']:>6.4f}"
                    )

            if log_analysis:
                log_analysis("Weighted Portfolio Correlation:")
            if log_data:
                log_data(f"  {new_symbol} vs Portfolio: {weighted_corr:>6.4f}")

        return weighted_corr, position_details

    def calculate_portfolio_return_correlation(
        self,
        new_symbol: str,
        min_points: int = DEFAULT_CORRELATION_MIN_POINTS,
        verbose: bool = True,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Calculate correlation between the portfolio's aggregated return and the new symbol.

        This method calculates the correlation between the weighted portfolio return
        (aggregated across all positions) and the new symbol's return. Returns are
        adjusted for LONG/SHORT directions.

        Args:
            new_symbol: Symbol to calculate correlation with
            min_points: Minimum number of data points required
            verbose: Whether to print detailed output

        Returns:
            Tuple containing:
                - correlation: Correlation value (None if insufficient data)
                - metadata: Dictionary with additional info (e.g., samples used)
        """
        if verbose and log_analysis:
            log_analysis("Portfolio Return Correlation Analysis:")

        if not self.positions:
            if verbose and log_warn:
                log_warn("No positions in portfolio to compare against.")
            return None, {}

        new_df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(new_symbol)
        new_series = self.data_fetcher.dataframe_to_close_series(new_df)
        if new_series is None:
            if verbose and log_error:
                log_error(f"Could not fetch price history for {new_symbol}")
            return None, {}

        symbol_series = self._get_portfolio_series_dict()

        if not symbol_series:
            if verbose and log_warn:
                log_warn("Unable to fetch history for existing positions.")
            return None, {}

        price_df = pd.DataFrame(symbol_series).dropna(how="all")
        if price_df.empty:
            if verbose and log_warn:
                log_warn("Insufficient overlapping data among current positions.")
            return None, {}

        portfolio_returns_df = price_df.pct_change().dropna(how="all")
        new_returns = new_series.pct_change().dropna()

        if portfolio_returns_df.empty or new_returns.empty:
            if verbose and log_warn:
                log_warn("Insufficient price history to compute returns.")
            return None, {}

        common_index = portfolio_returns_df.index.intersection(new_returns.index)
        if len(common_index) < min_points:
            if verbose and log_warn:
                log_warn(f"Need at least {min_points} overlapping points, found {len(common_index)}.")
            return None, {}

        # Vectorized approach: Adjust returns for LONG/SHORT and calculate weighted portfolio returns
        # Create adjusted returns DataFrame (invert SHORT positions)
        adjusted_returns_df = portfolio_returns_df.copy()
        for pos in self.positions:
            if pos.symbol in adjusted_returns_df.columns and pos.direction == "SHORT":
                adjusted_returns_df[pos.symbol] = -adjusted_returns_df[pos.symbol]

        # Create weights Series for each position
        position_weights = {}
        for pos in self.positions:
            if pos.symbol in adjusted_returns_df.columns:
                position_weights[pos.symbol] = abs(pos.size_usdt)

        # Filter to only symbols that exist in both adjusted_returns_df and position_weights
        valid_symbols = [sym for sym in adjusted_returns_df.columns if sym in position_weights]
        if not valid_symbols:
            if verbose and log_warn:
                log_warn("No valid positions for portfolio return calculation.")
            return None, {}

        # Calculate weighted portfolio returns using vectorization
        # Select only valid symbols and common index
        adjusted_common = adjusted_returns_df.loc[common_index, valid_symbols]
        weights_array = np.array([position_weights[sym] for sym in valid_symbols])

        # Calculate weighted sum for each row (index)
        weighted_sums = (adjusted_common * weights_array).sum(axis=1)
        total_weights = adjusted_common.notna().dot(weights_array)

        # Calculate weighted average portfolio returns (avoid division by zero)
        portfolio_return_series = weighted_sums / total_weights.replace(0, np.nan)

        # Align new returns with common index
        new_return_series = new_returns.loc[common_index]

        # Remove rows where either series has NaN
        valid_mask = portfolio_return_series.notna() & new_return_series.notna()
        portfolio_return_series = portfolio_return_series[valid_mask]
        new_return_series = new_return_series[valid_mask]

        if len(portfolio_return_series) < min_points:
            if verbose and log_warn:
                log_warn("Not enough aligned return samples for correlation.")
            return None, {"samples": len(portfolio_return_series)}

        correlation = portfolio_return_series.corr(new_return_series)

        if pd.isna(correlation):
            if verbose and log_warn:
                log_warn("Unable to compute correlation (insufficient variance).")
            return None, {"samples": len(portfolio_return_series)}

        if verbose:
            if log_data:
                log_data(f"  Portfolio Return vs {new_symbol}: {correlation:>6.4f}")
                log_data(f"  Samples used: {len(portfolio_return_series)}")

        return correlation, {"samples": len(portfolio_return_series)}
