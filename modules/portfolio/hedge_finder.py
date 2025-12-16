"""
Hedge finder for discovering and analyzing hedge candidates.

This module provides the HedgeFinder class for automatically discovering and
analyzing hedge candidates for cryptocurrency portfolios. It uses correlation
analysis and beta-weighted calculations to recommend optimal hedging strategies.
"""

import os
from math import ceil
from typing import List, Optional, Dict, Tuple, Any, Set, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

if TYPE_CHECKING:
    from modules.common.models.position import Position
    from modules.common.core.exchange_manager import ExchangeManager
    from modules.common.core.data_fetcher import DataFetcher
    from modules.portfolio.correlation_analyzer import PortfolioCorrelationAnalyzer
    from modules.portfolio.risk_calculator import PortfolioRiskCalculator

try:
    from modules.common.models.position import Position
    from modules.common.utils import (
        normalize_symbol,
        log_warn,
        log_error,
        log_info,
        log_analysis,
        log_model,
        log_success,
        log_data,
        log_system,
    )
    from modules.common.ui.progress_bar import ProgressBar
    from modules.common.core.exchange_manager import ExchangeManager
    from modules.common.core.data_fetcher import DataFetcher
    from modules.portfolio.correlation_analyzer import PortfolioCorrelationAnalyzer
    from modules.portfolio.risk_calculator import PortfolioRiskCalculator
    from config import (
        BENCHMARK_SYMBOL,
        HEDGE_CORRELATION_HIGH_THRESHOLD,
        HEDGE_CORRELATION_MEDIUM_THRESHOLD,
        HEDGE_CORRELATION_DIFF_THRESHOLD,
    )
except ImportError:
    Position = None
    normalize_symbol = None
    log_warn = None
    log_error = None
    log_info = None
    log_analysis = None
    log_model = None
    log_success = None
    log_data = None
    log_system = None
    ProgressBar = None
    ExchangeManager = None
    DataFetcher = None
    PortfolioCorrelationAnalyzer = None
    PortfolioRiskCalculator = None
    BENCHMARK_SYMBOL = "BTC/USDT"
    HEDGE_CORRELATION_HIGH_THRESHOLD = 0.7
    HEDGE_CORRELATION_MEDIUM_THRESHOLD = 0.4
    HEDGE_CORRELATION_DIFF_THRESHOLD = 0.1


class HedgeFinder:
    """
    Finds and analyzes hedge candidates for cryptocurrency portfolios.

    This class provides methods to automatically discover hedge candidates from
    Binance Futures, score them based on correlation metrics, and analyze potential
    trades with beta-weighted hedging recommendations.

    Attributes:
        exchange_manager: Exchange manager for accessing exchange APIs
        correlation_analyzer: Analyzer for calculating portfolio correlations
        risk_calculator: Calculator for risk metrics (beta, VaR, etc.)
        positions: List of current portfolio positions
        benchmark_symbol: Benchmark symbol for beta calculations
        shutdown_event: Event for graceful shutdown during long operations
        data_fetcher: Data fetcher for retrieving market data
    """

    def __init__(
        self,
        exchange_manager: "ExchangeManager",
        correlation_analyzer: "PortfolioCorrelationAnalyzer",
        risk_calculator: "PortfolioRiskCalculator",
        positions: List["Position"],
        benchmark_symbol: str = BENCHMARK_SYMBOL,
        shutdown_event: Optional[Any] = None,
        data_fetcher: Optional["DataFetcher"] = None,
    ) -> None:
        """
        Initialize the HedgeFinder.

        Args:
            exchange_manager: Exchange manager instance
            correlation_analyzer: Portfolio correlation analyzer instance
            risk_calculator: Portfolio risk calculator instance
            positions: List of current portfolio positions
            benchmark_symbol: Benchmark symbol for beta calculations (default: BENCHMARK_SYMBOL)
            shutdown_event: Optional shutdown event for graceful interruption
            data_fetcher: Optional data fetcher instance (creates new one if not provided)
        """
        self.exchange_manager = exchange_manager
        self.correlation_analyzer = correlation_analyzer
        self.risk_calculator = risk_calculator
        self.positions = positions
        self.benchmark_symbol = benchmark_symbol
        self.shutdown_event = shutdown_event
        if data_fetcher is not None:
            self.data_fetcher = data_fetcher
        elif DataFetcher is not None:
            self.data_fetcher = DataFetcher(exchange_manager, shutdown_event)
        else:
            self.data_fetcher = None

    def should_stop(self) -> bool:
        """Check if shutdown was requested."""
        if self.shutdown_event:
            return self.shutdown_event.is_set()
        return False

    def _list_candidate_symbols(
        self,
        exclude_symbols: Optional[Set[str]] = None,
        max_candidates: Optional[int] = None,
    ) -> List[str]:
        """
        Fetch potential hedge symbols from Binance Futures.

        Args:
            exclude_symbols: Set of symbols to exclude from candidates
            max_candidates: Maximum number of candidates to return

        Returns:
            List of candidate symbol strings

        Raises:
            ImportError: If DataFetcher is not available
        """
        if self.data_fetcher is None:
            raise ImportError("DataFetcher is required to list candidate symbols.")
        progress_label = "Symbol Discovery"
        return self.data_fetcher.list_binance_futures_symbols(
            exclude_symbols=exclude_symbols,
            max_candidates=max_candidates,
            progress_label=progress_label,
        )

    def _score_candidate(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Score a hedge candidate based on correlation metrics.

        Args:
            symbol: Symbol to score

        Returns:
            Dictionary with symbol, weighted_corr, return_corr, and score, or None if insufficient data
        """
        weighted_corr, _ = self.correlation_analyzer.calculate_weighted_correlation_with_new_symbol(
            symbol, verbose=False
        )
        return_corr, _ = (
            self.correlation_analyzer.calculate_portfolio_return_correlation(
                symbol, verbose=False
            )
        )

        if weighted_corr is None and return_corr is None:
            return None

        score_components = [
            abs(x) for x in [weighted_corr, return_corr] if x is not None
        ]
        if not score_components:
            return None

        score = sum(score_components) / len(score_components)
        return {
            "symbol": symbol,
            "weighted_corr": weighted_corr,
            "return_corr": return_corr,
            "score": score,
        }

    def find_best_hedge_candidate(
        self,
        total_delta: float,
        total_beta_delta: float,
        max_candidates: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Automatically scan Binance futures symbols to find the best hedge candidate.

        This method discovers candidate symbols, scores them based on correlation
        metrics, and returns the best candidate for hedging the portfolio.

        Args:
            total_delta: Current total portfolio delta
            total_beta_delta: Current total portfolio beta-weighted delta
            max_candidates: Maximum number of candidates to scan (None = all)

        Returns:
            Dictionary with best candidate info (symbol, weighted_corr, return_corr, score),
            or None if no suitable candidate found
        """
        if not self.positions:
            if log_warn:
                log_warn("No positions loaded. Cannot search for hedge candidates.")
            return None

        existing_symbols = {normalize_symbol(p.symbol) for p in self.positions} if normalize_symbol else set()
        if normalize_symbol:
            existing_symbols.add(normalize_symbol(self.benchmark_symbol))
        candidate_symbols = self._list_candidate_symbols(
            existing_symbols, max_candidates=None
        )

        if not candidate_symbols:
            if log_warn:
                log_warn("Could not find candidate symbols from Binance.")
            return None
        if self.should_stop():
            if log_warn:
                log_warn("Hedge scan aborted before start.")
            return None

        if max_candidates is not None:
            candidate_symbols = candidate_symbols[:max_candidates]
        scan_count = len(candidate_symbols)
        if log_analysis:
            log_analysis(f"Scanning {scan_count} candidate(s) for optimal hedge...")

        core_count = max(1, int((os.cpu_count() or 1) * 0.8))
        batch_size = ceil(scan_count / core_count) if scan_count else 0
        batches = [
            candidate_symbols[i : i + batch_size]
            for i in range(0, scan_count, batch_size)
        ] or [[]]
        total_batches = len([b for b in batches if b])
        progress_bar = ProgressBar(total_batches or 1, "Batch Progress")

        best_candidate = None

        def process_batch(batch_symbols: List[str]) -> Optional[Dict]:
            local_best = None
            for sym in batch_symbols:
                if self.should_stop():
                    return None
                result = self._score_candidate(sym)
                if result is None:
                    continue
                if local_best is None or result["score"] > local_best["score"]:
                    local_best = result
            return local_best

        with ThreadPoolExecutor(max_workers=core_count) as executor:
            futures = {}
            for idx, batch in enumerate(batches, start=1):
                if not batch:
                    continue
                if log_info:
                    log_info(f"Starting batch {idx}/{total_batches} (size {len(batch)})")
                futures[executor.submit(process_batch, batch)] = idx
            for future in as_completed(futures):
                if self.should_stop():
                    break
                batch_id = futures[future]
                try:
                    batch_best = future.result()
                except Exception as exc:
                    if log_error:
                        log_error(f"Batch {batch_id} failed: {exc}")
                    continue
                if batch_best is None:
                    if log_warn:
                        log_warn(f"Batch {batch_id}: no viable candidate.")
                    progress_bar.update()
                    continue
                if log_info:
                    log_info(f"Batch {batch_id}: best {batch_best['symbol']} (score {batch_best['score']:.4f})")
                if (
                    best_candidate is None
                    or batch_best["score"] > best_candidate["score"]
                ):
                    best_candidate = batch_best
                progress_bar.update()
        if total_batches:
            progress_bar.finish()

        if best_candidate is None:
            if log_warn:
                log_warn("No suitable hedge candidate found (insufficient data).")
        else:
            if log_model:
                log_model(f"Best candidate: {best_candidate['symbol']} (score {best_candidate['score']:.4f})")
            if best_candidate["weighted_corr"] is not None:
                if log_data:
                    log_data(f"  Weighted Correlation: {best_candidate['weighted_corr']:.4f}")
            if best_candidate["return_corr"] is not None:
                if log_data:
                    log_data(f"  Portfolio Return Correlation: {best_candidate['return_corr']:.4f}")

        return best_candidate

    def analyze_new_trade(
        self,
        new_symbol: str,
        total_delta: float,
        total_beta_delta: float,
        last_var_value: Optional[float] = None,
        last_var_confidence: Optional[float] = None,
        correlation_mode: str = "weighted",  # Currently unused, reserved for future use
    ) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """
        Analyze a potential new trade and automatically recommend direction for beta-weighted hedging.

        This method performs comprehensive analysis including:
        - Symbol normalization
        - Beta calculation
        - Delta/beta-weighted delta analysis
        - Correlation analysis (weighted and portfolio return)
        - VaR insights
        - Hedge recommendations

        Args:
            new_symbol: Symbol to analyze
            total_delta: Current total portfolio delta
            total_beta_delta: Current total portfolio beta-weighted delta
            last_var_value: Last computed VaR value (optional)
            last_var_confidence: Confidence level for VaR (optional)
            correlation_mode: Correlation analysis mode (default: "weighted")

        Returns:
            Tuple containing:
                - Recommended direction ("LONG" or "SHORT") or None
                - Recommended size in USDT or None
                - Final correlation value (weighted or portfolio return) or None
        """
        if normalize_symbol is None:
            normalized_symbol = new_symbol
        else:
            normalized_symbol = normalize_symbol(new_symbol)
        if normalized_symbol != new_symbol:
            if log_info:
                log_info(f"Symbol normalized: '{new_symbol}' -> '{normalized_symbol}'")

        new_symbol = normalized_symbol
        if log_analysis:
            log_analysis(f"Analyzing potential trade on {new_symbol}...")
        if log_data:
            log_data(f"Current Total Delta: {total_delta:+.2f} USDT")
            log_data(f"Current Total Beta Delta (vs {self.benchmark_symbol}): {total_beta_delta:+.2f} USDT")

        new_symbol_beta = self.risk_calculator.calculate_beta(new_symbol)
        beta_available = new_symbol_beta is not None and abs(new_symbol_beta) > 1e-6
        if beta_available:
            if log_analysis:
                log_analysis(f"{new_symbol} beta vs {self.benchmark_symbol}: {new_symbol_beta:.4f}")
        else:
            if log_warn:
                log_warn(f"Could not compute beta for {new_symbol}. Falling back to simple delta hedging.")

        hedge_mode = "beta" if beta_available else "delta"
        metric_label = "Beta Delta" if beta_available else "Delta"
        current_metric = total_beta_delta if beta_available else total_delta
        target_metric = -current_metric

        recommended_direction = None
        recommended_size = None

        if abs(current_metric) < 0.01:
            if log_success:
                log_success(f"Portfolio is already {metric_label} Neutral ({metric_label} ≈ 0).")
        else:
            if beta_available:
                beta_sign = np.sign(new_symbol_beta)
                if beta_sign == 0:
                    beta_available = False
                    hedge_mode = "delta"
                    metric_label = "Delta"
                    current_metric = total_delta
                    target_metric = -current_metric
                else:
                    direction_multiplier = -np.sign(current_metric) * beta_sign
                    recommended_direction = (
                        "LONG" if direction_multiplier >= 0 else "SHORT"
                    )
                    recommended_size = abs(current_metric) / max(
                        abs(new_symbol_beta), 1e-6
                    )
                    if log_analysis:
                        log_analysis(f"Targeting Beta Neutrality using {metric_label}.")
            if not beta_available:
                if current_metric > 0:
                    recommended_direction = "SHORT"
                    recommended_size = abs(target_metric)
                    if log_warn:
                        log_warn("Portfolio has excess LONG delta exposure.")
                else:
                    recommended_direction = "LONG"
                    recommended_size = abs(target_metric)
                    if log_warn:
                        log_warn("Portfolio has excess SHORT delta exposure.")
                if log_analysis:
                    log_analysis("Targeting simple Delta Neutrality.")

        if recommended_direction and recommended_size is not None:
            if log_model:
                log_model(f"Recommended {hedge_mode.upper()} hedge:")
            if log_data:
                log_data(f"  Direction: {recommended_direction}")
                log_data(f"  Size: {recommended_size:.2f} USDT")

        if not self.positions:
            if log_info:
                log_info("No existing positions for correlation analysis.")
            return (
                recommended_direction,
                recommended_size if recommended_direction else None,
                None,
            )

        if log_analysis:
            log_analysis("=" * 70)
            log_analysis("CORRELATION ANALYSIS - COMPARING BOTH METHODS")
            log_analysis("=" * 70)

        weighted_corr, weighted_details = (
            self.correlation_analyzer.calculate_weighted_correlation_with_new_symbol(new_symbol)
        )
        portfolio_return_corr, portfolio_return_details = (
            self.correlation_analyzer.calculate_portfolio_return_correlation(new_symbol)
        )

        if log_analysis:
            log_analysis("=" * 70)
            log_analysis("CORRELATION SUMMARY")
            log_analysis("=" * 70)

        if weighted_corr is not None:
            if log_data:
                log_data(f"1. Weighted Correlation (by Position Size):")
                log_data(f"   {new_symbol} vs Portfolio: {weighted_corr:>6.4f}")

            if abs(weighted_corr) > HEDGE_CORRELATION_HIGH_THRESHOLD:
                if log_success:
                    log_success("   → High correlation. Good for hedging.")
            elif abs(weighted_corr) > HEDGE_CORRELATION_MEDIUM_THRESHOLD:
                if log_warn:
                    log_warn("   → Moderate correlation. Partial hedging effect.")
            else:
                if log_error:
                    log_error("   → Low correlation. Limited hedging effectiveness.")
        else:
            if log_warn:
                log_warn("1. Weighted Correlation: N/A (insufficient data)")

        if portfolio_return_corr is not None:
            samples_info = (
                portfolio_return_details.get("samples", "N/A")
                if isinstance(portfolio_return_details, dict)
                else "N/A"
            )
            if log_data:
                log_data("2. Portfolio Return Correlation (includes direction):")
                log_data(f"   {new_symbol} vs Portfolio Return: {portfolio_return_corr:>6.4f}")
                log_data(f"   Samples used: {samples_info}")

            if abs(portfolio_return_corr) > HEDGE_CORRELATION_HIGH_THRESHOLD:
                if log_success:
                    log_success("   → High correlation. Excellent for hedging.")
            elif abs(portfolio_return_corr) > HEDGE_CORRELATION_MEDIUM_THRESHOLD:
                if log_warn:
                    log_warn("   → Moderate correlation. Acceptable hedging effect.")
            else:
                if log_error:
                    log_error("   → Low correlation. Poor hedging effectiveness.")
        else:
            if log_warn:
                log_warn("2. Portfolio Return Correlation: N/A (insufficient data)")

        if log_analysis:
            log_analysis("-" * 70)
            log_analysis("OVERALL ASSESSMENT:")

        if weighted_corr is not None and portfolio_return_corr is not None:
            diff = abs(weighted_corr - portfolio_return_corr)
            if diff < HEDGE_CORRELATION_DIFF_THRESHOLD:
                if log_success:
                    log_success("   ✓ Both methods show similar correlation → Consistent result")
            else:
                if log_warn:
                    log_warn(f"   ⚠ Methods differ by {diff:.4f} → Check if portfolio has SHORT positions")

            avg_corr = (abs(weighted_corr) + abs(portfolio_return_corr)) / 2
            if avg_corr > HEDGE_CORRELATION_HIGH_THRESHOLD:
                if log_success:
                    log_success("   [OK] High correlation detected. This pair is suitable for statistical hedging.")
            elif avg_corr > HEDGE_CORRELATION_MEDIUM_THRESHOLD:
                if log_warn:
                    log_warn("   [!] Moderate correlation. Hedge may be partially effective.")
            else:
                if log_error:
                    log_error("   [X] Low correlation. This hedge might be less effective systematically.")

        elif weighted_corr is not None:
            if abs(weighted_corr) > HEDGE_CORRELATION_HIGH_THRESHOLD:
                if log_success:
                    log_success("   [OK] High correlation detected. This pair is suitable for statistical hedging.")
            elif abs(weighted_corr) > HEDGE_CORRELATION_MEDIUM_THRESHOLD:
                if log_warn:
                    log_warn("   [!] Moderate correlation. Hedge may be partially effective.")
            else:
                if log_error:
                    log_error("   [X] Low correlation. This hedge might be less effective systematically.")

        elif portfolio_return_corr is not None:
            if abs(portfolio_return_corr) > HEDGE_CORRELATION_HIGH_THRESHOLD:
                if log_success:
                    log_success("   [OK] High correlation detected. This pair is suitable for statistical hedging.")
            elif abs(portfolio_return_corr) > HEDGE_CORRELATION_MEDIUM_THRESHOLD:
                if log_warn:
                    log_warn("   [!] Moderate correlation. Hedge may be partially effective.")
            else:
                if log_error:
                    log_error("   [X] Low correlation. This hedge might be less effective systematically.")

        if last_var_value is not None and last_var_confidence is not None:
            conf_pct = int(last_var_confidence * 100)
            if log_model:
                log_model("VaR INSIGHT:")
            if log_data:
                log_data(f"  With {conf_pct}% confidence, daily loss is unlikely to exceed {last_var_value:.2f} USDT.")
                log_data("  Use this ceiling to judge whether the proposed hedge keeps risk tolerable.")
        else:
            if log_warn:
                log_warn("VaR INSIGHT: N/A (insufficient historical data for VaR).")

        if log_analysis:
            log_analysis("=" * 70)

        final_corr = (
            weighted_corr if weighted_corr is not None else portfolio_return_corr
        )

        return (
            recommended_direction,
            recommended_size if recommended_direction else None,
            final_corr,
        )
