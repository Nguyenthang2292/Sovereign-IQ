"""
Pairs trading analyzer for identifying and validating pairs trading opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

try:
    from modules.config import (
        PAIRS_TRADING_MIN_SPREAD,
        PAIRS_TRADING_MAX_SPREAD,
        PAIRS_TRADING_MIN_CORRELATION,
        PAIRS_TRADING_MAX_CORRELATION,
        PAIRS_TRADING_CORRELATION_MIN_POINTS,
        PAIRS_TRADING_TIMEFRAME,
        PAIRS_TRADING_LIMIT,
        PAIRS_TRADING_ADF_PVALUE_THRESHOLD,
        PAIRS_TRADING_MAX_HALF_LIFE,
        PAIRS_TRADING_ZSCORE_LOOKBACK,
        PAIRS_TRADING_HURST_THRESHOLD,
        PAIRS_TRADING_MIN_SPREAD_SHARPE,
        PAIRS_TRADING_MAX_DRAWDOWN,
        PAIRS_TRADING_MIN_CALMAR,
        PAIRS_TRADING_JOHANSEN_CONFIDENCE,
        PAIRS_TRADING_PERIODS_PER_YEAR,
        PAIRS_TRADING_CLASSIFICATION_ZSCORE,
        PAIRS_TRADING_OLS_FIT_INTERCEPT,
        PAIRS_TRADING_KALMAN_DELTA,
        PAIRS_TRADING_KALMAN_OBS_COV,
        PAIRS_TRADING_PAIR_COLUMNS,
        PAIRS_TRADING_ADX_PERIOD,
    )
    from modules.common.utils import (
        log_warn,
        log_info,
        log_success,
        log_progress,
    )
    from modules.common.ProgressBar import ProgressBar
    from modules.common.indicators import calculate_adx
except ImportError:
    PAIRS_TRADING_MIN_SPREAD = 0.01
    PAIRS_TRADING_MAX_SPREAD = 0.50
    PAIRS_TRADING_MIN_CORRELATION = 0.3
    PAIRS_TRADING_MAX_CORRELATION = 0.9
    PAIRS_TRADING_CORRELATION_MIN_POINTS = 50
    PAIRS_TRADING_TIMEFRAME = "1h"
    PAIRS_TRADING_LIMIT = 200
    PAIRS_TRADING_ADF_PVALUE_THRESHOLD = 0.05
    PAIRS_TRADING_MAX_HALF_LIFE = 50
    PAIRS_TRADING_ZSCORE_LOOKBACK = 60
    PAIRS_TRADING_HURST_THRESHOLD = 0.5
    PAIRS_TRADING_MIN_SPREAD_SHARPE = 1.0
    PAIRS_TRADING_MAX_DRAWDOWN = 0.3
    PAIRS_TRADING_MIN_CALMAR = 1.0
    PAIRS_TRADING_JOHANSEN_CONFIDENCE = 0.95
    PAIRS_TRADING_PERIODS_PER_YEAR = 365 * 24
    PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5
    PAIRS_TRADING_OLS_FIT_INTERCEPT = True
    PAIRS_TRADING_KALMAN_DELTA = 1e-5
    PAIRS_TRADING_KALMAN_OBS_COV = 1.0
    PAIRS_TRADING_PAIR_COLUMNS = [
        # Core pair information
        'long_symbol',
        'short_symbol',
        'long_score',
        'short_score',
        'spread',
        'correlation',
        'opportunity_score',
        'quantitative_score',
        # OLS-based metrics
        'hedge_ratio',
        'adf_pvalue',
        'is_cointegrated',
        'half_life',
        'mean_zscore',
        'std_zscore',
        'skewness',
        'kurtosis',
        'current_zscore',
        'hurst_exponent',
        'spread_sharpe',
        'max_drawdown',
        'calmar_ratio',
        'classification_f1',
        'classification_precision',
        'classification_recall',
        'classification_accuracy',
        # Johansen test (independent of hedge ratio method)
        'johansen_trace_stat',
        'johansen_critical_value',
        'is_johansen_cointegrated',
        # Kalman hedge ratio
        'kalman_hedge_ratio',
        # Kalman-based metrics
        'kalman_half_life',
        'kalman_mean_zscore',
        'kalman_std_zscore',
        'kalman_skewness',
        'kalman_kurtosis',
        'kalman_current_zscore',
        'kalman_hurst_exponent',
        'kalman_spread_sharpe',
        'kalman_max_drawdown',
        'kalman_calmar_ratio',
        'kalman_classification_f1',
        'kalman_classification_precision',
        'kalman_classification_recall',
        'kalman_classification_accuracy',
        'long_adx',
        'short_adx',
    ]
    PAIRS_TRADING_ADX_PERIOD = 14
    
    # Fallback logging functions if modules.common.utils is not available
    def log_warn(msg: str) -> None:
        print(f"[WARN] {msg}")
    
    def log_info(msg: str) -> None:
        print(f"[INFO] {msg}")
    
    def log_success(msg: str) -> None:
        print(f"[SUCCESS] {msg}")
    
    def log_progress(msg: str) -> None:
        print(f"[PROGRESS] {msg}")
    
    ProgressBar = None

    def calculate_adx(ohlcv: pd.DataFrame, period: int = 14) -> Optional[float]:
        if ohlcv is None or len(ohlcv) < period * 2:
            return None

        required = {"high", "low", "close"}
        if not required.issubset(ohlcv.columns):
            return None

        data = ohlcv[list(required)].astype(float)
        high = data["high"]
        low = data["low"]
        close = data["close"]

        up_move = high.diff()
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_components = pd.concat(
            [
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)

        atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = (
            pd.Series(plus_dm, index=data.index)
            .ewm(alpha=1 / period, adjust=False)
            .mean()
            * 100
            / atr.replace(0, pd.NA)
        )
        minus_di = (
            pd.Series(minus_dm, index=data.index)
            .ewm(alpha=1 / period, adjust=False)
            .mean()
            * 100
            / atr.replace(0, pd.NA)
        )

        denom = (plus_di + minus_di).replace(0, pd.NA)
        dx = ((plus_di - minus_di).abs() / denom) * 100
        adx = dx.ewm(alpha=1 / period, adjust=False).mean().dropna()

        if adx.empty:
            return None

        last_value = adx.iloc[-1]
        if pd.isna(last_value) or np.isinf(last_value):
            return None

        return float(last_value)

from modules.pairs_trading.core.pair_metrics_computer import PairMetricsComputer
from modules.pairs_trading.core.opportunity_scorer import OpportunityScorer


def _get_all_pair_columns() -> list:
    """
    Get all column names for pair DataFrames.
    
    Returns column list from PAIRS_TRADING_PAIR_COLUMNS constant.
    This includes core pair information (long_symbol, short_symbol, spread, etc.)
    and all quantitative metrics from PairMetricsComputer.
    """
    try:
        return PAIRS_TRADING_PAIR_COLUMNS.copy()
    except NameError:
        # Fallback if constant not imported
        return [
            'long_symbol',
            'short_symbol',
            'long_score',
            'short_score',
            'spread',
            'correlation',
            'opportunity_score',
            'quantitative_score',
            'hedge_ratio',
            'adf_pvalue',
            'is_cointegrated',
            'half_life',
            'mean_zscore',
            'std_zscore',
            'skewness',
            'kurtosis',
            'current_zscore',
            'hurst_exponent',
            'spread_sharpe',
            'max_drawdown',
            'calmar_ratio',
            'classification_f1',
            'classification_precision',
            'classification_recall',
            'classification_accuracy',
            'johansen_trace_stat',
            'johansen_critical_value',
            'is_johansen_cointegrated',
            'kalman_hedge_ratio',
            'kalman_half_life',
            'kalman_mean_zscore',
            'kalman_std_zscore',
            'kalman_skewness',
            'kalman_kurtosis',
            'kalman_current_zscore',
            'kalman_hurst_exponent',
            'kalman_spread_sharpe',
            'kalman_max_drawdown',
            'kalman_calmar_ratio',
            'kalman_classification_f1',
            'kalman_classification_precision',
            'kalman_classification_recall',
            'kalman_classification_accuracy',
            'long_adx',
            'short_adx',
        ]


class PairsTradingAnalyzer:
    """
    Analyzes pairs trading opportunities from best and worst performing symbols.
    
    Pairs trading strategy:
    - Long worst performers (expect mean reversion upward)
    - Short best performers (expect mean reversion downward)
    """

    def __init__(
        self,
        min_spread: float = PAIRS_TRADING_MIN_SPREAD,
        max_spread: float = PAIRS_TRADING_MAX_SPREAD,
        min_correlation: float = PAIRS_TRADING_MIN_CORRELATION,
        max_correlation: float = PAIRS_TRADING_MAX_CORRELATION,
        correlation_min_points: int = PAIRS_TRADING_CORRELATION_MIN_POINTS,
        require_cointegration: bool = False,
        max_half_life: float = PAIRS_TRADING_MAX_HALF_LIFE,
        hurst_threshold: float = PAIRS_TRADING_HURST_THRESHOLD,
        min_spread_sharpe: Optional[float] = None,
        max_drawdown_threshold: Optional[float] = None,
        min_quantitative_score: Optional[float] = None,
        ols_fit_intercept: bool = PAIRS_TRADING_OLS_FIT_INTERCEPT,
        kalman_delta: float = PAIRS_TRADING_KALMAN_DELTA,
        kalman_obs_cov: float = PAIRS_TRADING_KALMAN_OBS_COV,
        scoring_multipliers: Optional[Dict[str, float]] = None,
        strategy: str = "reversion",
    ):
        """
        Initialize PairsTradingAnalyzer.

        Args:
            min_spread: Minimum spread (%) between long and short symbols
            max_spread: Maximum spread (%) between long and short symbols
            min_correlation: Minimum correlation to consider a pair
            max_correlation: Maximum correlation (avoid over-correlated pairs)
            correlation_min_points: Minimum data points for correlation calculation
            require_cointegration: If True, only accept cointegrated pairs
            max_half_life: Maximum acceptable half-life for mean reversion
            hurst_threshold: Maximum Hurst exponent (should be < 0.5 for mean reversion)
            min_spread_sharpe: Minimum Sharpe ratio (None to disable)
            max_drawdown_threshold: Maximum drawdown threshold (None to disable)
            min_quantitative_score: Minimum quantitative score (0-100, None to disable)
            strategy: Trading strategy ('reversion' or 'momentum')
        """
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.correlation_min_points = correlation_min_points
        self.require_cointegration = require_cointegration
        self.max_half_life = max_half_life
        self.hurst_threshold = hurst_threshold
        self.min_spread_sharpe = min_spread_sharpe if min_spread_sharpe is not None else PAIRS_TRADING_MIN_SPREAD_SHARPE
        self.max_drawdown_threshold = max_drawdown_threshold if max_drawdown_threshold is not None else PAIRS_TRADING_MAX_DRAWDOWN
        self.min_quantitative_score = min_quantitative_score
        self.ols_fit_intercept = ols_fit_intercept
        self.kalman_delta = kalman_delta
        self.kalman_obs_cov = kalman_obs_cov
        self.scoring_multipliers = scoring_multipliers
        self.adf_pvalue_threshold = PAIRS_TRADING_ADF_PVALUE_THRESHOLD
        self.johansen_confidence = PAIRS_TRADING_JOHANSEN_CONFIDENCE
        self.min_calmar = PAIRS_TRADING_MIN_CALMAR
        self._correlation_cache: Dict[Tuple[str, str], float] = {}
        self._price_cache: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
        self._adx_cache: Dict[str, Optional[float]] = {}
        self.adx_period = PAIRS_TRADING_ADX_PERIOD
        
        # Initialize metrics computer and opportunity scorer
        self.metrics_computer = PairMetricsComputer(
            adf_pvalue_threshold=self.adf_pvalue_threshold,
            periods_per_year=PAIRS_TRADING_PERIODS_PER_YEAR,
            zscore_lookback=PAIRS_TRADING_ZSCORE_LOOKBACK,
            classification_zscore=PAIRS_TRADING_CLASSIFICATION_ZSCORE,
            johansen_confidence=self.johansen_confidence,
            correlation_min_points=correlation_min_points,
            ols_fit_intercept=self.ols_fit_intercept,
            kalman_delta=self.kalman_delta,
            kalman_obs_cov=self.kalman_obs_cov,
        )
        
        self.opportunity_scorer = OpportunityScorer(
            min_correlation=self.min_correlation,
            max_correlation=self.max_correlation,
            adf_pvalue_threshold=self.adf_pvalue_threshold,
            max_half_life=self.max_half_life,
            hurst_threshold=self.hurst_threshold,
            min_spread_sharpe=self.min_spread_sharpe,
            max_drawdown_threshold=self.max_drawdown_threshold,
            min_calmar=self.min_calmar,
            scoring_multipliers=self.scoring_multipliers,
            strategy=strategy,
        )

    def _fetch_aligned_prices(
        self,
        symbol1: str,
        symbol2: str,
        data_fetcher,
        timeframe: str = PAIRS_TRADING_TIMEFRAME,
        limit: int = PAIRS_TRADING_LIMIT,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch and align price series for two symbols.
        Returns DataFrame with columns ['close1', 'close2'] aligned by timestamp/index.
        """
        cache_key = tuple(sorted([symbol1, symbol2]))
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        if data_fetcher is None:
            return None

        try:
            df1, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol1, limit=limit, timeframe=timeframe, check_freshness=False
            )
            df2, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol2, limit=limit, timeframe=timeframe, check_freshness=False
            )
        except Exception:
            self._price_cache[cache_key] = None
            return None

        if (
            df1 is None
            or df2 is None
            or df1.empty
            or df2.empty
            or "close" not in df1.columns
            or "close" not in df2.columns
        ):
            self._price_cache[cache_key] = None
            return None

        if "timestamp" not in df1.columns or "timestamp" not in df2.columns:
            log_warn(
                f"Missing timestamp column for {symbol1} or {symbol2}. "
                "Cannot align series reliably."
            )
            self._price_cache[cache_key] = None
            return None

        df1 = df1.set_index("timestamp")
        df2 = df2.set_index("timestamp")
        df_combined = pd.concat(
            [df1[["close"]], df2[["close"]]], axis=1, join="inner"
        )
        df_combined.columns = ["close1", "close2"]

        if len(df_combined) < self.correlation_min_points:
            self._price_cache[cache_key] = None
            return None

        self._price_cache[cache_key] = df_combined
        return df_combined

    def _get_symbol_adx(self, symbol: str, data_fetcher) -> Optional[float]:
        """
        Retrieve cached ADX for symbol, computing if necessary.
        """
        if symbol in self._adx_cache:
            return self._adx_cache[symbol]

        if data_fetcher is None:
            self._adx_cache[symbol] = None
            return None
        try:
            ohlcv, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=PAIRS_TRADING_LIMIT,
                timeframe=PAIRS_TRADING_TIMEFRAME,
                check_freshness=False,
            )
        except Exception:
            return None

        if (
            ohlcv is None
            or ohlcv.empty
            or not {"high", "low", "close"}.issubset(ohlcv.columns)
        ):
            self._adx_cache[symbol] = None
            return None

        adx_value = calculate_adx(ohlcv, self.adx_period)
        self._adx_cache[symbol] = adx_value
        return adx_value

    def calculate_correlation(
        self,
        symbol1: str,
        symbol2: str,
        data_fetcher,
        timeframe: str = PAIRS_TRADING_TIMEFRAME,
        limit: int = PAIRS_TRADING_LIMIT,
    ) -> Optional[float]:
        """
        Calculate correlation between two symbols based on returns.

        Args:
            symbol1: First trading symbol
            symbol2: Second trading symbol
            data_fetcher: DataFetcher instance
            timeframe: Timeframe for OHLCV data
            limit: Number of candles to fetch

        Returns:
            Correlation coefficient (float between -1 and 1), or None if calculation fails
        """
        # Check cache
        cache_key = tuple(sorted([symbol1, symbol2]))
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]

        df_combined = self._fetch_aligned_prices(
            symbol1, symbol2, data_fetcher, timeframe=timeframe, limit=limit
        )

        if df_combined is None:
            return None

        try:
            returns = df_combined.pct_change().dropna()

            if len(returns) < self.correlation_min_points:
                return None

            # Calculate correlation
            correlation = returns["close1"].corr(returns["close2"])

            if pd.isna(correlation):
                return None

            # Cache result
            self._correlation_cache[cache_key] = float(correlation)

            return float(correlation)

        except Exception:
            return None

    def _compute_pair_metrics(
        self,
        symbol1: str,
        symbol2: str,
        data_fetcher,
    ) -> Dict[str, Optional[float]]:
        """Compute Phase 1 quantitative metrics for a pair."""
        aligned_prices = self._fetch_aligned_prices(
            symbol1, symbol2, data_fetcher, timeframe=PAIRS_TRADING_TIMEFRAME
        )
        if aligned_prices is None:
            return {}

        price1 = aligned_prices["close1"]
        price2 = aligned_prices["close2"]

        metrics = self.metrics_computer.compute_pair_metrics(price1, price2)
        metrics["long_adx"] = self._get_symbol_adx(symbol1, data_fetcher)
        metrics["short_adx"] = self._get_symbol_adx(symbol2, data_fetcher)
        return metrics

    def calculate_spread(
        self, long_symbol: str, short_symbol: str, long_score: float, short_score: float
    ) -> float:
        """
        Calculate spread between long and short symbols based on performance scores.

        Args:
            long_symbol: Symbol to long (worst performer)
            short_symbol: Symbol to short (best performer)
            long_score: Performance score of long symbol (typically negative)
            short_score: Performance score of short symbol (typically positive)

        Returns:
            Spread as percentage (positive value)
        """
        # Spread = difference between short score and long score
        # Since long_score is negative and short_score is positive,
        # spread = short_score - long_score (which is positive)
        spread = short_score - long_score
        return abs(spread)  # Ensure positive

    def analyze_pairs_opportunity(
        self,
        best_performers: pd.DataFrame,
        worst_performers: pd.DataFrame,
        data_fetcher=None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Analyze pairs trading opportunities from best and worst performers.

        Args:
            best_performers: DataFrame with top performers (columns: symbol, score, etc.)
            worst_performers: DataFrame with worst performers (columns: symbol, score, etc.)
            data_fetcher: Optional DataFetcher for correlation calculation
            verbose: If True, print progress messages

        Returns:
            DataFrame with columns: ['long_symbol', 'short_symbol', 'long_score', 
                                   'short_score', 'spread', 'correlation', 'opportunity_score']
        """
        empty_df = pd.DataFrame(columns=_get_all_pair_columns())
        
        if best_performers is None or best_performers.empty:
            if verbose:
                log_warn("No best performers provided for pairs analysis.")
            return empty_df

        if worst_performers is None or worst_performers.empty:
            if verbose:
                log_warn("No worst performers provided for pairs analysis.")
            return empty_df

        pairs = []

        if verbose:
            log_info(
                f"Analyzing pairs: {len(worst_performers)} worst × {len(best_performers)} best = "
                f"{len(worst_performers) * len(best_performers)} combinations..."
            )

        progress = None
        total_combinations = len(worst_performers) * len(best_performers)
        if verbose and ProgressBar:
            progress = ProgressBar(total_combinations, "Pairs Analysis")

        for _, worst_row in worst_performers.iterrows():
            long_symbol = worst_row['symbol']
            long_score = worst_row['score']

            for _, best_row in best_performers.iterrows():
                short_symbol = best_row['symbol']
                short_score = best_row['score']

                # Skip if same symbol
                if long_symbol == short_symbol:
                    if progress:
                        progress.update()
                    continue

                # Calculate spread
                spread = self.calculate_spread(long_symbol, short_symbol, long_score, short_score)

                # Calculate correlation if data_fetcher provided
                correlation = None
                quant_metrics = {}
                if data_fetcher is not None:
                    correlation = self.calculate_correlation(
                        long_symbol, short_symbol, data_fetcher
                    )
                    # Không loại bỏ pair ngay cả khi tương quan ngoài vùng lý tưởng;
                    # OpportunityScorer sẽ tự áp dụng hình phạt thông qua hệ số.
                    quant_metrics = self._compute_pair_metrics(
                        long_symbol, short_symbol, data_fetcher
                    )

                # Calculate opportunity score
                opportunity_score = self.opportunity_scorer.calculate_opportunity_score(
                    spread, correlation, quant_metrics
                )
                
                # Calculate quantitative score
                quantitative_score = self.opportunity_scorer.calculate_quantitative_score(
                    quant_metrics
                )

                # Build pair record
                pair_record = {
                    'long_symbol': long_symbol,
                    'short_symbol': short_symbol,
                    'long_score': float(long_score),
                    'short_score': float(short_score),
                    'spread': float(spread),
                    'correlation': float(correlation) if correlation is not None else None,
                    'opportunity_score': float(opportunity_score),
                    'quantitative_score': float(quantitative_score),
                }
                
                # Add all quant metrics
                for key in _get_all_pair_columns():
                    if key not in pair_record:
                        pair_record[key] = quant_metrics.get(key)

                pairs.append(pair_record)

                if progress:
                    progress.update()

        if progress:
            progress.finish()

        if not pairs:
            if verbose:
                log_warn("No pairs opportunities found.")
            return empty_df

        df_pairs = pd.DataFrame(pairs)
        # Sort by opportunity_score descending
        df_pairs = df_pairs.sort_values('opportunity_score', ascending=False).reset_index(
            drop=True
        )

        if verbose:
            log_success(f"Found {len(df_pairs)} pairs opportunities.")

        return df_pairs

    def validate_pairs(
        self,
        pairs_df: pd.DataFrame,
        data_fetcher,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Validate pairs trading opportunities.

        Checks:
        - Spread is within acceptable range (min_spread to max_spread)
        - Correlation is within acceptable range (if available)
        - Symbols are available and have sufficient volume

        Args:
            pairs_df: DataFrame from analyze_pairs_opportunity()
            data_fetcher: DataFetcher instance for validation
            verbose: If True, print validation messages

        Returns:
            DataFrame with validated pairs only
        """
        if pairs_df is None or pairs_df.empty:
            return pd.DataFrame(columns=_get_all_pair_columns())

        if verbose:
            log_progress(f"Validating {len(pairs_df)} pairs...")

        validated_pairs = []

        progress = None
        if verbose and ProgressBar:
            progress = ProgressBar(len(pairs_df), "Validation")

        for _, row in pairs_df.iterrows():
            long_symbol = row['long_symbol']
            short_symbol = row['short_symbol']
            spread = row['spread']
            correlation = row.get('correlation')

            is_valid = True
            validation_errors = []

            # Check spread
            if spread < self.min_spread:
                is_valid = False
                validation_errors.append(f"Spread too small ({spread*100:.2f}% < {self.min_spread*100:.2f}%)")
            elif spread > self.max_spread:
                is_valid = False
                validation_errors.append(f"Spread too large ({spread*100:.2f}% > {self.max_spread*100:.2f}%)")

            # Check correlation if available
            if correlation is not None and not pd.isna(correlation):
                abs_corr = abs(correlation)
                if abs_corr < self.min_correlation:
                    is_valid = False
                    validation_errors.append(
                        f"Correlation too low ({abs_corr:.3f} < {self.min_correlation:.3f})"
                    )
                elif abs_corr > self.max_correlation:
                    is_valid = False
                    validation_errors.append(
                        f"Correlation too high ({abs_corr:.3f} > {self.max_correlation:.3f})"
                    )

            # Check quantitative metrics if available
            # Cointegration requirement
            if self.require_cointegration:
                is_cointegrated = row.get('is_cointegrated')
                # Fall back to Johansen cointegration flag if ADF-based flag is missing
                if (is_cointegrated is None or pd.isna(is_cointegrated)) and 'is_johansen_cointegrated' in row:
                    alt_coint = row.get('is_johansen_cointegrated')
                    if alt_coint is not None and not pd.isna(alt_coint):
                        is_cointegrated = bool(alt_coint)
                if is_cointegrated is None or pd.isna(is_cointegrated) or not is_cointegrated:
                    is_valid = False
                    validation_errors.append("Not cointegrated (required)")

            # Half-life check
            half_life = row.get('half_life')
            if half_life is not None and not pd.isna(half_life):
                if half_life > self.max_half_life:
                    is_valid = False
                    validation_errors.append(
                        f"Half-life too high ({half_life:.1f} > {self.max_half_life})"
                    )

            # Hurst exponent check
            hurst = row.get('hurst_exponent')
            if hurst is not None and not pd.isna(hurst):
                if hurst >= self.hurst_threshold:
                    is_valid = False
                    validation_errors.append(
                        f"Hurst exponent too high ({hurst:.3f} >= {self.hurst_threshold}, not mean-reverting)"
                    )

            # Sharpe ratio check
            spread_sharpe = row.get('spread_sharpe')
            if spread_sharpe is not None and not pd.isna(spread_sharpe):
                if spread_sharpe < self.min_spread_sharpe:
                    is_valid = False
                    validation_errors.append(
                        f"Sharpe ratio too low ({spread_sharpe:.2f} < {self.min_spread_sharpe})"
                    )

            # Max drawdown check
            max_dd = row.get('max_drawdown')
            if max_dd is not None and not pd.isna(max_dd):
                if abs(max_dd) > self.max_drawdown_threshold:
                    is_valid = False
                    validation_errors.append(
                        f"Max drawdown too high ({abs(max_dd)*100:.2f}% > {self.max_drawdown_threshold*100:.2f}%)"
                    )

            # Quantitative score check
            if self.min_quantitative_score is not None:
                quant_score = row.get('quantitative_score')
                if quant_score is not None and not pd.isna(quant_score):
                    if quant_score < self.min_quantitative_score:
                        is_valid = False
                        validation_errors.append(
                            f"Quantitative score too low ({quant_score:.1f} < {self.min_quantitative_score})"
                        )

            if is_valid:
                validated_pairs.append(row.to_dict())
            elif verbose:
                log_warn(f"Rejected {long_symbol} / {short_symbol}: {', '.join(validation_errors)}")

            if progress:
                progress.update()

        if progress:
            progress.finish()

        if not validated_pairs:
            if verbose:
                log_warn("No pairs passed validation.")
            return pd.DataFrame(columns=_get_all_pair_columns())

        df_validated = pd.DataFrame(validated_pairs)
        # Maintain sort order by opportunity_score
        df_validated = df_validated.sort_values('opportunity_score', ascending=False).reset_index(
            drop=True
        )

        if verbose:
            log_success(f"{len(df_validated)}/{len(pairs_df)} pairs passed validation.")

        return df_validated
