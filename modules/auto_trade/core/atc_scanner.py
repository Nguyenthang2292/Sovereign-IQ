"""
ATC Multi-Timeframe Scanner

Responsible for:
- Scanning symbols across multiple timeframes (5m, 15m, 1h) using ATC.
- Aggregating signals with weighted voting.
- Returning a unified signal score.

Score semantics:
- Score range: -1.0 to +1.0 (when weights sum to 1.0)
- LONG signal: score > threshold
- SHORT signal: score < -threshold
- NEUTRAL: otherwise
"""

from typing import Dict, List, NamedTuple, Optional, Tuple, TypedDict

import pandas as pd

from modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols import scan_all_symbols
from modules.adaptive_trend_LTS_mini.utils.config import create_atc_config_from_dict
from modules.common.core.data_fetcher import DataFetcher
from modules.common.ui.logging import log_error, log_info, log_warn


class ATCScannerConfig(TypedDict, total=False):
    """Type definition for ATCScanner configuration."""

    weights: Dict[str, float]  # Timeframe weights, e.g., {"1h": 0.5, "15m": 0.3, "5m": 0.2}
    threshold: float  # Signal threshold (0.0 to 1.0)
    timeframes: List[str]  # Timeframes to scan, e.g., ["1h", "15m", "5m"]
    min_signal: float  # Minimum signal for individual scans


# Signal Structure
class SignalResult(NamedTuple):
    """Result of multi-timeframe signal aggregation."""

    symbol: str
    score: float  # Aggregated score from -1.0 to +1.0
    signal_type: str  # "LONG", "SHORT", "NEUTRAL"
    details: Dict[str, str]  # Per-timeframe signals: {"5m": "LONG", "15m": "SHORT"}


class ATCScanner:
    """Scans symbols across multiple timeframes and aggregates results.

    Uses weighted voting across timeframes to generate unified signals.
    """

    def __init__(self, data_fetcher: DataFetcher, config: Optional[ATCScannerConfig] = None):
        """
        Initialize ATCScanner.

        Args:
            data_fetcher: DataFetcher instance for fetching market data
            config: Configuration dictionary containing ATC params and weights

        Raises:
            ValueError: If weights are invalid or threshold is out of range
        """
        self.data_fetcher = data_fetcher
        self.config: ATCScannerConfig = config or {}

        # Configurable timeframes
        self.timeframes: List[str] = self.config.get("timeframes", ["1h", "15m", "5m"])

        # Minimum signal for individual scans
        self.min_signal: float = self.config.get("min_signal", 0.0)

        # Default Weights
        self.weights: Dict[str, float] = self.config.get(
            "weights", {"1h": 0.5, "15m": 0.3, "5m": 0.2}
        )

        # Validate weights
        self._validate_weights()

        # Threshold for final decision
        self.threshold: float = self.config.get("threshold", 0.6)

        # Validate threshold
        if not 0 <= self.threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {self.threshold}")

    def _validate_weights(self) -> None:
        """Validate weights configuration.

        Raises:
            ValueError: If weights are negative or sum to zero
        """
        if not all(w >= 0 for w in self.weights.values()):
            raise ValueError("All weights must be non-negative")

        total_weight = sum(self.weights.values())
        if total_weight == 0:
            raise ValueError("Weights cannot sum to zero")

        # Warn if weights don't sum to 1.0
        if abs(total_weight - 1.0) > 0.01:
            log_warn(f"Weights sum to {total_weight}, not 1.0. Consider normalizing.")

    def scan_symbols(self, symbols: List[str]) -> List[SignalResult]:
        """
        Scan a list of symbols across configured timeframes.

        Uses weighted voting to aggregate signals across timeframes.
        Score range is -1.0 to +1.0 when weights sum to 1.0.

        Args:
            symbols: List of symbol strings to scan.

        Returns:
            List of SignalResult objects for symbols exceeding threshold.
        """
        results_by_tf: Dict[str, Dict[str, set]] = {}

        # Run scans for each timeframe sequentially
        # scan_all_symbols handles internal parallelism
        for tf in self.timeframes:
            log_info(f"ATCScanner: Scanning timeframe {tf}...")
            longs, shorts = self._run_single_scan(symbols, tf)
            results_by_tf[tf] = {
                "longs": set(longs["symbol"]) if not longs.empty else set(),
                "shorts": set(shorts["symbol"]) if not shorts.empty else set(),
            }

        # Aggregate results using weighted voting
        final_signals: List[SignalResult] = []
        for symbol in symbols:
            score = 0.0
            details: Dict[str, str] = {}

            for tf in self.timeframes:
                res = results_by_tf[tf]
                if symbol in res["longs"]:
                    score += self.weights.get(tf, 0.0)
                    details[tf] = "LONG"
                elif symbol in res["shorts"]:
                    score -= self.weights.get(tf, 0.0)
                    details[tf] = "SHORT"
                else:
                    details[tf] = "NEUTRAL"

            # Apply threshold to determine final signal
            signal_type = "NEUTRAL"
            if score > self.threshold:
                signal_type = "LONG"
            elif score < -self.threshold:
                signal_type = "SHORT"

            # Only include non-neutral signals
            if signal_type != "NEUTRAL":
                final_signals.append(
                    SignalResult(
                        symbol=symbol,
                        score=round(score, 2),
                        signal_type=signal_type,
                        details=details,
                    )
                )

        log_info(f"ATCScanner: Found {len(final_signals)} signals exceeding threshold {self.threshold}.")
        return final_signals

    def _run_single_scan(
        self, symbols: List[str], timeframe: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run ATC scan for a single timeframe.

        Args:
            symbols: List of symbols to scan
            timeframe: Timeframe string (e.g., "15m", "1h")

        Returns:
            Tuple of (long_signals_df, short_signals_df)
        """
        # Filter out scanner-specific config keys
        excluded_keys = {"weights", "threshold", "timeframes", "min_signal"}
        clean_params = {k: v for k, v in self.config.items() if k not in excluded_keys}

        try:
            atc_config = create_atc_config_from_dict(clean_params, timeframe=timeframe)
        except (ValueError, KeyError) as e:
            # Configuration errors should propagate
            raise ValueError(f"Invalid ATC config for {timeframe}: {e}") from e

        try:
            long_signals, short_signals = scan_all_symbols(
                data_fetcher=self.data_fetcher,
                atc_config=atc_config,
                symbols=symbols,
                min_signal=self.min_signal,
            )
            return long_signals, short_signals
        except Exception as e:
            log_error(f"ATCScanner: Error scanning {timeframe}: {e}")
            # Return empty DataFrames to allow other timeframes to complete
            return pd.DataFrame(), pd.DataFrame()
