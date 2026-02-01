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

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypedDict, cast

import polars as pl

from modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols import scan_all_symbols
from modules.adaptive_trend_LTS_mini.utils.config import create_atc_config_from_dict
from modules.common.core.data_fetcher import DataFetcher
from modules.common.system import get_hardware_manager
from modules.common.ui.logging import log_error, log_info, log_warn

try:
    import sovereign_prime

    USE_RUST_AGGREGATION = True
except ImportError:
    sovereign_prime = None
    USE_RUST_AGGREGATION = False
    log_warn("ATCScanner: sovereign_prime (Rust) not found. Using Python fallback.")

from threading import RLock


class ATCScannerConfig(TypedDict, total=False):
    """Type definition for ATCScanner configuration."""

    weights: Dict[str, float]  # Timeframe weights, e.g., {"1h": 0.5, "15m": 0.3, "5m": 0.2}
    threshold: float  # Signal threshold (0.0 to 1.0)
    timeframes: List[str]  # Timeframes to scan, e.g., ["1h", "15m", "5m"]
    min_signal: float  # Minimum signal for individual scans
    use_signal_strength: bool  # Whether to use signal strength for scoring (default: False)
    enable_cache: bool  # Enable caching of scan results (default: True)
    cache_ttl_seconds: int  # Cache time-to-live in seconds (default: 60)
    batch_size: int  # Batch size for processing large symbol lists (default: 50)
    use_rust_cache: bool  # Use Rust ScanCache for 10-20x performance improvement (default: True)


# Signal Structure
class SignalResult(NamedTuple):
    """Result of multi-timeframe signal aggregation."""

    symbol: str
    score: float  # Aggregated score from -1.0 to +1.0
    signal_type: str  # "LONG", "SHORT", "NEUTRAL"
    details: Dict[str, Any]  # Per-timeframe signals: {"5m": "LONG", "15m": "SHORT"}
    strengths: Dict[str, float]  # Per-timeframe signal strengths: {"5m": 0.8, "15m": -0.4}


class ATCScanner:
    """Scans symbols across multiple timeframes and aggregates results.

    Uses weighted voting across timeframes to generate unified signals.

    Architecture:
    - Data Processing: Polars DataFrames (converted from Pandas at boundary).
    - Hot Path: Rust-optimized signal aggregation (via sovereign_prime) if available.
    - Fallback: Pure Python implementation if Rust extension is missing.

    Security Note:
    - Data Privacy: Scan results and cache entries are stored in-memory without encryption.
    - DoS Protection: Both Rust and Python caches implement capacity limits (1000 and 100 entries respectively)
      with LRU eviction to prevent memory exhaustion attacks.
    """

    # Class constant for empty scan result
    _EMPTY_SCAN_RESULT: Dict[str, Any] = {"longs": set(), "shorts": set(), "strengths": {}}
    _EMPTY_DF_SCHEMA = {"symbol": pl.Utf8, "signal": pl.Float64}

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
        self.config: ATCScannerConfig = cast(ATCScannerConfig, config or {})

        # Configurable timeframes
        self.timeframes: List[str] = self.config.get("timeframes", ["1h", "15m", "5m"])

        # Minimum signal for individual scans
        self.min_signal: float = self.config.get("min_signal", 0.0)

        # Caching configuration
        self.enable_cache: bool = self.config.get("enable_cache", True)
        self.cache_ttl_seconds: int = self.config.get("cache_ttl_seconds", 60)

        # Initialize Cache Strategy (Task 11)
        self._use_rust_cache = False
        self._cache = {}
        self._cache_lock = RLock()
        self._rust_cache = None

        # Check if user wants Rust cache (default: True if available)
        use_rust_cache_config = self.config.get("use_rust_cache", True)

        if self.enable_cache and use_rust_cache_config and USE_RUST_AGGREGATION and sovereign_prime:
            try:
                self._rust_cache = sovereign_prime.ScanCache(capacity=1000, ttl_seconds=float(self.cache_ttl_seconds))
                self._use_rust_cache = True
                log_info("ATCScanner: Using Rust ScanCache (thread-safe, high-performance)")
            except Exception as e:
                log_warn(f"ATCScanner: Rust cache initialization failed: {e}. Using Python cache.")
                self._use_rust_cache = False
        elif self.enable_cache and not use_rust_cache_config:
            log_info("ATCScanner: use_rust_cache=False, using Python cache")

        # Batch processing configuration
        self.batch_size: int = self.config.get("batch_size", 50)

        # Max workers for parallel scanning (auto-detected)
        self.max_workers: Optional[int] = None
        try:
            self._configure_max_workers()
        except Exception as e:
            log_warn(f"Failed to auto-detect max_workers: {e}. Using default.")
            self.max_workers = len(self.timeframes)

        # Use signal strength for scoring
        self.use_signal_strength: bool = self.config.get("use_signal_strength", False)

        # Default Weights
        self.weights: Dict[str, float] = self.config.get("weights", {"1h": 0.5, "15m": 0.3, "5m": 0.2})

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

        # Validate timeframe-weight alignment
        weight_tfs = set(self.weights.keys())
        config_tfs = set(self.timeframes)

        missing_weights = config_tfs - weight_tfs
        if missing_weights:
            log_warn(f"Timeframes without weights (will use 0.0): {missing_weights}")

        extra_weights = weight_tfs - config_tfs
        if extra_weights:
            log_warn(f"Weights for unused timeframes (will be ignored): {extra_weights}")

    def _calculate_weighted_score(self, signal_type: str, tf_weight: float, strength: float) -> float:
        """Calculate weighted score for a timeframe signal.

        Args:
            signal_type: "LONG", "SHORT", or "NEUTRAL"
            tf_weight: Weight for this timeframe (0.0 to 1.0)
            strength: Signal strength (-1.0 to +1.0)

        Returns:
            Weighted score contribution (positive for LONG, negative for SHORT)

        Examples:
            >>> _calculate_weighted_score("LONG", 0.5, 0.8)
            0.4  # 0.5 * 0.8 = +0.4 (bullish)

            >>> _calculate_weighted_score("SHORT", 0.5, -0.8)
            -0.4  # 0.5 * -0.8 = -0.4 (bearish)
        """
        if USE_RUST_AGGREGATION and sovereign_prime:
            try:
                return sovereign_prime.calculate_weighted_score(
                    signal_type, tf_weight, strength, self.use_signal_strength
                )
            except Exception as e:
                log_warn(f"Rust calculate_weighted_score failed: {e}. Falling back to Python.")

        if signal_type == "LONG":
            if self.use_signal_strength:
                return tf_weight * abs(strength)
            else:
                return tf_weight
        elif signal_type == "SHORT":
            if self.use_signal_strength:
                # strength is already negative for shorts
                return tf_weight * strength
            else:
                return -tf_weight
        else:
            return 0.0

    def _configure_max_workers(self) -> None:
        """Configure max_workers using hardware manager (auto-detected)."""

        # Auto-detect based on hardware
        hw_manager = get_hardware_manager()
        # Use a representative workload size (number of timeframes as proxy)
        workload_config = hw_manager.get_optimal_workload_config(
            workload_size=len(self.timeframes),
            prefer_gpu=False,  # ATCScanner uses thread-based I/O, not GPU
        )
        self.max_workers = workload_config.num_threads
        cpu_cores = hw_manager.get_cpu_cores()
        log_info(f"ATCScanner: Auto-detected max_workers={self.max_workers} (based on {cpu_cores} CPU cores)")

    def _get_cache_key(self, symbols: List[str], timeframe: str) -> str:
        """Generate cache key for scan results.

        Args:
            symbols: List of symbols being scanned
            timeframe: Timeframe string

        Returns:
            Cache key string based on symbols and timeframe with minute precision
        """
        # Cache valid for TTL duration (aligned to minute boundaries)
        minute = datetime.now().replace(second=0, microsecond=0)
        # Sort symbols for consistent key generation
        symbol_key = ",".join(sorted(symbols))
        return f"{symbol_key}_{timeframe}_{minute}"

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """Get cached scan result if valid.

        Args:
            cache_key: Cache key generated by _get_cache_key

        Returns:
            Cached result dict if valid, None otherwise
        """
        if not self.enable_cache:
            return None

        if self._use_rust_cache and self._rust_cache:
            try:
                result = self._rust_cache.get(cache_key)
                if result:
                    log_info(f"ATCScanner: Rust cache HIT for {cache_key.split('_')[-2]}")
                    return result
                return None
            except Exception as e:
                log_error(f"ATCScanner: Rust cache get failed: {e}")
                return None

        # Fallback to Python cache with lock
        with self._cache_lock:
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                # Check if cache is still valid
                if time.time() - timestamp < self.cache_ttl_seconds:
                    log_info(f"ATCScanner: Using cached result for {cache_key.split('_')[-2]}")
                    return cached_data
                else:
                    # Remove expired cache entry
                    del self._cache[cache_key]

        return None

    def _set_cache(self, cache_key: str, data: Dict[str, Dict[str, Any]]) -> None:
        """Store scan result in cache.

        Args:
            cache_key: Cache key generated by _get_cache_key
            data: Scan result data to cache
        """
        if not self.enable_cache:
            return

        if self._use_rust_cache and self._rust_cache:
            try:
                # Extract components for Rust cache
                longs = data.get("longs", set())
                shorts = data.get("shorts", set())
                strengths = data.get("strengths", {})
                self._rust_cache.set(cache_key, longs, shorts, strengths)
            except Exception as e:
                log_error(f"ATCScanner: Rust cache set failed: {e}")
            return

        # Fallback to Python cache with lock
        with self._cache_lock:
            self._cache[cache_key] = (data, time.time())
            # Clean up old cache entries (simple LRU-like approach)
            if len(self._cache) > 100:  # Keep max 100 entries
                # Remove oldest entries
                sorted_keys = sorted(self._cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_keys[:20]:  # Remove 20 oldest
                    del self._cache[key]

    def clear_cache(self) -> None:
        """Clear all cached scan results."""
        if self._use_rust_cache and self._rust_cache:
            try:
                self._rust_cache.clear()
                log_info("ATCScanner: Rust cache cleared")
            except Exception as e:
                log_error(f"ATCScanner: Rust cache clear failed: {e}")

        with self._cache_lock:
            self._cache.clear()
            log_info("ATCScanner: Python cache cleared")

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        if self._use_rust_cache and self._rust_cache:
            return {
                "type": "rust",
                "size": self._rust_cache.len(),
                "capacity": self._rust_cache.capacity(),
            }
        else:
            with self._cache_lock:
                return {
                    "type": "python",
                    "size": len(self._cache),
                    "capacity": 100,
                }

    def scan_symbols(self, symbols: List[str]) -> List[SignalResult]:
        """
        Scan a list of symbols across configured timeframes.

        Uses weighted voting to aggregate signals across timeframes.
        Score range is -1.0 to +1.0 when weights sum to 1.0.

        Supports batch processing for large symbol lists and caching for performance.

        Args:
            symbols: List of symbol strings to scan.

        Returns:
            List of SignalResult objects for symbols exceeding threshold.
        """
        # Handle large symbol lists with batch processing
        if len(symbols) > self.batch_size:
            log_info(f"ATCScanner: Processing {len(symbols)} symbols in batches of {self.batch_size}")
            return self._scan_symbols_batched(symbols)

        # Process normally for small lists
        return self._scan_symbols_internal(symbols)

    def _scan_symbols_batched(self, symbols: List[str]) -> List[SignalResult]:
        """Scan symbols in batches to control memory usage.

        Args:
            symbols: List of symbol strings to scan

        Returns:
            Aggregated list of SignalResult objects from all batches
        """
        all_results: List[SignalResult] = []
        total_batches = (len(symbols) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(symbols), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch = symbols[i : i + self.batch_size]
            log_info(f"ATCScanner: Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)")

            batch_results = self._scan_symbols_internal(batch)
            all_results.extend(batch_results)

        log_info(f"ATCScanner: Completed batch processing. Total signals: {len(all_results)}")
        return all_results

    def _scan_symbols_internal(self, symbols: List[str]) -> List[SignalResult]:
        """Internal method to scan symbols (with caching support).

        Args:
            symbols: List of symbol strings to scan

        Returns:
            List of SignalResult objects for symbols exceeding threshold
        """
        results_by_tf: Dict[str, Dict[str, Any]] = {}

        # Run scans for each timeframe in parallel

        # Use configured max_workers (always set in __init__)
        workers = self.max_workers or len(self.timeframes)
        workers = max(1, workers)  # Ensure at least 1 worker

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_tf = {executor.submit(self._run_single_scan, symbols, tf): tf for tf in self.timeframes}

            for future in as_completed(future_to_tf):
                tf = future_to_tf[future]
                try:
                    longs, shorts = future.result()
                    log_info(f"ATCScanner: Finished scanning {tf}.")

                    # Extract signals with strength
                    long_signals = {}
                    if not longs.is_empty():
                        # Provide default if 'signal' column missing (regression safety)
                        if "signal" in longs.columns:
                            long_signals = dict(zip(longs["symbol"].to_list(), longs["signal"].to_list()))
                        else:
                            long_signals = {sym: 1.0 for sym in longs["symbol"].to_list()}

                    short_signals = {}
                    if not shorts.is_empty():
                        if "signal" in shorts.columns:
                            short_signals = dict(zip(shorts["symbol"].to_list(), shorts["signal"].to_list()))
                        else:
                            short_signals = {sym: -1.0 for sym in shorts["symbol"].to_list()}

                    results_by_tf[tf] = {
                        "longs": set(long_signals.keys()),
                        "shorts": set(short_signals.keys()),
                        "strengths": {**long_signals, **short_signals},
                    }
                except Exception as e:
                    log_error(f"ATCScanner: Parallel scan failed for {tf}: {e}")
                    results_by_tf[tf] = self._EMPTY_SCAN_RESULT.copy()

        # Aggregate results using weighted voting
        final_signals: List[SignalResult] = []

        if USE_RUST_AGGREGATION and sovereign_prime:
            try:
                rust_results = sovereign_prime.aggregate_signals(
                    symbols,
                    results_by_tf,
                    self.weights,
                    self.threshold,
                    self.use_signal_strength,
                )

                # Convert Rust dictionaries to SignalResult objects
                for res in rust_results:
                    final_signals.append(
                        SignalResult(
                            symbol=res["symbol"],
                            score=res["score"],
                            signal_type=res["signal_type"],
                            details=res["details"],
                            strengths=res["strengths"],
                        )
                    )

                log_info(
                    f"ATCScanner: Found {len(final_signals)} signals (via Rust) exceeding threshold {self.threshold}."
                )
                return final_signals
            except Exception as e:
                log_error(f"ATCScanner: Rust aggregation failed: {e}. Falling back to Python.")
                final_signals = []  # Reset for Python fallback

        for symbol in symbols:
            score = 0.0
            details: Dict[str, Any] = {}
            strengths: Dict[str, float] = {}

            for tf in self.timeframes:
                res = results_by_tf.get(tf, self._EMPTY_SCAN_RESULT.copy())
                tf_weight = self.weights.get(tf, 0.0)

                # Get signal strength if available, else use default direction
                strength = res.get("strengths", {}).get(symbol, 0.0)
                strengths[tf] = strength

                if symbol in res["longs"]:
                    score += self._calculate_weighted_score("LONG", tf_weight, strength)
                    details[tf] = "LONG"
                elif symbol in res["shorts"]:
                    score += self._calculate_weighted_score("SHORT", tf_weight, strength)
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
                        strengths=strengths,
                    )
                )

        log_info(f"ATCScanner: Found {len(final_signals)} signals exceeding threshold {self.threshold}.")
        return final_signals

    def _run_single_scan(self, symbols: List[str], timeframe: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Run ATC scan for a single timeframe (with caching support).

        Args:
            symbols: List of symbols to scan
            timeframe: Timeframe string (e.g., "15m", "1h")

        Returns:
            Tuple of (long_signals_df, short_signals_df)
        """
        # Check cache first
        cache_key = self._get_cache_key(symbols, timeframe)
        cached_result = self._get_cached_result(cache_key)

        if cached_result is not None:
            # Reconstruct DataFrames from cached data
            longs_data = [
                {"symbol": sym, "signal": strength}
                for sym, strength in cached_result.get("strengths", {}).items()
                if sym in cached_result.get("longs", set())
            ]
            shorts_data = [
                {"symbol": sym, "signal": strength}
                for sym, strength in cached_result.get("strengths", {}).items()
                if sym in cached_result.get("shorts", set())
            ]

            # Task 4: Reconstruct using Polars
            if not longs_data:
                longs_df = pl.DataFrame(schema=self._EMPTY_DF_SCHEMA)
            else:
                longs_df = pl.DataFrame(longs_data)

            if not shorts_data:
                shorts_df = pl.DataFrame(schema=self._EMPTY_DF_SCHEMA)
            else:
                shorts_df = pl.DataFrame(shorts_data)

            return longs_df, shorts_df

        # Filter out scanner-specific config keys
        excluded_keys = {
            "weights",
            "threshold",
            "timeframes",
            "min_signal",
            "max_workers",
            "use_signal_strength",
            "enable_cache",
            "cache_ttl_seconds",
            "batch_size",
        }
        clean_params = {k: v for k, v in self.config.items() if k not in excluded_keys}

        try:
            atc_config = create_atc_config_from_dict(clean_params, timeframe=timeframe)
        except (ValueError, KeyError) as e:
            # Configuration errors should propagate
            raise ValueError(f"Invalid ATC config for {timeframe}: {e}") from e

        try:
            long_pd, short_pd = scan_all_symbols(
                data_fetcher=self.data_fetcher,
                atc_config=atc_config,
                symbols=symbols,
                min_signal=self.min_signal,
            )

            # Task 5: Convert results to Polars
            long_signals = pl.from_pandas(long_pd)
            short_signals = pl.from_pandas(short_pd)

            # Schema validation
            expected_cols = {"symbol", "signal"}
            if not long_signals.is_empty() and not expected_cols.issubset(long_signals.columns):
                log_warn(f"Missing columns in longs: {expected_cols - set(long_signals.columns)}")

            # Cache the results
            if not long_signals.is_empty() or not short_signals.is_empty():
                # Extract data for caching
                long_data = {}
                if not long_signals.is_empty():
                    if "signal" in long_signals.columns:
                        long_data = dict(zip(long_signals["symbol"].to_list(), long_signals["signal"].to_list()))
                    else:
                        long_data = {sym: 1.0 for sym in long_signals["symbol"].to_list()}

                short_data = {}
                if not short_signals.is_empty():
                    if "signal" in short_signals.columns:
                        short_data = dict(zip(short_signals["symbol"].to_list(), short_signals["signal"].to_list()))
                    else:
                        short_data = {sym: -1.0 for sym in short_signals["symbol"].to_list()}

                cache_data = {
                    "longs": set(long_data.keys()),
                    "shorts": set(short_data.keys()),
                    "strengths": {**long_data, **short_data},
                }
                self._set_cache(cache_key, cache_data)

            return long_signals, short_signals
        except Exception as e:
            log_error(f"ATCScanner: Error scanning {timeframe}: {e}")
            # Task 6: Return empty Polars DataFrames
            return pl.DataFrame(schema=self._EMPTY_DF_SCHEMA), pl.DataFrame(schema=self._EMPTY_DF_SCHEMA)
