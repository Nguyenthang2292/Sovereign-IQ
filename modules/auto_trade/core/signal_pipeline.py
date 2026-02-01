"""
Signal Pipeline Orchestrator

Coordinates the entire auto-trading process:
1. Refresh Symbols
2. Scan Market (ATC)
3. Filter Signals (XGBoost)
4. AI Analysis (Gemini)
5. Select Final Signal

Example:
    >>> from modules.common.core.data_fetcher import DataFetcher
    >>> data_fetcher = DataFetcher()
    >>> pipeline = SignalPipeline(
    ...     symbol_manager=symbol_manager,
    ...     atc_scanner=atc_scanner,
    ...     xgboost_filter=xgboost_filter,
    ...     gemini_integration=gemini_integration,
    ...     signal_selector=signal_selector,
    ...     config={"max_symbols_to_scan": 20, "pipeline_timeout": 300}
    ... )
    >>> final_signal = pipeline.run_pipeline()
"""

import asyncio
import time
import traceback
from typing import Dict, Optional, TypedDict

from modules.auto_trade.core.atc_scanner import ATCScanner
from modules.auto_trade.core.caching import Cache
from modules.auto_trade.core.circuit_breaker import CircuitBreaker, CircuitState
from modules.auto_trade.core.gemini_integration import GeminiIntegration, GeminiSignal
from modules.auto_trade.core.health import HealthRegistry, HealthStatus
from modules.auto_trade.core.persistence import SignalPersistence
from modules.auto_trade.core.signal_selector import FinalSignal, SignalSelector
from modules.auto_trade.core.symbol_manager import SymbolManager
from modules.auto_trade.core.xgboost_filter import XGBoostFilter
from modules.common.ui.logging import log_error, log_info, log_warn


class PipelineConfig(TypedDict, total=False):
    """Pipeline configuration options.

    Attributes:
        max_symbols_to_scan: Maximum symbols to scan (default: 20)
        pipeline_timeout: Timeout in seconds (default: 300)
    """

    max_symbols_to_scan: int
    pipeline_timeout: int


class SignalPipeline:
    """Signal Pipeline Orchestrator.

    Coordinates the complete auto-trading workflow by cascading through multiple
    analysis stages to find the single best trading opportunity.

    Attributes:
        symbol_manager: Manages tradeable symbols
        atc_scanner: Multi-timeframe trend scanner
        xgboost_filter: ML signal filter
        gemini_integration: AI chart analyzer
        signal_selector: Final signal selector
        signal_persistence: Optional signal storage
        config: Pipeline configuration
        max_symbols: Maximum symbols to scan (default: 20)
        pipeline_timeout: Timeout in seconds (default: 300)
        cache: Cache for ATC results
        circuit_breaker: Circuit breaker for external APIs
        health_registry: Registry for system health checks
    """

    def __init__(
        self,
        symbol_manager: SymbolManager,
        atc_scanner: ATCScanner,
        xgboost_filter: XGBoostFilter,
        gemini_integration: GeminiIntegration,
        signal_selector: SignalSelector,
        signal_persistence: Optional[SignalPersistence] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.symbol_manager = symbol_manager
        self.atc_scanner = atc_scanner
        self.xgboost_filter = xgboost_filter
        self.gemini_integration = gemini_integration
        self.signal_selector = signal_selector
        self.signal_persistence = signal_persistence
        self.config = config or {}

        self.max_symbols = self.config.get("max_symbols_to_scan", 20)
        if self.max_symbols <= 0:
            raise ValueError(f"max_symbols_to_scan must be positive, got {self.max_symbols}")

        self.pipeline_timeout = self.config.get("pipeline_timeout", 300)
        if self.pipeline_timeout <= 0:
            raise ValueError(f"pipeline_timeout must be positive, got {self.pipeline_timeout}")

        # Optimization components
        self.cache = Cache()
        self.circuit_breaker = CircuitBreaker(name="GeminiAPI", failure_threshold=3, recovery_timeout=300)
        self.health_registry = HealthRegistry()

        # Register Health Checks
        self.health_registry.register_check("Memory", self._check_memory)
        self.health_registry.register_check("GeminiAPI", self._check_gemini_circuit)

    def _check_memory(self) -> tuple[HealthStatus, str]:
        """Simple memory usage check (placeholder)."""
        # In a real scenario, use psutil
        try:
            import psutil

            mem = psutil.virtual_memory()
            if mem.percent > 90:
                return HealthStatus.UNHEALTHY, f"High Memory Usage: {mem.percent}%"
            elif mem.percent > 80:
                return HealthStatus.DEGRADED, f"Memory Usage: {mem.percent}%"
            return HealthStatus.HEALTHY, f"Memory Usage: {mem.percent}%"
        except ImportError:
            return HealthStatus.HEALTHY, "psutil not installed, memory check skipped"

    def _check_gemini_circuit(self) -> tuple[HealthStatus, str]:
        """Check Gemini circuit breaker status."""
        if self.circuit_breaker.state == CircuitState.OPEN:
            return HealthStatus.UNHEALTHY, "Circuit Breaker OPEN"
        return HealthStatus.HEALTHY, "Circuit Breaker CLOSED"

    def run_pipeline(self) -> Optional[FinalSignal]:
        """
        Execute the full trading pipeline to find the single best trading opportunity.

        Returns:
            FinalSignal object if a valid trade is found, else None.
        """
        start_time = time.time()
        log_info("Starting Signal Pipeline...")

        # Run Health Checks
        health_status = self.health_registry.check_health()
        if not self.health_registry.is_healthy():
            log_warn(f"System Unhealthy: {health_status}. Proceeding with caution or aborting.")
            # For now, we just log, but we could abort if critical
            # return None

        try:
            # 1. Refresh Symbols
            log_info("Step 1: Refreshing Symbols...")
            self.symbol_manager.refresh_symbols()
            symbols = self.symbol_manager.get_symbols()

            if len(symbols) > self.max_symbols:
                log_info(f"Limiting scan to top {self.max_symbols} from {len(symbols)} candidates.")
                symbols = symbols[: self.max_symbols]

            if not symbols:
                log_warn("No symbols available to scan.")
                return None

            log_info(f"Scanning {len(symbols)} candidate symbols.")

            if time.time() - start_time > self.pipeline_timeout:
                log_warn("Pipeline timeout before scanning.")
                return None

            # 2. ATC Scan
            log_info("Step 2: Scanners (ATC)...")

            # Use Cache for ATC results
            # Key based on number of symbols and timestamp rounded to 5 mins
            # This is a basic key; ideally hash the symbols list.
            # But for simplicity, we assume the list of top symbols is stable enough or we just cache based on time.
            # Wait, if symbols change, we want fresh results.
            # Let's rely on the fact that refresh_symbols() might update the list.
            # The cache key should definitely include the symbols to be safe, or just cache "last_scan_result" and expiry handles it.

            # Simple approach: Cache the result of specific scan call if inputs match?
            # Or just "global_atc_scan" key if we assume refresh_symbols doesn't change wildly in 5 mins.
            cache_key = f"atc_scan_{len(symbols)}_{hash(tuple(sorted(symbols)))}"
            atc_signals = self.cache.get(cache_key)

            if atc_signals is None:
                atc_signals = self.atc_scanner.scan_symbols(symbols)
                self.cache.set(cache_key, atc_signals, ttl=300)  # 5 mins TTL
            else:
                log_info("Using cached ATC results.")

            if not atc_signals:
                log_info("No ATC signals found.")
                return None

            log_info(f"ATC Found {len(atc_signals)} candidates.")

            if time.time() - start_time > self.pipeline_timeout:
                log_warn("Pipeline timeout after ATC scan.")
                return None

            # 3. XGBoost Filter
            log_info("Step 3: Filtering (XGBoost)...")
            xgboost_signals = self.xgboost_filter.filter_signals(atc_signals)

            if not xgboost_signals:
                log_info("No signals passed XGBoost filter.")
                return None

            log_info(f"XGBoost passed {len(xgboost_signals)} candidates.")

            # 4. Gemini Analysis
            log_info(f"Step 4: AI Analysis (Gemini) for {len(xgboost_signals)} candidates...")

            if not self.gemini_integration.is_available():
                log_warn("Gemini API not configured. Skipping AI analysis.")
                gemini_results: Dict[str, GeminiSignal] = {}
            else:
                try:
                    # Wrapped in Circuit Breaker
                    def call_gemini():
                        return asyncio.run(
                            self.gemini_integration.analyze_candidates_batch_async(xgboost_signals, max_concurrency=3)
                        )

                    gemini_results_raw = self.circuit_breaker.call(call_gemini)

                    gemini_results = {k: v for k, v in gemini_results_raw.items() if v is not None}
                    log_info(f"Gemini analyzed {len(gemini_results)} candidates successfully.")
                except Exception as e:
                    log_error(f"Gemini batch analysis failed or circuit open: {e}. Falling back to no AI analysis.")
                    gemini_results = {}

            # 5. Signal Selection
            log_info("Step 5: Final Selection...")
            final_signal = self.signal_selector.select_best_signal(xgboost_signals, gemini_results)

            # 6. Persistence
            if final_signal and self.signal_persistence:
                self.signal_persistence.save_signal(final_signal)

            duration = time.time() - start_time
            if final_signal:
                log_info(f"Pipeline SUCCESS in {duration:.2f}s: Selected {final_signal.symbol}")
            else:
                log_info(f"Pipeline COMPLETED in {duration:.2f}s: No final signal selected.")

            return final_signal

        except Exception as e:
            log_error(f"Pipeline Failed: {e}")
            log_error(traceback.format_exc())
            return None
