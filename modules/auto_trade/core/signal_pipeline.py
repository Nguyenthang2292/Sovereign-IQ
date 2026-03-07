"""
Signal Pipeline Orchestrator

Coordinates the entire auto-trading process:
1. Refresh Symbols
2. Scan Market (ATC) - generates LONG/SHORT signals
3. Filter Signals (XGBoost) - trains per-symbol models to confirm ATC signals
4. AI Analysis (Gemini)
5. Select Final Signal

Pipeline Flow:
    ATC Scanner -> XGBoost Filter -> Gemini Analysis -> Signal Selection

    - ATC scans symbols across multiple timeframes and generates directional signals
    - XGBoost trains a fresh model for each ATC-filtered symbol
    - XGBoost prediction must confirm ATC direction with sufficient confidence
    - Gemini provides additional AI analysis for final selection

XGBoost Modes:
    - "per_symbol" (default): Trains a fresh XGBoost model for each symbol
    - "pretrained": Uses a pre-trained model loaded from disk (legacy behavior)

Example:
    >>> from modules.common.core.data_fetcher import DataFetcher
    >>> data_fetcher = DataFetcher()
    >>> pipeline = SignalPipeline(
    ...     symbol_manager=symbol_manager,
    ...     atc_scanner=atc_scanner,
    ...     xgboost_filter=xgboost_filter,  # Can be XGBoostFilter or XGBoostPerSymbolFilter
    ...     gemini_integration=gemini_integration,
    ...     signal_selector=signal_selector,
    ...     config={
    ...         "max_symbols_to_scan": 20,
    ...         "xgboost_mode": "per_symbol",  # or "pretrained"
    ...     }
    ... )
    >>> final_signal = pipeline.run_pipeline()
"""

import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, TypedDict, cast

from modules.auto_trade.core.atc_scanner import SignalResult
from modules.auto_trade.core.circuit_breaker import CircuitBreaker, CircuitState
from modules.auto_trade.core.gann_square_filter import GannSquareFilter
from modules.auto_trade.core.gemini_integration import GeminiIntegration, GeminiSignal

if TYPE_CHECKING:
    from modules.gemini_gann_square.core.gann_signal_engine import GannAnalysisResult
from modules.auto_trade.core.health import HealthRegistry, HealthStatus
from modules.auto_trade.core.signal_selector import FinalSignal, SignalSelector, SignalSources
from modules.auto_trade.core.symbol_manager import SymbolManager
from modules.auto_trade.monitoring.alerts import AlertManager
from modules.auto_trade.monitoring.logger import setup_logging
from modules.common.ui.logging import log_error, log_info, log_warn


class XGBoostFilterLike(Protocol):
    """Protocol for any XGBoost-style filter (including passthrough when no model is available)."""

    def filter_signals(self, signals: List[SignalResult]) -> List[SignalResult]: ...


class ATCScannerLike(Protocol):
    """Protocol for any ATC scanner implementation used by the pipeline."""

    def scan_symbols(self, symbols: List[str]) -> List[SignalResult]: ...


from modules.auto_trade.monitoring.audit import AuditLogger
from modules.auto_trade.monitoring.event_system import EventSystem, EventType
from modules.auto_trade.monitoring.metrics import MetricsCollector


class PipelineConfig(TypedDict, total=False):
    """Pipeline configuration options.

    Attributes:
        max_symbols_to_scan: Maximum symbols to scan (default: 20)
        monitoring_enabled: Whether monitoring components are enabled
        max_ai_candidates: Maximum candidates for AI analysis (default: 5)
        xgboost_mode: XGBoost filter mode - "per_symbol" or "pretrained" (default: "per_symbol")
        enable_gann_square: Enable Gann Square filter (default: False)
    """

    max_symbols_to_scan: int
    monitoring_enabled: bool
    max_ai_candidates: int
    xgboost_mode: str  # "per_symbol" (train fresh) or "pretrained" (use existing model)
    enable_gann_square: bool


class SignalPipeline:
    """Signal Pipeline Orchestrator.

    Coordinates the complete auto-trading workflow by cascading through multiple
    analysis stages to find the single best trading opportunity.

    Pipeline Flow:
        1. Refresh Symbols - Get list of tradeable symbols
        2. ATC Scan - Multi-timeframe trend analysis generates LONG/SHORT signals
        3. XGBoost Filter - Confirms ATC signals with ML predictions
           - "per_symbol" mode: Trains a fresh model for each ATC-filtered symbol
           - "pretrained" mode: Uses a pre-trained model loaded from disk
        4. Gemini Analysis - AI-powered chart analysis
        5. Signal Selection - Final signal selection based on all analyses

    Attributes:
        symbol_manager: Manages tradeable symbols
        atc_scanner: Multi-timeframe trend scanner
        xgboost_filter: ML signal filter (XGBoostFilter, XGBoostPerSymbolFilter, or passthrough)
        gemini_integration: AI chart analyzer
        signal_selector: Final signal selector
        signal_persistence: Optional signal storage
        config: Pipeline configuration
        max_symbols: Maximum symbols to scan (default: 20)
        max_ai_candidates: Maximum candidates for AI analysis (default: 5)
        xgboost_mode: XGBoost filter mode - "per_symbol" or "pretrained"
        circuit_breaker: Circuit breaker for external APIs
        health_registry: Registry for system health checks
    """

    symbol_manager: SymbolManager
    atc_scanner: ATCScannerLike
    xgboost_filter: XGBoostFilterLike
    gemini_integration: GeminiIntegration
    signal_selector: SignalSelector
    config: PipelineConfig
    max_symbols: int
    max_ai_candidates: int
    xgboost_mode: str
    circuit_breaker: CircuitBreaker
    health_registry: HealthRegistry
    event_bus: EventSystem
    metrics: MetricsCollector
    audit: AuditLogger
    alerts: AlertManager
    signal_persistence: Optional[Any]

    def __init__(
        self,
        symbol_manager: SymbolManager,
        atc_scanner: ATCScannerLike,
        xgboost_filter: XGBoostFilterLike,
        gemini_integration: GeminiIntegration,
        signal_selector: SignalSelector,
        config: Optional[PipelineConfig] = None,
        gann_square_filter: Optional[GannSquareFilter] = None,
    ) -> None:
        self.symbol_manager = symbol_manager
        self.atc_scanner = atc_scanner
        self.xgboost_filter = xgboost_filter
        self.gemini_integration = gemini_integration
        self.signal_selector = signal_selector
        self.config = config or {}
        self.gann_square_filter = gann_square_filter
        self.signal_persistence = None

        # Initialize Logging
        setup_logging()

        self.max_symbols = self.config.get("max_symbols_to_scan", 20)
        if self.max_symbols <= 0:
            raise ValueError(f"max_symbols_to_scan must be positive, got {self.max_symbols}")

        self.max_ai_candidates = self.config.get("max_ai_candidates", 5)
        if self.max_ai_candidates <= 0:
            raise ValueError(f"max_ai_candidates must be positive, got {self.max_ai_candidates}")

        # XGBoost mode tracking (for logging)
        self.xgboost_mode = self.config.get("xgboost_mode", "per_symbol")
        if self.xgboost_mode not in ["per_symbol", "pretrained"]:
            raise ValueError(f"xgboost_mode must be 'per_symbol' or 'pretrained', got {self.xgboost_mode}")

        # Optimization components (Cache removed - use ATCScanner's Rust cache)
        self.circuit_breaker = CircuitBreaker(name="GeminiAPI", failure_threshold=3, recovery_timeout=300)
        self.health_registry = HealthRegistry()

        # Monitoring foundation
        self.event_bus = EventSystem()
        self.metrics = MetricsCollector()
        self.audit = AuditLogger()
        self.alerts = AlertManager(self.event_bus)

        # Register Health Checks
        self.health_registry.register_check("Memory", self._check_memory)
        self.health_registry.register_check("GeminiAPI", self._check_gemini_circuit)

    def _check_memory(self) -> tuple[HealthStatus, str]:
        """Simple memory usage check (placeholder)."""
        # In a real scenario, use psutil
        try:
            import psutil

            mem = psutil.virtual_memory()
            self.metrics.gauge("system_memory_percent", mem.percent)  # Report metric
            if mem.percent > 90:
                return HealthStatus.UNHEALTHY, f"High Memory Usage: {mem.percent}%"
            elif mem.percent > 80:
                return HealthStatus.DEGRADED, f"Memory Usage: {mem.percent}%"
            return HealthStatus.HEALTHY, f"Memory Usage: {mem.percent}%"
        except ImportError:
            return HealthStatus.HEALTHY, "psutil not installed, memory check skipped"

    def _check_gemini_circuit(self) -> tuple[HealthStatus, str]:
        """Check Gemini circuit breaker status."""
        status = HealthStatus.HEALTHY if self.circuit_breaker.state == CircuitState.CLOSED else HealthStatus.UNHEALTHY
        # Publish event if unhealthy (though AlertManager listens to CIRCUIT_OPEN directly)
        return status, f"Circuit Breaker {self.circuit_breaker.state.name}"

    def _gann_result_to_final_signal(self, gann_result: Any, original: FinalSignal) -> FinalSignal:
        """
        Convert GannAnalysisResult to FinalSignal.

        Args:
            gann_result: The Gann analysis result.
            original: The original FinalSignal from ranking.

        Returns:
            FinalSignal with Gann-derived price levels and updated sources.

        Raises:
            ValueError: If price levels are invalid.
        """
        if gann_result is None:
            raise ValueError("gann_result is None")

        if not gann_result.is_tradeable():
            raise ValueError(f"Gann result is not tradeable: {gann_result.signal}")

        signal_type = gann_result.signal
        entry_price = gann_result.entry_price
        stop_loss = gann_result.stop_loss
        take_profit = gann_result.take_profit_1

        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            raise ValueError(f"Invalid Gann price levels: entry={entry_price}, sl={stop_loss}, tp={take_profit}")

        sources = dict(original.sources)
        sources["gann_confidence"] = gann_result.confidence_pct / 100.0
        sources["gann_reasoning"] = gann_result.reasoning

        return FinalSignal(
            symbol=gann_result.symbol,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=original.leverage,
            confidence=original.confidence,
            score=original.score,
            sources=cast(SignalSources, sources),
            timestamp=original.timestamp,
        )

    def run_pipeline(self) -> Optional[FinalSignal]:
        """
        Execute the full trading pipeline to find the single best trading opportunity.

        Returns:
            FinalSignal object if a valid trade is found, else None.
        """
        start_time = time.time()
        log_info("Starting Signal Pipeline...")
        self.event_bus.publish(EventType.PIPELINE_START, {}, source="SignalPipeline")
        self.metrics.increment("pipeline_runs")

        # Run Health Checks
        health_status = self.health_registry.check_health()
        if not self.health_registry.is_healthy():
            msg = f"System Unhealthy: {health_status}"
            log_warn(msg)
            self.event_bus.publish(
                EventType.HEALTH_CHECK_FAILED,
                {"details": str(health_status)},
                source="SignalPipeline",
            )
            # For now, we just log, but we could abort if critical
            # return None

        try:
            # 1. Refresh Symbols
            log_info("Step 1: Refreshing Symbols...")
            self.symbol_manager.refresh_symbols()

            # Log sampling configuration
            sample_pct = getattr(self.symbol_manager, "sample_percentage", 100.0)
            strategy = getattr(self.symbol_manager, "sampling_strategy", "random")
            log_info(f"Sampling config: {sample_pct}% using '{strategy}' strategy")

            symbols = self.symbol_manager.get_symbols()
            log_info(f"Sampled {len(symbols)} symbols using {strategy} strategy")

            if len(symbols) > self.max_symbols:
                log_info(f"Limiting scan to top {self.max_symbols} from {len(symbols)} candidates.")
                symbols = symbols[: self.max_symbols]

            if not symbols:
                log_warn("No symbols available to scan.")
                return None

            self.metrics.gauge("candidate_symbols", len(symbols))

            log_info(f"Scanning {len(symbols)} candidate symbols.")

            # 2. ATC Scan (uses internal Rust cache)
            log_info("Step 2: ATC Scan...")
            atc_signals = self.atc_scanner.scan_symbols(symbols)

            if not atc_signals:
                log_info("No ATC signals found.")
                return None

            self.metrics.gauge("atc_signals_found", len(atc_signals))

            log_info(f"ATC Found {len(atc_signals)} candidates.")

            # 3. XGBoost Filter
            xgboost_mode_label = "per-symbol training" if self.xgboost_mode == "per_symbol" else "pre-trained model"
            log_info(f"Step 3: XGBoost Filter ({xgboost_mode_label})...")
            xgboost_signals = self.xgboost_filter.filter_signals(atc_signals)

            self.metrics.gauge("xgboost_signals_passed", len(xgboost_signals))

            if not xgboost_signals:
                log_info("No signals passed XGBoost filter.")
                return None

            log_info(f"XGBoost passed {len(xgboost_signals)} candidates.")

            # 4. Filter top candidates for AI Analysis
            # Sort by XGBoost confidence (descending)
            xgboost_signals.sort(key=lambda x: float(x.details.get("xgboost_conf", 0.0)), reverse=True)

            if len(xgboost_signals) > self.max_ai_candidates:
                log_info(
                    f"Limiting AI analysis to top {self.max_ai_candidates} candidates (from {len(xgboost_signals)})."
                )
                xgboost_signals = xgboost_signals[: self.max_ai_candidates]

            # 5. Gemini Analysis
            log_info(f"Step 5: Gemini AI Analysis for {len(xgboost_signals)} candidates...")

            gemini_results: Dict[str, GeminiSignal] = {}
            if not self.gemini_integration.is_available():
                log_warn("Gemini API not configured. Skipping AI analysis.")
            else:
                try:
                    # Wrapped in Circuit Breaker - handle both sync and async contexts
                    def call_gemini():
                        maybe_result = self.gemini_integration.analyze_candidates_batch_async(
                            xgboost_signals, max_concurrency=3
                        )

                        # Test doubles may return a plain dict instead of a coroutine.
                        if not asyncio.iscoroutine(maybe_result):
                            return cast(Dict[str, Optional[GeminiSignal]], maybe_result)

                        try:
                            # Check if we're already in an event loop
                            loop = asyncio.get_running_loop()
                            # We're in an async context, use run_coroutine_threadsafe
                            future = asyncio.run_coroutine_threadsafe(maybe_result, loop)
                            return future.result(timeout=60.0)
                        except RuntimeError:
                            # No running loop, safe to use asyncio.run
                            return asyncio.run(maybe_result)

                    gemini_results_raw = self.circuit_breaker.call(call_gemini)

                    gemini_results = {k: v for k, v in gemini_results_raw.items() if v is not None}
                    log_info(f"Gemini analyzed {len(gemini_results)} candidates successfully.")
                except Exception as e:
                    log_error(f"Gemini analysis failed: {e}")
                    self.event_bus.publish(EventType.CIRCUIT_OPEN, {"error": str(e)}, source="SignalPipeline")

            # 6. Signal Selection (rank signals)
            log_info("Step 6: Final Signal Selection...")
            ranked_signals: List[FinalSignal] = []
            if hasattr(self.signal_selector, "rank_signals"):
                ranked_signals = self.signal_selector.rank_signals(xgboost_signals, gemini_results)

            final_signal: Optional[FinalSignal] = None
            if hasattr(self.signal_selector, "select_best_signal"):
                final_signal = self.signal_selector.select_best_signal(xgboost_signals, gemini_results)
            elif ranked_signals:
                final_signal = ranked_signals[0]

            # 7. Optional Gann Square Filter
            if self.gann_square_filter is not None and ranked_signals:
                log_info("Step 7: Gann Square Filter...")
                gann_result = self.gann_square_filter.run(ranked_signals)
                if gann_result is None:
                    log_warn("GannSquareFilter: All candidates rejected.")
                else:
                    try:
                        original_signal = next(
                            (candidate for candidate in ranked_signals if candidate.symbol == gann_result.symbol),
                            ranked_signals[0],
                        )
                        final_signal = self._gann_result_to_final_signal(gann_result, original_signal)
                    except ValueError as e:
                        log_warn(f"GannSquareFilter: Invalid result rejected: {e}")
                        final_signal = None

            # 7. Audit & Metric
            if final_signal:
                if self.signal_persistence is not None:
                    try:
                        self.signal_persistence.save_signal(final_signal)
                    except Exception as e:
                        log_warn(f"Signal persistence failed: {e}")

                # Audit Log
                self.audit.log_event("SIGNAL_GENERATED", final_signal.__dict__)

                # Event & Metric
                self.event_bus.publish(
                    EventType.SIGNAL_GENERATED,
                    {"symbol": final_signal.symbol, "type": final_signal.signal_type},
                    source="SignalPipeline",
                )
                self.metrics.increment("signals_generated")

                duration = time.time() - start_time
                self.metrics.histogram("pipeline_duration", duration)
                log_info(f"Pipeline SUCCESS in {duration:.2f}s: Selected {final_signal.symbol}")
            else:
                log_info(f"Pipeline COMPLETED in {time.time() - start_time:.2f}s: No final signal selected.")

            self.event_bus.publish(EventType.PIPELINE_COMPLETE, {}, source="SignalPipeline")
            return final_signal

        except Exception as e:
            log_error(f"Pipeline Failed: {e}", exc_info=True)
            self.event_bus.publish(EventType.PIPELINE_ERROR, {"error": str(e)}, source="SignalPipeline")
            return None
