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
    ...         "pipeline_timeout": 300,
    ...         "xgboost_mode": "per_symbol",  # or "pretrained"
    ...     }
    ... )
    >>> final_signal = pipeline.run_pipeline()
"""

import asyncio
import time
from typing import Dict, Optional, TypedDict, Union

from modules.auto_trade.core.atc_scanner import ATCScanner
from modules.auto_trade.core.circuit_breaker import CircuitBreaker, CircuitState
from modules.auto_trade.core.gemini_integration import GeminiIntegration, GeminiSignal
from modules.auto_trade.core.health import HealthRegistry, HealthStatus
from modules.auto_trade.core.persistence_sqlite import SignalPersistenceSQLite
from modules.auto_trade.core.signal_selector import FinalSignal, SignalSelector
from modules.auto_trade.core.symbol_manager import SymbolManager
from modules.auto_trade.core.xgboost_filter import XGBoostFilter
from modules.auto_trade.core.xgboost_per_symbol import XGBoostPerSymbolFilter
from modules.auto_trade.monitoring.alerts import AlertManager
from modules.auto_trade.monitoring.audit import AuditLogger
from modules.auto_trade.monitoring.events import Event, EventBus, EventType
from modules.auto_trade.monitoring.logger import get_logger, setup_logging
from modules.auto_trade.monitoring.metrics import MetricsCollector

logger = get_logger("modules.auto_trade.core.signal_pipeline")


class PipelineConfig(TypedDict, total=False):
    """Pipeline configuration options.

    Attributes:
        max_symbols_to_scan: Maximum symbols to scan (default: 20)
        pipeline_timeout: Timeout in seconds (default: 300)
        monitoring_enabled: Whether monitoring components are enabled
        max_ai_candidates: Maximum candidates for AI analysis (default: 5)
        xgboost_mode: XGBoost filter mode - "per_symbol" or "pretrained" (default: "per_symbol")
    """

    max_symbols_to_scan: int
    pipeline_timeout: int
    monitoring_enabled: bool
    max_ai_candidates: int
    xgboost_mode: str  # "per_symbol" (train fresh) or "pretrained" (use existing model)


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
        xgboost_filter: ML signal filter (XGBoostFilter or XGBoostPerSymbolFilter)
        gemini_integration: AI chart analyzer
        signal_selector: Final signal selector
        signal_persistence: Optional signal storage
        config: Pipeline configuration
        max_symbols: Maximum symbols to scan (default: 20)
        max_ai_candidates: Maximum candidates for AI analysis (default: 5)
        pipeline_timeout: Timeout in seconds (default: 300)
        xgboost_mode: XGBoost filter mode - "per_symbol" or "pretrained"
        cache: Cache for ATC results
        circuit_breaker: Circuit breaker for external APIs
        health_registry: Registry for system health checks
    """

    def __init__(
        self,
        symbol_manager: SymbolManager,
        atc_scanner: ATCScanner,
        xgboost_filter: Union[XGBoostFilter, XGBoostPerSymbolFilter],
        gemini_integration: GeminiIntegration,
        signal_selector: SignalSelector,
        signal_persistence: Optional[SignalPersistenceSQLite] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.symbol_manager = symbol_manager
        self.atc_scanner = atc_scanner
        self.xgboost_filter = xgboost_filter
        self.gemini_integration = gemini_integration
        self.signal_selector = signal_selector
        self.signal_persistence = signal_persistence
        self.config = config or {}

        # Initialize Logging
        setup_logging()

        self.max_symbols = self.config.get("max_symbols_to_scan", 20)
        if self.max_symbols <= 0:
            raise ValueError(f"max_symbols_to_scan must be positive, got {self.max_symbols}")

        self.max_ai_candidates = self.config.get("max_ai_candidates", 5)
        if self.max_ai_candidates <= 0:
            raise ValueError(f"max_ai_candidates must be positive, got {self.max_ai_candidates}")

        self.pipeline_timeout = self.config.get("pipeline_timeout", 300)
        if self.pipeline_timeout <= 0:
            raise ValueError(f"pipeline_timeout must be positive, got {self.pipeline_timeout}")

        # XGBoost mode tracking (for logging)
        self.xgboost_mode = self.config.get("xgboost_mode", "per_symbol")
        if self.xgboost_mode not in ["per_symbol", "pretrained"]:
            raise ValueError(f"xgboost_mode must be 'per_symbol' or 'pretrained', got {self.xgboost_mode}")

        # Optimization components (Cache removed - use ATCScanner's Rust cache)
        self.circuit_breaker = CircuitBreaker(name="GeminiAPI", failure_threshold=3, recovery_timeout=300)
        self.health_registry = HealthRegistry()

        # Monitoring foundation
        self.event_bus = EventBus()
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

    def run_pipeline(self) -> Optional[FinalSignal]:
        """
        Execute the full trading pipeline to find the single best trading opportunity.

        Returns:
            FinalSignal object if a valid trade is found, else None.
        """
        start_time = time.time()
        logger.info("Starting Signal Pipeline...")
        self.event_bus.publish(Event(EventType.PIPELINE_START))
        self.metrics.increment("pipeline_runs")

        # Run Health Checks
        health_status = self.health_registry.check_health()
        if not self.health_registry.is_healthy():
            msg = f"System Unhealthy: {health_status}"
            logger.warning(msg)
            self.event_bus.publish(Event(EventType.HEALTH_CHECK_FAILED, {"details": str(health_status)}))
            # For now, we just log, but we could abort if critical
            # return None

        try:
            # 1. Refresh Symbols
            logger.info("Step 1: Refreshing Symbols...")
            self.symbol_manager.refresh_symbols()

            # Log sampling configuration
            sample_pct = getattr(self.symbol_manager, "sample_percentage", 100.0)
            strategy = getattr(self.symbol_manager, "sampling_strategy", "random")
            logger.info(f"Sampling config: {sample_pct}% using '{strategy}' strategy")

            symbols = self.symbol_manager.get_symbols()
            logger.info(f"Sampled {len(symbols)} symbols using {strategy} strategy")

            if len(symbols) > self.max_symbols:
                logger.info(f"Limiting scan to top {self.max_symbols} from {len(symbols)} candidates.")
                symbols = symbols[: self.max_symbols]

            if not symbols:
                logger.warning("No symbols available to scan.")
                return None

            self.metrics.gauge("candidate_symbols", len(symbols))

            logger.info(f"Scanning {len(symbols)} candidate symbols.")

            if time.time() - start_time > self.pipeline_timeout:
                logger.warning("Pipeline timeout before scanning.")
                return None

            # 2. ATC Scan (uses internal Rust cache)
            logger.info("Step 2: Scanners (ATC)...")
            atc_signals = self.atc_scanner.scan_symbols(symbols)

            if not atc_signals:
                logger.info("No ATC signals found.")
                return None

            self.metrics.gauge("atc_signals_found", len(atc_signals))

            logger.info(f"ATC Found {len(atc_signals)} candidates.")

            if time.time() - start_time > self.pipeline_timeout:
                logger.warning("Pipeline timeout after ATC scan.")
                return None

            # 3. XGBoost Filter
            xgboost_mode_label = "per-symbol training" if self.xgboost_mode == "per_symbol" else "pre-trained model"
            logger.info(f"Step 3: Filtering (XGBoost - {xgboost_mode_label})...")
            xgboost_signals = self.xgboost_filter.filter_signals(atc_signals)

            self.metrics.gauge("xgboost_signals_passed", len(xgboost_signals))

            if not xgboost_signals:
                logger.info("No signals passed XGBoost filter.")
                return None

            logger.info(f"XGBoost passed {len(xgboost_signals)} candidates.")

            # 4. Filter top candidates for AI Analysis
            # Sort by XGBoost confidence (descending)
            xgboost_signals.sort(key=lambda x: float(x.details.get("xgboost_conf", 0.0)), reverse=True)

            if len(xgboost_signals) > self.max_ai_candidates:
                logger.info(
                    f"Limiting AI analysis to top {self.max_ai_candidates} candidates (from {len(xgboost_signals)})."
                )
                xgboost_signals = xgboost_signals[: self.max_ai_candidates]

            # 5. Gemini Analysis
            logger.info(f"Step 4: AI Analysis (Gemini) for {len(xgboost_signals)} candidates...")

            gemini_results: Dict[str, GeminiSignal] = {}
            if not self.gemini_integration.is_available():
                logger.warning("Gemini API not configured. Skipping AI analysis.")
            else:
                try:
                    # Wrapped in Circuit Breaker
                    def call_gemini():
                        return asyncio.run(
                            self.gemini_integration.analyze_candidates_batch_async(xgboost_signals, max_concurrency=3)
                        )

                    gemini_results_raw = self.circuit_breaker.call(call_gemini)

                    gemini_results = {k: v for k, v in gemini_results_raw.items() if v is not None}
                    logger.info(f"Gemini analyzed {len(gemini_results)} candidates successfully.")
                except Exception as e:
                    logger.error(f"Gemini analysis failed: {e}")
                    self.event_bus.publish(Event(EventType.CIRCUIT_OPEN, {"error": str(e)}))

            # 5. Signal Selection
            logger.info("Step 5: Final Selection...")
            final_signal = self.signal_selector.select_best_signal(xgboost_signals, gemini_results)

            # 6. Persistence & Audit
            if final_signal:
                if self.signal_persistence:
                    self.signal_persistence.save_signal(final_signal)

                # Audit Log
                self.audit.log_event("SIGNAL_GENERATED", final_signal.__dict__)

                # Event & Metric
                self.event_bus.publish(
                    Event(EventType.SIGNAL_GENERATED, {"symbol": final_signal.symbol, "type": final_signal.signal_type})
                )
                self.metrics.increment("signals_generated")

                duration = time.time() - start_time
                self.metrics.histogram("pipeline_duration", duration)
                logger.info(f"Pipeline SUCCESS in {duration:.2f}s: Selected {final_signal.symbol}")
            else:
                logger.info(f"Pipeline COMPLETED in {time.time() - start_time:.2f}s: No final signal selected.")

            self.event_bus.publish(Event(EventType.PIPELINE_COMPLETE))
            return final_signal

        except Exception as e:
            logger.error(f"Pipeline Failed: {e}", exc_info=True)
            self.event_bus.publish(Event(EventType.PIPELINE_ERROR, {"error": str(e)}))
            return None
