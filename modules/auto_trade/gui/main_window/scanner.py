"""Scanner management and configuration."""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

if TYPE_CHECKING:
    from modules.auto_trade.core.signal_pipeline import SignalPipeline

    from .main_window import AutoTradeDashboard

# Create logger for scanner - this will be captured by GUI log handler
logger = logging.getLogger("auto_trade.scanner")
logger.setLevel(logging.INFO)


class ScannerManager:
    """Manages market scanner operations with full SignalPipeline integration."""

    def __init__(self, parent: "AutoTradeDashboard"):
        self.parent = parent
        self.updater = None
        self.pipeline: Optional[SignalPipeline] = None
        self._pipeline_initialized = False
        self._manual_scan_running = False
        self._scan_running = False
        self._scan_lock = threading.Lock()

    def handle_scan_toggle(self, action):
        """Handle scanner start/stop from ScannerControl."""
        try:
            logger.info(f"Scanner action: {action}")

            if action == "manual":
                self._manual_scan()
            elif action is True:
                self._start_scanner()
            elif action is False:
                self._stop_scanner()

        except Exception as e:
            logger.error(f"Error handling scanner toggle: {e}")

    def handle_config_change(self, config: dict):
        """Handle scanner configuration change."""
        try:
            logger.info(f"Scanner config changed: {config}")
            self.parent.settings_manager.set("scanner", config)
            self.parent.settings_manager.save()
            # Reset pipeline so it picks up new config on next scan
            self._pipeline_initialized = False
            self.pipeline = None
        except Exception as e:
            logger.error(f"Error handling scanner config change: {e}")

    def _start_scanner(self):
        """Start scanner loop."""
        config = self.parent.settings_manager.get("scanner", {})
        interval = config.get("scan_interval", 5) * 60

        # Use the parent's existing updater_manager instead of creating a new one
        self.updater = self.parent.updater_manager.create_scanner_updater(self._scanner_cycle, interval=interval)
        logger.info(f"Scanner started (interval: {interval}s)")

    def _stop_scanner(self):
        """Stop scanner loop."""
        if self.updater:
            self.updater.stop()
            self.updater = None
            logger.info("Scanner stopped")

    def _manual_scan(self):
        """Trigger manual scan in background thread."""
        # Prevent multiple manual scans from running simultaneously
        if hasattr(self, "_manual_scan_running") and self._manual_scan_running:
            logger.warning("Manual scan already in progress, skipping...")
            return

        # Mark as running
        self._manual_scan_running = True

        # Update UI to show scanning
        scanner_control = getattr(self.parent, "scanner_control", None)
        if scanner_control is not None:
            scanner_control.progress_label.configure(text="Scanning...")

        # Run in background thread
        def run_scan():
            try:
                self._scanner_cycle()
            finally:
                # Mark as complete
                self._manual_scan_running = False

                # Update timestamp on main thread
                def _update_ui_after_scan():
                    ctrl = getattr(self.parent, "scanner_control", None)
                    if ctrl is not None:
                        ctrl.update_last_scan_time()

                self.parent.after(0, _update_ui_after_scan)

                # Clear progress after 2 seconds
                def _clear_progress():
                    ctrl = getattr(self.parent, "scanner_control", None)
                    if ctrl is not None:
                        ctrl.progress_label.configure(text="")

                self.parent.after(2000, _clear_progress)

        scan_thread = threading.Thread(target=run_scan, daemon=True, name="ManualScan")
        scan_thread.start()

    def _initialize_pipeline(self):
        """Initialize the SignalPipeline with all components."""
        if self._pipeline_initialized and self.pipeline:
            return True

        try:
            logger.info("Initializing SignalPipeline...")

            from config import (
                ATC_SCANNER_DEFAULTS,
                SIGNAL_SELECTOR_DEFAULTS,
                XGBOOST_FILTER_DEFAULTS,
                XGBOOST_PER_SYMBOL_DEFAULTS,
            )
            from modules.auto_trade.core.atc_scanner import ATCScanner, ATCScannerConfig
            from modules.auto_trade.core.gemini_integration import GeminiIntegration
            from modules.auto_trade.core.persistence_sqlite import SignalPersistenceSQLite
            from modules.auto_trade.core.signal_pipeline import SignalPipeline, XGBoostFilterLike
            from modules.auto_trade.core.signal_selector import SignalSelector
            from modules.auto_trade.core.symbol_manager import SymbolManager
            from modules.auto_trade.core.xgboost_filter import XGBoostFilter, XGBoostFilterConfig
            from modules.auto_trade.core.xgboost_per_symbol import (
                XGBoostPerSymbolConfig,
                XGBoostPerSymbolFilter,
            )

            # Get scanner config from settings
            scanner_config = self.parent.settings_manager.get("scanner", {})
            timeframe = scanner_config.get("timeframe", "1h")
            sample_percentage = scanner_config.get("sample_percentage", 20)
            sampling_strategy = scanner_config.get("sampling_strategy", "stratified")

            # Get XGBoost mode from settings (default: per_symbol)
            xgboost_mode = scanner_config.get("xgboost_mode", "per_symbol")

            # Get or create data_fetcher from data_service
            data_fetcher = self.parent.data_service.data_fetcher
            if not data_fetcher:
                logger.error("DataFetcher not available (required for pipeline)")
                return False

            # 1. Symbol Manager
            symbol_manager = SymbolManager(
                data_fetcher=data_fetcher,
                sample_percentage=sample_percentage,
                sampling_strategy=sampling_strategy,
            )
            logger.info(f"SymbolManager ready (sample: {sample_percentage}%, strategy: {sampling_strategy})")

            # 2. ATC Scanner - threshold from Signal Filters tab
            atc_config = ATC_SCANNER_DEFAULTS.copy()
            filters = self.parent.settings_manager.get("filters", {})
            atc_config["threshold"] = float(filters.get("atc_threshold", 0.6))
            atc_scanner = ATCScanner(data_fetcher=data_fetcher, config=cast(ATCScannerConfig, atc_config))
            logger.info(f"ATCScanner ready (timeframes: {atc_config['timeframes']})")

            # 3. XGBoost Filter (per-symbol or pre-trained based on config)
            xgboost_filter: XGBoostFilterLike = self._create_passthrough_xgboost_filter()
            if filters.get("enable_xgboost", True):
                if xgboost_mode == "per_symbol":
                    # Per-symbol training mode - trains fresh XGBoost for each symbol
                    xgboost_config = XGBOOST_PER_SYMBOL_DEFAULTS.copy()
                    xgboost_config["training_timeframe"] = timeframe
                    xgboost_filter = XGBoostPerSymbolFilter(
                        data_fetcher=data_fetcher,
                        config=cast(XGBoostPerSymbolConfig, xgboost_config),
                    )
                    logger.info(f"XGBoostPerSymbolFilter ready (per-symbol training, timeframe: {timeframe})")
                else:
                    # Pre-trained model mode (legacy behavior)
                    model_path = self._find_xgboost_model()
                    if model_path:
                        xgboost_filter = XGBoostFilter(
                            data_fetcher=data_fetcher,
                            model_path=model_path,
                            config=cast(XGBoostFilterConfig, XGBOOST_FILTER_DEFAULTS),
                        )
                        logger.info(f"XGBoostFilter ready (pre-trained model: {Path(model_path).name})")
                    else:
                        logger.warning("XGBoost model not found, using passthrough filter")
                        xgboost_filter = self._create_passthrough_xgboost_filter()
            else:
                logger.info("XGBoost disabled in filters, using passthrough")

            # 4. Gemini Integration
            gemini_integration = GeminiIntegration(data_fetcher=data_fetcher, analysis_timeframe=timeframe)
            if gemini_integration.is_available():
                logger.info("GeminiIntegration ready (API configured)")
            else:
                logger.warning("GeminiIntegration (API not configured, will skip AI analysis)")

            # 5. Signal Selector
            selector_config = SIGNAL_SELECTOR_DEFAULTS.copy()
            selector_config["min_confidence_threshold"] = float(filters.get("min_signal_score", 0.7))
            signal_selector = SignalSelector(config=selector_config)
            logger.info(f"SignalSelector ready (min_confidence: {selector_config['min_confidence_threshold']})")

            # 6. Persistence (SQLite)
            # Use separate database for SignalPipeline to avoid schema conflicts
            db_path = Path("data/signals/pipeline_signals.db")
            persistence = SignalPersistenceSQLite(db_path=str(db_path))
            logger.info(f"SignalPersistence ready ({db_path.name})")

            # 7. Create Pipeline
            self.pipeline = SignalPipeline(
                symbol_manager=symbol_manager,
                atc_scanner=atc_scanner,
                xgboost_filter=xgboost_filter,
                gemini_integration=gemini_integration,
                signal_selector=signal_selector,
                signal_persistence=persistence,
                config={
                    "max_symbols_to_scan": 30,
                    "max_ai_candidates": 5,
                    "xgboost_mode": xgboost_mode,
                },
            )
            logger.info(f"SignalPipeline initialized (xgboost_mode: {xgboost_mode})")

            self._pipeline_initialized = True
            return True

        except ImportError as e:
            logger.error(f"Failed to import pipeline components: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _find_xgboost_model(self) -> Optional[str]:
        """Find the latest XGBoost model. Prefers native .json over legacy .joblib."""
        # Check default locations (native first, then legacy)
        for ext in (".json", ".joblib"):
            default_path = Path(f"models/xgboost_model{ext}")
            if default_path.exists():
                return str(default_path)

        # Check artifacts directory: prefer .json (native), then .joblib
        artifacts_dir = Path("artifacts/xgboost/models")
        if artifacts_dir.exists():
            for pattern in ("*.json", "*.joblib"):
                models = list(artifacts_dir.glob(pattern))
                if models:
                    latest = sorted(models, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                    return str(latest)

        # Check auto_trade data directory
        for ext in (".json", ".joblib"):
            auto_trade_model = Path(f"data/models/xgboost_model{ext}")
            if auto_trade_model.exists():
                return str(auto_trade_model)

        return None

    def _create_passthrough_xgboost_filter(self):
        """Create a passthrough filter when XGBoost model is not available."""

        class PassthroughXGBoostFilter:
            """Passthrough filter that adds mock confidence scores."""

            def filter_signals(self, signals):
                """Pass through all signals with default confidence."""
                for signal in signals:
                    signal.details["xgboost_conf"] = 0.7
                    signal.details["xgboost_dir"] = signal.signal_type
                return signals

        return PassthroughXGBoostFilter()

    def _scanner_cycle(self):
        """Scanner cycle (runs in background thread)."""
        start_time = None
        with self._scan_lock:
            if self._scan_running:
                logger.warning("Scanner cycle already in progress, skipping...")
                return
            self._scan_running = True
        start_time = time.perf_counter()
        try:
            logger.info("=" * 50)
            logger.info("Running scanner cycle...")
            logger.info(f"Mode: {self.parent.mode}")
            logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("-" * 50)

            new_signal = None
            scan_skipped = False
            open_count = 0

            # Gate: skip expensive scan (Gemini) when already at max open positions (DB)
            if self.parent.mode in ["PRODUCTION", "DEMO"]:
                try:
                    from modules.auto_trade.database import get_open_positions, session_scope

                    max_open = self.parent.settings_manager.get("risk.max_open_positions", 3)
                    if not isinstance(max_open, int) or max_open <= 0:
                        max_open = 1
                    with session_scope() as session:
                        open_orders = get_open_positions(session)
                        open_count = len(open_orders)
                    if open_count >= max_open:
                        scan_skipped = True
                        logger.info(
                            f"Scanner cycle skipped: open position(s) present ({open_count}/{max_open}), "
                            "no Gemini call. Will run full scan again when position(s) close."
                        )
                except Exception as e:
                    logger.warning(f"Could not check open positions for scanner gate: {e}. Skipping scan (no Gemini).")
                    scan_skipped = True

            if not scan_skipped and self.parent.mode in ["PRODUCTION", "DEMO"]:
                logger.info("Scanning market for new signals...")
                new_signal = self._run_signal_scan()
                if new_signal:
                    logger.info(f"New signal generated: {new_signal.symbol} {new_signal.signal_type}")
                    logger.info(f"Confidence: {new_signal.confidence:.1%}")
                    logger.info(f"Score: {new_signal.score:.2f}/100")
                else:
                    logger.warning("No new signals generated from scan")
            elif not scan_skipped:
                logger.info("DRY_RUN mode - skipping live market scan")

            # Get signals from database (existing + newly generated)
            logger.info("Fetching signals from database...")
            signals = self.parent.data_service.get_signals()
            logger.info(f"Total signals in database: {len(signals)}")

            if signals:
                logger.info("Top signals:")
                for i, sig in enumerate(signals[:3]):
                    sym, sgn, sc = sig.get("symbol", "N/A"), sig.get("signal", "N/A"), sig.get("score", 0)
                    logger.info(f"  {i + 1}. {sym} - {sgn} ({sc:.2f})")
            else:
                logger.info("No signals found in database")

            self.parent._update_queue.put(("signals", signals))
            self.parent._update_queue.put(
                ("scanner_done", {"skipped": scan_skipped, "count": open_count} if scan_skipped else None)
            )
            logger.info("-" * 50)
            logger.info("Scanner cycle completed")
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"Error in scanner cycle: {e}")
            import traceback

            logger.error(traceback.format_exc())
        finally:
            with self._scan_lock:
                self._scan_running = False
            if start_time is not None:
                duration = time.perf_counter() - start_time
                logger.info("Scanner cycle completed in %.1fs", duration)

    def _run_signal_scan(self):
        """Run actual signal pipeline scan.

        Returns:
            FinalSignal object if a signal was generated, None otherwise.
        """
        try:
            # Initialize pipeline if not already done
            if not self._initialize_pipeline():
                logger.warning("Pipeline not available, reading from database only")
                return None

            assert self.pipeline is not None  # guaranteed after _initialize_pipeline() returns True
            # Run the pipeline
            logger.info("Running SignalPipeline...")
            final_signal = self.pipeline.run_pipeline()

            # If signal generated, also save to main GUI database
            if final_signal:
                self._save_signal_to_gui_db(final_signal)

            return final_signal

        except Exception as e:
            logger.error(f"Error running signal scan: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def _save_signal_to_gui_db(self, signal):
        """Save signal to main GUI database for display."""
        try:
            import uuid

            from modules.auto_trade.database import get_db_manager, save_signal

            db_manager = get_db_manager()
            with db_manager.session_scope() as session:
                correlation_id = f"scan-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
                save_signal(
                    session,
                    correlation_id=correlation_id,
                    symbol=signal.symbol,
                    signal_type=signal.signal_type,
                    confidence=signal.confidence,
                    atc_score=signal.sources.get("atc_score"),
                    xgboost_score=signal.sources.get("xgboost_score"),
                    gemini_score=signal.sources.get("gemini_score"),
                    final_score=signal.score / 100.0,
                    market_context=str(signal.sources) if signal.sources else None,
                )
                logger.info("Signal saved to GUI database")
                try:
                    from modules.auto_trade.monitoring.event_system import EventType

                    self.parent.event_bus.publish(
                        EventType.SIGNAL_GENERATED,
                        {
                            "symbol": signal.symbol,
                            "signal_type": signal.signal_type,
                            "score": signal.score,
                            "correlation_id": correlation_id,
                        },
                        source="scanner",
                    )
                    logger.info(f"Published SIGNAL_GENERATED event for {signal.symbol}")
                except Exception as e:
                    logger.warning(f"Failed to publish signal event: {e}")

        except Exception as e:
            logger.warning(f"Could not save to GUI database: {e}")
            # Don't fail the whole scan if GUI DB save fails
