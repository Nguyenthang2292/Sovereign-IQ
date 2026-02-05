"""Scanner management and configuration."""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .main_window import AutoTradeDashboard

# Create logger for scanner - this will be captured by GUI log handler
logger = logging.getLogger("auto_trade.scanner")
logger.setLevel(logging.INFO)


class ScannerManager:
    """Manages market scanner operations with full SignalPipeline integration."""

    def __init__(self, parent: "AutoTradeDashboard"):
        self.parent = parent
        self.updater = None
        self.pipeline = None
        self._pipeline_initialized = False
        self._manual_scan_running = False

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
        import threading

        # Prevent multiple manual scans from running simultaneously
        if hasattr(self, "_manual_scan_running") and self._manual_scan_running:
            logger.warning("Manual scan already in progress, skipping...")
            return

        # Mark as running
        self._manual_scan_running = True

        # Update UI to show scanning
        if hasattr(self.parent, "scanner_control"):
            self.parent.scanner_control.progress_label.configure(text="Scanning...")

        # Run in background thread
        def run_scan():
            try:
                self._scanner_cycle()
            finally:
                # Mark as complete
                self._manual_scan_running = False

                # Update timestamp on main thread
                self.parent.after(
                    0,
                    lambda: (
                        self.parent.scanner_control.update_last_scan_time()
                        if hasattr(self.parent, "scanner_control")
                        else None
                    ),
                )

                # Clear progress after 2 seconds
                self.parent.after(
                    2000,
                    lambda: (
                        self.parent.scanner_control.progress_label.configure(text="")
                        if hasattr(self.parent, "scanner_control")
                        else None
                    ),
                )

        scan_thread = threading.Thread(target=run_scan, daemon=True, name="ManualScan")
        scan_thread.start()

    def _initialize_pipeline(self):
        """Initialize the SignalPipeline with all components."""
        if self._pipeline_initialized and self.pipeline:
            return True

        try:
            logger.info("Initializing SignalPipeline...")

            from config import ATC_SCANNER_DEFAULTS, SIGNAL_SELECTOR_DEFAULTS, XGBOOST_FILTER_DEFAULTS
            from modules.auto_trade.core.atc_scanner import ATCScanner
            from modules.auto_trade.core.gemini_integration import GeminiIntegration
            from modules.auto_trade.core.persistence_sqlite import SignalPersistenceSQLite
            from modules.auto_trade.core.signal_pipeline import SignalPipeline
            from modules.auto_trade.core.signal_selector import SignalSelector
            from modules.auto_trade.core.symbol_manager import SymbolManager
            from modules.auto_trade.core.xgboost_filter import XGBoostFilter

            # Get scanner config from settings
            scanner_config = self.parent.settings_manager.get("scanner", {})
            timeframe = scanner_config.get("timeframe", "1h")
            sample_percentage = scanner_config.get("sample_percentage", 20)
            sampling_strategy = scanner_config.get("sampling_strategy", "stratified")

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

            # 2. ATC Scanner - adjust config based on GUI settings
            atc_config = ATC_SCANNER_DEFAULTS.copy()
            # Lower threshold for more signals during testing
            atc_config["threshold"] = 0.1
            atc_scanner = ATCScanner(data_fetcher=data_fetcher, config=atc_config)
            logger.info(f"ATCScanner ready (timeframes: {atc_config['timeframes']})")

            # 3. XGBoost Filter
            model_path = self._find_xgboost_model()
            if model_path:
                xgboost_filter = XGBoostFilter(
                    data_fetcher=data_fetcher, model_path=model_path, config=XGBOOST_FILTER_DEFAULTS
                )
                logger.info(f"XGBoostFilter ready (model: {Path(model_path).name})")
            else:
                logger.warning("XGBoost model not found, using passthrough filter")
                xgboost_filter = self._create_passthrough_xgboost_filter()

            # 4. Gemini Integration
            gemini_integration = GeminiIntegration(data_fetcher=data_fetcher, analysis_timeframe=timeframe)
            if gemini_integration.is_available():
                logger.info("GeminiIntegration ready (API configured)")
            else:
                logger.warning("GeminiIntegration (API not configured, will skip AI analysis)")

            # 5. Signal Selector
            signal_selector = SignalSelector(config=SIGNAL_SELECTOR_DEFAULTS)
            logger.info("SignalSelector ready")

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
                    "pipeline_timeout": 300,
                },
            )
            logger.info("SignalPipeline initialized successfully")

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
        """Find the latest XGBoost model."""
        # Check default location first
        default_path = Path("models/xgboost_model.joblib")
        if default_path.exists():
            return str(default_path)

        # Check artifacts directory
        artifacts_dir = Path("artifacts/xgboost/models")
        if artifacts_dir.exists():
            models = list(artifacts_dir.glob("*.joblib"))
            if models:
                latest_model = sorted(models, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                return str(latest_model)

        # Check auto_trade data directory
        auto_trade_model = Path("data/models/xgboost_model.joblib")
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
        try:
            logger.info("=" * 50)
            logger.info("Running scanner cycle...")
            logger.info(f"Mode: {self.parent.mode}")
            logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("-" * 50)

            new_signal = None

            # In PRODUCTION/DEMO mode, run the full SignalPipeline
            if self.parent.mode in ["PRODUCTION", "DEMO"]:
                logger.info("Scanning market for new signals...")
                new_signal = self._run_signal_scan()
                if new_signal:
                    logger.info(f"New signal generated: {new_signal.symbol} {new_signal.signal_type}")
                    logger.info(f"Confidence: {new_signal.confidence:.1%}")
                    logger.info(f"Score: {new_signal.score:.2f}/100")
                else:
                    logger.warning("No new signals generated from scan")
            else:
                logger.info("DRY_RUN mode - skipping live market scan")

            # Get signals from database (existing + newly generated)
            logger.info("Fetching signals from database...")
            signals = self.parent.data_service.get_signals()
            logger.info(f"Total signals in database: {len(signals)}")

            if signals:
                logger.info("Top signals:")
                for i, sig in enumerate(signals[:3]):
                    logger.info(
                        f"  {i + 1}. {sig.get('symbol', 'N/A')} - {sig.get('signal', 'N/A')} ({sig.get('score', 0):.2f})"
                    )
            else:
                logger.info("No signals found in database")

            self.parent._update_queue.put(("signals", signals))
            self.parent._update_queue.put(("scanner_done", None))
            logger.info("-" * 50)
            logger.info("Scanner cycle completed")
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"Error in scanner cycle: {e}")
            import traceback

            logger.error(traceback.format_exc())

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
            from modules.auto_trade.database import get_db_manager, save_signal
            from datetime import datetime

            db_manager = get_db_manager()
            with db_manager.session_scope() as session:
                # Convert FinalSignal to GUI database format
                signal_data = {
                    "symbol": signal.symbol,
                    "signal_type": signal.signal_type,
                    "confidence": signal.confidence,
                    "atc_score": signal.sources.get("atc_score"),
                    "xgboost_score": signal.sources.get("xgboost_score"),
                    "gemini_score": signal.sources.get("gemini_score"),
                    "final_score": signal.score / 100.0,  # Convert 0-100 to 0-1
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "leverage": signal.leverage,
                    "market_context": signal.sources,
                }

                save_signal(session, signal_data)
                logger.info("Signal saved to GUI database")

        except Exception as e:
            logger.warning(f"Could not save to GUI database: {e}")
            # Don't fail the whole scan if GUI DB save fails
