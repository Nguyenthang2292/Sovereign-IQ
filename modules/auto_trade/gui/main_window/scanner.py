"""Scanner management and configuration."""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

from modules.common.ui.logging import log_error, log_info, log_warn

if TYPE_CHECKING:
    from modules.auto_trade.core.signal_pipeline import SignalPipeline

    from .main_window import AutoTradeDashboard

# Scanner manager handles the trading loop and signal generation


class ScannerManager:
    """Manages market scanner operations with full SignalPipeline integration."""

    def __init__(self, parent: "AutoTradeDashboard"):
        self.parent = parent
        self.updater = None
        self.pipeline: Optional[SignalPipeline] = None
        self._pipeline_initialized = False
        self._scan_running = False
        self._scan_lock = threading.Lock()

    def handle_scan_toggle(self, action) -> None:
        """Handle scanner start/stop from ScannerControl."""
        try:
            log_info(f"Scanner action: {action}")

            if action is True:
                self._start_scanner()
            elif action is False:
                self._stop_scanner()

        except Exception as e:
            log_error(f"Error handling scanner toggle: {e}")

    def handle_config_change(self, config: dict):
        """Handle scanner configuration change."""
        try:
            log_info(f"Scanner config changed: {config}")
            self.parent.settings_manager.set("scanner", config)
            self.parent.settings_manager.save()
            # Reset pipeline so it picks up new config on next scan
            self._pipeline_initialized = False
            self.pipeline = None
        except Exception as e:
            log_error(f"Error handling scanner config change: {e}")

    def _start_scanner(self):
        """Start scanner loop."""
        config = self.parent.settings_manager.get("scanner", {})
        interval = config.get("scan_interval", 5) * 60

        # Use the parent's existing updater_manager instead of creating a new one
        self.updater = self.parent.updater_manager.create_scanner_updater(self._scanner_cycle, interval=interval)
        log_info(f"Scanner started (interval: {interval}s)")

        # Prime the countdown so users see it from the very first tick.
        # PeriodicUpdater runs the callback immediately on start, so the first
        # "real" next-scan will be after `interval` seconds.
        import time as _time

        if hasattr(self.parent, "updater_manager"):
            self.parent.updater_manager._next_scan_time = _time.monotonic() + interval

    def _stop_scanner(self):
        """Stop scanner loop."""
        if self.updater:
            self.updater.stop()
            self.updater = None
            log_info("Scanner stopped")
            # Clear countdown
            if hasattr(self.parent, "updater_manager"):
                self.parent.updater_manager._next_scan_time = 0.0

    def _initialize_pipeline(self) -> bool:
        """Initialize the SignalPipeline with all components."""
        if self._pipeline_initialized and self.pipeline:
            return True

        try:
            log_info("Initializing SignalPipeline...")

            from config import (
                ATC_SCANNER_DEFAULTS,
                SIGNAL_SELECTOR_DEFAULTS,
                XGBOOST_FILTER_DEFAULTS,
                XGBOOST_PER_SYMBOL_DEFAULTS,
            )
            from modules.auto_trade.core.atc_scanner import ATCScanner, ATCScannerConfig
            from modules.auto_trade.core.atc_serverless_scanner import ATCServerlessScanner
            from modules.auto_trade.core.gemini_integration import GeminiIntegration
            from modules.auto_trade.core.signal_pipeline import ATCScannerLike, SignalPipeline, XGBoostFilterLike
            from modules.auto_trade.core.signal_selector import SignalSelector
            from modules.auto_trade.core.symbol_manager import SymbolManager
            from modules.auto_trade.core.xgboost_filter import XGBoostFilter, XGBoostFilterConfig
            from modules.auto_trade.core.xgboost_per_symbol import (
                XGBoostPerSymbolConfig,
                XGBoostPerSymbolFilter,
            )
            from modules.auto_trade.core.xgboost_serverless_filter import XGBoostServerlessFilter

            # Get scanner config from settings
            scanner_config = self.parent.settings_manager.get("scanner", {})
            timeframe = scanner_config.get("timeframe", "1h")
            sample_percentage = scanner_config.get("sample_percentage", 20)
            sampling_strategy = scanner_config.get("sampling_strategy", "stratified")
            atc_backend = str(scanner_config.get("atc_backend", "local")).lower()
            xgboost_backend = str(scanner_config.get("xgboost_backend", "local")).lower()

            # Get XGBoost mode from settings (default: per_symbol)
            xgboost_mode = scanner_config.get("xgboost_mode", "per_symbol")

            # Get or create data_fetcher from data_service
            data_fetcher = self.parent.data_service.data_fetcher
            if not data_fetcher:
                log_error("DataFetcher not available (required for pipeline)")
                return False

            # 1. Symbol Manager
            symbol_manager = SymbolManager(
                data_fetcher=data_fetcher,
                sample_percentage=sample_percentage,
                sampling_strategy=sampling_strategy,
            )
            log_info(f"SymbolManager ready (sample: {sample_percentage}%, strategy: {sampling_strategy})")

            # 2. ATC Scanner - threshold from Scanner Configuration
            atc_config = ATC_SCANNER_DEFAULTS.copy()
            atc_config["threshold"] = float(scanner_config.get("atc_threshold", 0.6))

            atc_scanner: ATCScannerLike | None = None
            if atc_backend == "serverless":
                serverless_config = {
                    **atc_config,
                    "serverless_function_name": scanner_config.get("atc_serverless_function_name", "atc-serverless"),
                    "serverless_sqs_queue": scanner_config.get("atc_serverless_sqs_queue", "atc-results"),
                    "serverless_region": scanner_config.get("atc_serverless_region", "us-east-1"),
                    "serverless_sqs_poll_timeout": scanner_config.get("atc_serverless_sqs_poll_timeout", 60),
                    "serverless_sqs_poll_interval": scanner_config.get("atc_serverless_sqs_poll_interval", 2.0),
                    "serverless_ohlcv_limit": scanner_config.get("atc_serverless_ohlcv_limit", 220),
                    "serverless_min_candles_per_tf": scanner_config.get("atc_serverless_min_candles_per_tf", 50),
                    "serverless_mock_mode": scanner_config.get("atc_serverless_mock_mode", False),
                }
                try:
                    atc_scanner = ATCServerlessScanner(data_fetcher=data_fetcher, config=serverless_config)
                    log_info(
                        "ATCServerlessScanner ready "
                        f"(function={serverless_config['serverless_function_name']}, "
                        f"region={serverless_config['serverless_region']})"
                    )
                except Exception as exc:
                    log_error("ATC serverless backend init failed: %s", exc)
                    log_info("Falling back to local ATC scanner backend")

            if atc_scanner is None:
                atc_scanner = ATCScanner(data_fetcher=data_fetcher, config=cast(ATCScannerConfig, atc_config))
                log_info(f"ATCScanner ready (backend=local, timeframes: {atc_config['timeframes']})")

            # 3. XGBoost Filter (serverless, per-symbol, or pre-trained based on config)
            xgboost_filter: XGBoostFilterLike = self._create_passthrough_xgboost_filter()
            if scanner_config.get("enable_xgboost", True):
                if xgboost_backend == "serverless":
                    # Serverless mode - delegate to AWS Lambda
                    serverless_xgb_config = {
                        "xgboost_serverless_function_name": scanner_config.get(
                            "xgboost_serverless_function_name", "xgboost-serverless-predict"
                        ),
                        "xgboost_serverless_region": scanner_config.get("xgboost_serverless_region", "us-east-1"),
                        "xgboost_serverless_model_version": scanner_config.get(
                            "xgboost_serverless_model_version", "v1"
                        ),
                        "xgboost_serverless_timeframe": timeframe,
                        "xgboost_serverless_candle_limit": scanner_config.get("xgboost_serverless_candle_limit", 200),
                        "xgboost_serverless_min_confidence": float(scanner_config.get("xgboost_min_confidence", 0.55)),
                        "xgboost_serverless_min_candles": scanner_config.get("xgboost_serverless_min_candles", 50),
                        "xgboost_serverless_mock_mode": scanner_config.get("xgboost_serverless_mock_mode", False),
                    }
                    try:
                        xgboost_filter = XGBoostServerlessFilter(
                            data_fetcher=data_fetcher,
                            config=serverless_xgb_config,
                        )
                        log_info(
                            "XGBoostServerlessFilter ready "
                            f"(function={serverless_xgb_config['xgboost_serverless_function_name']}, "
                            f"region={serverless_xgb_config['xgboost_serverless_region']})"
                        )
                    except Exception as exc:
                        log_error("XGBoost serverless backend init failed: %s", exc)
                        log_info("Falling back to local XGBoost filter")
                        xgboost_backend = "local"  # fall through to local logic below

                if xgboost_backend == "local":
                    if xgboost_mode == "per_symbol":
                        # Per-symbol training mode - trains fresh XGBoost for each symbol
                        xgboost_config = XGBOOST_PER_SYMBOL_DEFAULTS.copy()
                        xgboost_config["training_timeframe"] = timeframe
                        xgboost_filter = XGBoostPerSymbolFilter(
                            data_fetcher=data_fetcher,
                            config=cast(XGBoostPerSymbolConfig, xgboost_config),
                        )
                        log_info(f"XGBoostPerSymbolFilter ready (per-symbol training, timeframe: {timeframe})")
                    else:
                        # Pre-trained model mode (legacy behavior)
                        model_path = self._find_xgboost_model()
                        if model_path:
                            xgboost_filter = XGBoostFilter(
                                data_fetcher=data_fetcher,
                                model_path=model_path,
                                config=cast(XGBoostFilterConfig, XGBOOST_FILTER_DEFAULTS),
                            )
                            log_info(f"XGBoostFilter ready (pre-trained model: {Path(model_path).name})")
                        else:
                            log_warn("XGBoost model not found, using passthrough filter")
                            xgboost_filter = self._create_passthrough_xgboost_filter()
            else:
                log_info("XGBoost disabled in scanner config, using passthrough")

            # 4. Gemini Integration
            gemini_integration = GeminiIntegration(data_fetcher=data_fetcher, analysis_timeframe=timeframe)
            if gemini_integration.is_available():
                log_info("GeminiIntegration ready (API configured)")
            else:
                log_warn("GeminiIntegration (API not configured, will skip AI analysis)")

            # 5. Signal Selector
            selector_config = SIGNAL_SELECTOR_DEFAULTS.copy()
            selector_config["min_confidence_threshold"] = float(scanner_config.get("min_signal_score", 0.7))
            signal_selector = SignalSelector(config=selector_config)
            log_info(f"SignalSelector ready (min_confidence: {selector_config['min_confidence_threshold']})")

            # 6. Gann Square Filter (optional)
            gann_filter = None
            if scanner_config.get("enable_gann_square", False):
                try:
                    from modules.auto_trade.core.gann_square_filter import GannSquareFilter

                    gann_filter = GannSquareFilter(
                        timeframe=scanner_config.get("gann_timeframe", "1h"),
                        limit=int(scanner_config.get("gann_candle_limit", 200)),
                        lookback=int(scanner_config.get("gann_lookback", 5)),
                    )
                    log_info(
                        f"GannSquareFilter ready (tf={gann_filter.timeframe}, limit={gann_filter.limit}, lookback={gann_filter.lookback})"
                    )
                except ImportError as e:
                    log_warn(f"GannSquareFilter not available: {e}")

            # 7. Create Pipeline
            assert atc_scanner is not None, "atc_scanner must be initialized before building the pipeline"
            self.pipeline = SignalPipeline(
                symbol_manager=symbol_manager,
                atc_scanner=atc_scanner,
                xgboost_filter=xgboost_filter,
                gemini_integration=gemini_integration,
                signal_selector=signal_selector,
                config={
                    "max_symbols_to_scan": 30,
                    "max_ai_candidates": 5,
                    "xgboost_mode": xgboost_mode,
                },
                gann_square_filter=gann_filter,
            )
            log_info(f"SignalPipeline initialized (xgboost_mode: {xgboost_mode}, gann: {gann_filter is not None})")

            self._pipeline_initialized = True
            return True

        except ImportError as e:
            log_error(f"Failed to import pipeline components: {e}")
            return False
        except Exception as e:
            log_error(f"Failed to initialize pipeline: {e}")
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
            """Passthrough filter that adds mock confidence scores.

            Uses _replace() to create new SignalResult instances instead of
            mutating the shared details dict (safe for NamedTuple + shared refs).
            """

            def filter_signals(self, signals):
                """Pass through all signals with default confidence scores."""
                result = []
                for signal in signals:
                    # Build a new merged dict — never mutate the original
                    new_details = {**signal.details, "xgboost_conf": 0.7, "xgboost_dir": signal.signal_type}
                    result.append(signal._replace(details=new_details))
                return result

        return PassthroughXGBoostFilter()

    def _scanner_cycle(self):
        """Scanner cycle (runs in background thread)."""
        start_time = None
        with self._scan_lock:
            if self._scan_running:
                log_warn("Scanner cycle already in progress, skipping...")
                return
            self._scan_running = True
        start_time = time.perf_counter()
        try:
            log_info("=" * 50)
            log_info("Running scanner cycle...")
            log_info(f"Mode: {self.parent.mode}")
            log_info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            log_info("-" * 50)

            new_signal = None
            scan_skipped = False
            open_count = 0

            # Gate: skip expensive scan (Gemini) when already at max open positions
            # Check BOTH database AND Binance positions for accuracy
            if self.parent.mode in ["PRODUCTION", "DEMO"]:
                try:
                    max_open = self.parent.settings_manager.get("risk.max_open_positions", 3)
                    if not isinstance(max_open, int) or max_open <= 0:
                        max_open = 1

                    # Check database positions
                    db_open_count = 0
                    try:
                        from modules.auto_trade.database import get_open_positions

                        open_orders = get_open_positions()
                        db_open_count = len(open_orders)
                    except Exception as e:
                        log_warn(f"Could not check DB positions: {e}")

                    # Check live Binance positions (source of truth for PRODUCTION/DEMO)
                    binance_open_count = 0
                    try:
                        positions = self.parent.data_service.get_positions()
                        if positions:
                            binance_open_count = len(positions)
                    except Exception as e:
                        log_warn(f"Could not check Binance positions: {e}")

                    # Use the HIGHER count (most conservative check)
                    open_count = max(db_open_count, binance_open_count)

                    log_info(f"Position check: DB={db_open_count}, Binance={binance_open_count}, Using={open_count}")

                    if open_count >= max_open:
                        scan_skipped = True
                        log_info(
                            f"Scanner cycle skipped: open position(s) present ({open_count}/{max_open}), "
                            "no Gemini call. Will run full scan again when position(s) close."
                        )
                except Exception as e:
                    log_warn(f"Could not check open positions for scanner gate: {e}. Skipping scan (no Gemini).")
                    scan_skipped = True

            if not scan_skipped and self.parent.mode in ["PRODUCTION", "DEMO"]:
                log_info("Scanning market for new signals...")
                new_signal = self._run_signal_scan()
                if new_signal:
                    log_info(f"New signal generated: {new_signal.symbol} {new_signal.signal_type}")
                    log_info(f"Confidence: {new_signal.confidence:.1%}")
                    log_info(f"Score: {new_signal.score:.2f}/100")
                else:
                    log_warn("No new signals generated from scan")
            elif not scan_skipped:
                log_info("DRY_RUN mode - skipping live market scan")

            # Get signals from database (existing + newly generated)
            log_info("Fetching signals from database...")
            signals = self.parent.data_service.get_signals()
            log_info(f"Total signals in database: {len(signals)}")

            if signals:
                log_info("Top signals:")
                for i, sig in enumerate(signals[:3]):
                    sym = sig.get("symbol", "N/A")
                    sgn = sig.get("signal", "N/A")
                    sc = sig.get("score", 0)
                    ts = sig.get("created_at_ts", 0.0)
                    sig_time = sig.get("time", "")
                    if ts:
                        age_hours = (time.time() - ts) / 3600
                        if age_hours < 1:
                            age_str = f"{age_hours * 60:.0f}min ago"
                        else:
                            age_str = f"{age_hours:.1f}h ago"
                        log_info(f"  {i + 1}. {sym} - {sgn} ({sc:.2f}) | {sig_time} [{age_str}]")
                    else:
                        log_info(f"  {i + 1}. {sym} - {sgn} ({sc:.2f}) | time unknown")
            else:
                log_info("No signals found in database (or all signals are stale)")

            self.parent._update_queue.put(("signals", signals))
            self.parent._update_queue.put(
                ("scanner_done", {"skipped": scan_skipped, "count": open_count} if scan_skipped else None)
            )
            log_info("-" * 50)
            log_info("Scanner cycle completed")
            log_info("=" * 50)
        except Exception as e:
            log_error(f"Error in scanner cycle: {e}")
            import traceback

            log_error(traceback.format_exc())
        finally:
            with self._scan_lock:
                self._scan_running = False
            if start_time is not None:
                duration = time.perf_counter() - start_time
                log_info("Scanner cycle completed in %.1fs", duration)

    def _run_signal_scan(self):
        """Run actual signal pipeline scan.

        Returns:
            FinalSignal object if a signal was generated, None otherwise.
        """
        try:
            # Initialize pipeline if not already done
            if not self._initialize_pipeline():
                log_warn("Pipeline not available, reading from database only")
                return None

            assert self.pipeline is not None  # guaranteed after _initialize_pipeline() returns True
            # Run the pipeline
            log_info("Running SignalPipeline...")
            final_signal = self.pipeline.run_pipeline()

            # If signal generated, also save to main GUI database
            if final_signal:
                self._save_signal_to_gui_db(final_signal)

            return final_signal

        except Exception as e:
            log_error(f"Error running signal scan: {e}")
            import traceback

            log_error(traceback.format_exc())
            return None

    def _save_signal_to_gui_db(self, signal):
        """Save signal to main GUI database for display."""
        try:
            import json
            import uuid

            from modules.auto_trade.database.repository.context import RepositoryContext

            repo_context = RepositoryContext.from_env()
            correlation_id = f"scan-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

            repo_context.signals.save_signal(
                {
                    "correlation_id": correlation_id,
                    "symbol": signal.symbol,
                    "signal_type": signal.signal_type,
                    "confidence": signal.confidence,
                    "atc_score": signal.sources.get("atc_score"),
                    "xgboost_score": signal.sources.get("xgboost_score"),
                    "gemini_score": signal.sources.get("gemini_score"),
                    "final_score": signal.score / 100.0,
                    "market_context": json.dumps(signal.sources) if signal.sources else None,
                }
            )
            log_info("Signal saved to GUI database")
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
                log_info(f"Published SIGNAL_GENERATED event for {signal.symbol}")
            except Exception as e:
                log_warn(f"Failed to publish signal event: {e}")

        except Exception as e:
            log_warn(f"Could not save to GUI database: {e}")
            # Don't fail the whole scan if GUI DB save fails
