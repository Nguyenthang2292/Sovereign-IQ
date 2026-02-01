"""
Signal Pipeline Orchestrator

Coordinates the entire auto-trading process:
1. Refresh Symbols
2. Scan Market (ATC)
3. Filter Signals (XGBoost)
4. AI Analysis (Gemini)
5. Select Final Signal
"""

import time
import traceback
from typing import Dict, Optional

from modules.auto_trade.core.atc_scanner import ATCScanner
from modules.auto_trade.core.gemini_integration import GeminiIntegration, GeminiSignal
from modules.auto_trade.core.persistence import SignalPersistence
from modules.auto_trade.core.signal_selector import FinalSignal, SignalSelector
from modules.auto_trade.core.symbol_manager import SymbolManager
from modules.auto_trade.core.xgboost_filter import XGBoostFilter
from modules.common.ui.logging import log_error, log_info, log_warn


class SignalPipeline:
    def __init__(
        self,
        symbol_manager: SymbolManager,
        atc_scanner: ATCScanner,
        xgboost_filter: XGBoostFilter,
        gemini_integration: GeminiIntegration,
        signal_selector: SignalSelector,
        signal_persistence: Optional[SignalPersistence] = None,
        config: Optional[Dict] = None,
    ):
        self.symbol_manager = symbol_manager
        self.atc_scanner = atc_scanner
        self.xgboost_filter = xgboost_filter
        self.gemini_integration = gemini_integration
        self.signal_selector = signal_selector
        self.signal_persistence = signal_persistence
        self.config = config or {}

        self.max_symbols = self.config.get("max_symbols_to_scan", 20)
        self.pipeline_timeout = self.config.get("pipeline_timeout", 300)  # seconds

    def run_pipeline(self) -> Optional[FinalSignal]:
        """
        Execute the full trading pipeline to find the single best trading opportunity.

        Returns:
            FinalSignal object if a valid trade is found, else None.
        """
        start_time = time.time()
        log_info("Starting Signal Pipeline...")

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
            atc_signals = self.atc_scanner.scan_symbols(symbols)

            if not atc_signals:
                log_info("No ATC signals found.")
                return None

            log_info(f"ATC Found {len(atc_signals)} candidates.")

            # 3. XGBoost Filter
            log_info("Step 3: Filtering (XGBoost)...")
            xgboost_signals = self.xgboost_filter.filter_signals(atc_signals)

            if not xgboost_signals:
                log_info("No signals passed XGBoost filter.")
                return None

            log_info(f"XGBoost passed {len(xgboost_signals)} candidates.")

            # 4. Gemini Analysis
            log_info("Step 4: AI Analysis (Gemini)...")
            gemini_results: Dict[str, GeminiSignal] = {}

            for signal in xgboost_signals:
                if time.time() - start_time > self.pipeline_timeout:
                    log_warn("Pipeline timeout during Gemini analysis.")
                    break

                gemini_sig = self.gemini_integration.analyze_candidate(signal)
                if gemini_sig:
                    gemini_results[signal.symbol] = gemini_sig

            log_info(f"Gemini analyzed {len(gemini_results)} candidates successfully.")

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
