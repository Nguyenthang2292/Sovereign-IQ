"""Service layer for batch market scanning operations."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from modules.common.ui.logging import log_error, log_info, log_success, log_warn
from modules.gemini_chart_analyzer.core.exceptions import (
    ChartGenerationError,
    DataFetchError,
    GeminiAnalysisError,
    ReportGenerationError,
    ScanConfigurationError,
)
from modules.gemini_chart_analyzer.core.reporting.html_report_generator import generate_html_report
from modules.gemini_chart_analyzer.core.scanner_types import BatchScanResult
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_analysis_results_dir


@dataclass
class BatchScanConfig:
    """Configuration for batch scan."""

    timeframe: Optional[str] = None
    timeframes: Optional[List[str]] = None
    max_symbols: Optional[int] = None
    limit: int = 700
    cooldown: float = 2.5
    enable_pre_filter: bool = False
    pre_filter_mode: str = "voting"
    pre_filter_percentage: Optional[float] = None
    pre_filter_auto_skip_threshold: int = 10
    fast_mode: bool = True
    initial_symbols: Optional[List[str]] = None
    spc_config: Optional[Dict[str, Any]] = None
    rf_model_path: Optional[str] = None
    skip_cleanup: bool = False
    output_dir: Optional[str] = None
    stage0_sample_percentage: Optional[float] = None  # Stage 0: Random sampling before ATC
    rf_training: Optional[Dict[str, Any]] = None  # RF model training configuration
    atc_performance: Optional[Dict[str, Any]] = None  # ATC high-performance parameters


def run_batch_scan(config: BatchScanConfig) -> BatchScanResult:
    """Run batch scan service.
    Execute batch market scan with provided configuration.

    Args:
        config: BatchScanConfig object

    Returns:
        Dictionary containing scan results
    """
    try:
        log_info("Starting batch scan service...")

        # Handle RF Training if requested in config
        if config.rf_training and config.rf_training.get("auto_train"):
            from modules.gemini_chart_analyzer.cli.batch_scanner.utils import init_components
            from modules.gemini_chart_analyzer.services.model_training_service import run_rf_model_training

            log_info("Initializing components for RF training...")
            _, data_fetcher = init_components()

            # Resolve symbols for training
            training_symbols = []
            if config.rf_training.get("training_symbols_mode") == "manual":
                training_symbols = config.rf_training.get("manual_symbols", [])
                log_info(f"Using {len(training_symbols)} manual symbols for training.")
            else:
                # Auto mode: fetch top N symbols by volume
                count = config.rf_training.get("training_symbols_count", 10)
                log_info(f"Fetching top {count} symbols from Binance for RF training...")
                # Use list_binance_futures_symbols to get top symbols by volume
                training_symbols = data_fetcher.list_binance_futures_symbols(max_candidates=count)

            if training_symbols:
                success, new_path = run_rf_model_training(
                    data_fetcher=data_fetcher,
                    symbols=training_symbols,
                    timeframe=config.rf_training.get("training_timeframe", "1h"),
                    limit=config.rf_training.get("training_limit", 1500),
                )
                if success:
                    config.rf_model_path = new_path
                    log_success(f"RF model training successful! New model: {new_path}")
                else:
                    log_error("RF model training failed. Proceeding with existing model if available.")
            else:
                log_warn("No symbols found for RF training. Skipping training.")

        # Initialize scanner
        scanner = MarketBatchScanner(cooldown_seconds=config.cooldown, rf_model_path=config.rf_model_path)

        # Run scan
        results = scanner.scan_market(
            timeframe=config.timeframe,
            timeframes=config.timeframes,
            max_symbols=config.max_symbols,
            limit=config.limit,
            initial_symbols=config.initial_symbols,
            enable_pre_filter=config.enable_pre_filter,
            pre_filter_mode=config.pre_filter_mode,
            pre_filter_percentage=config.pre_filter_percentage,
            pre_filter_auto_skip_threshold=config.pre_filter_auto_skip_threshold,
            fast_mode=config.fast_mode,
            spc_config=config.spc_config,
            skip_cleanup=config.skip_cleanup,
            stage0_sample_percentage=config.stage0_sample_percentage,
            atc_performance=config.atc_performance,
        )

        # Generate HTML report if results were found
        # BatchScanResult is a dataclass, access attributes directly
        has_signals = results and (results.long_symbols or results.short_symbols)
        if has_signals:
            output_dir = config.output_dir or str(get_analysis_results_dir() / "batch_scan")

            log_info(f"Generating HTML report in {output_dir}...")
            # Convert BatchScanResult to dict for HTML report generation
            from dataclasses import asdict

            results_dict = asdict(results)
            html_path = generate_html_report(analysis_data=results_dict, output_dir=output_dir, report_type="batch")
            results.html_report_path = html_path
            log_success(f"Batch scan report generated: {html_path}")

        return results

    except ScanConfigurationError as e:
        log_error(f"Invalid scan configuration: {e}")
        raise
    except DataFetchError as e:
        log_error(f"Data fetching failed: {e}")
        raise
    except GeminiAnalysisError as e:
        log_error(f"Gemini analysis failed: {e}")
        raise
    except ChartGenerationError as e:
        log_error(f"Chart generation failed: {e}")
        raise
    except ReportGenerationError as e:
        log_error(f"Failed to generate reports: {e}")
        raise
    except Exception as e:
        log_error(f"Unexpected error in batch scan service: {e}")
        import traceback

        traceback.print_exc()
        raise
