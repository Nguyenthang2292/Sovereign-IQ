"""Service layer for batch market scanning operations."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from modules.common.ui.logging import log_error, log_info, log_success
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
    fast_mode: bool = True
    initial_symbols: Optional[List[str]] = None
    spc_config: Optional[Dict[str, Any]] = None
    rf_model_path: Optional[str] = None
    skip_cleanup: bool = False
    output_dir: Optional[str] = None


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
            fast_mode=config.fast_mode,
            spc_config=config.spc_config,
            skip_cleanup=config.skip_cleanup,
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
