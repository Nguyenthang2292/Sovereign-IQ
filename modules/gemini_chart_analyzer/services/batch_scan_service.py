"""Service layer for batch market scanning operations."""

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner, ScanConfig
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_analysis_results_dir


class Stage0Config(BaseModel):
    """Stage 0 sampling configuration."""

    model_config = ConfigDict(extra="ignore")

    sample_percentage: Optional[float] = None
    sampling_strategy: str = "random"
    stratified_strata_count: int = 3
    hybrid_top_percentage: float = 50.0


class PreFilterConfig(BaseModel):
    """Pre-filter workflow configuration."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    mode: str = "voting"
    percentage: Optional[float] = None
    auto_skip_threshold: int = 10
    fast_mode: bool = True
    spc_config: Optional[Dict[str, Any]] = None
    stage0: Stage0Config = Field(default_factory=Stage0Config)


class ATCPerformanceConfig(BaseModel):
    """ATC performance module configuration."""

    model_config = ConfigDict(extra="ignore")

    settings: Optional[Dict[str, Any]] = None
    approximate_ma_scanner: Optional[Dict[str, Any]] = None
    use_performance: bool = True
    use_mini: bool = False


class XGBoostConfig(BaseModel):
    """XGBoost configuration for pre-filter stage."""

    model_config = ConfigDict(extra="ignore")

    settings: Optional[Dict[str, Any]] = None
    use_performance: bool = True


class BatchScanConfig(BaseModel):
    """Configuration for batch scan."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    timeframe: Optional[str] = None
    timeframes: Optional[List[str]] = None
    max_symbols: Optional[int] = None
    limit: int = 700
    cooldown: float = 2.5
    cancelled_callback: Optional[Callable[[], bool]] = None
    initial_symbols: Optional[List[str]] = None
    rf_model_path: Optional[str] = None
    skip_cleanup: bool = False
    output_dir: Optional[str] = None
    rf_training: Optional[Dict[str, Any]] = None
    pre_filter: PreFilterConfig = Field(default_factory=PreFilterConfig)
    atc: ATCPerformanceConfig = Field(default_factory=ATCPerformanceConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_flat_fields(cls, data: Any) -> Any:
        """Map legacy flat config fields to nested config models."""
        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        pre_filter_data = dict(normalized.get("pre_filter") or {})
        stage0_data = dict(pre_filter_data.get("stage0") or {})

        if "enable_pre_filter" in normalized:
            pre_filter_data.setdefault("enabled", normalized.pop("enable_pre_filter"))
        if "pre_filter_mode" in normalized:
            pre_filter_data.setdefault("mode", normalized.pop("pre_filter_mode"))
        if "pre_filter_percentage" in normalized:
            pre_filter_data.setdefault("percentage", normalized.pop("pre_filter_percentage"))
        if "pre_filter_auto_skip_threshold" in normalized:
            pre_filter_data.setdefault("auto_skip_threshold", normalized.pop("pre_filter_auto_skip_threshold"))
        if "pre_filter_fast_mode" in normalized:
            pre_filter_data.setdefault("fast_mode", normalized.pop("pre_filter_fast_mode"))
        if "spc_config" in normalized:
            pre_filter_data.setdefault("spc_config", normalized.pop("spc_config"))

        if "stage0_sample_percentage" in normalized:
            stage0_data.setdefault("sample_percentage", normalized.pop("stage0_sample_percentage"))
        if "stage0_sampling_strategy" in normalized:
            stage0_data.setdefault("sampling_strategy", normalized.pop("stage0_sampling_strategy"))
        if "stage0_stratified_strata_count" in normalized:
            stage0_data.setdefault("stratified_strata_count", normalized.pop("stage0_stratified_strata_count"))
        if "stage0_hybrid_top_percentage" in normalized:
            stage0_data.setdefault("hybrid_top_percentage", normalized.pop("stage0_hybrid_top_percentage"))

        if stage0_data:
            pre_filter_data["stage0"] = stage0_data
        if pre_filter_data:
            normalized["pre_filter"] = pre_filter_data

        atc_data = dict(normalized.get("atc") or {})
        if "atc_performance" in normalized:
            atc_data.setdefault("settings", normalized.pop("atc_performance"))
        if "approximate_ma_scanner" in normalized:
            atc_data.setdefault("approximate_ma_scanner", normalized.pop("approximate_ma_scanner"))
        if "use_atc_performance" in normalized:
            atc_data.setdefault("use_performance", normalized.pop("use_atc_performance"))
        if "use_atc_performance_mini" in normalized:
            atc_data.setdefault("use_mini", normalized.pop("use_atc_performance_mini"))
        if atc_data:
            normalized["atc"] = atc_data

        xgboost_data = dict(normalized.get("xgboost") or {})
        if "xgboost_lts" in normalized:
            xgboost_data.setdefault("settings", normalized.pop("xgboost_lts"))
        if "use_xgboost_performance" in normalized:
            xgboost_data.setdefault("use_performance", normalized.pop("use_xgboost_performance"))
        if xgboost_data:
            normalized["xgboost"] = xgboost_data

        return normalized


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

        # Execute pre-filter before scanning if enabled
        initial_symbols = config.initial_symbols
        if config.pre_filter.enabled and not initial_symbols:
            # Need to get all symbols first
            # We already initialized components for RF training, but if not we can use a fresh one
            from modules.gemini_chart_analyzer.cli.batch_scanner.utils import init_components
            from modules.gemini_chart_analyzer.core.prefilter.workflow import run_prefilter_worker

            _, data_fetcher = init_components()

            log_info(f"Pre-filter enabled ({config.pre_filter.mode}). Fetching all symbols...")
            try:
                all_symbols = scanner.get_all_symbols()

                initial_symbols = run_prefilter_worker(
                    all_symbols=all_symbols,
                    percentage=config.pre_filter.percentage if config.pre_filter.percentage is not None else 100.0,
                    timeframe=config.timeframe or "1h",
                    limit=config.limit,
                    fast_mode=config.pre_filter.fast_mode,
                    spc_config=config.pre_filter.spc_config,
                    rf_model_path=config.rf_model_path,
                    stage0_sample_percentage=config.pre_filter.stage0.sample_percentage,
                    stage0_sampling_strategy=config.pre_filter.stage0.sampling_strategy,
                    stage0_stratified_strata_count=config.pre_filter.stage0.stratified_strata_count,
                    stage0_hybrid_top_percentage=config.pre_filter.stage0.hybrid_top_percentage,
                    atc_performance=config.atc.settings,
                    approximate_ma_scanner=config.atc.approximate_ma_scanner,
                    auto_skip_threshold=config.pre_filter.auto_skip_threshold,
                    use_atc_performance=config.atc.use_performance,
                    use_atc_performance_mini=config.atc.use_mini,
                    xgboost_lts=config.xgboost.settings,
                    use_xgboost_performance=config.xgboost.use_performance,
                )
                log_success(f"Pre-filter complete. {len(initial_symbols)} symbols selected.")
            except Exception as e:
                log_error(f"Error during pre-filtering: {e}")
                log_warn("Proceeding with empty initial_symbols to scan all fetched symbols.")
                initial_symbols = None

        # Run scan
        scan_config = ScanConfig(
            timeframe=config.timeframe,
            timeframes=config.timeframes,
            max_symbols=config.max_symbols,
            limit=config.limit,
            cancelled_callback=config.cancelled_callback,
            initial_symbols=initial_symbols,
            skip_cleanup=config.skip_cleanup,
        )
        results = scanner.scan_market(scan_config)

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
