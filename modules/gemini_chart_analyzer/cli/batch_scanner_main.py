"""
CLI Main Program for Market Batch Scanner.

Interactive menu for batch scanning entire market with Gemini.
"""

import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from colorama import Fore
from colorama import init as colorama_init

# Add project root to sys.path
if "__file__" in globals():
    project_root = Path(__file__).parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

# Ensure stdin is available on Windows before configuring stdio
if sys.platform == "win32":
    try:
        if sys.stdin is None or (hasattr(sys.stdin, "closed") and sys.stdin.closed):
            sys.stdin = open("CON", "r", encoding="utf-8", errors="replace")
    except (OSError, IOError, AttributeError):
        pass

# Fix encoding issues on Windows
from modules.common.utils import configure_windows_stdio

configure_windows_stdio()

from modules.common.ui.logging import log_info
from modules.common.utils import color_text, log_error, log_warn, safe_input
from modules.gemini_chart_analyzer.cli.config.display import display_loaded_configuration
from modules.gemini_chart_analyzer.cli.config.exporter import export_configuration_to_json
from modules.gemini_chart_analyzer.cli.config.loader import list_configuration_files, load_configuration_from_json
from modules.gemini_chart_analyzer.cli.models.random_forest_manager import (
    check_random_forest_model_status,
)
from modules.gemini_chart_analyzer.cli.prompts.model_training import (
    confirm_training,
    prompt_model_action,
    prompt_training_limit,
    prompt_training_symbols,
    prompt_training_timeframe,
)
from modules.gemini_chart_analyzer.cli.prompts.pre_filter import (
    prompt_enable_pre_filter,
    prompt_fast_mode,
    prompt_pre_filter_mode,
    prompt_pre_filter_percentage,
)
from modules.gemini_chart_analyzer.cli.prompts.spc import (
    prompt_spc_config_mode,
    prompt_spc_custom_config,
    prompt_spc_preset,
)
from modules.gemini_chart_analyzer.cli.prompts.timeframe import (
    prompt_analysis_mode,
    prompt_cooldown,
    prompt_limit,
    prompt_max_symbols,
)
from modules.gemini_chart_analyzer.cli.runners.scanner_runner import display_scan_results
from modules.gemini_chart_analyzer.core.exceptions import (
    ChartGenerationError,
    DataFetchError,
    GeminiAnalysisError,
    ReportGenerationError,
    ScanConfigurationError,
)
from modules.gemini_chart_analyzer.core.scanner_types import BatchScanResult
from modules.gemini_chart_analyzer.services.batch_scan_service import BatchScanConfig, run_batch_scan
from modules.gemini_chart_analyzer.services.model_training_service import run_rf_model_training

# Suppress specific noisy warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PIL.Image")
colorama_init(autoreset=True)


def init_components():
    """Initialize ExchangeManager and DataFetcher."""
    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.core.exchange_manager import ExchangeManager

    log_info("Initializing ExchangeManager and DataFetcher...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    return exchange_manager, data_fetcher


def gather_scan_configuration() -> Dict[str, Any]:
    """Gather all scan configuration through prompts or file loading."""
    # Load configuration from JSON (optional)
    loaded_config = None
    use_loaded_config = False

    config_files = list_configuration_files()
    if config_files:
        print("\nLoad Configuration:")
        print("  Load configuration from a previously saved JSON file")
        load_config_input = safe_input(
            color_text("Load configuration from JSON? (y/n) [n]: ", Fore.YELLOW), default="n"
        ).lower()
        if load_config_input in ["y", "yes"]:
            print("\nAvailable configuration files:")
            for idx, config_file in enumerate(config_files[:10], 1):
                try:
                    mtime = datetime.fromtimestamp(config_file.stat().st_mtime)
                    mtime_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"  {idx}. {config_file.name} ({mtime_str})")
                except (OSError, ValueError):
                    print(f"  {idx}. {config_file.name}")

            print("  Or enter full path to a configuration file")
            file_choice = safe_input(
                color_text("Select file number or enter path (press Enter to skip): ", Fore.YELLOW),
                default="",
            ).strip()

            config_path = None
            if file_choice:
                try:
                    file_idx = int(file_choice)
                    if 1 <= file_idx <= len(config_files):
                        config_path = config_files[file_idx - 1]
                except ValueError:
                    config_path = Path(file_choice)

                if config_path:
                    loaded_config = load_configuration_from_json(config_path)
                    if loaded_config:
                        display_loaded_configuration(loaded_config)
                        print("\nConfiguration Options:")
                        print("  1. Use loaded configuration as-is")
                        print("  2. Use as defaults and adjust")
                        print("  3. Start fresh (ignore loaded config)")
                        use_choice = safe_input(color_text("Select option (1/2/3) [2]: ", Fore.YELLOW), default="2")
                        if not use_choice:
                            use_choice = "2"
                        if use_choice == "1":
                            use_loaded_config = True
                        elif use_choice == "2":
                            use_loaded_config = False
                        else:
                            loaded_config = None
                            use_loaded_config = False
            else:
                log_info("Skipping configuration load")

    # Configuration extraction
    if use_loaded_config and loaded_config:
        analysis_mode = loaded_config.get("analysis_mode", "multi-timeframe")
        timeframe = loaded_config.get("timeframe", "1h")
        timeframes = loaded_config.get("timeframes", None)
        max_symbols = loaded_config.get("max_symbols", None)
        cooldown = loaded_config.get("cooldown", 2.5)
        limit = loaded_config.get("limit", 700)
        enable_pre_filter = loaded_config.get("enable_pre_filter", False)
        pre_filter_mode = loaded_config.get("pre_filter_mode", "voting")
        pre_filter_percentage = loaded_config.get("pre_filter_percentage", None)
        fast_mode = loaded_config.get("fast_mode", True)
        spc_config_data = loaded_config.get("spc_config", {})

        spc_preset = spc_config_data.get("preset")
        spc_volatility_adjustment = spc_config_data.get("volatility_adjustment", False)
        spc_use_correlation_weights = spc_config_data.get("use_correlation_weights", False)
        spc_time_decay_factor = spc_config_data.get("time_decay_factor")
        spc_interpolation_mode = spc_config_data.get("interpolation_mode")
        spc_min_flip_duration = spc_config_data.get("min_flip_duration")
        spc_flip_confidence_threshold = spc_config_data.get("flip_confidence_threshold")
        spc_enable_mtf = spc_config_data.get("enable_mtf", False)
        spc_mtf_timeframes = spc_config_data.get("mtf_timeframes")
        spc_mtf_require_alignment = spc_config_data.get("mtf_require_alignment")

        rf_model_config = loaded_config.get("random_forest_model", {})
        rf_model_status = rf_model_config.get("status")
        rf_model_retrained = rf_model_config.get("retrained", False)
        log_info("Using loaded configuration as-is...")
    else:
        analysis_mode, timeframe, timeframes = prompt_analysis_mode(default="2", loaded_config=loaded_config)
        max_symbols = prompt_max_symbols(default=None, loaded_config=loaded_config)
        cooldown = prompt_cooldown(default=2.5, loaded_config=loaded_config)
        limit = prompt_limit(default=700, loaded_config=loaded_config)
        enable_pre_filter = prompt_enable_pre_filter(default=True, loaded_config=loaded_config)

        pre_filter_mode = "voting"
        pre_filter_percentage = None
        fast_mode = True
        spc_preset = None
        spc_volatility_adjustment = False
        spc_use_correlation_weights = False
        spc_time_decay_factor = None
        spc_interpolation_mode = None
        spc_min_flip_duration = None
        spc_flip_confidence_threshold = None
        spc_enable_mtf = False
        spc_mtf_timeframes = None
        spc_mtf_require_alignment = None
        rf_model_status = None
        rf_model_retrained = False

        if enable_pre_filter:
            pre_filter_mode = prompt_pre_filter_mode(default="voting", loaded_config=loaded_config)
            pre_filter_percentage = prompt_pre_filter_percentage(default=None, loaded_config=loaded_config)
            fast_mode = prompt_fast_mode(default=True, loaded_config=loaded_config)
            spc_config_mode = prompt_spc_config_mode(default="3", loaded_config=loaded_config)

            if spc_config_mode == "1":
                spc_preset = prompt_spc_preset()
            elif spc_config_mode == "2":
                spc_config_res = prompt_spc_custom_config(loaded_config=loaded_config)
                spc_volatility_adjustment = spc_config_res.get("volatility_adjustment", False)
                spc_use_correlation_weights = spc_config_res.get("use_correlation_weights", False)
                spc_time_decay_factor = spc_config_res.get("time_decay_factor")
                spc_interpolation_mode = spc_config_res.get("interpolation_mode")
                spc_min_flip_duration = spc_config_res.get("min_flip_duration")
                spc_flip_confidence_threshold = spc_config_res.get("flip_confidence_threshold")
                spc_enable_mtf = spc_config_res.get("enable_mtf", False)
                spc_mtf_timeframes = spc_config_res.get("mtf_timeframes")
                spc_mtf_require_alignment = spc_config_res.get("mtf_require_alignment")

            if not fast_mode:
                rf_model_status = check_random_forest_model_status()
                rf_action = prompt_model_action(rf_model_status)
                if rf_action == "retrain":
                    training_symbols = prompt_training_symbols()
                    training_tf = prompt_training_timeframe()
                    training_limit = prompt_training_limit()
                    if confirm_training(training_symbols, training_tf, training_limit):
                        _, data_fetcher = init_components()
                        success, new_path = run_rf_model_training(
                            data_fetcher, training_symbols, training_tf, training_limit
                        )
                        if success:
                            rf_model_retrained = True
                            rf_model_status = check_random_forest_model_status(new_path)
                        else:
                            log_error("Model training failed.")

    # Export Configuration
    print("\nExport Configuration:")
    export_config_input = safe_input(
        color_text("Export configuration to JSON? (y/n) [n]: ", Fore.YELLOW), default="n"
    ).lower()
    if export_config_input in ["y", "yes"]:
        config_data = {
            "analysis_mode": analysis_mode,
            "timeframe": timeframe,
            "timeframes": timeframes,
            "max_symbols": max_symbols,
            "cooldown": cooldown,
            "limit": limit,
            "enable_pre_filter": enable_pre_filter,
            "pre_filter_mode": pre_filter_mode,
            "pre_filter_percentage": pre_filter_percentage,
            "fast_mode": fast_mode,
            "spc_config": {
                "preset": spc_preset,
                "volatility_adjustment": spc_volatility_adjustment,
                "use_correlation_weights": spc_use_correlation_weights,
                "time_decay_factor": spc_time_decay_factor,
                "interpolation_mode": spc_interpolation_mode,
                "min_flip_duration": spc_min_flip_duration,
                "flip_confidence_threshold": spc_flip_confidence_threshold,
                "enable_mtf": spc_enable_mtf,
                "mtf_timeframes": spc_mtf_timeframes,
                "mtf_require_alignment": spc_mtf_require_alignment,
            },
            "random_forest_model": {"status": rf_model_status, "retrained": rf_model_retrained},
            "export_timestamp": datetime.now().isoformat(),
        }
        export_configuration_to_json(config_data)

    # Return complete configuration dict
    return {
        "analysis_mode": analysis_mode,
        "timeframe": timeframe,
        "timeframes": timeframes,
        "max_symbols": max_symbols,
        "cooldown": cooldown,
        "limit": limit,
        "enable_pre_filter": enable_pre_filter,
        "pre_filter_mode": pre_filter_mode,
        "pre_filter_percentage": pre_filter_percentage,
        "fast_mode": fast_mode,
        "spc_config": {
            "preset": spc_preset,
            "volatility_adjustment": spc_volatility_adjustment,
            "use_correlation_weights": spc_use_correlation_weights,
            "time_decay_factor": spc_time_decay_factor,
            "interpolation_mode": spc_interpolation_mode,
            "min_flip_duration": spc_min_flip_duration,
            "flip_confidence_threshold": spc_flip_confidence_threshold,
            "enable_mtf": spc_enable_mtf,
            "mtf_timeframes": spc_mtf_timeframes,
            "mtf_require_alignment": spc_mtf_require_alignment,
        },
        "random_forest_model": {
            "status": rf_model_status,
            "retrained": rf_model_retrained,
            "model_path": rf_model_status.get("model_path") if rf_model_status else None,
        },
    }


def display_configuration_summary(config: Dict[str, Any]) -> None:
    """Display configuration summary before scan."""
    print("\n" + color_text("=" * 50, Fore.CYAN))
    print(color_text("CONFIGURATION SUMMARY", Fore.CYAN))
    print(color_text("=" * 50, Fore.CYAN))

    print(f"Analysis Mode: {config.get('analysis_mode', 'N/A')}")
    print(f"Timeframe: {config.get('timeframe', 'N/A')}")
    if config.get("timeframes"):
        print(f"Timeframes: {', '.join(config['timeframes'])}")
    print(f"Max Symbols: {config.get('max_symbols', 'All')}")
    print(f"Cooldown: {config.get('cooldown', 'N/A')}s")
    print(f"Limit: {config.get('limit', 'N/A')}")

    if config.get("enable_pre_filter"):
        print(f"Pre-filter Mode: {config.get('pre_filter_mode', 'N/A')}")
        pre_filter_percentage = config.get("pre_filter_percentage")
        if pre_filter_percentage is not None:
            print(f"Pre-filter Percentage: {pre_filter_percentage}%")
        else:
            print("Pre-filter Percentage: 10% (default)")
        print(f"Fast Mode: {config.get('fast_mode', 'N/A')}")
        spc_config = config.get("spc_config", {})
        if spc_config.get("preset"):
            print(f"SPC Preset: {spc_config['preset']}")
        else:
            print("SPC Config: Custom")
    else:
        print("Pre-filter: Disabled")

    rf_model = config.get("random_forest_model", {})
    if rf_model.get("status"):
        print(f"RF Model: Available ({rf_model['status'].get('model_path', 'Unknown')})")
    else:
        print("RF Model: None")

    print(color_text("=" * 50, Fore.CYAN))


def execute_scan(config: Dict[str, Any]) -> BatchScanResult:
    """Execute the scan with given configuration."""
    try:
        # Create BatchScanConfig from dict
        batch_config = BatchScanConfig(
            timeframe=config["timeframe"],
            timeframes=config["timeframes"],
            max_symbols=config["max_symbols"],
            limit=config["limit"],
            cooldown=config["cooldown"],
            enable_pre_filter=config["enable_pre_filter"],
            pre_filter_mode=config["pre_filter_mode"],
            pre_filter_percentage=config.get("pre_filter_percentage"),
            fast_mode=config["fast_mode"],
            spc_config=config["spc_config"] if config["enable_pre_filter"] else None,
            rf_model_path=config["random_forest_model"]["model_path"],
        )

        # Run the scan
        results = run_batch_scan(batch_config)
        return results
    except ScanConfigurationError as e:
        log_error(f"Configuration error: {e}")
        raise
    except DataFetchError as e:
        log_error(f"Market data fetch error: {e}")
        raise
    except GeminiAnalysisError as e:
        log_error(f"Gemini AI analysis error: {e}")
        raise
    except ChartGenerationError as e:
        log_error(f"Chart generation error: {e}")
        raise
    except ReportGenerationError as e:
        log_error(f"Report generation error: {e}")
        raise
    except Exception as e:
        log_error(f"Unexpected error during batch scan: {e}")
        traceback.print_exc()
        raise


def interactive_batch_scan():
    """Interactive menu for batch scanning."""
    print()
    print(color_text("=" * 60, Fore.CYAN))
    print(color_text("MARKET BATCH SCANNER", Fore.CYAN))
    print(color_text("=" * 60, Fore.CYAN))
    print()

    # 1. Load or gather configuration
    config = gather_scan_configuration()

    # 2. Display configuration summary
    display_configuration_summary(config)

    # 3. Confirm and run scan
    confirm = safe_input(color_text("Start batch scan? (y/n) [y]: ", Fore.YELLOW), default="y").lower()
    if confirm not in ["y", "yes", ""]:
        log_warn("Cancelled by user")
        return

    # 4. Run scan
    results = execute_scan(config)

    # 5. Display results
    display_scan_results(results)


def main():
    """Main entry point."""
    if sys.platform == "win32":
        try:
            if sys.stdin is None or (hasattr(sys.stdin, "closed") and sys.stdin.closed):
                sys.stdin = open("CON", "r", encoding="utf-8", errors="replace")
        except (OSError, IOError, AttributeError):
            pass

    try:
        interactive_batch_scan()
    except KeyboardInterrupt:
        log_warn("\nExiting...")
        sys.exit(0)
    except Exception as e:
        log_error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
