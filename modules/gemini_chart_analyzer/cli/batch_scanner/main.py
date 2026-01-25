"""
CLI Main Program for Market Batch Scanner.

Interactive menu for batch scanning entire market with Gemini.
"""

import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict

from colorama import Fore
from colorama import init as colorama_init

# Add project root to sys.path
if "__file__" in globals():
    project_root = Path(__file__).parent.parent.parent.parent.parent
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


from modules.common.utils import color_text, log_error, log_warn, safe_input
from modules.gemini_chart_analyzer.cli.batch_scanner.config_builder import gather_scan_configuration
from modules.gemini_chart_analyzer.cli.batch_scanner.display import display_configuration_summary
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

# Suppress specific noisy warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PIL.Image")
colorama_init(autoreset=True)


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
            pre_filter_auto_skip_threshold=config.get("pre_filter_auto_skip_threshold", 10),
            fast_mode=config["fast_mode"],
            spc_config=config["spc_config"] if config["enable_pre_filter"] else None,
            rf_model_path=config["random_forest_model"]["model_path"],
            stage0_sample_percentage=config.get("stage0_sample_percentage"),
            rf_training=config.get("rf_training"),
            atc_performance=config.get("atc_performance"),
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
