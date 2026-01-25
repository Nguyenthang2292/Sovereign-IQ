"""
CLI Main Program for Gemini Chart Analyzer

Workflow:
1. Enter symbol names and timeframe
2. Generate chart images based on the entered information (can add indicators such as MA, RSI, etc.)
3. Save the image
4. Access Google Gemini using the API configured in config/config_api.py
5. Upload the image for Google Gemini to analyze the chart image (LONG/SHORT - TP/SL)
"""

import os
import sys
import warnings
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to sys.path to ensure modules can be imported
if "__file__" in globals():
    project_root = Path(__file__).parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

# Fix encoding issues on Windows
from modules.common.utils import configure_windows_stdio

configure_windows_stdio()

import json

from colorama import Fore
from colorama import init as colorama_init

from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.utils import (
    cleanup_old_files,
    color_text,
    log_error,
    log_info,
    log_success,
    log_warn,
    normalize_timeframe,
)
from modules.gemini_chart_analyzer.cli.argument_parser import parse_args
from modules.gemini_chart_analyzer.cli.interactive_menu import interactive_config_menu
from modules.gemini_chart_analyzer.core.reporting.html_report_generator import (
    generate_html_report as centralized_generate_html_report,
)
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_analysis_results_dir
from modules.gemini_chart_analyzer.services.chart_analysis_service import SingleAnalysisConfig, run_chart_analysis

# Suppress specific warnings from third-party libraries
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="matplotlib", message=r"(?i).*(findfont|font family|glyph|backend).*"
)
colorama_init(autoreset=True)


def _convert_args_to_config(args):
    """Convert parsed arguments to configuration format."""
    symbol = args.symbol
    timeframe = getattr(args, "timeframe", None)
    timeframes_list = getattr(args, "timeframes_list", None)

    indicators = {}
    if not args.no_ma:
        if args.ma_periods_list:
            indicators["MA"] = {"periods": args.ma_periods_list}
        else:
            indicators["MA"] = {"periods": [20, 50, 200]}

    if not args.no_rsi:
        indicators["RSI"] = {"period": args.rsi_period}

    if not args.no_macd:
        indicators["MACD"] = {"fast": 12, "slow": 26, "signal": 9}

    if args.enable_bb:
        indicators["BB"] = {"period": args.bb_period, "std": 2}

    prompt_type = args.prompt_type
    custom_prompt = getattr(args, "custom_prompt", None)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "timeframes_list": timeframes_list,
        "indicators": indicators,
        "prompt_type": prompt_type,
        "custom_prompt": custom_prompt,
        "limit": args.limit,
        "chart_figsize": args.chart_figsize_tuple,
        "chart_dpi": args.chart_dpi,
        "no_cleanup": args.no_cleanup,
    }


def _convert_menu_to_config(config):
    """Convert interactive menu config to format used by main()."""
    return {
        "symbol": config.symbol,
        "timeframe": getattr(config, "timeframe", None),
        "timeframes_list": getattr(config, "timeframes_list", None),
        "indicators": getattr(config, "indicators", {}),
        "prompt_type": config.prompt_type,
        "custom_prompt": getattr(config, "custom_prompt", None),
        "limit": getattr(config, "limit", 500),
        "chart_figsize": getattr(config, "chart_figsize_tuple", (16, 10)),
        "chart_dpi": getattr(config, "chart_dpi", 150),
        "no_cleanup": getattr(config, "no_cleanup", False),
    }


def format_text_to_html(text: str) -> str:
    """Delegates to centralized html_report_generator."""
    from modules.gemini_chart_analyzer.core.reporting.html_report_generator import _format_text_to_html

    return _format_text_to_html(text)


def _sanitize_chart_path(chart_path: str, output_dir: str) -> str:
    """Delegates to centralized html_report_generator."""
    from modules.gemini_chart_analyzer.core.reporting.html_report_generator import _sanitize_chart_path

    return _sanitize_chart_path(chart_path, output_dir)


def _find_chart_paths_for_timeframes(symbol: str, timeframes: List[str], charts_dir: str) -> Dict[str, str]:
    """Delegates to centralized html_report_generator."""
    from modules.gemini_chart_analyzer.core.reporting.html_report_generator import _find_chart_paths_for_timeframes

    return _find_chart_paths_for_timeframes(symbol, timeframes, charts_dir)


def generate_html_report(
    symbol: str, timeframe: str, chart_path: str, analysis_result: str, report_datetime: datetime, output_dir: str
) -> str:
    """Delegates to centralized html_report_generator."""
    from modules.gemini_chart_analyzer.core.reporting.html_report_generator import (
        generate_html_report as centralized_gen,
    )

    return centralized_gen(
        analysis_data={"symbol": symbol, "timeframe": timeframe, "analysis": analysis_result},
        output_dir=output_dir,
        report_type="single",
        chart_path=chart_path,
        report_datetime=report_datetime,
    )


def generate_multi_tf_html_report(
    symbol: str, timeframes_list: List[str], results: Dict, report_datetime: datetime, output_dir: str
) -> str:
    """Delegates to centralized html_report_generator."""
    from modules.gemini_chart_analyzer.core.reporting.html_report_generator import (
        generate_html_report as centralized_gen,
    )

    return centralized_gen(
        analysis_data=results,
        output_dir=output_dir,
        report_type="multi",
        timeframes_list=timeframes_list,
        report_datetime=report_datetime,
    )


def parse_and_build_config(args):
    """Parse arguments or show interactive menu and build configuration."""
    if args is None:
        config = interactive_config_menu()
        cfg = _convert_menu_to_config(config)
    else:
        cfg = _convert_args_to_config(args)
    return cfg


def init_components():
    """Initialize ExchangeManager and DataFetcher."""
    log_info("Initializing ExchangeManager and DataFetcher...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    return exchange_manager, data_fetcher


def save_and_open_reports(
    symbol: str,
    primary_timeframe: str,
    results_or_analysis,
    chart_path: Optional[str],
    output_dir: str,
    report_datetime: datetime,
    is_multi_tf: bool,
    timeframes_list: Optional[List[str]],
    prompt_type: str,
    no_cleanup: bool,
):
    """Save analysis results to files and open HTML report in the browser."""
    os.makedirs(output_dir, exist_ok=True)
    if not no_cleanup:
        cleanup_old_files(output_dir)

    timestamp = report_datetime.strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace("/", "_").replace(":", "_")

    if is_multi_tf:
        results = results_or_analysis
        json_file = os.path.join(output_dir, f"{safe_symbol}_multi_tf_{timestamp}.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "symbol": symbol,
                    "timestamp": report_datetime.isoformat(),
                    "timeframes_list": timeframes_list,
                    "timeframes": results["timeframes"],
                    "aggregated": results["aggregated"],
                    "prompt_type": prompt_type,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        log_success(f"Saved JSON results: {json_file}")

        result_file = os.path.join(output_dir, f"{safe_symbol}_multi_tf_{timestamp}.txt")
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Timeframes: {', '.join(timeframes_list) if timeframes_list else 'N/A'}\n")
            f.write(f"Prompt Type: {prompt_type}\n")
            f.write(f"\n{'=' * 60}\n")
            f.write("MULTI-TIMEFRAME ANALYSIS RESULTS\n")
            f.write(f"{'=' * 60}\n\n")
            if timeframes_list:
                for tf in timeframes_list:
                    if tf in results["timeframes"]:
                        tf_result = results["timeframes"][tf]
                        f.write(
                            f"{tf}: {tf_result.get('signal', 'NONE')} (confidence: {tf_result.get('confidence', 0.0):.2f})\n"
                        )
                        if "analysis" in tf_result:
                            f.write(f"  Analysis: {tf_result['analysis'][:200]}...\n")
                        f.write("\n")
            f.write(
                f"\nAGGREGATED: {results['aggregated'].get('signal', 'NONE')} (confidence: {results['aggregated'].get('confidence', 0.0):.2f})\n"
            )
        log_success(f"Saved summary: {result_file}")

        log_info("Generating HTML report for multi-timeframe...")
        html_path = centralized_generate_html_report(
            analysis_data=results,
            output_dir=output_dir,
            report_type="multi",
            timeframes_list=timeframes_list,
            report_datetime=report_datetime,
        )
        log_success(f"HTML report generated: {html_path}")
    else:
        analysis_result = results_or_analysis
        result_file = os.path.join(output_dir, f"{safe_symbol}_{primary_timeframe}_{timestamp}.txt")
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Timeframe: {primary_timeframe}\n")
            f.write(f"Chart Path: {chart_path}\n")
            f.write(f"Prompt Type: {prompt_type}\n")
            f.write(f"\n{'=' * 60}\n")
            f.write("ANALYSIS RESULTS\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(analysis_result if isinstance(analysis_result, str) else str(analysis_result))
        log_success(f"Analysis result saved: {result_file}")

        log_info("Generating HTML report...")
        html_path = centralized_generate_html_report(
            analysis_data={
                "symbol": symbol,
                "timeframe": primary_timeframe,
                "analysis": analysis_result if isinstance(analysis_result, str) else str(analysis_result),
            },
            output_dir=output_dir,
            report_type="single",
            chart_path=chart_path,
            report_datetime=report_datetime,
        )
        log_success(f"HTML report generated: {html_path}")

    try:
        html_uri = Path(html_path).resolve().as_uri()
        webbrowser.open(html_uri)
        log_success("Opened HTML report in browser")
    except Exception as e:
        log_warn(f"Could not open browser automatically: {e}")


def main():
    """Main function for Gemini Chart Analyzer."""
    print()
    print(color_text("=" * 60, Fore.CYAN))
    print(color_text("GEMINI CHART ANALYZER", Fore.CYAN))
    print(color_text("=" * 60, Fore.CYAN))

    try:
        args = parse_args()
        cfg = parse_and_build_config(args)

        symbol = cfg["symbol"]
        timeframe = cfg.get("timeframe")
        timeframes_list = cfg.get("timeframes_list")
        indicators = cfg["indicators"]
        prompt_type = cfg["prompt_type"]
        custom_prompt = cfg.get("custom_prompt")
        limit = cfg.get("limit", 500)
        chart_figsize = cfg.get("chart_figsize", (16, 10))
        chart_dpi = cfg.get("chart_dpi", 150)
        no_cleanup = cfg.get("no_cleanup", False)

        if not symbol:
            log_error("Symbol is required. Please provide --symbol or use interactive menu.")
            return

        is_multi_tf = timeframes_list is not None and len(timeframes_list) > 0
        if not is_multi_tf:
            timeframe = normalize_timeframe(timeframe) if timeframe else "1h"
        else:
            timeframe = timeframe or (timeframes_list[0] if timeframes_list else "1h")

        exchange_manager, data_fetcher = init_components()

        # Execute analysis using Service Layer
        config = SingleAnalysisConfig(
            symbol=symbol,
            timeframe=timeframe,
            timeframes_list=timeframes_list,
            indicators=indicators,
            prompt_type=prompt_type,
            custom_prompt=custom_prompt,
            limit=limit,
            chart_figsize=chart_figsize,
            chart_dpi=chart_dpi,
        )

        analysis_result = run_chart_analysis(config, data_fetcher)

        output_dir = str(get_analysis_results_dir())
        report_datetime = datetime.now()

        # Determine primary timeframe and chart path for reporting
        if is_multi_tf:
            primary_timeframe = timeframes_list[0] if timeframes_list else "1h"
            chart_path = None
        else:
            primary_timeframe = timeframe or "1h"
            chart_path = analysis_result.get("chart_path")

        save_and_open_reports(
            symbol=symbol,
            primary_timeframe=primary_timeframe,
            results_or_analysis=analysis_result,
            chart_path=chart_path,
            output_dir=output_dir,
            report_datetime=report_datetime,
            is_multi_tf=is_multi_tf,
            timeframes_list=timeframes_list,
            prompt_type=prompt_type,
            no_cleanup=no_cleanup,
        )

    except KeyboardInterrupt:
        log_warn("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        log_error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
