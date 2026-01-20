"""Service for individual chart analysis operations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from modules.common.core.data_fetcher import DataFetcher
from modules.common.ui.logging import log_error, log_info
from modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer import GeminiChartAnalyzer
from modules.gemini_chart_analyzer.core.analyzers.multi_timeframe_coordinator import MultiTimeframeCoordinator
from modules.gemini_chart_analyzer.core.exceptions import (
    ChartGenerationError,
    DataFetchError,
    GeminiAnalysisError,
    ReportGenerationError,
    ScanConfigurationError,
)
from modules.gemini_chart_analyzer.core.generators.chart_generator import ChartGenerator
from modules.gemini_chart_analyzer.core.reporting.html_report_generator import generate_html_report
from modules.gemini_chart_analyzer.core.utils.chart_paths import get_analysis_results_dir


@dataclass
class SingleAnalysisConfig:
    """Configuration for individual chart analysis."""

    symbol: str
    timeframe: Optional[str] = "1h"
    timeframes_list: Optional[List[str]] = None
    indicators: Optional[Dict[str, Any]] = None
    prompt_type: str = "general"
    custom_prompt: Optional[str] = None
    limit: int = 500
    chart_figsize: Tuple[int, int] = (16, 10)
    chart_dpi: int = 150
    output_dir: Optional[str] = None


def run_chart_analysis(config: SingleAnalysisConfig, data_fetcher: DataFetcher) -> Dict[str, Any]:
    """
    Execute single or multi-timeframe chart analysis.

    Args:
        config: SingleAnalysisConfig object
        data_fetcher: Initialized DataFetcher instance

    Returns:
        Dictionary containing analysis results and metadata
    """
    try:
        is_multi_tf = config.timeframes_list is not None and len(config.timeframes_list) > 0
        report_datetime = datetime.now()
        output_dir = config.output_dir or str(get_analysis_results_dir())

        chart_gen = ChartGenerator(figsize=config.chart_figsize, style="dark_background", dpi=config.chart_dpi)
        gemini_analyzer = GeminiChartAnalyzer()

        if is_multi_tf:
            log_info(f"Running multi-timeframe analysis for {config.symbol}...")
            mtf_coordinator = MultiTimeframeCoordinator()

            def fetch_data_func(sym, tf):
                df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                    symbol=sym, timeframe=tf, limit=config.limit, check_freshness=False
                )
                return df

            def generate_chart_func(df, sym, tf):
                return chart_gen.create_chart(
                    df=df, symbol=sym, timeframe=tf, indicators=config.indicators, show_volume=True
                )

            def analyze_chart_func(path, sym, tf):
                return gemini_analyzer.analyze_chart(
                    image_path=path,
                    symbol=sym,
                    timeframe=tf,
                    prompt_type=config.prompt_type,
                    custom_prompt=config.custom_prompt,
                )

            results = mtf_coordinator.analyze_deep(
                symbol=config.symbol,
                timeframes=config.timeframes_list,
                fetch_data_func=fetch_data_func,
                generate_chart_func=generate_chart_func,
                analyze_chart_func=analyze_chart_func,
            )

            html_path = generate_html_report(
                analysis_data=results,
                output_dir=output_dir,
                report_type="multi",
                timeframes_list=config.timeframes_list,
                report_datetime=report_datetime,
            )
            results["html_report_path"] = html_path
            return results

        else:
            log_info(f"Running single timeframe analysis for {config.symbol} ({config.timeframe})...")

            df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol=config.symbol, timeframe=config.timeframe, limit=config.limit, check_freshness=False
            )

            if df is None or df.empty:
                raise DataFetchError(f"No data found for {config.symbol} on {config.timeframe}")

            chart_path = chart_gen.create_chart(
                df=df, symbol=config.symbol, timeframe=config.timeframe, indicators=config.indicators, show_volume=True
            )

            analysis_text = gemini_analyzer.analyze_chart(
                image_path=chart_path,
                symbol=config.symbol,
                timeframe=config.timeframe,
                prompt_type=config.prompt_type,
                custom_prompt=config.custom_prompt,
            )

            analysis_data = {
                "symbol": config.symbol,
                "timeframe": config.timeframe,
                "analysis": analysis_text,
                "chart_path": chart_path,
                "exchange": exchange_id,
            }

            html_path = generate_html_report(
                analysis_data=analysis_data,
                output_dir=output_dir,
                report_type="single",
                chart_path=chart_path,
                report_datetime=report_datetime,
            )
            analysis_data["html_report_path"] = html_path
            return analysis_data

    except ScanConfigurationError as e:
        log_error(f"Invalid configuration: {e}")
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
        log_error(f"Report generation failed: {e}")
        raise
    except Exception as e:
        log_error(f"Unexpected error in chart analysis service: {e}")
        raise
