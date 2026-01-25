from pathlib import Path

"""
Chart path utilities for gemini chart analyzer.

Provides functions for determining chart directory paths.
"""


def get_charts_dir() -> Path:
    """
    Get the charts directory path relative to module root.

    Creates the charts directory if it does not exist.

    Returns:
        Path object pointing to the charts directory
    """
    module_root = Path(__file__).resolve().parent.parent.parent
    # Go up one more level to reach project root (crypto-probability)
    project_root = module_root.parent.parent
    charts_dir = project_root / "outputs" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    return charts_dir


def get_analysis_results_dir() -> Path:
    """
    Get the analysis results directory path relative to project root.

    Creates the analysis results directory if it does not exist.

    Returns:
        Path object pointing to the analysis_results directory
    """
    module_root = Path(__file__).resolve().parent.parent.parent
    # Go up one more level to reach project root (crypto-probability)
    project_root = module_root.parent.parent
    results_dir = project_root / "outputs" / "analysis_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
