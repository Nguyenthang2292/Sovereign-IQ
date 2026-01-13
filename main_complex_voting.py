"""
ATC + Range Oscillator + SPC Pure Voting System (Phương án 2).

Entry point for the Voting Analyzer that combines signals from multiple indicators
using a pure voting system approach without sequential filtering.

Indicators:
    1. ATC (Adaptive Trend Classification): Market trend classification
    2. Range Oscillator: Overbought/oversold zone detection
    3. SPC (Simplified Percentile Clustering): Percentile-based clustering (3 strategies)
    4. XGBoost (optional): Machine learning prediction
    5. HMM (optional): Hidden Markov Model signal prediction
    6. Random Forest (optional): Random Forest model prediction

Workflow:
    1. Run ATC auto scan to find initial LONG/SHORT signals
    2. Calculate signals from all indicators in parallel for all symbols
    3. Apply voting system with weighted impact and cumulative vote
    4. Filter symbols based on voting results
    5. Display final results with voting metadata

Approach:
    Pure voting system. All indicators calculate signals simultaneously,
    then voting system decides which symbols to keep. No early filtering
    means all information is preserved for voting decision.

Key Features:
    - Parallel signal calculation for all indicators
    - All symbols evaluated with full indicator information
    - Voting system considers all indicators simultaneously
    - More flexible, not dependent on filtering order
    - Higher resource usage but potentially better accuracy

See Also:
    - core.voting_analyzer.VotingAnalyzer: Main analyzer class
    - core.README.md: Detailed workflow comparison with HybridAnalyzer
    - main_hybrid.py: Alternative sequential filtering approach (Phương án 1)

Example:
    Run from command line:
        $ python main_voting.py --timeframe 1h --enable-spc
"""

import os
import sys
import warnings

from modules.common.utils import configure_windows_stdio

configure_windows_stdio()

from colorama import Fore
from colorama import init as colorama_init

from cli.argument_parser import parse_args
from core.voting_analyzer import VotingAnalyzer
from modules.common.utils import (
    color_text,
    initialize_components,
    log_error,
)

# Configure warning filters with targeted approach
# Environment variable ENABLE_DEPRECATION_WARNINGS can be set to "1" or "true" to show all warnings
# Useful for CI/CD pipelines or periodic audits to detect breaking changes early
ENABLE_ALL_WARNINGS = os.getenv("ENABLE_DEPRECATION_WARNINGS", "").lower() in ("1", "true", "yes")

if not ENABLE_ALL_WARNINGS:
    # Targeted suppression: Only filter specific noisy warnings from data science libraries
    # This approach allows important warnings to surface while reducing noise

    # Pandas-specific deprecation warnings (common in data processing)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
    # Pandas FutureWarning about upcoming behavior changes
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

    # NumPy deprecation warnings (common in numerical operations)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

    # scikit-learn deprecation warnings (common in ML workflows)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

    # XGBoost deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="xgboost")

    # Common pandas FutureWarning patterns that don't affect current functionality
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*DataFrame.*append.*",  # DataFrame.append is deprecated
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*Series.*append.*",  # Series.append is deprecated
    )
else:
    # All warnings enabled - useful for periodic audits
    # Run with: ENABLE_DEPRECATION_WARNINGS=1 python main_complex_voting.py
    warnings.filterwarnings("default")  # Reset to default behavior

# NOTE: Periodic Audit Schedule
# It's recommended to periodically run with ENABLE_DEPRECATION_WARNINGS=1 to:
# 1. Detect upcoming breaking changes in dependencies
# 2. Plan migration paths before dependencies are updated
# 3. Identify code that needs refactoring
# Suggested schedule: Monthly or before major dependency updates
# Example: ENABLE_DEPRECATION_WARNINGS=1 python main_complex_voting.py --timeframe 1h

colorama_init(autoreset=True)


def main() -> None:
    """
    Main entry point for Voting Analyzer workflow.

    Initializes components, creates VotingAnalyzer instance, and runs the
    complete pure voting system workflow.

    Workflow steps:
        1. Parse command-line arguments (voting mode)
        2. Initialize ExchangeManager and DataFetcher
        3. Create VotingAnalyzer with parsed args
        4. Execute analyzer.run() to perform:
           - ATC scan
           - Parallel calculation of all indicator signals
           - Voting system application
           - Results display with voting metadata

    Note:
        Decision Matrix is always enabled in voting mode (it's the core mechanism).
        SPC can be enabled/disabled via command-line arguments.

    Raises:
        SystemExit: On KeyboardInterrupt or unhandled exceptions
    """
    args = parse_args(mode="voting", force_enable_spc=False, force_enable_decision_matrix=True)
    _, data_fetcher = initialize_components()
    analyzer = VotingAnalyzer(args, data_fetcher)
    analyzer.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\nExiting program by user request.", Fore.YELLOW))
        sys.exit(0)
    except Exception as e:
        log_error(f"Error: {type(e).__name__}: {e}")
        import traceback

        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
