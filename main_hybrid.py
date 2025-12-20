"""
ATC + Range Oscillator + SPC Hybrid Approach (Phương án 1).

Entry point for the Hybrid Analyzer that combines signals from multiple indicators
using a sequential filtering approach followed by optional voting system.

Indicators:
    1. ATC (Adaptive Trend Classification): Market trend classification
    2. Range Oscillator: Overbought/oversold zone detection
    3. SPC (Simplified Percentile Clustering): Percentile-based clustering (3 strategies)
    4. XGBoost (optional): Machine learning prediction
    5. HMM (optional): Hidden Markov Model signal prediction
    6. Random Forest (optional): Random Forest model prediction

Workflow:
    1. Run ATC auto scan to find initial LONG/SHORT signals
    2. Filter by Range Oscillator confirmation (sequential filtering)
    3. Calculate SPC signals for remaining symbols (if enabled)
    4. Apply Decision Matrix voting system (if enabled)
    5. Display final filtered results

Approach:
    Sequential filtering + voting system. Symbols are filtered step-by-step
    to reduce workload before applying voting. Range Oscillator acts as an
    early filter to remove false positives.

Key Features:
    - Early filtering reduces computational load
    - Fallback mechanism when no symbols pass Range Oscillator filter
    - Optional SPC and Decision Matrix for additional filtering
    - Sequential approach is easier to debug and monitor

See Also:
    - core.hybrid_analyzer.HybridAnalyzer: Main analyzer class
    - core.README.md: Detailed workflow comparison with VotingAnalyzer
    - main_voting.py: Alternative pure voting approach (Phương án 2)

Example:
    Run from command line:
        $ python main_hybrid.py --timeframe 1h --enable-spc --use-decision-matrix
"""

import warnings
import sys

from modules.common.utils import configure_windows_stdio

# Fix encoding issues on Windows for interactive CLI runs only
configure_windows_stdio()

from colorama import Fore, init as colorama_init

from modules.common.utils import (
    color_text,
    log_error,
    log_progress,
)
from cli.argument_parser import parse_args
from core.hybrid_analyzer import HybridAnalyzer
from modules.common.utils import initialize_components

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


def main() -> None:
    """
    Main entry point for Hybrid Analyzer workflow.
    
    Initializes components, creates HybridAnalyzer instance, and runs the
    complete sequential filtering + voting workflow.
    
    Workflow steps:
        1. Parse command-line arguments
        2. Initialize ExchangeManager and DataFetcher
        3. Create HybridAnalyzer with parsed args
        4. Execute analyzer.run() to perform:
           - ATC scan
           - Range Oscillator filtering
           - SPC signal calculation (if enabled)
           - Decision Matrix voting (if enabled)
           - Results display
    
    Raises:
        SystemExit: On KeyboardInterrupt or unhandled exceptions
    """
    args = parse_args()
    exchange_manager, data_fetcher = initialize_components()
    analyzer = HybridAnalyzer(args, data_fetcher)
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

