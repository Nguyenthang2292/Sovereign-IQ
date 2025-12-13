"""
ATC + Range Oscillator + SPC Pure Voting System (Phương án 2).

Entry point for the Voting Analyzer that combines signals from multiple indicators
using a pure voting system approach without sequential filtering.

Indicators:
    1. ATC (Adaptive Trend Classification): Market trend classification
    2. Range Oscillator: Overbought/oversold zone detection
    3. SPC (Simplified Percentile Clustering): Percentile-based clustering (3 strategies)
    4. XGBoost (optional): Machine learning prediction

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

import warnings
import sys

from modules.common.utils import configure_windows_stdio

configure_windows_stdio()

from colorama import Fore, init as colorama_init

from modules.common.utils import (
    color_text,
    log_error,
    log_progress,
)
from cli.argument_parser import parse_args
from core.voting_analyzer import VotingAnalyzer
from modules.common.utils import initialize_components

warnings.filterwarnings("ignore")
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
    args = parse_args(mode="voting", force_enable_spc=False, force_enable_decision_matrix=False)
    exchange_manager, data_fetcher = initialize_components()
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

