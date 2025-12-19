"""
Position Sizing Calculator - Entry Point.

Calculate optimal position sizes using Bayesian Kelly Criterion and Regime Switching
based on results from main_hybrid.py or main_voting.py.

Example:
    Run from command line:
        $ python main_position_sizing.py --source hybrid --account-balance 10000
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
)
from modules.position_sizing.cli.main import main

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


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

