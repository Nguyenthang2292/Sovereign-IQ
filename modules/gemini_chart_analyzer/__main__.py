"""
Allow running as: python -m modules.gemini_chart_analyzer

Provides an interactive menu or accepts command line arguments to
launch either the Single Chart Analyzer or the Batch Scanner.
"""

import sys

from colorama import Fore, init


def run_single_analyzer():
    """Runs the single chart analyzer."""
    from modules.gemini_chart_analyzer.cli.chart_analyzer_main import main

    main()


def run_batch_scanner():
    """Runs the batch market scanner."""
    # We load the main wrapper which handles Rust module checks and environment setup.
    # It runs a lot of top-level code on import, so we import it lazily.
    import main_gemini_chart_batch_scanner
    from modules.gemini_chart_analyzer.cli.batch_scanner.main import main

    # Run the model cleanup similar to the script
    main_gemini_chart_batch_scanner._cleanup_old_models(main_gemini_chart_batch_scanner.MODEL_CLEANUP_DIR, keep_count=5)

    log_file = main_gemini_chart_batch_scanner._build_log_file_path()
    with main_gemini_chart_batch_scanner._tee_output(log_file):
        main()
        print(f"\nLog saved to: {log_file}")


def main():
    init(autoreset=True)

    # If the user passed arguments indicating the mode directly
    if len(sys.argv) > 1:
        mode_arg = sys.argv[1].lower()
        if mode_arg in ("--batch", "batch", "scanner"):
            sys.argv.pop(1)  # Remove the mode arg so the downstream parser doesn't break
            run_batch_scanner()
            return
        elif mode_arg in ("--single", "single", "analyzer"):
            sys.argv.pop(1)
            run_single_analyzer()
            return
        # If they passed CLI flags like --symbol, default to single analyzer
        elif mode_arg.startswith("-"):
            run_single_analyzer()
            return

    # No arguments provided, show an interactive menu
    print("\n" + Fore.CYAN + "=" * 60)
    print(Fore.CYAN + "   GEMINI CHART ANALYZER - SELECT MODE")
    print(Fore.CYAN + "=" * 60)
    print("1. " + Fore.YELLOW + "Single Chart Analyzer" + Fore.RESET + " (Analyze 1 symbol with multiple TFs)")
    print("2. " + Fore.YELLOW + "Batch Market Scanner" + Fore.RESET + "  (Scan hundreds of symbols at once)")
    print(Fore.CYAN + "=" * 60 + "\n")

    try:
        choice = input("Select mode (1 or 2) [default=1]: ").strip()

        if choice == "2":
            # Clear sys.argv so batch scanner's default menu is invoked correctly
            sys.argv = [sys.argv[0]]
            run_batch_scanner()
        else:
            # Default to single
            sys.argv = [sys.argv[0]]
            run_single_analyzer()

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(0)


if __name__ == "__main__":
    main()
