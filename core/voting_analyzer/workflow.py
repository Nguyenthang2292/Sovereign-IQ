"""Workflow helpers for VotingAnalyzer."""

from colorama import Fore, Style
import pandas as pd

from modules.adaptive_trend_LTS.cli import prompt_timeframe
from modules.common.utils import (
    color_text,
    log_progress,
    log_success,
    log_warn,
)
from modules.range_oscillator.cli import display_final_results


class VotingWorkflowMixin:
    """Mixin for workflow orchestration, display, and runtime steps."""

    def determine_timeframe(self) -> str:
        """Determine timeframe from arguments and interactive menu."""
        self.selected_timeframe = self.args.timeframe

        if not self.args.no_menu:
            print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
            print(color_text("TIMEFRAME SELECTION", Fore.CYAN, Style.BRIGHT))
            print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
            self.selected_timeframe = prompt_timeframe(default_timeframe=self.selected_timeframe)
            print(color_text(f"\nSelected timeframe: {self.selected_timeframe}", Fore.GREEN))

        self.atc_analyzer.selected_timeframe = self.selected_timeframe
        return self.selected_timeframe

    def display_config(self) -> None:
        """Display configuration information."""
        # Lazy import to avoid circular dependency
        from cli.display import display_config

        display_config(
            selected_timeframe=self.selected_timeframe,
            args=self.args,
            get_oscillator_params=self.get_oscillator_params,
            get_spc_params=self.get_spc_params,
            mode="voting",
        )

    def run_atc_scan(self) -> bool:
        """
        Run ATC auto scan to get LONG/SHORT signals.

        Returns:
            True if signals found, False otherwise
        """
        log_progress("\nStep 1: Running ATC auto scan...")
        log_progress("=" * 80)

        self.long_signals_atc, self.short_signals_atc = self.atc_analyzer.run_auto_scan()

        original_long_count = len(self.long_signals_atc)
        original_short_count = len(self.short_signals_atc)

        log_success(f"\nATC Scan Complete: Found {original_long_count} LONG + {original_short_count} SHORT signals")

        if self.long_signals_atc.empty and self.short_signals_atc.empty:
            log_warn("No ATC signals found. Cannot proceed with analysis.")
            log_warn("Please try:")
            log_warn("  - Different timeframe")
            log_warn("  - Different market conditions")
            log_warn("  - Check ATC configuration parameters")
            return False

        return True

    def calculate_and_vote(self) -> None:
        """
        Calculate signals from all indicators and apply voting system.

        This is the main step for Phương án 2.
        """
        log_progress("\nStep 2: Calculating signals from all indicators...")
        log_progress("=" * 80)

        # Calculate all signals in parallel
        if not self.long_signals_atc.empty:
            long_with_signals = self.calculate_signals_for_all_indicators(
                atc_signals_df=self.long_signals_atc,
                signal_type="LONG",
            )

            # Apply voting system
            log_progress("\nStep 3: Applying voting system to LONG signals...")
            self.long_signals_final = self.apply_voting_system(long_with_signals, "LONG")
            log_progress(f"LONG signals: {len(self.long_signals_atc)} → {len(self.long_signals_final)} after voting")
        else:
            self.long_signals_final = pd.DataFrame()

        if not self.short_signals_atc.empty:
            short_with_signals = self.calculate_signals_for_all_indicators(
                atc_signals_df=self.short_signals_atc,
                signal_type="SHORT",
            )

            # Apply voting system
            log_progress("\nStep 3: Applying voting system to SHORT signals...")
            self.short_signals_final = self.apply_voting_system(short_with_signals, "SHORT")
            log_progress(f"SHORT signals: {len(self.short_signals_atc)} → {len(self.short_signals_final)} after voting")
        else:
            self.short_signals_final = pd.DataFrame()

    def display_results(self) -> None:
        """Display final results with voting metadata."""
        log_progress("\nStep 4: Displaying final results...")
        display_final_results(
            long_signals=self.long_signals_final,
            short_signals=self.short_signals_final,
            original_long_count=len(self.long_signals_atc),
            original_short_count=len(self.short_signals_atc),
            long_uses_fallback=False,
            short_uses_fallback=False,
        )

        # Display voting metadata
        if not self.long_signals_final.empty:
            self._display_voting_metadata(self.long_signals_final, "LONG")

        if not self.short_signals_final.empty:
            self._display_voting_metadata(self.short_signals_final, "SHORT")

    def _display_voting_metadata(self, signals_df, signal_type: str) -> None:
        """Display voting metadata for signals."""
        # Lazy import to avoid circular dependency
        from cli.display import display_voting_metadata

        display_voting_metadata(
            signals_df=signals_df,
            signal_type=signal_type,
            show_spc_debug=False,  # No debug info for voting mode
        )

    def run(self) -> None:
        """
        Run the complete Pure Voting System workflow.

        Workflow:
        1. Determine timeframe
        2. Display configuration
        3. Run ATC auto scan
        4. Calculate signals from all indicators in parallel
        5. Apply voting system
        6. Display final results
        """
        self.determine_timeframe()
        self.display_config()
        log_progress("Initializing components...")

        # Run ATC scan - exit early if no signals found
        if not self.run_atc_scan():
            log_warn("\nAnalysis terminated: No ATC signals found.")
            return

        self.calculate_and_vote()
        self.display_results()

        log_success("\nAnalysis complete!")
