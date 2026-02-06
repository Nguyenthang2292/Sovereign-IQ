"""
Mode Management for ATC Analysis.

This module handles mode determination and timeframe selection
for ATC analysis workflows.
"""

from argparse import Namespace
from typing import Tuple

from modules.adaptive_trend_LTS_mini.cli.input_utils import determine_mode_and_timeframe

__all__ = ["ModeManager"]


class ModeManager:
    """
    Manages execution mode and timeframe for ATC analysis.

    Handles mode determination (auto/manual) and timeframe selection
    based on command-line arguments and user interaction.
    """

    def __init__(self, args: Namespace):
        """
        Initialize ModeManager.

        Args:
            args: Command-line arguments namespace
        """
        self.args = args
        self.mode: str = "manual"
        self.timeframe: str = args.timeframe

    def determine_mode_and_timeframe(self) -> Tuple[str, str]:
        """
        Determine execution mode and timeframe.

        Returns:
            Tuple of (mode, timeframe) strings
        """
        self.mode, self.timeframe = determine_mode_and_timeframe(self.args)
        return self.mode, self.timeframe

    def get_mode(self) -> str:
        """
        Get current execution mode.

        Returns:
            Current mode ("auto" or "manual")
        """
        return self.mode

    def get_timeframe(self) -> str:
        """
        Get current timeframe.

        Returns:
            Current timeframe string (e.g., "1h", "4h")
        """
        return self.timeframe

    def set_timeframe(self, timeframe: str) -> None:
        """
        Set timeframe.

        Args:
            timeframe: New timeframe string
        """
        self.timeframe = timeframe
