
from modules.pairs_trading.cli.display.display_pairs_opportunities import (

"""
Display formatters for pairs trading analysis.

This package provides formatted display functions for showing performance data
and pairs trading opportunities in user-friendly table formats.
"""

    display_pairs_opportunities,
)
from modules.pairs_trading.cli.display.display_performers import (
    display_performers,
)

__all__ = [
    "display_performers",
    "display_pairs_opportunities",
]
