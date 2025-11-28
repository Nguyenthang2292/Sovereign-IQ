"""
Display formatters for pairs trading analysis.

This package provides formatted display functions for showing performance data
and pairs trading opportunities in user-friendly table formats.
"""

from modules.pairs_trading.cli.formatters.performance_formatter import (
    display_performers,
)
from modules.pairs_trading.cli.formatters.pairs_formatter import (
    display_pairs_opportunities,
)

__all__ = [
    'display_performers',
    'display_pairs_opportunities',
]
