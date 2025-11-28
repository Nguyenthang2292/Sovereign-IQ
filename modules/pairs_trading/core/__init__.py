"""Core pairs trading analysis components."""

from modules.pairs_trading.core.pairs_analyzer import PairsTradingAnalyzer
from modules.pairs_trading.core.pair_metrics_computer import PairMetricsComputer
from modules.pairs_trading.core.opportunity_scorer import OpportunityScorer

__all__ = [
    'PairsTradingAnalyzer',
    'PairMetricsComputer',
    'OpportunityScorer',
]

