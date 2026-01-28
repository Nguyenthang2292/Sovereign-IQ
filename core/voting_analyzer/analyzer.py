"""
VotingAnalyzer core class with shared initialization.
"""

import threading
from typing import Dict, Optional

import pandas as pd

from config.spc import (
    SPC_AGGREGATION_ADAPTIVE_PERFORMANCE_WINDOW,
    SPC_AGGREGATION_ENABLE_ADAPTIVE_WEIGHTS,
    SPC_AGGREGATION_ENABLE_SIMPLE_FALLBACK,
    SPC_AGGREGATION_MIN_SIGNAL_STRENGTH,
    SPC_AGGREGATION_MODE,
    SPC_AGGREGATION_SIMPLE_MIN_ACCURACY_TOTAL,
    SPC_AGGREGATION_STRATEGY_WEIGHTS,
    SPC_AGGREGATION_THRESHOLD,
    SPC_AGGREGATION_WEIGHTED_MIN_DIFF,
    SPC_AGGREGATION_WEIGHTED_MIN_TOTAL,
)
from modules.common.core.data_fetcher import DataFetcher
from modules.common.utils import log_info
from modules.simplified_percentile_clustering.aggregation import SPCVoteAggregator
from modules.simplified_percentile_clustering.config import SPCAggregationConfig

from .params import VotingParamsMixin
from .signal_calculation import VotingSignalCalculationMixin
from .voting import VotingVotingMixin
from .workflow import VotingWorkflowMixin


class VotingAnalyzer(
    VotingParamsMixin,
    VotingSignalCalculationMixin,
    VotingVotingMixin,
    VotingWorkflowMixin,
):
    """
    ATC + Range Oscillator + SPC Pure Voting Analyzer.

    Option 2: Completely replace sequential filtering with a voting system.
    """

    def __init__(self, args, data_fetcher: DataFetcher, ohlcv_cache: Optional[Dict[str, pd.DataFrame]] = None):
        """Initialize analyzer."""
        self.args = args
        self.data_fetcher = data_fetcher
        self.ohlcv_cache = ohlcv_cache

        # Dynamic import of ATCAnalyzer based on configuration
        use_performance = getattr(args, "use_atc_performance", True)
        if use_performance:
            from modules.adaptive_trend_LTS.cli import ATCAnalyzer

            log_info("Using High-Performance ATC (LTS) module")
        else:
            from modules.adaptive_trend.cli import ATCAnalyzer

            log_info("Using Standard ATC (Legacy) module")

        self.atc_analyzer = ATCAnalyzer(args, data_fetcher, ohlcv_cache=ohlcv_cache)
        self.selected_timeframe = args.timeframe
        self.atc_analyzer.selected_timeframe = args.timeframe

        # Initialize SPC Vote Aggregator
        aggregation_config = SPCAggregationConfig(
            mode=SPC_AGGREGATION_MODE,
            threshold=SPC_AGGREGATION_THRESHOLD,
            weighted_min_total=SPC_AGGREGATION_WEIGHTED_MIN_TOTAL,
            weighted_min_diff=SPC_AGGREGATION_WEIGHTED_MIN_DIFF,
            enable_adaptive_weights=SPC_AGGREGATION_ENABLE_ADAPTIVE_WEIGHTS,
            adaptive_performance_window=SPC_AGGREGATION_ADAPTIVE_PERFORMANCE_WINDOW,
            min_signal_strength=SPC_AGGREGATION_MIN_SIGNAL_STRENGTH,
            enable_simple_fallback=SPC_AGGREGATION_ENABLE_SIMPLE_FALLBACK,
            simple_min_accuracy_total=SPC_AGGREGATION_SIMPLE_MIN_ACCURACY_TOTAL,
            strategy_weights=SPC_AGGREGATION_STRATEGY_WEIGHTS,
        )
        self.spc_aggregator = SPCVoteAggregator(aggregation_config)

        # Thread-safe lock for mode changes
        self._mode_lock = threading.Lock()

        # Results storage
        self.long_signals_atc = pd.DataFrame()
        self.short_signals_atc = pd.DataFrame()
        self.long_signals_final = pd.DataFrame()
        self.short_signals_final = pd.DataFrame()
