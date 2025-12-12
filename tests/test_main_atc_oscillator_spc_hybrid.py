"""
Tests for main_atc_oscillator_spc_hybrid.py (Hybrid Approach).
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add parent directory to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_atc_oscillator_spc_hybrid import (
    ATCOscillatorSPCHybridAnalyzer,
    get_range_oscillator_signal,
    get_spc_signal,
)
from modules.common.DataFetcher import DataFetcher
from modules.config import (
    SPC_AGGREGATION_MODE,
    SPC_AGGREGATION_THRESHOLD,
    SPC_AGGREGATION_WEIGHTED_MIN_TOTAL,
    SPC_AGGREGATION_WEIGHTED_MIN_DIFF,
    SPC_AGGREGATION_ENABLE_ADAPTIVE_WEIGHTS,
    SPC_AGGREGATION_ADAPTIVE_PERFORMANCE_WINDOW,
    SPC_AGGREGATION_MIN_SIGNAL_STRENGTH,
    SPC_AGGREGATION_STRATEGY_WEIGHTS,
)


class TestATCOscillatorSPCHybridAnalyzer:
    """Test suite for ATCOscillatorSPCHybridAnalyzer."""
    
    @pytest.fixture
    def mock_args(self):
        """Create mock args object."""
        args = Mock()
        args.timeframe = "1h"
        args.no_menu = True
        args.limit = 500
        args.max_workers = 10
        args.osc_length = 50
        args.osc_mult = 2.0
        args.osc_strategies = None
        args.enable_spc = True
        args.use_decision_matrix = True
        args.spc_k = 2
        args.spc_lookback = 1000
        args.spc_p_low = 5.0
        args.spc_p_high = 95.0
        args.voting_threshold = 0.5
        args.min_votes = 2
        args.min_signal = 0.01
        args.max_symbols = None
        return args
    
    @pytest.fixture
    def mock_data_fetcher(self):
        """Create mock DataFetcher."""
        return Mock(spec=DataFetcher)
    
    @pytest.fixture
    def analyzer(self, mock_args, mock_data_fetcher):
        """Create analyzer instance."""
        with patch('main_atc_oscillator_spc_hybrid.ATCAnalyzer'):
            analyzer = ATCOscillatorSPCHybridAnalyzer(mock_args, mock_data_fetcher)
            return analyzer
    
    def test_init(self, analyzer, mock_args):
        """Test analyzer initialization."""
        assert analyzer.args == mock_args
        assert analyzer.selected_timeframe == mock_args.timeframe
        assert analyzer.spc_aggregator is not None
        assert analyzer.long_signals_atc.empty
        assert analyzer.short_signals_atc.empty
        assert analyzer.long_uses_fallback is False
        assert analyzer.short_uses_fallback is False
    
    def test_get_oscillator_params(self, analyzer, mock_args):
        """Test get_oscillator_params."""
        params = analyzer.get_oscillator_params()
        
        assert params['osc_length'] == mock_args.osc_length
        assert params['osc_mult'] == mock_args.osc_mult
        assert params['max_workers'] == mock_args.max_workers
    
    def test_get_spc_params(self, analyzer, mock_args):
        """Test get_spc_params."""
        params = analyzer.get_spc_params()
        
        assert params['k'] == mock_args.spc_k
        assert params['lookback'] == mock_args.spc_lookback
        assert 'cluster_transition_params' in params
        assert 'regime_following_params' in params
        assert 'mean_reversion_params' in params
    
    def test_aggregate_spc_votes(self, analyzer):
        """Test _aggregate_spc_votes."""
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': 1,
            'spc_mean_reversion_strength': 0.6,
        }
        
        vote, strength = analyzer._aggregate_spc_votes(symbol_data, "LONG")
        
        assert vote in [0, 1]
        assert 0.0 <= strength <= 1.0
    
    def test_calculate_indicator_votes(self, analyzer):
        """Test calculate_indicator_votes."""
        symbol_data = {
            'signal': 1,  # ATC signal
            'osc_signal': 1,
            'osc_confidence': 0.8,
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.8,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.7,
            'spc_mean_reversion_signal': 1,
            'spc_mean_reversion_strength': 0.6,
        }
        
        votes = analyzer.calculate_indicator_votes(symbol_data, "LONG")
        
        assert 'atc' in votes
        assert 'oscillator' in votes
        assert 'spc' in votes
        assert all(isinstance(v, tuple) and len(v) == 2 for v in votes.values())
    
    def test_apply_decision_matrix_all_agree(self, analyzer):
        """Test apply_decision_matrix when all indicators agree."""
        signals_df = pd.DataFrame([
            {
                'symbol': 'BTC/USDT',
                'signal': 1,
                'trend': 'UPTREND',
                'price': 50000.0,
                'exchange': 'binance',
                'osc_signal': 1,
                'osc_confidence': 0.8,
                'spc_cluster_transition_signal': 1,
                'spc_cluster_transition_strength': 0.8,
                'spc_regime_following_signal': 1,
                'spc_regime_following_strength': 0.7,
                'spc_mean_reversion_signal': 1,
                'spc_mean_reversion_strength': 0.6,
            }
        ])
        
        result = analyzer.apply_decision_matrix(signals_df, "LONG")
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert 'cumulative_vote' in result.columns
            assert 'weighted_score' in result.columns
    
    def test_apply_decision_matrix_empty_input(self, analyzer):
        """Test apply_decision_matrix with empty DataFrame."""
        signals_df = pd.DataFrame()
        
        result = analyzer.apply_decision_matrix(signals_df, "LONG")
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_get_indicator_accuracy(self, analyzer):
        """Test _get_indicator_accuracy."""
        atc_accuracy = analyzer._get_indicator_accuracy('atc', 'LONG')
        osc_accuracy = analyzer._get_indicator_accuracy('oscillator', 'LONG')
        spc_accuracy = analyzer._get_indicator_accuracy('spc', 'LONG')
        
        assert 0.0 <= atc_accuracy <= 1.0
        assert 0.0 <= osc_accuracy <= 1.0
        assert 0.0 <= spc_accuracy <= 1.0


class TestHelperFunctionsHybrid:
    """Test suite for helper functions in hybrid approach."""
    
    @pytest.fixture
    def mock_data_fetcher(self):
        """Create mock DataFetcher."""
        fetcher = Mock(spec=DataFetcher)
        return fetcher
    
    def test_get_range_oscillator_signal_success(self, mock_data_fetcher):
        """Test get_range_oscillator_signal with successful result."""
        mock_df = pd.DataFrame({
            'high': [100, 101, 102],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
        })
        
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, 'binance')
        
        with patch('main_atc_oscillator_spc_hybrid.generate_signals_combined_all_strategy') as mock_gen:
            mock_gen.return_value = (
                pd.Series([0, 0, 1], name='signal'),
                pd.Series([0.0, 0.0, 0.8], name='strength'),
                None,
                pd.Series([0.0, 0.0, 0.75], name='confidence'),
            )
            
            result = get_range_oscillator_signal(
                mock_data_fetcher, 'BTC/USDT', '1h', 500
            )
            
            assert result is not None
            assert isinstance(result, tuple)
            assert len(result) == 2
    
    def test_get_spc_signal_success(self, mock_data_fetcher):
        """Test get_spc_signal with successful result."""
        mock_df = pd.DataFrame({
            'high': [100, 101, 102],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
        })
        
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, 'binance')
        
        with patch('main_atc_oscillator_spc_hybrid.SimplifiedPercentileClustering') as mock_spc, \
             patch('main_atc_oscillator_spc_hybrid.generate_signals_cluster_transition') as mock_gen:
            
            mock_clustering_result = Mock()
            mock_spc.return_value.compute.return_value = mock_clustering_result
            
            mock_gen.return_value = (
                pd.Series([0, 0, 1], name='signal'),
                pd.Series([0.0, 0.0, 0.8], name='strength'),
                {},
            )
            
            result = get_spc_signal(
                mock_data_fetcher, 'BTC/USDT', '1h', 500, 'cluster_transition'
            )
            
            assert result is not None
            assert isinstance(result, tuple)
            assert len(result) == 2

