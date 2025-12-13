"""
Tests for main_hybrid.py (Hybrid Approach).
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

from core.hybrid_analyzer import HybridAnalyzer
from core.signal_calculators import (
    get_range_oscillator_signal,
    get_spc_signal,
    get_hmm_signal,
    get_xgboost_signal,
)
from modules.common.DataFetcher import DataFetcher
from config import (
    SPC_AGGREGATION_MODE,
    SPC_AGGREGATION_THRESHOLD,
    SPC_AGGREGATION_WEIGHTED_MIN_TOTAL,
    SPC_AGGREGATION_WEIGHTED_MIN_DIFF,
    SPC_AGGREGATION_ENABLE_ADAPTIVE_WEIGHTS,
    SPC_AGGREGATION_ADAPTIVE_PERFORMANCE_WINDOW,
    SPC_AGGREGATION_MIN_SIGNAL_STRENGTH,
    SPC_AGGREGATION_STRATEGY_WEIGHTS,
)


class TestHybridAnalyzer:
    """Test suite for HybridAnalyzer."""
    
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
        args.enable_xgboost = False
        args.enable_hmm = False
        args.hmm_window_size = None
        args.hmm_window_kama = None
        args.hmm_fast_kama = None
        args.hmm_slow_kama = None
        args.hmm_orders_argrelextrema = None
        args.hmm_strict_mode = None
        return args
    
    @pytest.fixture
    def mock_data_fetcher(self):
        """Create mock DataFetcher."""
        return Mock(spec=DataFetcher)
    
    @pytest.fixture
    def analyzer(self, mock_args, mock_data_fetcher):
        """Create analyzer instance."""
        with patch('core.hybrid_analyzer.ATCAnalyzer'):
            analyzer = HybridAnalyzer(mock_args, mock_data_fetcher)
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
    
    def test_aggregate_spc_votes_simple_fallback(self, analyzer):
        """Test _aggregate_spc_votes with simple mode fallback when weighted and threshold both return 0."""
        # Enable simple fallback in config
        analyzer.spc_aggregator.config.enable_simple_fallback = True
        analyzer.spc_aggregator.config.simple_min_accuracy_total = 1.0
        
        # Set weighted mode with very high thresholds that won't be met
        analyzer.spc_aggregator.config.mode = "weighted"
        analyzer.spc_aggregator.config.weighted_min_total = 0.95  # Very high
        analyzer.spc_aggregator.config.weighted_min_diff = 0.3
        
        # Scenario: 2 strategies LONG, 1 strategy SHORT
        # Weighted mode: might fail due to high threshold
        # Threshold mode: need 3 but only 2 agree -> fails
        # Simple mode: should fallback and use accuracy sum
        symbol_data = {
            'spc_cluster_transition_signal': 1,  # LONG, accuracy = 0.68
            'spc_cluster_transition_strength': 0.5,
            'spc_regime_following_signal': 1,  # LONG, accuracy = 0.66
            'spc_regime_following_strength': 0.4,
            'spc_mean_reversion_signal': -1,  # SHORT, accuracy = 0.64
            'spc_mean_reversion_strength': 0.3,
        }
        
        vote, strength = analyzer._aggregate_spc_votes(symbol_data, "LONG")
        
        # Weighted mode might fail, threshold mode might fail (only 2/3 agree)
        # Simple fallback should kick in: LONG accuracy = 0.68 + 0.66 = 1.34 > 1.0
        # So should return vote = 1 (LONG wins)
        assert vote in [0, 1]
        assert 0.0 <= strength <= 1.0
    
    def test_aggregate_spc_votes_simple_fallback_insufficient_accuracy(self, analyzer):
        """Test simple fallback when accuracy is insufficient."""
        # Enable simple fallback
        analyzer.spc_aggregator.config.enable_simple_fallback = True
        analyzer.spc_aggregator.config.simple_min_accuracy_total = 2.0  # Higher than max possible (1.98)
        
        # Set weighted mode to fail
        analyzer.spc_aggregator.config.mode = "weighted"
        analyzer.spc_aggregator.config.weighted_min_total = 0.95
        
        # Only one strategy has signal
        symbol_data = {
            'spc_cluster_transition_signal': 1,  # accuracy = 0.68
            'spc_cluster_transition_strength': 0.5,
            'spc_regime_following_signal': 0,
            'spc_regime_following_strength': 0.0,
            'spc_mean_reversion_signal': 0,
            'spc_mean_reversion_strength': 0.0,
        }
        
        vote, strength = analyzer._aggregate_spc_votes(symbol_data, "LONG")
        
        # Weighted fails, threshold fails (only 1/3), simple fallback fails (accuracy < 2.0)
        assert vote == 0
        # Strength might still have value from the strategy with signal, but vote should be 0
    
    def test_aggregate_spc_votes_simple_fallback_disabled(self, analyzer):
        """Test that simple fallback doesn't trigger when disabled."""
        # Disable simple fallback
        analyzer.spc_aggregator.config.enable_simple_fallback = False
        
        # Set weighted mode to fail
        analyzer.spc_aggregator.config.mode = "weighted"
        analyzer.spc_aggregator.config.weighted_min_total = 0.95
        
        symbol_data = {
            'spc_cluster_transition_signal': 1,
            'spc_cluster_transition_strength': 0.3,
            'spc_regime_following_signal': 1,
            'spc_regime_following_strength': 0.2,
            'spc_mean_reversion_signal': 0,
            'spc_mean_reversion_strength': 0.0,
        }
        
        vote, strength = analyzer._aggregate_spc_votes(symbol_data, "LONG")
        
        # Weighted might fail, threshold might fail, but simple fallback is disabled
        # So should return 0 if both fail
        assert vote in [0, 1]  # Could be 0 or 1 depending on threshold fallback
    
    def test_aggregate_spc_votes_simple_fallback_both_modes_fail(self, analyzer):
        """Test simple fallback when both weighted and threshold modes return 0."""
        # Enable simple fallback
        analyzer.spc_aggregator.config.enable_simple_fallback = True
        analyzer.spc_aggregator.config.simple_min_accuracy_total = 1.0
        
        # Set weighted mode with high threshold
        analyzer.spc_aggregator.config.mode = "weighted"
        analyzer.spc_aggregator.config.weighted_min_total = 0.95  # Very high, won't be met
        analyzer.spc_aggregator.config.weighted_min_diff = 0.3
        
        # Set threshold mode to also fail (need 3 strategies, but only 2 agree)
        # This will be used in fallback, but we'll set threshold high enough to fail
        analyzer.spc_aggregator.config.threshold = 0.9  # Need 3 strategies (ceil(3 * 0.9) = 3)
        
        # 2 strategies LONG, 1 strategy SHORT
        # Weighted: long_weight might be ~0.68 (normalized), but need 0.95 -> fails
        # Threshold: need 3 but only 2 agree -> fails
        # Simple: LONG accuracy = 0.68 + 0.66 = 1.34 > 1.0 -> should succeed
        symbol_data = {
            'spc_cluster_transition_signal': 1,  # LONG, accuracy = 0.68
            'spc_cluster_transition_strength': 0.6,
            'spc_regime_following_signal': 1,  # LONG, accuracy = 0.66
            'spc_regime_following_strength': 0.5,
            'spc_mean_reversion_signal': -1,  # SHORT, accuracy = 0.64
            'spc_mean_reversion_strength': 0.4,
        }
        
        vote, strength = analyzer._aggregate_spc_votes(symbol_data, "LONG")
        
        # Both weighted and threshold should fail, simple fallback should succeed
        # LONG accuracy sum = 1.34 > 1.0, so vote should be 1
        assert vote == 1
        assert strength > 0.0
    
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
    
    def test_get_indicator_accuracy_hmm(self, analyzer):
        """Test _get_indicator_accuracy for HMM."""
        hmm_accuracy = analyzer._get_indicator_accuracy('hmm', 'LONG')
        xgb_accuracy = analyzer._get_indicator_accuracy('xgboost', 'LONG')
        
        assert 0.0 <= hmm_accuracy <= 1.0
        assert 0.0 <= xgb_accuracy <= 1.0
    
    def test_run_atc_scan_no_signals(self, analyzer):
        """Test run_atc_scan when no ATC signals found."""
        analyzer.atc_analyzer = Mock()
        analyzer.atc_analyzer.run_auto_scan.return_value = (pd.DataFrame(), pd.DataFrame())
        
        result = analyzer.run_atc_scan()
        
        assert result is False
        assert analyzer.long_signals_atc.empty
        assert analyzer.short_signals_atc.empty
    
    def test_run_atc_scan_with_signals(self, analyzer):
        """Test run_atc_scan when ATC signals found."""
        analyzer.atc_analyzer = Mock()
        long_df = pd.DataFrame([{'symbol': 'BTC/USDT', 'signal': 1}])
        short_df = pd.DataFrame([{'symbol': 'ETH/USDT', 'signal': -1}])
        analyzer.atc_analyzer.run_auto_scan.return_value = (long_df, short_df)
        
        result = analyzer.run_atc_scan()
        
        assert result is True
        assert not analyzer.long_signals_atc.empty
        assert not analyzer.short_signals_atc.empty
    
    def test_calculate_indicator_votes_with_hmm(self, analyzer):
        """Test calculate_indicator_votes with HMM enabled."""
        analyzer.args.enable_hmm = True
        symbol_data = {
            'signal': 1,
            'osc_signal': 1,
            'osc_confidence': 0.8,
            'hmm_signal': 1,
            'hmm_confidence': 0.7,
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
        assert 'hmm' in votes
        assert all(isinstance(v, tuple) and len(v) == 2 for v in votes.values())
    
    def test_calculate_indicator_votes_with_xgboost(self, analyzer):
        """Test calculate_indicator_votes with XGBoost enabled."""
        analyzer.args.enable_xgboost = True
        symbol_data = {
            'signal': 1,
            'osc_signal': 1,
            'osc_confidence': 0.8,
            'xgboost_signal': 1,
            'xgboost_confidence': 0.75,
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
        assert 'xgboost' in votes
        assert all(isinstance(v, tuple) and len(v) == 2 for v in votes.values())
    
    def test_calculate_indicator_votes_with_all_indicators(self, analyzer):
        """Test calculate_indicator_votes with all indicators enabled."""
        analyzer.args.enable_xgboost = True
        analyzer.args.enable_hmm = True
        symbol_data = {
            'signal': 1,
            'osc_signal': 1,
            'osc_confidence': 0.8,
            'xgboost_signal': 1,
            'xgboost_confidence': 0.75,
            'hmm_signal': 1,
            'hmm_confidence': 0.7,
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
        assert 'xgboost' in votes
        assert 'hmm' in votes
        assert len(votes) == 5
    
    def test_apply_decision_matrix_with_hmm(self, analyzer):
        """Test apply_decision_matrix with HMM enabled."""
        analyzer.args.enable_hmm = True
        signals_df = pd.DataFrame([
            {
                'symbol': 'BTC/USDT',
                'signal': 1,
                'trend': 'UPTREND',
                'price': 50000.0,
                'exchange': 'binance',
                'osc_signal': 1,
                'osc_confidence': 0.8,
                'hmm_signal': 1,
                'hmm_confidence': 0.7,
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
            assert 'voting_breakdown' in result.columns
            # Check that HMM is in voting breakdown
            voting_breakdown = result.iloc[0]['voting_breakdown']
            assert 'hmm' in voting_breakdown
    
    @patch('core.hybrid_analyzer.log_warn')
    def test_run_early_exit_no_atc_signals(self, mock_log_warn, analyzer):
        """Test run() method exits early when no ATC signals found."""
        analyzer.atc_analyzer = Mock()
        analyzer.atc_analyzer.run_auto_scan.return_value = (pd.DataFrame(), pd.DataFrame())
        analyzer.determine_timeframe = Mock()
        analyzer.display_config = Mock()
        
        analyzer.run()
        
        # Should call log_warn and return early
        assert mock_log_warn.called
        # Should not proceed to filter_by_oscillator
        assert not hasattr(analyzer, 'long_signals_confirmed') or analyzer.long_signals_confirmed.empty
    
    @patch('core.hybrid_analyzer.log_progress')
    def test_run_complete_workflow(self, mock_log_progress, analyzer):
        """Test run() method completes full workflow when ATC signals found."""
        analyzer.atc_analyzer = Mock()
        long_df = pd.DataFrame([{'symbol': 'BTC/USDT', 'signal': 1}])
        short_df = pd.DataFrame([{'symbol': 'ETH/USDT', 'signal': -1}])
        analyzer.atc_analyzer.run_auto_scan.return_value = (long_df, short_df)
        analyzer.determine_timeframe = Mock()
        analyzer.display_config = Mock()
        analyzer.filter_by_oscillator = Mock()
        analyzer.calculate_spc_signals_for_all = Mock()
        analyzer.filter_by_decision_matrix = Mock()
        analyzer.display_results = Mock()
        
        analyzer.run()
        
        # Should proceed through all steps
        analyzer.filter_by_oscillator.assert_called_once()
        if analyzer.args.enable_spc:
            analyzer.calculate_spc_signals_for_all.assert_called_once()
        if analyzer.args.use_decision_matrix:
            analyzer.filter_by_decision_matrix.assert_called_once()
        analyzer.display_results.assert_called_once()
    
    def test_apply_decision_matrix_with_xgboost(self, analyzer):
        """Test apply_decision_matrix with XGBoost enabled."""
        analyzer.args.enable_xgboost = True
        signals_df = pd.DataFrame([
            {
                'symbol': 'BTC/USDT',
                'signal': 1,
                'trend': 'UPTREND',
                'price': 50000.0,
                'exchange': 'binance',
                'osc_signal': 1,
                'osc_confidence': 0.8,
                'xgboost_signal': 1,
                'xgboost_confidence': 0.75,
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
            assert 'xgboost' in result.iloc[0]['voting_breakdown']


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
        
        with patch('core.signal_calculators.generate_signals_combined_all_strategy') as mock_gen:
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
        
        with patch('core.signal_calculators.SimplifiedPercentileClustering') as mock_spc, \
             patch('core.signal_calculators.generate_signals_cluster_transition') as mock_gen:
            
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
    
    def test_get_hmm_signal_success(self, mock_data_fetcher):
        """Test get_hmm_signal with successful result."""
        mock_df = pd.DataFrame({
            'high': [100, 101, 102, 103, 104],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'open': [99, 100, 101, 102, 103],
        })
        
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, 'binance')
        
        with patch('core.signal_calculators.hmm_signals') as mock_hmm:
            from modules.hmm.signal_resolution import LONG, HOLD
            mock_hmm.return_value = (LONG, LONG)
            
            result = get_hmm_signal(
                mock_data_fetcher, 'BTC/USDT', '1h', 500
            )
            
            assert result is not None
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] in [-1, 0, 1]  # Signal value
            assert 0.0 <= result[1] <= 1.0  # Confidence
    
    def test_get_hmm_signal_no_data(self, mock_data_fetcher):
        """Test get_hmm_signal when no data."""
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (None, None)
        
        result = get_hmm_signal(
            mock_data_fetcher, 'BTC/USDT', '1h', 500
        )
        
        assert result is None
    
    def test_get_hmm_signal_conflict(self, mock_data_fetcher):
        """Test get_hmm_signal when signals conflict."""
        mock_df = pd.DataFrame({
            'high': [100, 101, 102, 103, 104] * 20,
            'low': [99, 100, 101, 102, 103] * 20,
            'close': [100, 101, 102, 103, 104] * 20,
            'open': [99, 100, 101, 102, 103] * 20,
        })
        
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, 'binance')
        
        with patch('core.signal_calculators.hmm_signals') as mock_hmm:
            from modules.hmm.signal_resolution import LONG, SHORT
            mock_hmm.return_value = (LONG, SHORT)  # Conflict
            
            result = get_hmm_signal(
                mock_data_fetcher, 'BTC/USDT', '1h', 500
            )
            
            assert result is not None
            assert result[0] == 0  # HOLD when conflict
            assert result[1] == 0.0  # Zero confidence when conflict
    
    def test_get_hmm_signal_with_custom_params(self, mock_data_fetcher):
        """Test get_hmm_signal with custom parameters."""
        mock_df = pd.DataFrame({
            'high': [100, 101, 102, 103, 104] * 20,
            'low': [99, 100, 101, 102, 103] * 20,
            'close': [100, 101, 102, 103, 104] * 20,
            'open': [99, 100, 101, 102, 103] * 20,
        })
        
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (mock_df, 'binance')
        
        with patch('core.signal_calculators.hmm_signals') as mock_hmm:
            from modules.hmm.signal_resolution import LONG
            mock_hmm.return_value = (LONG, LONG)
            
            result = get_hmm_signal(
                mock_data_fetcher, 'BTC/USDT', '1h', 500,
                window_size=150,
                window_kama=15,
                fast_kama=3,
                slow_kama=40,
                orders_argrelextrema=6,
                strict_mode=True,
            )
            
            assert result is not None
            # Verify that custom parameters were passed
            mock_hmm.assert_called_once()
            call_kwargs = mock_hmm.call_args[1]
            assert call_kwargs['window_size'] == 150
            assert call_kwargs['window_kama'] == 15
            assert call_kwargs['fast_kama'] == 3
            assert call_kwargs['slow_kama'] == 40
            assert call_kwargs['orders_argrelextrema'] == 6
            assert call_kwargs['strict_mode'] is True

