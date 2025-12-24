"""
Tests for MultiTimeframeCoordinator class.

Tests cover:
- Initialization with default and custom weights
- Timeframe validation
- Weight calculation
- Deep analysis mode
- Batch analysis mode
- Error handling
"""

import pytest
import pandas as pd
from unittest.mock import Mock
import numpy as np

from modules.gemini_chart_analyzer.core.analyzers.multi_timeframe_coordinator import MultiTimeframeCoordinator
from config.gemini_chart_analyzer import TIMEFRAME_WEIGHTS


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV DataFrame for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    np.random.seed(42)
    
    base_price = 50000
    prices = []
    for i in range(100):
        change = np.random.randn() * 100
        base_price = max(base_price + change, 1000)
        prices.append(base_price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': [p * 1.005 for p in prices],
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    return df


@pytest.fixture
def default_analyzer():
    """Create MultiTimeframeCoordinator with default weights."""
    return MultiTimeframeCoordinator()


@pytest.fixture
def custom_analyzer():
    """Create MultiTimeframeCoordinator with custom weights."""
    custom_weights = {
        '15m': 0.2,
        '1h': 0.3,
        '4h': 0.5
    }
    return MultiTimeframeCoordinator(timeframe_weights=custom_weights)


class TestMultiTimeframeCoordinatorInit:
    """Test MultiTimeframeCoordinator initialization."""
    
    def test_init_default_weights(self, default_analyzer):
        """Test initialization with default weights."""
        assert default_analyzer.signal_aggregator is not None
        assert default_analyzer.timeframe_weights is not None
        # Assert both expected timeframes are present (logical AND)
        assert '15m' in default_analyzer.timeframe_weights
        assert '1h' in default_analyzer.timeframe_weights
        # Assert exact set equality to ensure no extra or missing timeframes
        expected_timeframes = set(TIMEFRAME_WEIGHTS.keys())
        actual_timeframes = set(default_analyzer.timeframe_weights.keys())
        assert actual_timeframes == expected_timeframes, \
            f"Expected timeframes {expected_timeframes}, but got {actual_timeframes}"
    
    def test_init_custom_weights(self, custom_analyzer):
        """Test initialization with custom weights."""
        assert custom_analyzer.timeframe_weights['15m'] == 0.2
        assert custom_analyzer.timeframe_weights['1h'] == 0.3
        assert custom_analyzer.timeframe_weights['4h'] == 0.5


class TestValidateTimeframes:
    """Test _validate_timeframes method."""
    
    def test_validate_valid_timeframes(self, default_analyzer):
        """Test validation with valid timeframes."""
        result = default_analyzer._validate_timeframes(['15m', '1h', '4h'])
        assert len(result) == 3
        assert '15m' in result
        assert '1h' in result
        assert '4h' in result
    
    def test_validate_empty_list(self, default_analyzer):
        """Test validation raises error for empty list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            default_analyzer._validate_timeframes([])
    
    def test_validate_duplicates(self, default_analyzer):
        """Test validation raises error for duplicates."""
        with pytest.raises(ValueError):
            default_analyzer._validate_timeframes(['1h', '1h', '4h'])
    
    def test_validate_normalizes_timeframes(self, default_analyzer):
        """Test that timeframes are normalized."""
        result = default_analyzer._validate_timeframes(['1H', '4H'])
        assert all(isinstance(tf, str) and tf == tf.lower() for tf in result)


class TestCalculateTimeframeWeights:
    """Test _calculate_timeframe_weights method."""
    
    def test_calculate_weights_default(self, default_analyzer):
        """Test weight calculation with default weights."""
        weights = default_analyzer._calculate_timeframe_weights(['15m', '1h', '4h'])
        assert len(weights) == 3
        assert '15m' in weights
        assert '1h' in weights
        assert '4h' in weights
        # Weights should be normalized (sum to 1.0)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.0001
    
    def test_calculate_weights_custom(self, custom_analyzer):
        """Test weight calculation with custom weights."""
        weights = custom_analyzer._calculate_timeframe_weights(['15m', '1h', '4h'])
        assert len(weights) == 3
        # Should use custom weights
        assert weights['15m'] > 0
        assert weights['1h'] > 0
        assert weights['4h'] > 0
    
    def test_calculate_weights_unknown_timeframe(self, default_analyzer):
        """Test weight calculation with unknown timeframe."""
        weights = default_analyzer._calculate_timeframe_weights(['unknown_tf'])
        assert 'unknown_tf' in weights
        # After normalization, single weight becomes 1.0 (0.1 / 0.1 = 1.0)
        assert weights['unknown_tf'] == 1.0


class TestAnalyzeDeep:
    """Test analyze_deep method."""
    
    def test_analyze_deep_success(self, default_analyzer, sample_ohlcv_data):
        """Test successful deep analysis."""
        # Mock functions
        fetch_data_func = Mock(return_value=sample_ohlcv_data)
        generate_chart_func = Mock(return_value='/path/to/chart.png')
        analyze_chart_func = Mock(return_value='Analysis result text')
        
        result = default_analyzer.analyze_deep(
            symbol='BTC/USDT',
            timeframes=['15m', '1h'],
            fetch_data_func=fetch_data_func,
            generate_chart_func=generate_chart_func,
            analyze_chart_func=analyze_chart_func
        )
        
        assert result['symbol'] == 'BTC/USDT'
        assert 'timeframes' in result
        assert 'aggregated' in result
        assert len(result['timeframes']) == 2
        assert '15m' in result['timeframes']
        assert '1h' in result['timeframes']
        
        # Verify functions were called
        assert fetch_data_func.call_count == 2
        assert generate_chart_func.call_count == 2
        assert analyze_chart_func.call_count == 2
    
    def test_analyze_deep_no_data(self, default_analyzer):
        """Test deep analysis when no data is available."""
        fetch_data_func = Mock(return_value=pd.DataFrame())  # Empty DataFrame
        generate_chart_func = Mock()
        analyze_chart_func = Mock()
        
        result = default_analyzer.analyze_deep(
            symbol='BTC/USDT',
            timeframes=['15m'],
            fetch_data_func=fetch_data_func,
            generate_chart_func=generate_chart_func,
            analyze_chart_func=analyze_chart_func
        )
        
        assert result['symbol'] == 'BTC/USDT'
        assert '15m' in result['timeframes']
        assert result['timeframes']['15m']['signal'] == 'NONE'
        assert result['timeframes']['15m']['confidence'] == 0.0
        assert 'No data' in result['timeframes']['15m']['analysis']
        
        # Chart generation should not be called for empty data
        assert generate_chart_func.call_count == 0
    
    def test_analyze_deep_fetch_error(self, default_analyzer):
        """Test deep analysis when fetch_data_func raises error."""
        fetch_data_func = Mock(side_effect=Exception("Fetch error"))
        generate_chart_func = Mock()
        analyze_chart_func = Mock()
        
        result = default_analyzer.analyze_deep(
            symbol='BTC/USDT',
            timeframes=['15m'],
            fetch_data_func=fetch_data_func,
            generate_chart_func=generate_chart_func,
            analyze_chart_func=analyze_chart_func
        )
        
        assert result['symbol'] == 'BTC/USDT'
        assert '15m' in result['timeframes']
        assert result['timeframes']['15m']['signal'] == 'NONE'
        assert 'error' in result['timeframes']['15m']
    
    def test_analyze_deep_aggregates_signals(self, default_analyzer, sample_ohlcv_data):
        """Test that signals are aggregated correctly."""
        fetch_data_func = Mock(return_value=sample_ohlcv_data)
        generate_chart_func = Mock(return_value='/path/to/chart.png')
        analyze_chart_func = Mock(return_value='Analysis result')
        
        result = default_analyzer.analyze_deep(
            symbol='BTC/USDT',
            timeframes=['15m', '1h'],
            fetch_data_func=fetch_data_func,
            generate_chart_func=generate_chart_func,
            analyze_chart_func=analyze_chart_func
        )
        
        assert 'aggregated' in result
        assert 'signal' in result['aggregated']
        assert 'confidence' in result['aggregated']
        assert result['aggregated']['signal'] in ['LONG', 'SHORT', 'NONE']
        assert 0.0 <= result['aggregated']['confidence'] <= 1.0
    
    def test_analyze_deep_multiple_timeframes(self, default_analyzer, sample_ohlcv_data):
        """Test deep analysis with multiple timeframes."""
        fetch_data_func = Mock(return_value=sample_ohlcv_data)
        generate_chart_func = Mock(return_value='/path/to/chart.png')
        analyze_chart_func = Mock(return_value='Analysis result')
        
        result = default_analyzer.analyze_deep(
            symbol='BTC/USDT',
            timeframes=['15m', '1h', '4h', '1d'],
            fetch_data_func=fetch_data_func,
            generate_chart_func=generate_chart_func,
            analyze_chart_func=analyze_chart_func
        )
        
        assert len(result['timeframes']) == 4
        assert fetch_data_func.call_count == 4
        assert generate_chart_func.call_count == 4
        assert analyze_chart_func.call_count == 4


class TestAnalyzeBatch:
    """Test analyze_batch method."""
    
    def test_analyze_batch_success(self, default_analyzer, sample_ohlcv_data):
        """Test successful batch analysis."""
        symbols = ['BTC/USDT', 'ETH/USDT']
        timeframes = ['15m', '1h']
        
        # Mock functions
        fetch_data_func = Mock(return_value=sample_ohlcv_data)
        generate_batch_chart_func = Mock(return_value='/path/to/batch_chart.png')
        
        # Mock batch results from Gemini
        batch_results = {
            'BTC/USDT': {
                '15m': {'signal': 'LONG', 'confidence': 0.7},
                '1h': {'signal': 'LONG', 'confidence': 0.8}
            },
            'ETH/USDT': {
                '15m': {'signal': 'SHORT', 'confidence': 0.6},
                '1h': {'signal': 'SHORT', 'confidence': 0.7}
            }
        }
        analyze_batch_chart_func = Mock(return_value=batch_results)
        
        result = default_analyzer.analyze_batch(
            symbols=symbols,
            timeframes=timeframes,
            fetch_data_func=fetch_data_func,
            generate_batch_chart_func=generate_batch_chart_func,
            analyze_batch_chart_func=analyze_batch_chart_func
        )
        
        assert len(result) == 2
        assert 'BTC/USDT' in result
        assert 'ETH/USDT' in result
        
        # Check structure
        for symbol in symbols:
            assert 'timeframes' in result[symbol]
            assert 'aggregated' in result[symbol]
            assert 'signal' in result[symbol]['aggregated']
            assert 'confidence' in result[symbol]['aggregated']
        
        # Verify functions were called
        assert fetch_data_func.call_count == 4  # 2 symbols × 2 timeframes
        assert generate_batch_chart_func.call_count == 1
        assert analyze_batch_chart_func.call_count == 1
    
    def test_analyze_batch_missing_symbol(self, default_analyzer, sample_ohlcv_data):
        """Test batch analysis when some symbols are missing from results."""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        timeframes = ['15m', '1h']
        
        fetch_data_func = Mock(return_value=sample_ohlcv_data)
        generate_batch_chart_func = Mock(return_value='/path/to/batch_chart.png')
        
        # Only BTC/USDT in results
        batch_results = {
            'BTC/USDT': {
                '15m': {'signal': 'LONG', 'confidence': 0.7},
                '1h': {'signal': 'LONG', 'confidence': 0.8}
            }
        }
        analyze_batch_chart_func = Mock(return_value=batch_results)
        
        result = default_analyzer.analyze_batch(
            symbols=symbols,
            timeframes=timeframes,
            fetch_data_func=fetch_data_func,
            generate_batch_chart_func=generate_batch_chart_func,
            analyze_batch_chart_func=analyze_batch_chart_func
        )
        
        assert len(result) == 3
        assert 'BTC/USDT' in result
        assert 'ETH/USDT' in result
        assert 'BNB/USDT' in result
        
        # Missing symbols should have NONE signal
        assert result['ETH/USDT']['aggregated']['signal'] == 'NONE'
        assert result['BNB/USDT']['aggregated']['signal'] == 'NONE'
    
    def test_analyze_batch_fetch_error(self, default_analyzer, sample_ohlcv_data):
        """Test batch analysis when fetch_data_func raises error."""
        symbols = ['BTC/USDT', 'ETH/USDT']
        timeframes = ['15m', '1h']
        
        # Mock fetch to raise error for ETH/USDT
        def fetch_with_error(symbol, timeframe):
            if symbol == 'ETH/USDT':
                raise Exception("Fetch error")
            return sample_ohlcv_data
        
        fetch_data_func = Mock(side_effect=fetch_with_error)
        generate_batch_chart_func = Mock(return_value='/path/to/batch_chart.png')
        analyze_batch_chart_func = Mock(return_value={})
        
        result = default_analyzer.analyze_batch(
            symbols=symbols,
            timeframes=timeframes,
            fetch_data_func=fetch_data_func,
            generate_batch_chart_func=generate_batch_chart_func,
            analyze_batch_chart_func=analyze_batch_chart_func
        )
        
        # Should still return results for all symbols
        assert len(result) == 2
        assert 'BTC/USDT' in result
        assert 'ETH/USDT' in result


class TestAnalyzeDeepLimitations:
    """Test deep analysis mode limitations and documentation."""
    
    def test_analyze_deep_returns_none_signals(self, default_analyzer, sample_ohlcv_data):
        """Test that analyze_deep returns NONE signals by default (limitation)."""
        def fetch_data(symbol, tf):
            return sample_ohlcv_data
        
        def generate_chart(df, symbol, tf):
            return "/tmp/test_chart.png"
        
        def analyze_chart(chart_path, symbol, tf):
            return "This is analysis text from Gemini"
        
        result = default_analyzer.analyze_deep(
            symbol="BTC/USDT",
            timeframes=['15m', '1h'],
            fetch_data_func=fetch_data,
            generate_chart_func=generate_chart,
            analyze_chart_func=analyze_chart
        )
        
        # Deep analysis mode should return NONE signals
        assert result['symbol'] == 'BTC/USDT'
        assert '15m' in result['timeframes']
        assert result['timeframes']['15m']['signal'] == 'NONE'
        assert result['timeframes']['15m']['confidence'] == 0.5
        assert 'analysis' in result['timeframes']['15m']
        assert result['aggregated']['signal'] == 'NONE'


class TestIntegration:
    """Integration tests."""
    
    def test_full_deep_analysis_workflow(self, default_analyzer, sample_ohlcv_data):
        """Test complete deep analysis workflow."""
        fetch_data_func = Mock(return_value=sample_ohlcv_data)
        generate_chart_func = Mock(return_value='/path/to/chart.png')
        analyze_chart_func = Mock(return_value='Detailed analysis text')
        
        result = default_analyzer.analyze_deep(
            symbol='BTC/USDT',
            timeframes=['15m', '1h', '4h'],
            fetch_data_func=fetch_data_func,
            generate_chart_func=generate_chart_func,
            analyze_chart_func=analyze_chart_func
        )
        
        # Verify complete structure
        assert result['symbol'] == 'BTC/USDT'
        assert len(result['timeframes']) == 3
        assert 'aggregated' in result
        assert 'signal' in result['aggregated']
        assert 'confidence' in result['aggregated']
        assert 'timeframe_breakdown' in result['aggregated']
    
    def test_full_batch_analysis_workflow(self, default_analyzer, sample_ohlcv_data):
        """Test complete batch analysis workflow."""
        symbols = ['BTC/USDT', 'ETH/USDT']
        timeframes = ['15m', '1h']
        
        fetch_data_func = Mock(return_value=sample_ohlcv_data)
        generate_batch_chart_func = Mock(return_value='/path/to/batch_chart.png')
        
        batch_results = {
            'BTC/USDT': {
                '15m': {'signal': 'LONG', 'confidence': 0.7},
                '1h': {'signal': 'LONG', 'confidence': 0.8}
            },
            'ETH/USDT': {
                '15m': {'signal': 'SHORT', 'confidence': 0.6},
                '1h': {'signal': 'SHORT', 'confidence': 0.7}
            }
        }
        analyze_batch_chart_func = Mock(return_value=batch_results)
        
        result = default_analyzer.analyze_batch(
            symbols=symbols,
            timeframes=timeframes,
            fetch_data_func=fetch_data_func,
            generate_batch_chart_func=generate_batch_chart_func,
            analyze_batch_chart_func=analyze_batch_chart_func
        )
        
        # Verify complete structure for each symbol
        for symbol in symbols:
            assert symbol in result
            assert 'timeframes' in result[symbol]
            assert 'aggregated' in result[symbol]
            assert len(result[symbol]['timeframes']) == 2
            assert result[symbol]['aggregated']['signal'] in ['LONG', 'SHORT', 'NONE']


