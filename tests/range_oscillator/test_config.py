import pytest

from modules.range_oscillator.config.consensus_config import ConsensusConfig
from modules.range_oscillator.config.dynamic_selection_config import DynamicSelectionConfig
from modules.range_oscillator.config.strategy_combine_config import CombinedStrategyConfig
from modules.range_oscillator.config.strategy_specific_config import StrategySpecificConfig

"""
Tests for range_oscillator config module.

Tests configuration classes:
- ConsensusConfig
- DynamicSelectionConfig
- StrategySpecificConfig
- CombinedStrategyConfig
"""


class TestConsensusConfig:
    """Tests for ConsensusConfig class."""

    def test_consensus_config_default(self):
        """Test ConsensusConfig with default values."""
        config = ConsensusConfig()

        assert config.mode == "threshold"
        assert config.threshold == 0.5
        assert config.adaptive_weights is False
        assert config.performance_window == 10
        assert config.weighted_min_diff == 0.1
        assert config.weighted_min_total == 0.5

    def test_consensus_config_threshold_mode(self):
        """Test ConsensusConfig with threshold mode."""
        config = ConsensusConfig(mode="threshold", threshold=0.6)

        assert config.mode == "threshold"
        assert config.threshold == 0.6

    def test_consensus_config_weighted_mode(self):
        """Test ConsensusConfig with weighted mode."""
        config = ConsensusConfig(
            mode="weighted", adaptive_weights=True, performance_window=20, weighted_min_diff=0.2, weighted_min_total=0.6
        )

        assert config.mode == "weighted"
        assert config.adaptive_weights is True
        assert config.performance_window == 20
        assert config.weighted_min_diff == 0.2
        assert config.weighted_min_total == 0.6

    def test_consensus_config_validation(self):
        """Test ConsensusConfig validation."""
        # Invalid mode
        with pytest.raises(ValueError, match="mode must be"):
            ConsensusConfig(mode="invalid")

        # Invalid threshold (out of range)
        with pytest.raises(ValueError, match="threshold must be between"):
            ConsensusConfig(threshold=1.5)
        with pytest.raises(ValueError, match="threshold must be between"):
            ConsensusConfig(threshold=-0.1)

        # Invalid performance_window
        with pytest.raises(ValueError, match="performance_window must be"):
            ConsensusConfig(performance_window=0)
        with pytest.raises(ValueError, match="performance_window must be"):
            ConsensusConfig(performance_window=-1)

        # Invalid weighted_min_diff
        with pytest.raises(ValueError, match="weighted_min_diff must be"):
            ConsensusConfig(weighted_min_diff=-0.1)

        # Invalid weighted_min_total
        with pytest.raises(ValueError, match="weighted_min_total must be"):
            ConsensusConfig(weighted_min_total=-0.1)


class TestDynamicSelectionConfig:
    """Tests for DynamicSelectionConfig class."""

    def test_dynamic_selection_config_default(self):
        """Test DynamicSelectionConfig with default values."""
        config = DynamicSelectionConfig()

        assert config.enabled is False
        assert config.lookback == 20
        assert config.volatility_threshold == 0.6
        assert config.trend_threshold == 0.5

    def test_dynamic_selection_config_enabled(self):
        """Test DynamicSelectionConfig with enabled=True."""
        config = DynamicSelectionConfig(enabled=True, lookback=30, volatility_threshold=0.7, trend_threshold=0.6)

        assert config.enabled is True
        assert config.lookback == 30
        assert config.volatility_threshold == 0.7
        assert config.trend_threshold == 0.6

    def test_dynamic_selection_config_validation(self):
        """Test DynamicSelectionConfig validation."""
        # Invalid lookback
        with pytest.raises(ValueError, match="lookback must be"):
            DynamicSelectionConfig(lookback=0)
        with pytest.raises(ValueError, match="lookback must be"):
            DynamicSelectionConfig(lookback=-1)

        # Invalid volatility_threshold
        with pytest.raises(ValueError, match="volatility_threshold must be between"):
            DynamicSelectionConfig(volatility_threshold=1.5)
        with pytest.raises(ValueError, match="volatility_threshold must be between"):
            DynamicSelectionConfig(volatility_threshold=-0.1)

        # Invalid trend_threshold
        with pytest.raises(ValueError, match="trend_threshold must be between"):
            DynamicSelectionConfig(trend_threshold=1.5)
        with pytest.raises(ValueError, match="trend_threshold must be between"):
            DynamicSelectionConfig(trend_threshold=-0.1)


class TestStrategySpecificConfig:
    """Tests for StrategySpecificConfig class."""

    def test_strategy_specific_config_default(self):
        """Test StrategySpecificConfig with default values."""
        config = StrategySpecificConfig()

        assert config.use_sustained is True
        assert config.min_bars_sustained == 3
        assert config.use_crossover is True
        assert config.confirmation_bars == 2
        assert config.use_momentum is True
        assert config.momentum_period == 3
        assert config.momentum_threshold == 5.0
        assert config.use_breakout is False
        assert config.use_divergence is False
        assert config.use_trend_following is False
        assert config.use_mean_reversion is False

    def test_strategy_specific_config_custom(self):
        """Test StrategySpecificConfig with custom values."""
        config = StrategySpecificConfig(
            use_sustained=False,
            min_bars_sustained=5,
            use_crossover=True,
            confirmation_bars=3,
            use_momentum=True,
            momentum_period=5,
            momentum_threshold=10.0,
        )

        assert config.use_sustained is False
        assert config.min_bars_sustained == 5
        assert config.confirmation_bars == 3
        assert config.momentum_period == 5
        assert config.momentum_threshold == 10.0

    def test_strategy_specific_config_validation(self):
        """Test StrategySpecificConfig validation."""
        # Invalid min_bars_sustained when use_sustained=True
        with pytest.raises(ValueError, match="min_bars_sustained must be"):
            StrategySpecificConfig(use_sustained=True, min_bars_sustained=0)

        # Invalid confirmation_bars when use_crossover=True
        with pytest.raises(ValueError, match="confirmation_bars must be"):
            StrategySpecificConfig(use_crossover=True, confirmation_bars=0)

        # Invalid momentum_period when use_momentum=True
        with pytest.raises(ValueError, match="momentum_period must be"):
            StrategySpecificConfig(use_momentum=True, momentum_period=0)

        # Invalid momentum_threshold when use_momentum=True
        with pytest.raises(ValueError, match="momentum_threshold must be"):
            StrategySpecificConfig(use_momentum=True, momentum_threshold=-1)

        # Invalid trend_filter_period when use_trend_following=True
        with pytest.raises(ValueError, match="trend_filter_period must be"):
            StrategySpecificConfig(use_trend_following=True, trend_filter_period=0)

        # Should not raise when strategy is disabled
        StrategySpecificConfig(use_sustained=False, min_bars_sustained=0)
        StrategySpecificConfig(use_crossover=False, confirmation_bars=0)
        StrategySpecificConfig(use_momentum=False, momentum_period=0)


class TestCombinedStrategyConfig:
    """Tests for CombinedStrategyConfig class."""

    def test_combined_strategy_config_default(self):
        """Test CombinedStrategyConfig with default values."""
        config = CombinedStrategyConfig()

        assert config.enabled_strategies == []
        assert isinstance(config.consensus, ConsensusConfig)
        assert isinstance(config.dynamic, DynamicSelectionConfig)
        assert isinstance(config.params, StrategySpecificConfig)
        assert config.min_signal_strength == 0.0
        assert config.strategy_weights is None
        assert config.return_confidence_score is False
        assert config.return_strategy_stats is False
        assert config.enable_debug is False

    def test_combined_strategy_config_custom(self):
        """Test CombinedStrategyConfig with custom values."""
        config = CombinedStrategyConfig(
            enabled_strategies=[2, 3, 4],
            min_signal_strength=0.1,
            strategy_weights={2: 1.0, 3: 1.5, 4: 0.8},
            return_confidence_score=True,
            return_strategy_stats=True,
            enable_debug=True,
        )

        assert config.enabled_strategies == [2, 3, 4]
        assert config.min_signal_strength == 0.1
        assert config.strategy_weights == {2: 1.0, 3: 1.5, 4: 0.8}
        assert config.return_confidence_score is True
        assert config.return_strategy_stats is True
        assert config.enable_debug is True

    def test_combined_strategy_config_nested_configs(self):
        """Test CombinedStrategyConfig with nested config objects."""
        consensus = ConsensusConfig(mode="weighted", threshold=0.6)
        dynamic = DynamicSelectionConfig(enabled=True, lookback=30)
        params = StrategySpecificConfig(use_sustained=False)

        config = CombinedStrategyConfig(consensus=consensus, dynamic=dynamic, params=params)

        assert config.consensus.mode == "weighted"
        assert config.consensus.threshold == 0.6
        assert config.dynamic.enabled is True
        assert config.dynamic.lookback == 30
        assert config.params.use_sustained is False

    def test_combined_strategy_config_validation(self):
        """Test CombinedStrategyConfig validation."""
        from config import VALID_STRATEGY_IDS

        # Invalid strategy IDs
        with pytest.raises(ValueError, match="Invalid strategy IDs"):
            CombinedStrategyConfig(enabled_strategies=[1, 5, 10])

        # Valid strategy IDs should not raise
        valid_ids = list(VALID_STRATEGY_IDS)
        config = CombinedStrategyConfig(enabled_strategies=valid_ids)
        assert config.enabled_strategies == valid_ids

        # Invalid min_signal_strength
        with pytest.raises(ValueError, match="min_signal_strength must be"):
            CombinedStrategyConfig(min_signal_strength=-0.1)

        # Invalid strategy_weights - invalid strategy ID
        with pytest.raises(ValueError, match="Invalid strategy ID in weights"):
            CombinedStrategyConfig(strategy_weights={1: 1.0})

        # Invalid strategy_weights - negative weight
        with pytest.raises(ValueError, match="Weight for strategy"):
            CombinedStrategyConfig(strategy_weights={2: -1.0})

        # Valid strategy_weights should not raise
        CombinedStrategyConfig(strategy_weights={2: 1.0, 3: 1.5, 4: 0.8})

    def test_combined_strategy_config_validation_cascade(self):
        """Test that validation in nested configs is triggered."""
        # Invalid consensus config should raise
        with pytest.raises(ValueError):
            CombinedStrategyConfig(consensus=ConsensusConfig(mode="invalid"))

        # Invalid dynamic config should raise
        with pytest.raises(ValueError):
            CombinedStrategyConfig(dynamic=DynamicSelectionConfig(lookback=0))

        # Invalid params config should raise
        with pytest.raises(ValueError):
            CombinedStrategyConfig(params=StrategySpecificConfig(use_sustained=True, min_bars_sustained=0))
