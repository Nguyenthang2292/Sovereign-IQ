"""
Integration tests for Decision Matrix module.

Tests complete workflow from data → training → classification → voting.
"""

from typing import List, Tuple

import pytest

from modules.decision_matrix import (
    DecisionMatrixClassifier,
    FeatureType,
    RandomForestConfig,
    RandomForestCore,
    TargetType,
    ThresholdCalculator,
    TrainingDataStorage,
)


class TestDataCollectionWorkflow:
    """Test workflow for collecting and storing training data."""

    def test_collect_training_samples(self):
        """Test collecting training samples over time."""
        storage = TrainingDataStorage()
        storage.training_length = 100

        # Simulate collecting data over time
        training_data = self._generate_mock_training_data(150)

        for x1, x2, y in training_data:
            storage.add_sample(x1, x2, y)

        assert storage.get_size() == 100  # Circular buffer capped at 100
        assert storage.is_full()

    def test_get_training_matrices(self):
        """Test retrieving training matrices."""
        storage = TrainingDataStorage()
        storage.training_length = 50

        training_data = self._generate_mock_training_data(50)
        for x1, x2, y in training_data:
            storage.add_sample(x1, x2, y)

        x1_matrix = storage.get_x1_matrix()
        x2_matrix = storage.get_x2_matrix()

        assert len(x1_matrix) == 50
        assert len(x2_matrix) == 50

        # Verify structure: each row is (value, label)
        for row in x1_matrix:
            assert len(row) == 2
            assert isinstance(row[0], (int, float))
            assert row[1] in (0, 1)

    def _generate_mock_training_data(self, n: int) -> List[Tuple[float, float, int]]:
        """Generate mock training data."""
        import random

        data = []
        for _ in range(n):
            x1 = random.uniform(0, 100)
            x2 = random.uniform(0, 1000)
            y = random.choice([0, 1])
            data.append((x1, x2, y))
        return data


class TestTrainingWorkflow:
    """Test workflow for training Random Forest model."""

    def test_collect_and_train(self):
        """Test complete data collection and training workflow."""
        # Setup
        config = RandomForestConfig(
            training_length=100,
            x1_type=FeatureType.RSI,
            x2_type=FeatureType.VOLUME,
            target_type=TargetType.RED_GREEN_CANDLE,
        )

        storage = TrainingDataStorage()
        storage.training_length = config.training_length
        classifier = DecisionMatrixClassifier()
        classifier.random_forest.training_length = config.training_length

        # Collect training data
        training_data = self._generate_mock_training_data(150)
        for x1, x2, y in training_data:
            storage.add_sample(x1, x2, y)

        # Verify storage
        assert storage.get_size() == 100
        assert storage.is_full()

    def test_training_data_persistence(self):
        """Test that training data persists across classifications."""
        storage = TrainingDataStorage()
        storage.training_length = 50
        rf_core = RandomForestCore()

        # Add training samples
        for i in range(50):
            storage.add_sample(float(i), float(i * 10), i % 2)

        # Perform multiple classifications
        x1_matrix = storage.get_x1_matrix()
        x2_matrix = storage.get_x2_matrix()

        results1 = rf_core.classify(x1_matrix, x2_matrix, 25.0, 250.0, 5.0, 50.0)
        results2 = rf_core.classify(x1_matrix, x2_matrix, 30.0, 300.0, 5.0, 50.0)

        # Verify storage unchanged
        assert storage.get_size() == 50

        # Verify both results are valid
        assert results1["vote"] in (0, 1)
        assert results2["vote"] in (0, 1)

    def _generate_mock_training_data(self, n: int) -> List[Tuple[float, float, int]]:
        """Generate mock training data."""
        import random

        data = []
        for _ in range(n):
            x1 = random.uniform(0, 100)
            x2 = random.uniform(0, 1000)
            y = random.choice([0, 1])
            data.append((x1, x2, y))
        return data


class TestClassificationWorkflow:
    """Test workflow for Random Forest classification."""

    def test_full_classification_workflow(self):
        """Test complete classification workflow."""
        # Setup
        config = RandomForestConfig(
            training_length=100,
            x1_type=FeatureType.RSI,
            x2_type=FeatureType.VOLUME,
        )

        storage = TrainingDataStorage()
        storage.training_length = config.training_length
        threshold_calc = ThresholdCalculator()
        rf_core = RandomForestCore()

        # Collect training data
        training_data = self._generate_mock_training_data(150)
        for x1, x2, y in training_data:
            storage.add_sample(x1, x2, y)

        # Prepare for classification
        x1_matrix = storage.get_x1_matrix()
        x2_matrix = storage.get_x2_matrix()

        current_x1, current_x2, current_y = training_data[-1]

        # Calculate thresholds
        historical_x1 = [row[0] for row in x1_matrix]
        historical_x2 = [row[0] for row in x2_matrix]

        x1_threshold = threshold_calc.calculate_threshold(config.x1_type.value, historical_x1)
        x2_threshold = threshold_calc.calculate_threshold(config.x2_type.value, historical_x2)

        # Classify
        results = rf_core.classify(x1_matrix, x2_matrix, current_x1, current_x2, x1_threshold, x2_threshold)

        # Verify results
        assert "vote" in results
        assert "accuracy" in results
        assert "y1_pass" in results
        assert "y1_fail" in results
        assert "y2_pass" in results
        assert "y2_fail" in results
        assert "x1_vote" in results
        assert "x2_vote" in results
        assert "x1_accuracy" in results
        assert "x2_accuracy" in results

        assert results["vote"] in (0, 1)
        assert 0.0 <= results["accuracy"] <= 100.0
        assert results["y1_pass"] >= 0
        assert results["y1_fail"] >= 0
        assert results["y2_pass"] >= 0
        assert results["y2_fail"] >= 0

    def test_classification_with_different_feature_types(self):
        """Test classification with various feature types."""
        feature_types = [
            (FeatureType.RSI, FeatureType.MFI),
            (FeatureType.STOCHASTIC, FeatureType.EMA),
            (FeatureType.Z_SCORE, FeatureType.SMA),
            (FeatureType.VOLUME, FeatureType.RSI),
        ]

        for x1_type, x2_type in feature_types:
            config = RandomForestConfig(
                training_length=50,
                x1_type=x1_type,
                x2_type=x2_type,
            )

            storage = TrainingDataStorage()
            storage.training_length = config.training_length
            threshold_calc = ThresholdCalculator()
            rf_core = RandomForestCore()

            # Collect training data
            for i in range(50):
                x1 = float(i % 100)
                x2 = float(i * 10)
                y = i % 2
                storage.add_sample(x1, x2, y)

            x1_matrix = storage.get_x1_matrix()
            x2_matrix = storage.get_x2_matrix()

            # Calculate thresholds
            historical_x1 = [row[0] for row in x1_matrix]
            historical_x2 = [row[0] for row in x2_matrix]

            x1_threshold = threshold_calc.calculate_threshold(config.x1_type.value, historical_x1)
            x2_threshold = threshold_calc.calculate_threshold(config.x2_type.value, historical_x2)

            # Classify
            results = rf_core.classify(x1_matrix, x2_matrix, 25.0, 250.0, x1_threshold, x2_threshold)

            assert results["vote"] in (0, 1)

    def _generate_mock_training_data(self, n: int) -> List[Tuple[float, float, int]]:
        """Generate mock training data."""
        import random

        data = []
        for _ in range(n):
            x1 = random.uniform(0, 100)
            x2 = random.uniform(0, 1000)
            y = random.choice([0, 1])
            data.append((x1, x2, y))
        return data


class TestVotingWorkflow:
    """Test workflow for voting system."""

    def test_complete_voting_workflow(self):
        """Test complete voting workflow with Random Forest integration."""
        # Setup
        config = RandomForestConfig(
            training_length=100,
            x1_type=FeatureType.RSI,
            x2_type=FeatureType.VOLUME,
        )

        classifier = DecisionMatrixClassifier(indicators=["atc", "oscillator"])
        classifier.random_forest.training_length = config.training_length

        # Add training samples
        training_data = self._generate_mock_training_data(150)
        for x1, x2, y in training_data:
            classifier.add_training_sample(x1, x2, y)

        # Add node votes (from ATC, Oscillator)
        classifier.add_node_vote("atc", vote=1, signal_strength=0.7, accuracy=0.65)
        classifier.add_node_vote("oscillator", vote=1, signal_strength=0.8, accuracy=0.72)

        # Calculate weighted impact
        classifier.calculate_weighted_impact()

        # Calculate cumulative vote
        vote, weighted_score, breakdown = classifier.calculate_cumulative_vote(threshold=0.5, min_votes=2)

        # Verify results
        assert vote in (0, 1)
        assert 0.0 <= weighted_score <= 1.0
        assert "atc" in breakdown
        assert "oscillator" in breakdown

    def test_voting_with_random_forest(self):
        """Test voting system combined with Random Forest."""
        classifier = DecisionMatrixClassifier(indicators=["atc", "oscillator", "random_forest"])

        # Add training samples
        for i in range(100):
            classifier.add_training_sample(float(i), float(i * 10), i % 2)

        # Add votes from all indicators
        classifier.add_node_vote("atc", vote=1, signal_strength=0.7, accuracy=0.65)
        classifier.add_node_vote("oscillator", vote=1, signal_strength=0.8, accuracy=0.72)
        classifier.add_node_vote("random_forest", vote=1, signal_strength=0.75, accuracy=0.68)

        # Calculate weighted impact (should cap at 40% since N=3)
        classifier.calculate_weighted_impact()

        # Verify no indicator exceeds 40%
        for indicator, weight in classifier.weighted_impact.items():
            assert weight <= 0.4, f"{indicator} has weight {weight} > 0.4"

        # Calculate cumulative vote
        vote, weighted_score, breakdown = classifier.calculate_cumulative_vote(threshold=0.5, min_votes=2)

        assert vote in (0, 1)
        assert sum(classifier.weighted_impact.values()) == pytest.approx(1.0, abs=0.01)

    def test_voting_with_n2_indicators(self):
        """Test voting with 2 indicators (cap at 60%)."""
        classifier = DecisionMatrixClassifier(indicators=["atc", "oscillator"])

        # Add votes with uneven accuracy
        classifier.add_node_vote("atc", vote=1, signal_strength=0.9, accuracy=0.85)
        classifier.add_node_vote("oscillator", vote=1, signal_strength=0.5, accuracy=0.55)

        # Calculate weighted impact
        classifier.calculate_weighted_impact()

        # Verify cap at 60%
        max_weight = max(classifier.weighted_impact.values())
        assert max_weight <= 0.6

    def _generate_mock_training_data(self, n: int) -> List[Tuple[float, float, int]]:
        """Generate mock training data."""
        import random

        data = []
        for _ in range(n):
            x1 = random.uniform(0, 100)
            x2 = random.uniform(0, 1000)
            y = random.choice([0, 1])
            data.append((x1, x2, y))
        return data


class TestEndToEndWorkflow:
    """Test end-to-end workflow from data to voting."""

    def test_complete_e2e_workflow(self):
        """Test complete end-to-end workflow."""
        # Step 1: Configuration
        config = RandomForestConfig(
            training_length=850,
            x1_type=FeatureType.RSI,
            x2_type=FeatureType.VOLUME,
            target_type=TargetType.RED_GREEN_CANDLE,
        )

        # Step 2: Initialize classifier
        classifier = DecisionMatrixClassifier(indicators=["atc", "oscillator"])
        classifier.random_forest.training_length = config.training_length

        # Step 3: Collect training data (simulate over time)
        training_data = self._generate_mock_training_data(1000)
        for x1, x2, y in training_data:
            classifier.add_training_sample(x1, x2, y)

        # Step 4: Add node votes from indicators
        classifier.add_node_vote("atc", vote=1, signal_strength=0.75, accuracy=0.70)
        classifier.add_node_vote("oscillator", vote=1, signal_strength=0.80, accuracy=0.72)

        # Step 5: Calculate weighted impact
        classifier.calculate_weighted_impact()

        # Step 6: Classify with Random Forest
        current_x1, current_x2, current_y = training_data[-1]

        rf_results = classifier.classify_with_random_forest(
            x1=current_x1,
            x2=current_x2,
            y=current_y,
            x1_type=config.x1_type.value,
            x2_type=config.x2_type.value,
            historical_x1=[row[0] for row in classifier.training_data.get_x1_matrix()],
            historical_x2=[row[0] for row in classifier.training_data.get_x2_matrix()],
        )

        # Step 7: Calculate cumulative vote from voting system
        vote, weighted_score, breakdown = classifier.calculate_cumulative_vote(threshold=0.5, min_votes=2)

        # Step 8: Verify results
        # Random Forest results
        assert rf_results["vote"] in (0, 1)
        assert 0.0 <= rf_results["accuracy"] <= 100.0

        # Voting system results
        assert vote in (0, 1)
        assert 0.0 <= weighted_score <= 1.0
        assert "atc" in breakdown
        assert "oscillator" in breakdown

        # Metadata
        metadata = classifier.get_metadata()
        assert "node_votes" in metadata
        assert "feature_importance" in metadata
        assert "random_forest" in metadata

    def test_e2e_workflow_with_errors(self):
        """Test end-to-end workflow with error handling."""
        classifier = DecisionMatrixClassifier(indicators=["atc", "oscillator"])

        # Try to calculate weighted impact before adding votes
        with pytest.raises(ValueError, match="Missing feature importance data"):
            classifier.calculate_weighted_impact()

        # Add votes
        classifier.add_node_vote("atc", vote=1, signal_strength=0.7, accuracy=0.65)
        classifier.add_node_vote("oscillator", vote=1, signal_strength=0.8, accuracy=0.72)

        # Calculate weighted impact
        classifier.calculate_weighted_impact()

        # Try to add vote for non-existent indicator
        with pytest.raises(ValueError):
            classifier.add_node_vote("spc", vote=1, signal_strength=0.5)

    def _generate_mock_training_data(self, n: int) -> List[Tuple[float, float, int]]:
        """Generate mock training data."""
        import random

        data = []
        for _ in range(n):
            x1 = random.uniform(0, 100)
            x2 = random.uniform(0, 1000)
            y = random.choice([0, 1])
            data.append((x1, x2, y))
        return data


class TestRealWorldScenario:
    """Test real-world scenarios."""

    def test_crypto_trading_scenario(self):
        """Test scenario similar to crypto trading."""
        # Setup
        config = RandomForestConfig(
            training_length=850,
            x1_type=FeatureType.RSI,
            x2_type=FeatureType.VOLUME,
        )

        classifier = DecisionMatrixClassifier(indicators=["atc", "oscillator"])

        # Simulate collecting data over multiple candles
        num_candles = 1000
        for i in range(num_candles):
            # Simulate OHLCV data
            rsi = float(30 + (i % 70))  # RSI between 30-100
            volume = float(1000 + (i * 10))  # Volume increasing over time
            is_bullish = 1 if i % 3 == 0 else 0  # Every 3rd candle is bullish

            classifier.add_training_sample(x1=rsi, x2=volume, y=is_bullish)

        # At the end, we have the latest signal
        latest_rsi = 75.0
        latest_volume = 10000.0
        latest_signal = 1

        # Add indicator votes
        classifier.add_node_vote("atc", vote=1, signal_strength=0.8, accuracy=0.70)
        classifier.add_node_vote("oscillator", vote=1, signal_strength=0.75, accuracy=0.68)

        # Calculate weighted impact
        classifier.calculate_weighted_impact()

        # Classify with Random Forest
        rf_results = classifier.classify_with_random_forest(
            x1=latest_rsi,
            x2=latest_volume,
            y=latest_signal,
            x1_type=config.x1_type.value,
            x2_type=config.x2_type.value,
            historical_x1=[row[0] for row in classifier.training_data.get_x1_matrix()],
            historical_x2=[row[0] for row in classifier.training_data.get_x2_matrix()],
        )

        # Calculate voting system result
        vote, weighted_score, breakdown = classifier.calculate_cumulative_vote(threshold=0.5, min_votes=2)

        # Verify we got meaningful results
        assert rf_results["vote"] in (0, 1)
        assert rf_results["accuracy"] > 0

        assert vote in (0, 1)
        assert weighted_score > 0

        # Get final decision
        final_decision = {
            "random_forest": rf_results,
            "voting_system": {
                "vote": vote,
                "weighted_score": weighted_score,
                "breakdown": breakdown,
            },
            "metadata": classifier.get_metadata(),
        }

        # Verify we have all decision components
        assert "random_forest" in final_decision
        assert "voting_system" in final_decision
        assert "metadata" in final_decision
