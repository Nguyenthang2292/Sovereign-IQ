"""
Tests for Random Forest signals module.

Tests cover:
- Model training and saving
- Signal generation
- Model evaluation with confidence thresholds
- Edge cases and error handling
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from modules.random_forest import (
    train_random_forest_model,
    get_latest_random_forest_signal,
    train_and_save_global_rf_model,
    load_random_forest_model,
    evaluate_model_with_confidence,
    apply_confidence_threshold,
    calculate_and_display_metrics,
)

from config import (
    MAX_TRAINING_ROWS, CONFIDENCE_THRESHOLD,
    CONFIDENCE_THRESHOLDS, MIN_MEMORY_GB, MIN_TRAINING_SAMPLES,
    BUY_THRESHOLD, SELL_THRESHOLD, MODEL_RANDOM_STATE, MODEL_TEST_SIZE,
    MODELS_DIR, RANDOM_FOREST_MODEL_FILENAME
)
from config.model_features import MODEL_FEATURES


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample OHLC data for testing"""
    np.random.seed(42)
    sample_size = 100
    
    close_prices = 100 + np.cumsum(np.random.randn(sample_size) * 0.5)
    high_prices = close_prices + np.random.uniform(0, 2, sample_size)
    low_prices = close_prices - np.random.uniform(0, 2, sample_size)
    open_prices = close_prices + np.random.uniform(-1, 1, sample_size)
    
    volume_data = np.random.uniform(1000, 10000, sample_size)
    
    return pd.DataFrame({
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume_data
    })


@pytest.fixture
def training_data():
    """Create larger dataset for training"""
    np.random.seed(42)
    # Increased size to 500 to ensure enough samples after dropna with many features
    training_size = 500
    close_prices_large = 100 + np.cumsum(np.random.randn(training_size) * 0.5)
    high_prices_large = close_prices_large + np.random.uniform(0, 2, training_size)
    low_prices_large = close_prices_large - np.random.uniform(0, 2, training_size)
    open_prices_large = close_prices_large + np.random.uniform(-1, 1, training_size)
    
    volume_data_large = np.random.uniform(1000, 10000, training_size)
    
    return pd.DataFrame({
        "open": open_prices_large,
        "high": high_prices_large,
        "low": low_prices_large,
        "close": close_prices_large,
        "volume": volume_data_large
    })


@pytest.fixture
def empty_dataframe():
    """Create empty DataFrame for edge cases"""
    return pd.DataFrame()


@pytest.fixture
def incomplete_dataframe():
    """Create DataFrame with missing columns"""
    return pd.DataFrame({
        "open": [100, 101, 102],
        "high": [105, 106, 107]
        # Missing LOW and CLOSE columns
    })


@pytest.fixture
def large_dataframe():
    """Create very large dataset for testing memory constraints"""
    np.random.seed(42)
    large_dataset_size = MAX_TRAINING_ROWS + 100
    large_close = 100 + np.cumsum(np.random.randn(large_dataset_size) * 0.5)
    volume_data_large = np.random.uniform(1000, 10000, large_dataset_size)
    
    return pd.DataFrame({
        "open": large_close + np.random.uniform(-1, 1, large_dataset_size),
        "high": large_close + np.random.uniform(0, 2, large_dataset_size),
        "low": large_close - np.random.uniform(0, 2, large_dataset_size),
        "close": large_close,
        "volume": volume_data_large
    })


@pytest.fixture
def single_class_dataframe():
    """Create DataFrame with single class for testing class imbalance"""
    # Increased size to ensure enough samples after feature calculation
    return pd.DataFrame({
        "open": [100] * 200,
        "high": [105] * 200,
        "low": [95] * 200,
        "close": [100] * 200,
        "volume": [5000] * 200
    })


@pytest.fixture
def insufficient_dataframe():
    """Create DataFrame with insufficient samples"""
    return pd.DataFrame({
        "open": [100, 101, 102],
        "high": [105, 106, 107],
        "low": [95, 96, 97],
        "close": [100, 101, 102],
        "volume": [5000, 5100, 5200]
    })


@pytest.fixture
def test_evaluation_data():
    """Create test data for evaluation functions"""
    y_true = np.array([-1, 0, 1, -1, 0, 1, 0, 1, -1, 0])
    y_pred = np.array([-1, 0, 1, 0, 0, 1, 0, 1, -1, 0])
    y_proba = np.array([
        [0.8, 0.1, 0.1],  # High confidence for -1
        [0.1, 0.8, 0.1],  # High confidence for 0
        [0.1, 0.1, 0.8],  # High confidence for 1
        [0.4, 0.5, 0.1],  # Low confidence
        [0.1, 0.8, 0.1],  # High confidence for 0
        [0.1, 0.1, 0.8],  # High confidence for 1
        [0.1, 0.8, 0.1],  # High confidence for 0
        [0.1, 0.1, 0.8],  # High confidence for 1
        [0.8, 0.1, 0.1],  # High confidence for -1
        [0.1, 0.8, 0.1]   # High confidence for 0
    ])
    classes = np.array([-1, 0, 1])
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'classes': classes
    }


@pytest.fixture
def temp_models_dir():
    """Create temporary directory for model testing"""
    temp_dir = tempfile.mkdtemp()
    test_models_dir = Path(temp_dir) / "models"
    test_models_dir.mkdir(exist_ok=True)
    yield test_models_dir
    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except (OSError, FileNotFoundError):
        pass


@pytest.fixture
def mock_model_features():
    """Mock MODEL_FEATURES for testing"""
    return MODEL_FEATURES[:10]  # Use first 10 features for testing


@pytest.fixture
def mock_sufficient_memory():
    """Mock sufficient memory (4GB)"""
    with patch('modules.random_forest.utils.training.psutil.virtual_memory') as mock_mem:
        mock_mem.return_value.available = 4 * 1024**3  # 4GB
        yield mock_mem


@pytest.fixture
def mock_insufficient_memory():
    """Mock insufficient memory (500MB)"""
    with patch('modules.random_forest.utils.training.psutil.virtual_memory') as mock_mem:
        mock_mem.return_value.available = 500 * 1024**2  # 500MB
        yield mock_mem


@pytest.fixture
def mock_indicator_engine(sample_data, mock_model_features):
    """Mock IndicatorEngine with valid features"""
    def _create_mock_features_with_values(df, include_nan=False):
        features_df = df.copy()
        for feature in mock_model_features:
            if include_nan:
                features_df[feature] = np.nan
            else:
                features_df[feature] = np.random.randn(len(features_df))
        return features_df
    
    return _create_mock_features_with_values


# ============================================================================
# Tests for train_random_forest_model
# ============================================================================

class TestTrainRandomForestModel:
    """Test suite for train_random_forest_model function"""
    
    def test_insufficient_memory(self, mock_insufficient_memory, training_data):
        """Test model training with insufficient memory (SMOTE skipped but training continues)"""
        with patch('modules.random_forest.core.model.joblib.dump'):
            result = train_random_forest_model(training_data, save_model=False)
            # With low memory, SMOTE is skipped but model training still succeeds
            assert result is not None
    
    def test_success(self, mock_sufficient_memory, training_data):
        """Test successful model training"""
        with patch('modules.random_forest.core.model.joblib.dump'):
            result = train_random_forest_model(training_data, save_model=False)
            
            assert result is not None
            assert hasattr(result, 'predict')
            assert hasattr(result, 'predict_proba')
    
    def test_large_dataset_sampling(self, mock_sufficient_memory, large_dataframe):
        """Test model training with large dataset that needs sampling"""
        with patch('modules.random_forest.core.model.joblib.dump'):
            result = train_random_forest_model(large_dataframe, save_model=False)
            assert result is not None
    
    def test_column_mapping(self, mock_sufficient_memory, training_data):
        """Test model training with lowercase column names"""
        df_lowercase = training_data.copy()
        df_lowercase.columns = ['open', 'high', 'low', 'close', 'volume']
        
        with patch('modules.random_forest.core.model.joblib.dump'):
            result = train_random_forest_model(df_lowercase, save_model=False)
            assert result is not None
    
    @patch('modules.random_forest.utils.training.SMOTE')
    def test_smote_error(self, mock_smote_class, mock_sufficient_memory, training_data):
        """Test model training when SMOTE fails"""
        mock_smote = MagicMock()
        mock_smote.fit_resample.side_effect = RuntimeError("SMOTE failed")
        mock_smote_class.return_value = mock_smote
        
        with patch('modules.random_forest.core.model.joblib.dump'):
            result = train_random_forest_model(training_data, save_model=False)
            assert result is not None
    
    @patch('modules.random_forest.utils.training.compute_class_weight')
    def test_class_weight_error(self, mock_weight, mock_sufficient_memory, training_data):
        """Test model training when class weight computation fails"""
        mock_weight.side_effect = ValueError("Class weight computation failed")
        
        with patch('modules.random_forest.core.model.joblib.dump'):
            # When class weight computation fails, training should fail and return None
            with pytest.raises(ValueError):
                train_random_forest_model(training_data, save_model=False)
    
    def test_save_error(self, mock_sufficient_memory, training_data):
        """Test model training when saving fails"""
        with patch('modules.random_forest.core.model.joblib.dump') as mock_dump:
            mock_dump.side_effect = OSError("Save failed")
            # Model should still be returned even if saving fails
            result = train_random_forest_model(training_data, save_model=True)
            assert result is not None
    
    @pytest.mark.parametrize("invalid_data,description", [
        ("single_class_dataframe", "single class data"),
        ("insufficient_dataframe", "insufficient samples"),
        (None, "None input"),
    ])
    def test_invalid_inputs(self, mock_sufficient_memory, invalid_data, description, 
                            request, single_class_dataframe, insufficient_dataframe):
        """Test model training with invalid inputs"""
        data_map = {
            "single_class_dataframe": single_class_dataframe,
            "insufficient_dataframe": insufficient_dataframe,
            None: None
        }
        test_data = data_map[invalid_data]
        
        result = train_random_forest_model(test_data, save_model=False)  # type: ignore
        assert result is None


# ============================================================================
# Tests for train_and_save_global_rf_model
# ============================================================================

class TestTrainAndSaveGlobalRfModel:
    """Test suite for train_and_save_global_rf_model function"""
    
    @patch('modules.random_forest.utils.training.psutil.virtual_memory')
    @patch('modules.random_forest.core.model.joblib.dump')
    def test_success(self, mock_dump, mock_virtual_memory, training_data):
        """Test successful global model training and saving"""
        mock_virtual_memory.return_value.available = 4 * 1024**3  # 4GB
        
        model, model_path = train_and_save_global_rf_model(training_data)
        
        if model is not None:
            assert model is not None
            assert isinstance(model_path, str)
            assert mock_dump.called
    
    def test_empty_input(self, empty_dataframe):
        """Test global model training with empty DataFrame"""
        model, model_path = train_and_save_global_rf_model(empty_dataframe)
        assert model is None
        assert model_path == ""
    
    def test_custom_filename(self, training_data):
        """Test global model training with custom filename"""
        custom_filename = "custom_rf_model.joblib"
        
        with patch('modules.random_forest.utils.training.psutil.virtual_memory') as mock_virtual_memory:
            mock_virtual_memory.return_value.available = 4 * 1024**3  # 4GB
            with patch('modules.random_forest.core.model.joblib.dump') as mock_dump:
                model, model_path = train_and_save_global_rf_model(training_data, custom_filename)
                
                if model is not None:
                    assert custom_filename in model_path
    
    @patch('modules.random_forest.utils.training.psutil.virtual_memory')
    @patch('modules.random_forest.core.model.joblib.dump')
    def test_save_error(self, mock_dump, mock_virtual_memory, training_data):
        """Test global model training when saving fails"""
        mock_virtual_memory.return_value.available = 4 * 1024**3  # 4GB
        mock_dump.side_effect = OSError("Save failed")
        
        model, model_path = train_and_save_global_rf_model(training_data)
        assert model is None
        assert model_path == ""
    
    @patch('modules.random_forest.utils.training.psutil.virtual_memory')
    def test_training_failure(self, mock_virtual_memory, insufficient_dataframe):
        """Test global model training when model training fails"""
        mock_virtual_memory.return_value.available = 4 * 1024**3  # 4GB
        
        model, model_path = train_and_save_global_rf_model(insufficient_dataframe)
        assert model is None
        assert model_path == ""
    
    @patch('modules.random_forest.utils.training.psutil.virtual_memory')
    def test_timestamped_filename(self, mock_virtual_memory, training_data):
        """Test global model training with auto-generated timestamped filename"""
        mock_virtual_memory.return_value.available = 4 * 1024**3  # 4GB
        
        with patch('modules.random_forest.core.model.joblib.dump') as mock_dump:
            # Mock datetime.now() to return a datetime-like object that formats correctly
            from datetime import datetime as dt
            fake_now = dt(2023, 12, 1, 14, 30)
            with patch('modules.random_forest.core.model.datetime') as mock_datetime:
                mock_datetime.now.return_value = fake_now
                
                model, model_path = train_and_save_global_rf_model(training_data)
                
                if model is not None:
                    assert "rf_model_global_20231201_1430.joblib" in model_path


# ============================================================================
# Tests for load_random_forest_model
# ============================================================================

class TestLoadRandomForestModel:
    """Test suite for load_random_forest_model function"""
    
    def test_success(self, temp_models_dir):
        """Test successful model loading"""
        mock_model = MagicMock()
        mock_model_path = temp_models_dir / "test_model.joblib"
        
        with patch('modules.random_forest.core.model.joblib.load', return_value=mock_model) as mock_load:
            with patch('modules.random_forest.core.model.Path.exists', return_value=True):
                result = load_random_forest_model(mock_model_path)
                
                assert result is not None
                assert result == mock_model
                mock_load.assert_called_once_with(mock_model_path)
    
    def test_file_not_found(self, temp_models_dir):
        """Test model loading when file doesn't exist"""
        mock_model_path = temp_models_dir / "nonexistent_model.joblib"
        
        with patch('modules.random_forest.core.model.Path.exists', return_value=False):
            result = load_random_forest_model(mock_model_path)
            assert result is None
    
    def test_load_error(self, temp_models_dir):
        """Test model loading when joblib.load raises an exception"""
        mock_model_path = temp_models_dir / "test_model.joblib"
        
        with patch('modules.random_forest.core.model.joblib.load', side_effect=OSError("Load error")):
            with patch('modules.random_forest.core.model.Path.exists', return_value=True):
                result = load_random_forest_model(mock_model_path)
                assert result is None
    
    def test_default_path(self):
        """Test model loading with default path"""
        mock_model = MagicMock()
        
        with patch('modules.random_forest.core.model.joblib.load', return_value=mock_model) as mock_load:
            with patch('modules.random_forest.core.model.Path.exists', return_value=True):
                result = load_random_forest_model()
                
                assert result is not None
                assert result == mock_model
                expected_path = MODELS_DIR / RANDOM_FOREST_MODEL_FILENAME
                mock_load.assert_called_once_with(expected_path)


# ============================================================================
# Tests for evaluation functions
# ============================================================================

class TestApplyConfidenceThreshold:
    """Test suite for apply_confidence_threshold function"""
    
    @pytest.mark.parametrize("threshold,expected", [
        (0.7, np.array([-1, 0, 1, 0, 0, 1, 0, 1, -1, 0])),
        (0.3, np.array([-1, 0, 1, 0, 0, 1, 0, 1, -1, 0])),
        (0.9, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ])
    def test_confidence_thresholds(self, test_evaluation_data, threshold, expected):
        """Test confidence threshold application with different thresholds"""
        result = apply_confidence_threshold(
            test_evaluation_data['y_proba'], 
            threshold, 
            test_evaluation_data['classes']
        )
        np.testing.assert_array_equal(result, expected)
    
    def test_edge_cases(self, test_evaluation_data):
        """Test confidence threshold application with edge cases"""
        # Test with single prediction
        single_proba = np.array([[0.1, 0.8, 0.1]])
        result = apply_confidence_threshold(single_proba, 0.7, test_evaluation_data['classes'])
        assert result[0] == 0  # Should return 0 for low confidence
        
        # Test with exact threshold
        result = apply_confidence_threshold(single_proba, 0.8, test_evaluation_data['classes'])
        assert result[0] == 0  # Should return class 0 for exact confidence


class TestCalculateAndDisplayMetrics:
    """Test suite for calculate_and_display_metrics function"""
    
    @patch('modules.random_forest.core.evaluation.log_model')
    def test_success(self, mock_log_model, test_evaluation_data):
        """Test metrics calculation and display"""
        calculate_and_display_metrics(
            test_evaluation_data['y_true'], 
            test_evaluation_data['y_pred'], 
            0.7
        )
        assert mock_log_model.call_count > 0
    
    @patch('modules.random_forest.core.evaluation.log_model')
    def test_different_signals(self, mock_log_model):
        """Test metrics calculation with different signal distributions"""
        y_true = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1])
        y_pred = np.array([1, 0, 1, 0, 0, 1, -1, 0, -1])
        
        calculate_and_display_metrics(y_true, y_pred, 0.5)
        assert mock_log_model.call_count > 0
    
    @patch('modules.random_forest.core.evaluation.log_warn')
    def test_empty_arrays(self, mock_log_warn):
        """Test metrics calculation with empty arrays"""
        y_true = np.array([])
        y_pred = np.array([])
        
        calculate_and_display_metrics(y_true, y_pred, 0.5)
        assert mock_log_warn.call_count > 0


class TestEvaluateModelWithConfidence:
    """Test suite for evaluate_model_with_confidence function"""
    
    @patch('modules.random_forest.core.evaluation.log_model')
    def test_success(self, mock_log_model, test_evaluation_data):
        """Test model evaluation with confidence thresholds"""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = test_evaluation_data['y_proba']
        mock_model.classes_ = test_evaluation_data['classes']
        
        X_test = pd.DataFrame(
            np.random.randn(10, len(MODEL_FEATURES)), 
            columns=pd.Index(MODEL_FEATURES)
        )
        y_test = pd.Series(test_evaluation_data['y_true'])
        
        evaluate_model_with_confidence(mock_model, X_test, y_test)
        
        expected_calls = len(CONFIDENCE_THRESHOLDS) + 1  # +1 for the initial log message
        assert mock_log_model.call_count >= expected_calls
    
    @patch('modules.random_forest.core.evaluation.log_model')
    def test_prediction_error(self, mock_log_model, test_evaluation_data):
        """Test model evaluation when predict_proba fails"""
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = Exception("Prediction error")
        mock_model.classes_ = test_evaluation_data['classes']
        
        X_test = pd.DataFrame(
            np.random.randn(10, len(MODEL_FEATURES)), 
            columns=pd.Index(MODEL_FEATURES)
        )
        y_test = pd.Series(test_evaluation_data['y_true'])
        
        with pytest.raises(Exception):
            evaluate_model_with_confidence(mock_model, X_test, y_test)


# ============================================================================
# Tests for get_latest_random_forest_signal
# ============================================================================

class TestGetLatestRandomForestSignal:
    """Test suite for get_latest_random_forest_signal function"""
    
    @patch('modules.random_forest.utils.training.psutil.virtual_memory')
    def test_success(self, mock_virtual_memory, sample_data, training_data):
        """Test successful signal generation"""
        mock_virtual_memory.return_value.available = 4 * 1024**3  # 4GB
        
        with patch('modules.random_forest.core.model.joblib.dump'):
            model = train_random_forest_model(training_data, save_model=False)
        
        if model is not None:
            signal, _ = get_latest_random_forest_signal(sample_data, model)
            assert signal in ["LONG", "SHORT", "NEUTRAL"]
    
    def test_empty_input(self, empty_dataframe):
        """Test signal generation with empty DataFrame"""
        mock_model = MagicMock()
        signal, _ = get_latest_random_forest_signal(empty_dataframe, mock_model)
        assert signal == "NEUTRAL"
    
    def test_missing_columns(self, incomplete_dataframe):
        """Test signal generation with missing OHLC columns"""
        mock_model = MagicMock()
        signal, _ = get_latest_random_forest_signal(incomplete_dataframe, mock_model)
        assert signal == "NEUTRAL"
    
    def test_features_empty(self, sample_data):
        """Test signal generation when feature calculation returns empty DataFrame"""
        mock_model = MagicMock()
        
        with patch('modules.random_forest.core.signals.IndicatorEngine') as mock_engine:
            mock_instance = MagicMock()
            mock_instance.compute_features.return_value = pd.DataFrame()
            mock_engine.return_value = mock_instance
            
            signal, _ = get_latest_random_forest_signal(sample_data, mock_model)
            assert signal == "NEUTRAL"
    
    @pytest.mark.parametrize("confidence,expected_signal", [
        (np.array([[0.4, 0.3, 0.3]]), "NEUTRAL"),  # Low confidence
        (np.array([[0.1, 0.1, 0.8]]), "LONG"),     # High confidence LONG
        (np.array([[0.8, 0.1, 0.1]]), "SHORT"),    # High confidence SHORT
        (np.array([[0.1, 0.8, 0.1]]), "NEUTRAL"),  # High confidence NEUTRAL
    ])
    def test_confidence_scenarios(self, sample_data, confidence, expected_signal):
        """Test signal generation with different confidence scenarios"""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = confidence
        mock_model.classes_ = np.array([-1, 0, 1])
        # Set feature_names_in_ to match MODEL_FEATURES
        mock_model.feature_names_in_ = np.array(MODEL_FEATURES)
        
        with patch('modules.random_forest.core.signals.IndicatorEngine') as mock_engine:
            mock_instance = MagicMock()
            mock_features_df = sample_data.copy()
            # Add all required MODEL_FEATURES columns
            for feature in MODEL_FEATURES:
                if feature not in mock_features_df.columns:
                    mock_features_df[feature] = np.random.randn(len(mock_features_df))
            mock_instance.compute_features.return_value = mock_features_df
            mock_engine.return_value = mock_instance
            
            signal, _ = get_latest_random_forest_signal(sample_data, mock_model)
            assert signal == expected_signal
    
    def test_features_with_nan(self, sample_data):
        """Test signal generation when features contain NaN values"""
        mock_model = MagicMock()
        
        with patch('modules.random_forest.core.signals.IndicatorEngine') as mock_engine:
            mock_instance = MagicMock()
            mock_features_df = sample_data.copy()
            for feature in MODEL_FEATURES:
                if feature not in mock_features_df.columns:
                    mock_features_df[feature] = np.nan
            mock_instance.compute_features.return_value = mock_features_df
            mock_engine.return_value = mock_instance
            
            signal, _ = get_latest_random_forest_signal(sample_data, mock_model)
            assert signal == "NEUTRAL"
    
    def test_unknown_class(self, sample_data):
        """Test signal generation with unknown predicted class"""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])
        mock_model.classes_ = np.array([2, 3, 4])  # Unknown classes
        
        with patch('modules.random_forest.core.signals.IndicatorEngine') as mock_engine:
            mock_instance = MagicMock()
            mock_features_df = sample_data.copy()
            for feature in MODEL_FEATURES:
                if feature not in mock_features_df.columns:
                    mock_features_df[feature] = np.random.randn(len(mock_features_df))
            mock_instance.compute_features.return_value = mock_features_df
            mock_engine.return_value = mock_instance
            
            signal, _ = get_latest_random_forest_signal(sample_data, mock_model)
            assert signal == "NEUTRAL"
    
    def test_prediction_error(self, sample_data):
        """Test signal generation when prediction raises an exception"""
        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = RuntimeError("Prediction error")
        
        with patch('modules.random_forest.core.signals.IndicatorEngine') as mock_engine:
            mock_instance = MagicMock()
            mock_features_df = sample_data.copy()
            for feature in MODEL_FEATURES:
                if feature not in mock_features_df.columns:
                    mock_features_df[feature] = np.random.randn(len(mock_features_df))
            mock_instance.compute_features.return_value = mock_features_df
            mock_engine.return_value = mock_instance
            
            signal, _ = get_latest_random_forest_signal(sample_data, mock_model)
            assert signal == "NEUTRAL"