from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from modules.auto_trade.execution.adaptive_close_calculator import AdaptiveCloseCalculator
from modules.detect_regime_change.regime_lambda_client import RegimeDurationResult


class MockSettings:
    def get(self, key, default=None):
        if key == "auto_close":
            return {
                "max_duration_hours": 4.0,
                "timeframe": "15m",
                "adaptive": {
                    "enabled": True,
                    "min_duration_hours": 1.0,
                    "max_duration_hours": 12.0,
                    "lookback_days": 60,
                    "use_lambda": True,
                    "lambda_endpoint": "http://dummy",
                    "lambda_timeout_seconds": 3.0,
                },
            }
        return default


@patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeLambdaClient")
@patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer")
def test_lambda_success(mock_local_analyzer, mock_lambda_client):
    calc = AdaptiveCloseCalculator(settings_manager=MockSettings())
    calc._fetch_ohlcv = MagicMock(return_value=[1, 2, 3] * 50)  # Mock 150 items length

    mock_lambda_instance = MagicMock()
    mock_lambda_instance.invoke.return_value = RegimeDurationResult(
        is_valid=True,
        recommended_duration_hours=6.0,
        pelt_avg_duration_hours=6.0,
        hmm_next_state_duration_hours=6.0,
    )
    mock_lambda_client.return_value = mock_lambda_instance

    result = calc.compute_adaptive_deadline_with_meta("BTC/USDT", datetime.now(timezone.utc), ohlcv_df=[...] * 100)

    assert result.source == "adaptive"
    assert result.duration_hours == 6.0
    mock_local_analyzer.assert_not_called()


@patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeLambdaClient")
@patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer")
def test_lambda_timeout_fallback_to_local_success(mock_local_analyzer, mock_lambda_client):
    calc = AdaptiveCloseCalculator(settings_manager=MockSettings())

    mock_lambda_instance = MagicMock()
    mock_lambda_instance.invoke.return_value = None  # Simulate timeout or error
    mock_lambda_client.return_value = mock_lambda_instance

    mock_local_instance = MagicMock()
    mock_local_instance.analyze.return_value = MagicMock(
        is_valid=True,
        recommended_duration_hours=5.0,
        pelt_avg_duration_hours=5.0,
        hmm_next_state_duration_hours=5.0,
    )
    mock_local_analyzer.return_value = mock_local_instance

    result = calc.compute_adaptive_deadline_with_meta("BTC/USDT", datetime.now(timezone.utc), ohlcv_df=[...] * 100)

    assert result.source == "adaptive"
    assert result.duration_hours == 5.0
    mock_local_instance.analyze.assert_called_once()


@patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeLambdaClient")
@patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer")
def test_lambda_invalid_result_fallback_to_local_success(mock_local_analyzer, mock_lambda_client):
    calc = AdaptiveCloseCalculator(settings_manager=MockSettings())

    mock_lambda_instance = MagicMock()
    mock_lambda_instance.invoke.return_value = RegimeDurationResult(
        is_valid=False,
        recommended_duration_hours=None,
        pelt_avg_duration_hours=None,
        hmm_next_state_duration_hours=None,
        error="Lambda analysis invalid",
    )
    mock_lambda_client.return_value = mock_lambda_instance

    mock_local_instance = MagicMock()
    mock_local_instance.analyze.return_value = MagicMock(
        is_valid=True,
        recommended_duration_hours=4.5,
        pelt_avg_duration_hours=4.2,
        hmm_next_state_duration_hours=4.8,
    )
    mock_local_analyzer.return_value = mock_local_instance

    result = calc.compute_adaptive_deadline_with_meta("BTC/USDT", datetime.now(timezone.utc), ohlcv_df=[...] * 100)

    assert result.source == "adaptive"
    assert result.duration_hours == 4.5
    mock_local_instance.analyze.assert_called_once()


@patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeLambdaClient")
@patch("modules.auto_trade.execution.adaptive_close_calculator.RegimeDurationAnalyzer")
def test_lambda_and_local_fail_static_fallback(mock_local_analyzer, mock_lambda_client):
    calc = AdaptiveCloseCalculator(settings_manager=MockSettings())

    mock_lambda_instance = MagicMock()
    mock_lambda_instance.invoke.return_value = None
    mock_lambda_client.return_value = mock_lambda_instance

    mock_local_instance = MagicMock()
    mock_local_instance.analyze.return_value = MagicMock(is_valid=False, error="Analysis failed")
    mock_local_analyzer.return_value = mock_local_instance

    result = calc.compute_adaptive_deadline_with_meta("BTC/USDT", datetime.now(timezone.utc), ohlcv_df=[...] * 100)

    assert result.source == "adaptive_fallback"
    assert result.duration_hours == 4.0
