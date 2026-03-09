from unittest.mock import MagicMock, patch

import pandas as pd
from requests.exceptions import RequestException, Timeout

from modules.detect_regime_change.regime_lambda_client import RegimeLambdaClient


def test_serialize_ohlcv():
    client = RegimeLambdaClient(endpoint="http://dummy")
    df = pd.DataFrame(
        {
            "timestamp": ["2026-03-09T00:00:00Z", "2026-03-09T00:15:00Z"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, 12.0],
        }
    )

    payload = client._serialize_ohlcv(df)
    assert payload["timestamps"] == ["2026-03-09T00:00:00Z", "2026-03-09T00:15:00Z"]
    assert payload["open"] == [100.0, 101.0]
    assert payload["close"] == [101.0, 102.0]


def test_build_payload_includes_config_block():
    client = RegimeLambdaClient(endpoint="http://dummy")
    df = pd.DataFrame(
        {
            "timestamp": ["2026-03-09T00:00:00Z"],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [10.0],
        }
    )

    payload = client._build_payload(
        df,
        "BTC/USDT",
        {
            "timeframe": "15m",
            "lookback_days": 60,
            "pelt_model": "normal",
            "pelt_min_segment": 12,
            "hmm_train_ratio": 0.75,
            "hmm_high_confidence_threshold": 0.8,
        },
    )

    assert payload["config"]["pelt_model"] == "normal"
    assert payload["config"]["pelt_min_segment"] == 12
    assert payload["config"]["hmm_train_ratio"] == 0.75
    assert payload["config"]["hmm_high_confidence_threshold"] == 0.8


def test_deserialize_result():
    client = RegimeLambdaClient(endpoint="http://dummy")
    data = {
        "is_valid": True,
        "pelt_avg_duration_hours": 4.5,
        "hmm_next_state_duration_hours": 2.2,
        "recommended_duration_hours": 4.0,
    }
    result = client._deserialize_result(data)
    assert result.is_valid is True
    assert result.pelt_avg_duration_hours == 4.5
    assert result.hmm_next_state_duration_hours == 2.2
    assert result.recommended_duration_hours == 4.0


@patch("modules.detect_regime_change.regime_lambda_client.requests.post")
def test_timeout_handling(mock_post):
    mock_post.side_effect = Timeout("Timed out")
    client = RegimeLambdaClient(endpoint="http://dummy")
    result = client.invoke(pd.DataFrame(), "BTC/USDT", {})
    assert result is None


@patch("modules.detect_regime_change.regime_lambda_client.requests.post")
def test_http_error_handling(mock_post):
    mock_post.side_effect = RequestException("HTTP Error")
    client = RegimeLambdaClient(endpoint="http://dummy")
    result = client.invoke(pd.DataFrame(), "BTC/USDT", {})
    assert result is None


@patch("modules.detect_regime_change.regime_lambda_client.requests.post")
def test_500_response_handling(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_post.return_value = mock_response
    client = RegimeLambdaClient(endpoint="http://dummy")
    result = client.invoke(pd.DataFrame(), "BTC/USDT", {})
    assert result is None
