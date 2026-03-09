"""
detect_regime_change/regime_lambda_client.py
==========================================
HTTP client for invoking AWS Lambda regime analysis.

Offloads regime duration analysis to AWS Lambda to reduce local CPU usage.
Implements graceful fallback to local analyzer when Lambda fails or times out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from modules.common.ui.logging import log_error, log_info, log_warn

# Type imports for DataFrame handling
try:
    import pandas as pd
    DataFrame = pd.DataFrame
except ImportError:
    DataFrame = Any


@dataclass
class RegimeDurationResult:
    """Result from regime duration analysis."""
    is_valid: bool
    recommended_duration_hours: Optional[float]
    pelt_avg_duration_hours: Optional[float]
    hmm_next_state_duration_hours: Optional[float]
    error: Optional[str] = None


class RegimeLambdaClient:
    """
    HTTP client for invoking AWS Lambda regime analysis.

    Usage:
        client = RegimeLambdaClient(
            endpoint="https://lambda.amazonaws.com/regime-analysis",
            timeout_seconds=3.0,
        )
        result = client.invoke(ohlcv_df, symbol="BTC/USDT", config={...})

        if result is None:
            # Fallback to local analyzer
            pass
    """

    def __init__(
        self,
        endpoint: str,
        timeout_seconds: float = 3.0,
    ):
        """
        Initialize Lambda client.

        Args:
            endpoint: Lambda function URL or API Gateway endpoint
            timeout_seconds: Request timeout in seconds (default: 3.0)
        """
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def invoke(
        self,
        ohlcv_df: DataFrame,
        symbol: str,
        config: Dict[str, Any],
    ) -> Optional[RegimeDurationResult]:
        """
        Invoke Lambda with OHLCV data for regime analysis.

        Args:
            ohlcv_df: OHLCV DataFrame with timestamp, open, high, low, close, volume
            symbol: Trading symbol (e.g., "BTC/USDT")
            config: Configuration dict with lookback_days, timeframe, etc.

        Returns:
            RegimeDurationResult if successful, None if failed/timed out.
            Does NOT raise exceptions - returns None on any error.
        """
        if not self.endpoint:
            log_warn("RegimeLambdaClient: No endpoint configured, skipping Lambda")
            return None

        try:
            # Serialize OHLCV data
            payload = self._build_payload(ohlcv_df, symbol, config)

            # Invoke Lambda
            log_info(
                f"RegimeLambdaClient: Invoking Lambda for {symbol} "
                f"(timeout={self.timeout_seconds}s)"
            )

            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout_seconds,
                headers={"Content-Type": "application/json"},
            )

            # Check HTTP status
            if response.status_code != 200:
                log_warn(
                    f"RegimeLambdaClient: HTTP {response.status_code} from Lambda, "
                    f"will fallback to local analyzer"
                )
                return None

            # Deserialize response
            result = self._deserialize_result(response.json())
            log_info(f"RegimeLambdaClient: Lambda analysis successful for {symbol}")
            return result

        except requests.exceptions.Timeout:
            log_warn(
                f"RegimeLambdaClient: Timeout after {self.timeout_seconds}s, "
                f"will fallback to local analyzer"
            )
            return None

        except requests.exceptions.RequestException as e:
            log_warn(f"RegimeLambdaClient: Request failed: {e}, will fallback")
            return None

        except Exception as e:
            log_error(f"RegimeLambdaClient: Unexpected error: {e}")
            return None

    def _build_payload(
        self,
        ohlcv_df: DataFrame,
        symbol: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build JSON payload for Lambda request.

        Args:
            ohlcv_df: OHLCV DataFrame
            symbol: Trading symbol
            config: Configuration dict

        Returns:
            Dict ready for JSON serialization.
        """
        return {
            "symbol": symbol,
            "timeframe": config.get("timeframe", "15m"),
            "lookback_days": config.get("lookback_days", 60),
            "ohlcv": self._serialize_ohlcv(ohlcv_df),
            "config": {
                "pelt_model": config.get("pelt_model", "l2"),
                "pelt_min_segment": config.get("pelt_min_segment", 10),
                "hmm_train_ratio": config.get("hmm_train_ratio", 0.8),
                "hmm_high_confidence_threshold": config.get("hmm_high_confidence_threshold", 0.7),
            },
        }

    def _serialize_ohlcv(self, df: DataFrame) -> Dict[str, Any]:
        """
        Serialize OHLCV DataFrame to JSON-compatible dict.

        Efficient serialization: only send essential columns as arrays
        to minimize payload size and Lambda cold start time.

        Args:
            df: DataFrame with timestamp, open, high, low, close, volume

        Returns:
            Dict with timestamps array and ohlcv arrays.
        """
        # Handle both pandas DataFrame and dict
        if hasattr(df, 'to_dict'):
            # Pandas DataFrame
            records = df.reset_index().to_dict('records')
        else:
            # Already a dict or list
            records = df if isinstance(df, list) else [df]

        # Extract columns
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        for record in records:
            # Handle both dict and DataFrame row
            if hasattr(record, 'to_dict'):
                record = record.to_dict()

            ts = record.get('timestamp') or record.get('index')
            if hasattr(ts, 'isoformat'):
                ts = ts.isoformat()
            timestamps.append(ts)

            opens.append(float(record.get('open', 0)))
            highs.append(float(record.get('high', 0)))
            lows.append(float(record.get('low', 0)))
            closes.append(float(record.get('close', 0)))
            volumes.append(float(record.get('volume', 0)))

        return {
            "timestamps": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }

    def _deserialize_result(self, data: Dict[str, Any]) -> RegimeDurationResult:
        """
        Deserialize Lambda response JSON into RegimeDurationResult.

        Args:
            data: Response JSON dict from Lambda

        Returns:
            RegimeDurationResult.
        """
        return RegimeDurationResult(
            is_valid=data.get("is_valid", False),
            recommended_duration_hours=data.get("recommended_duration_hours"),
            pelt_avg_duration_hours=data.get("pelt_avg_duration_hours"),
            hmm_next_state_duration_hours=data.get("hmm_next_state_duration_hours"),
            error=data.get("error"),
        )
