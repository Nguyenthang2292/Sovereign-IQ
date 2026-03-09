"""
Smoke test script for regime analysis Lambda endpoint.

Usage:
  python scripts/test_regime_lambda.py --endpoint https://.../analyze --symbol BTC/USDT
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import requests


def _build_synthetic_ohlcv(points: int = 160) -> dict[str, list[Any]]:
    now = datetime.now(timezone.utc)
    timestamps: list[str] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []

    price = 100.0
    for i in range(points):
        ts = now - timedelta(minutes=15 * (points - i))
        # Two simple regimes for a realistic-ish signal.
        drift = 0.05 if i < points // 2 else 0.35
        price = max(1.0, price + drift)

        o = price - 0.1
        h = price + 0.4
        l = price - 0.4
        c = price
        v = 100.0 + i

        timestamps.append(ts.isoformat())
        opens.append(round(o, 6))
        highs.append(round(h, 6))
        lows.append(round(l, 6))
        closes.append(round(c, 6))
        volumes.append(round(v, 6))

    return {
        "timestamps": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for regime Lambda endpoint")
    parser.add_argument("--endpoint", required=True, help="Lambda Function URL or API Gateway endpoint")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout seconds")
    args = parser.parse_args()

    payload = {
        "symbol": args.symbol,
        "timeframe": "15m",
        "lookback_days": 60,
        "ohlcv": _build_synthetic_ohlcv(),
        "config": {
            "pelt_model": "l2",
            "pelt_min_segment": 10,
            "hmm_train_ratio": 0.8,
            "hmm_high_confidence_threshold": 0.7,
        },
    }

    try:
        response = requests.post(
            args.endpoint,
            json=payload,
            timeout=args.timeout,
            headers={"Content-Type": "application/json"},
        )
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return 2

    print(f"HTTP {response.status_code}")
    if response.status_code != 200:
        print(response.text)
        return 1

    try:
        body = response.json()
    except ValueError:
        print("Response is not valid JSON")
        print(response.text)
        return 1

    print(json.dumps(body, indent=2, ensure_ascii=True))

    is_valid = bool(body.get("is_valid", False))
    recommended = body.get("recommended_duration_hours")
    if not is_valid or recommended is None:
        print("Lambda returned invalid regime analysis")
        return 1

    print(f"Smoke test OK: recommended_duration_hours={recommended}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
