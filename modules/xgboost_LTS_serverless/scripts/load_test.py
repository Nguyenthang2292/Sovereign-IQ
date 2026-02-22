"""Simple concurrent load test for deployed predict endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import time
from typing import Any

import requests

from lambda_demo import build_payload


def worker(endpoint: str, payload: dict[str, Any]) -> tuple[bool, float]:
    start = time.perf_counter()
    try:
        response = requests.post(endpoint, json=payload, timeout=30)
        ok = response.status_code == 200
    except Exception:
        ok = False
    duration_ms = (time.perf_counter() - start) * 1000
    return ok, duration_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Load test XGBoost predict endpoint")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    payload = build_payload(args.symbol, args.timeframe, args.model_version)

    durations: list[float] = []
    success = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker, args.endpoint, payload) for _ in range(args.requests)]
        for future in concurrent.futures.as_completed(futures):
            ok, duration_ms = future.result()
            durations.append(duration_ms)
            if ok:
                success += 1

    durations_sorted = sorted(durations)
    p50 = statistics.median(durations_sorted)
    p95_idx = max(0, min(len(durations_sorted) - 1, int(len(durations_sorted) * 0.95) - 1))
    p95 = durations_sorted[p95_idx]
    failure = args.requests - success

    print(f"requests={args.requests} concurrency={args.concurrency}")
    print(f"success={success} failure={failure} success_rate={success / args.requests:.2%}")
    print(f"latency_ms p50={p50:.2f} p95={p95:.2f}")

    return 0 if failure == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
