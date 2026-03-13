"""Generate deterministic parity fixtures for adaptive trend modules.

One-command usage:
    python scripts/generate_adaptive_trend_parity_fixtures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Dict

import numpy as np
import pandas as pd

# Ensure repository root is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.adaptive_trend.core.compute_atc_signals import compute_atc_signals


FixtureGenerator = Callable[[int, int], pd.Series]


DEFAULT_CONFIG = {
    "ema_len": 28,
    "hull_len": 28,
    "wma_len": 28,
    "dema_len": 28,
    "lsma_len": 28,
    "kama_len": 28,
    "ema_w": 1.0,
    "hma_w": 1.0,
    "wma_w": 1.0,
    "dema_w": 1.0,
    "lsma_w": 1.0,
    "kama_w": 1.0,
    "robustness": "Medium",
    "La": 0.02,
    "De": 0.03,
    "cutout": 0,
    "long_threshold": 0.1,
    "short_threshold": -0.1,
    "strategy_mode": False,
}


def _to_json_series(series: pd.Series) -> list[float | None]:
    out: list[float | None] = []
    for value in series.to_list():
        if pd.isna(value):
            out.append(None)
        else:
            out.append(float(value))
    return out


def _latest_classification(avg: pd.Series) -> str:
    if avg.empty:
        return "NEUTRAL"
    value = avg.iloc[-1]
    if pd.isna(value):
        return "NEUTRAL"
    if value > 0:
        return "LONG"
    if value < 0:
        return "SHORT"
    return "NEUTRAL"


def _trend_up(length: int, seed: int) -> pd.Series:
    idx = np.arange(length, dtype=np.float64)
    prices = 100.0 + (0.25 * idx) + (1.75 * np.sin(idx / 9.0))
    return pd.Series(prices, dtype="float64")


def _trend_down(length: int, seed: int) -> pd.Series:
    idx = np.arange(length, dtype=np.float64)
    prices = 180.0 - (0.23 * idx) + (1.5 * np.sin(idx / 11.0))
    return pd.Series(prices, dtype="float64")


def _sideways(length: int, seed: int) -> pd.Series:
    idx = np.arange(length, dtype=np.float64)
    prices = 125.0 + (2.2 * np.sin(idx / 7.0)) + (0.8 * np.sin(idx / 19.0))
    return pd.Series(prices, dtype="float64")


def _noisy(length: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.02, scale=1.35, size=length)
    prices = 110.0 + np.cumsum(steps)
    return pd.Series(prices, dtype="float64")


def _nan_gaps(length: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = np.arange(length, dtype=np.float64)
    prices = 105.0 + (0.11 * idx) + (1.9 * np.sin(idx / 8.5)) + rng.normal(0.0, 0.35, length)
    series = pd.Series(prices, dtype="float64")
    series.iloc[30:36] = np.nan
    series.iloc[90] = np.nan
    series.iloc[150:155] = np.nan
    series.iloc[210] = np.nan
    return series


SCENARIOS: Dict[str, tuple[int, FixtureGenerator]] = {
    "trend_up": (1001, _trend_up),
    "trend_down": (1002, _trend_down),
    "sideways": (1003, _sideways),
    "noisy": (1004, _noisy),
    "nan_gaps": (1005, _nan_gaps),
}


def build_fixture_payload(name: str, prices: pd.Series, seed: int) -> dict:
    atc = compute_atc_signals(prices=prices, src=prices, **DEFAULT_CONFIG)

    expected = {
        "EMA_Signal": _to_json_series(atc["EMA_Signal"]),
        "HMA_Signal": _to_json_series(atc["HMA_Signal"]),
        "WMA_Signal": _to_json_series(atc["WMA_Signal"]),
        "DEMA_Signal": _to_json_series(atc["DEMA_Signal"]),
        "LSMA_Signal": _to_json_series(atc["LSMA_Signal"]),
        "KAMA_Signal": _to_json_series(atc["KAMA_Signal"]),
        "Average_Signal": _to_json_series(atc["Average_Signal"]),
        "classification": _latest_classification(atc["Average_Signal"]),
    }

    return {
        "scenario": name,
        "seed": seed,
        "length": int(len(prices)),
        "config": DEFAULT_CONFIG,
        "prices": _to_json_series(prices),
        "expected": expected,
    }


def generate_fixtures(output_dir: Path, length: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (seed, generator) in SCENARIOS.items():
        prices = generator(length, seed)
        payload = build_fixture_payload(name=name, prices=prices, seed=seed)
        fixture_path = output_dir / f"{name}.json"
        fixture_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"generated: {fixture_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adaptive trend parity fixtures.")
    parser.add_argument(
        "--output-dir",
        default="tests/parity_fixtures",
        help="Output directory for fixture json files.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=260,
        help="Number of bars per generated scenario.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_fixtures(output_dir=Path(args.output_dir), length=args.length)


if __name__ == "__main__":
    main()
