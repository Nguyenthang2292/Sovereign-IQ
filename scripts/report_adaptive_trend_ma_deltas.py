"""Report MA-series parity deltas between source and Rust serverless implementations.

This script compares raw MA series (base length + 8 diflen variations) for:
EMA, HMA, WMA, DEMA, LSMA, and KAMA.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.adaptive_trend.core.compute_moving_averages import set_of_moving_averages

VARIATION_LABELS = ["MA", "MA1", "MA2", "MA3", "MA4", "MA_1", "MA_2", "MA_3", "MA_4"]
MA_CONFIG_KEYS = [
    ("EMA", "ema_len"),
    ("HMA", "hull_len"),
    ("WMA", "wma_len"),
    ("DEMA", "dema_len"),
    ("LSMA", "lsma_len"),
    ("KAMA", "kama_len"),
]


@dataclass
class DeltaRow:
    scenario: str
    ma_type: str
    max_abs_diff: float
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report source-vs-rust MA parity deltas.")
    parser.add_argument(
        "--fixtures-dir",
        default="tests/parity_fixtures",
        help="Directory containing fixture JSON files.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Maximum allowed absolute difference for MA parity.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any row exceeds tolerance.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-variation deltas.",
    )
    return parser.parse_args()


def load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_series(values: list[Any]) -> pd.Series:
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    return pd.Series(arr, dtype="float64")


def max_abs_diff(a: pd.Series, b: pd.Series) -> float:
    av = np.asarray(a.values, dtype=np.float64)
    bv = np.asarray(b.values, dtype=np.float64)
    mask = np.isfinite(av) & np.isfinite(bv)
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(av[mask] - bv[mask])))


def compute_source_ma_outputs(fixture: dict[str, Any]) -> dict[str, list[pd.Series]]:
    prices = to_series(fixture["prices"])
    cfg = fixture["config"]
    robustness = str(cfg["robustness"])

    outputs: dict[str, list[pd.Series]] = {}
    for ma_name, len_key in MA_CONFIG_KEYS:
        base_length = int(cfg[len_key])
        ma_tuple = set_of_moving_averages(base_length, prices, ma_name, robustness=robustness)
        if ma_tuple is None:
            raise RuntimeError(f"source MA calculation failed for {ma_name} length={base_length}")
        outputs[ma_name] = [series.astype("float64") for series in ma_tuple]
    return outputs


def run_rust_ma_runner(fixture_path: Path) -> dict[str, Any]:
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--bin",
        "ma_parity_runner",
        "--",
        str(fixture_path.resolve()),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT / "modules" / "adaptive_trend_LTS_serverless",
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("ma_parity_runner produced empty output")
    return json.loads(stdout)


def evaluate_fixture(path: Path, tolerance: float, verbose: bool) -> list[DeltaRow]:
    fixture = load_fixture(path)
    scenario = str(fixture["scenario"])

    source_outputs = compute_source_ma_outputs(fixture)
    rust_payload = run_rust_ma_runner(path)
    rust_outputs = rust_payload["ma_outputs"]

    rows: list[DeltaRow] = []
    for ma_name, _ in MA_CONFIG_KEYS:
        source_bundle = source_outputs[ma_name]
        rust_bundle = [to_series(series) for series in rust_outputs[ma_name]]

        per_variation: dict[str, float] = {}
        max_delta = 0.0
        for idx, label in enumerate(VARIATION_LABELS):
            delta = max_abs_diff(source_bundle[idx], rust_bundle[idx])
            per_variation[label] = delta
            max_delta = max(max_delta, delta)

        if verbose:
            detail = ", ".join(f"{k}={v:.6g}" for k, v in per_variation.items())
            print(f"  {scenario}/{ma_name} deltas: {detail}")

        rows.append(
            DeltaRow(
                scenario=scenario,
                ma_type=ma_name,
                max_abs_diff=max_delta,
                passed=max_delta <= tolerance,
            )
        )

    return rows


def print_summary(rows: list[DeltaRow]) -> None:
    print("MA Parity Delta Report")
    print("======================")
    for row in rows:
        status = "PASS" if row.passed else "FAIL"
        print(
            f"[{status}] {row.scenario:12s} | {row.ma_type:4s} | "
            f"max_abs_diff={row.max_abs_diff:.6g}"
        )
    total = len(rows)
    failed = sum(1 for row in rows if not row.passed)
    print("----------------------")
    print(f"total={total} passed={total - failed} failed={failed}")


def main() -> int:
    args = parse_args()
    fixtures_dir = Path(args.fixtures_dir)
    fixture_paths = sorted(
        [path for path in fixtures_dir.glob("*.json") if path.is_file()],
        key=lambda p: p.name,
    )
    if not fixture_paths:
        print(f"no fixture json files found in: {fixtures_dir}")
        return 2

    all_rows: list[DeltaRow] = []
    for path in fixture_paths:
        all_rows.extend(evaluate_fixture(path, args.tolerance, args.verbose))

    print_summary(all_rows)
    has_failures = any(not row.passed for row in all_rows)
    if has_failures and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
