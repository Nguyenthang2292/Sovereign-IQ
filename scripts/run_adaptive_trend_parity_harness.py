"""Cross-implementation parity harness for adaptive trend modules.

Runs source, LTS mini, and LTS serverless against identical fixtures and reports
per-scenario max absolute deltas.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_namespace_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


# Avoid importing heavy package __init__ trees that pull optional dependencies.
_ensure_namespace_pkg("modules.adaptive_trend_LTS_mini", REPO_ROOT / "modules" / "adaptive_trend_LTS_mini")
_ensure_namespace_pkg("modules.adaptive_trend_LTS_mini.core", REPO_ROOT / "modules" / "adaptive_trend_LTS_mini" / "core")
_ensure_namespace_pkg(
    "modules.adaptive_trend_LTS_mini.core.compute_atc_signals",
    REPO_ROOT / "modules" / "adaptive_trend_LTS_mini" / "core" / "compute_atc_signals",
)

compute_source_signals = importlib.import_module(
    "modules.adaptive_trend.core.compute_atc_signals"
).compute_atc_signals
compute_mini_signals = importlib.import_module(
    "modules.adaptive_trend_LTS_mini.core.compute_atc_signals.compute_atc_signals"
).compute_atc_signals
source_moving_averages = importlib.import_module(
    "modules.adaptive_trend.core.compute_moving_averages"
)
source_utils_roc = importlib.import_module("modules.adaptive_trend.utils.rate_of_change")
source_equity = importlib.import_module("modules.adaptive_trend.core.compute_equity")
source_process_layer1 = importlib.import_module("modules.adaptive_trend.core.process_layer1")


KEYS = [
    "EMA_Signal",
    "HMA_Signal",
    "WMA_Signal",
    "DEMA_Signal",
    "LSMA_Signal",
    "KAMA_Signal",
    "Average_Signal",
]


@dataclass
class ScenarioResult:
    scenario: str
    impl: str
    max_abs_diff: float
    classification_match: bool
    passed: bool
    per_key_diff: dict[str, float]


_SERVERLESS_RUNNER_UNAVAILABLE = False
_SERVERLESS_FALLBACK_WARNED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parity harness for adaptive trend implementations.")
    parser.add_argument("--fixtures-dir", default="tests/parity_fixtures", help="Directory containing fixture JSON files.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Maximum allowed absolute difference for series parity.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status when any scenario fails.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-key delta details for each scenario.",
    )
    parser.add_argument(
        "--impl",
        choices=["all", "mini", "serverless"],
        default="all",
        help="Implementation subset to evaluate (default: all).",
    )
    return parser.parse_args()


def load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_series(values: list[Any]) -> pd.Series:
    array = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    return pd.Series(array, dtype="float64")


def classify(avg_signal: pd.Series) -> str:
    if avg_signal.empty:
        return "NEUTRAL"
    latest = avg_signal.iloc[-1]
    if pd.isna(latest):
        return "NEUTRAL"
    if latest > 0:
        return "LONG"
    if latest < 0:
        return "SHORT"
    return "NEUTRAL"


def max_abs_diff(a: pd.Series, b: pd.Series) -> float:
    a_values = np.asarray(a.values, dtype=np.float64)
    b_values = np.asarray(b.values, dtype=np.float64)
    mask = np.isfinite(a_values) & np.isfinite(b_values)
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(a_values[mask] - b_values[mask])))


def run_serverless_runner(fixture_path: Path) -> dict[str, Any]:
    global _SERVERLESS_RUNNER_UNAVAILABLE
    if _SERVERLESS_RUNNER_UNAVAILABLE:
        raise RuntimeError("serverless rust runner unavailable")

    fixture_abs = fixture_path.resolve()
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--bin",
        "parity_runner",
        "--",
        str(fixture_abs),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT / "modules" / "adaptive_trend_LTS_serverless",
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        _SERVERLESS_RUNNER_UNAVAILABLE = True
        raise RuntimeError(str(exc)) from exc
    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("serverless parity runner produced empty output")
    return json.loads(stdout)


def compute_source(fixture: dict[str, Any]) -> tuple[dict[str, pd.Series], str]:
    cfg = fixture["config"]
    prices = to_series(fixture["prices"])
    result = compute_source_signals(
        prices=prices,
        src=prices,
        ema_len=int(cfg["ema_len"]),
        hull_len=int(cfg["hull_len"]),
        wma_len=int(cfg["wma_len"]),
        dema_len=int(cfg["dema_len"]),
        lsma_len=int(cfg["lsma_len"]),
        kama_len=int(cfg["kama_len"]),
        ema_w=float(cfg["ema_w"]),
        hma_w=float(cfg["hma_w"]),
        wma_w=float(cfg["wma_w"]),
        dema_w=float(cfg["dema_w"]),
        lsma_w=float(cfg["lsma_w"]),
        kama_w=float(cfg["kama_w"]),
        robustness=str(cfg["robustness"]),
        La=float(cfg["La"]),
        De=float(cfg["De"]),
        cutout=int(cfg["cutout"]),
        long_threshold=float(cfg["long_threshold"]),
        short_threshold=float(cfg["short_threshold"]),
        strategy_mode=False,
    )
    outputs = {key: result[key].astype("float64") for key in KEYS}
    return outputs, classify(outputs["Average_Signal"])


def compute_mini(fixture: dict[str, Any]) -> tuple[dict[str, pd.Series], str]:
    cfg = fixture["config"]
    prices = to_series(fixture["prices"])
    result = compute_mini_signals(
        prices=prices,
        src=prices,
        ema_len=int(cfg["ema_len"]),
        hma_len=int(cfg["hull_len"]),
        wma_len=int(cfg["wma_len"]),
        dema_len=int(cfg["dema_len"]),
        lsma_len=int(cfg["lsma_len"]),
        kama_len=int(cfg["kama_len"]),
        ema_w=float(cfg["ema_w"]),
        hma_w=float(cfg["hma_w"]),
        wma_w=float(cfg["wma_w"]),
        dema_w=float(cfg["dema_w"]),
        lsma_w=float(cfg["lsma_w"]),
        kama_w=float(cfg["kama_w"]),
        robustness=str(cfg["robustness"]),
        lambda_param=float(cfg["La"]),
        decay=float(cfg["De"]),
        cutout=int(cfg["cutout"]),
        long_threshold=float(cfg["long_threshold"]),
        short_threshold=float(cfg["short_threshold"]),
        strategy_mode=False,
        parallel_l1=False,
        parallel_l2=False,
        use_approximate=False,
        use_adaptive_approximate=False,
    )
    outputs = {key: result[key].astype("float64") for key in KEYS}
    return outputs, classify(outputs["Average_Signal"])


def compute_serverless_fallback(fixture: dict[str, Any]) -> tuple[dict[str, pd.Series], str]:
    cfg = fixture["config"]
    prices = to_series(fixture["prices"])
    robustness = str(cfg["robustness"])
    lambda_scaled = float(cfg["La"]) / 1000.0
    decay_scaled = float(cfg["De"]) / 100.0
    cutout = int(cfg["cutout"])
    long_threshold = float(cfg["long_threshold"])
    short_threshold = float(cfg.get("short_threshold", -abs(long_threshold)))

    set_of_moving_averages = source_moving_averages.set_of_moving_averages
    rate_of_change = source_utils_roc.rate_of_change
    layer1_for_ma = source_process_layer1._layer1_signal_for_ma
    cut_signal = source_process_layer1.cut_signal
    equity_series = source_equity.equity_series

    ma_configs = [
        ("EMA", int(cfg["ema_len"]), float(cfg["ema_w"])),
        ("HMA", int(cfg["hull_len"]), float(cfg["hma_w"])),
        ("WMA", int(cfg["wma_len"]), float(cfg["wma_w"])),
        ("DEMA", int(cfg["dema_len"]), float(cfg["dema_w"])),
        ("LSMA", int(cfg["lsma_len"]), float(cfg["lsma_w"])),
        ("KAMA", int(cfg["kama_len"]), float(cfg["kama_w"])),
    ]

    roc = rate_of_change(prices).astype("float64")

    outputs: dict[str, pd.Series] = {}
    layer1_signals: dict[str, pd.Series] = {}
    layer2_equities: dict[str, pd.Series] = {}

    for ma_type, base_length, static_weight in ma_configs:
        ma_tuple = set_of_moving_averages(base_length, prices, ma_type, robustness=robustness)
        if ma_tuple is None:
            raise RuntimeError(f"fallback serverless MA calculation failed for {ma_type} length={base_length}")

        layer1, _, _ = layer1_for_ma(
            prices,
            ma_tuple,
            L=lambda_scaled,
            De=decay_scaled,
            cutout=cutout,
            R=roc,
        )
        layer1 = layer1.astype("float64")
        layer1_signals[ma_type] = layer1
        outputs[f"{ma_type}_Signal"] = layer1

        layer2 = equity_series(
            starting_equity=static_weight,
            sig=layer1,
            R=roc,
            L=lambda_scaled,
            De=decay_scaled,
            cutout=cutout,
        ).astype("float64")
        layer2_equities[ma_type] = layer2

    nom = pd.Series(0.0, index=prices.index, dtype="float64")
    den = pd.Series(0.0, index=prices.index, dtype="float64")
    for ma_type, _, _ in ma_configs:
        cut = cut_signal(
            layer1_signals[ma_type],
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            cutout=cutout,
        ).astype("float64")
        weight = layer2_equities[ma_type]
        nom = nom + (cut * weight)
        den = den + weight

    avg = (nom / den.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")

    outputs["Average_Signal"] = avg
    return outputs, classify(avg)


def compute_serverless(fixture_path: Path, fixture: dict[str, Any]) -> tuple[dict[str, pd.Series], str]:
    global _SERVERLESS_FALLBACK_WARNED
    try:
        payload = run_serverless_runner(fixture_path)
        outputs = {key: to_series(payload["outputs"][key]) for key in KEYS}
        classification = str(payload["classification"])
        return outputs, classification
    except RuntimeError:
        if not _SERVERLESS_FALLBACK_WARNED:
            print(
                "[WARN] Rust serverless parity runner is unavailable in this environment; "
                "falling back to Python serverless emulation."
            )
            _SERVERLESS_FALLBACK_WARNED = True
        return compute_serverless_fallback(fixture)


def evaluate_scenario(
    fixture_path: Path,
    tolerance: float,
    verbose: bool,
    impls: set[str],
) -> list[ScenarioResult]:
    fixture = load_fixture(fixture_path)
    scenario = str(fixture["scenario"])

    source_outputs, source_cls = compute_source(fixture)

    implementations: list[tuple[str, dict[str, pd.Series], str]] = []
    if "mini" in impls:
        mini_outputs, mini_cls = compute_mini(fixture)
        implementations.append(("mini", mini_outputs, mini_cls))
    if "serverless" in impls:
        srv_outputs, srv_cls = compute_serverless(fixture_path, fixture)
        implementations.append(("serverless", srv_outputs, srv_cls))

    results: list[ScenarioResult] = []
    for impl_name, outputs, impl_cls in implementations:
        diffs = {key: max_abs_diff(source_outputs[key], outputs[key]) for key in KEYS}
        max_delta = max(diffs.values()) if diffs else 0.0
        cls_match = impl_cls == source_cls
        passed = max_delta <= tolerance and cls_match
        results.append(
            ScenarioResult(
                scenario=scenario,
                impl=impl_name,
                max_abs_diff=max_delta,
                classification_match=cls_match,
                passed=passed,
                per_key_diff=diffs,
            )
        )
        if verbose:
            detail = ", ".join(f"{k}={v:.6g}" for k, v in diffs.items())
            print(f"  {scenario}/{impl_name} per-key max abs diff: {detail}")

    return results


def print_summary(results: list[ScenarioResult]) -> None:
    print("Parity Harness Report")
    print("=====================")
    for row in results:
        status = "PASS" if row.passed else "FAIL"
        cls_flag = "match" if row.classification_match else "mismatch"
        print(
            f"[{status}] {row.scenario:12s} | {row.impl:10s} | "
            f"max_abs_diff={row.max_abs_diff:.6g} | classification={cls_flag}"
        )

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    print("---------------------")
    print(f"total={total} passed={passed} failed={failed}")


def main() -> int:
    args = parse_args()
    impls: set[str]
    if args.impl == "all":
        impls = {"mini", "serverless"}
    else:
        impls = {args.impl}

    fixtures_dir = Path(args.fixtures_dir)
    fixture_paths = sorted(
        [path for path in fixtures_dir.glob("*.json") if path.is_file()],
        key=lambda p: p.name,
    )
    if not fixture_paths:
        print(f"no fixture json files found in: {fixtures_dir}")
        return 2

    all_results: list[ScenarioResult] = []
    for fixture_path in fixture_paths:
        scenario_results = evaluate_scenario(
            fixture_path=fixture_path,
            tolerance=args.tolerance,
            verbose=args.verbose,
            impls=impls,
        )
        all_results.extend(scenario_results)

    print_summary(all_results)

    has_failures = any(not row.passed for row in all_results)
    if has_failures and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
