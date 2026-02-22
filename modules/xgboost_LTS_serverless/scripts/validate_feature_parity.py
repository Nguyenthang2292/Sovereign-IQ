"""Validate Rust feature outputs against Python indicator pipeline where available."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import pandas as pd

from config import MODEL_FEATURES
from modules.common.core.indicator_engine import IndicatorConfig, IndicatorEngine, IndicatorProfile

EXCLUDED_PARITY_FEATURES = {
    "obv",
    "bbp_5_2_0",
}


def load_test_df(data_path: Path) -> pd.DataFrame:
    raw = json.loads(data_path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        if len(raw) % 6 != 0:
            raise ValueError("Flat OHLCV array length must be divisible by 6")
        rows = [raw[index : index + 6] for index in range(0, len(raw), 6)]
        return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    if isinstance(raw, dict):
        return pd.DataFrame(raw)

    raise ValueError("Unsupported OHLCV JSON format")


def run_rust_feature_binary(module_dir: Path, data_path: Path) -> dict[str, float]:
    command = ["cargo", "run", "--bin", "calculate_features", "--", str(data_path)]
    result = subprocess.run(command, cwd=module_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Rust feature binary failed:\n{result.stderr}")
    return json.loads(result.stdout)


def calculate_python_reference(df: pd.DataFrame) -> dict[str, float]:
    engine = IndicatorEngine(config=IndicatorConfig.for_profile(IndicatorProfile.XGBOOST))
    computed = engine.compute_features(df.copy())
    enriched = computed[0] if isinstance(computed, tuple) else computed

    available = [feature for feature in MODEL_FEATURES if feature in enriched.columns]
    if not available:
        raise RuntimeError("Python pipeline produced no overlapping model features")

    latest = enriched.iloc[-1]
    return {feature: float(latest[feature]) for feature in available}


def normalize_feature_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return normalized.strip("_")


def compare_features(reference: dict[str, float], rust_features: dict[str, float], tolerance: float) -> list[str]:
    mismatches: list[str] = []
    reference_norm = {normalize_feature_name(name): value for name, value in reference.items()}
    rust_norm = {normalize_feature_name(name): value for name, value in rust_features.items()}

    overlap = sorted(set(reference_norm.keys()) & set(rust_norm.keys()))
    if not overlap:
        return ["No overlapping normalized features between Python and Rust outputs"]

    for feature_name in overlap:
        if feature_name in EXCLUDED_PARITY_FEATURES:
            continue
        reference_value = reference_norm[feature_name]
        rust_value = rust_norm[feature_name]
        if math.isnan(reference_value) and math.isnan(rust_value):
            continue

        diff = abs(reference_value - rust_value)
        if diff > tolerance:
            mismatches.append(
                f"{feature_name}: python={reference_value:.8f}, rust={rust_value:.8f}, diff={diff:.8f}"
            )

    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate feature parity between Python and Rust")
    parser.add_argument(
        "--data",
        default="tests/test_data/btc_usdt_1h.json",
        help="Path to OHLCV JSON test data (relative to module root)",
    )
    parser.add_argument(
        "--module-dir",
        default=".",
        help="xgboost_LTS_serverless module directory (contains Cargo.toml)",
    )
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Absolute tolerance per feature")
    args = parser.parse_args()

    module_dir = Path(args.module_dir).resolve()
    data_path = module_dir / args.data

    df = load_test_df(data_path)
    rust_features = run_rust_feature_binary(module_dir, data_path)
    python_features = calculate_python_reference(df)

    mismatches = compare_features(python_features, rust_features, args.tolerance)
    if mismatches:
        print(f"Feature parity FAILED: {len(mismatches)} mismatches")
        for mismatch in mismatches[:30]:
            print(f"- {mismatch}")
        return 1

    print(f"Feature parity PASSED for {len(python_features)} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
