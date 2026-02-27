"""Prototype Dask training script for Phase 5.

Usage:
    python -m modules.xgboost_LTS.scripts.prototype_dask_training --rows 5000 --partitions 4
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from config import MODEL_FEATURES, XGBOOST_PARAMS
from modules.xgboost_LTS.core.model_dask import train_and_predict_dask


def build_synthetic_dataframe(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data: dict[str, Any] = {feature: rng.normal(size=rows) for feature in MODEL_FEATURES}
    data["Target"] = rng.integers(low=0, high=3, size=rows)
    return pd.DataFrame(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype Dask training for xgboost_LTS Phase 5")
    parser.add_argument("--rows", type=int, default=5000, help="Number of synthetic rows to generate")
    parser.add_argument("--partitions", type=int, default=4, help="Number of Dask partitions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import dask.dataframe as dd
    except Exception as exc:
        raise SystemExit(f"Dask is required for this prototype script: {exc}")

    pdf = build_synthetic_dataframe(args.rows)
    ddf = dd.from_pandas(pdf, npartitions=max(1, args.partitions))

    params = XGBOOST_PARAMS.copy()
    params.update(
        {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 10,
            "max_depth": 4,
        }
    )

    model = train_and_predict_dask(ddf, model_features=MODEL_FEATURES, params=params)
    print(f"Dask prototype training completed. Model type: {type(model).__name__}")


if __name__ == "__main__":
    main()
