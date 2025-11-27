"""Shared types and helpers for indicator blocks."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import pandas as pd

IndicatorMetadata = Dict[str, str]
IndicatorResult = Tuple[pd.DataFrame, IndicatorMetadata]
IndicatorFunc = Callable[[pd.DataFrame], IndicatorResult]


def collect_metadata(
    before: Iterable[str],
    after: Iterable[str],
    category: str,
) -> IndicatorMetadata:
    """Map newly created columns to a metadata category."""
    before_set = set(before)
    return {col: category for col in after if col not in before_set}


__all__ = [
    "IndicatorMetadata",
    "IndicatorResult",
    "IndicatorFunc",
    "collect_metadata",
]
