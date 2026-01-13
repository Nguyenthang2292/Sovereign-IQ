"""
Feature calculations for Simplified Percentile Clustering.

Computes RSI, CCI, Fisher Transform, DMI, Z-Score, and MAR (Moving Average Ratio)
features with optional standardization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pandas_ta as ta

from modules.common.indicators.trend import (
    calculate_cci,
    calculate_dmi_difference,
)
from modules.common.quantitative_metrics import (
    calculate_fisher_transform,
    calculate_mar,
    calculate_zscore,
)
from modules.simplified_percentile_clustering.utils.validation import (
    validate_feature_config,
)


@dataclass
class FeatureConfig:
    """Configuration for feature calculations."""

    # RSI
    use_rsi: bool = True
    rsi_len: int = 14
    rsi_standardize: bool = True

    # CCI
    use_cci: bool = True
    cci_len: int = 20
    cci_standardize: bool = True

    # Fisher
    use_fisher: bool = True
    fisher_len: int = 9
    fisher_standardize: bool = True

    # DMI
    use_dmi: bool = True
    dmi_len: int = 9
    dmi_standardize: bool = True

    # Z-Score
    use_zscore: bool = True
    zscore_len: int = 20

    # MAR (Moving Average Ratio)
    use_mar: bool = True
    mar_len: int = 14
    mar_type: str = "SMA"  # "SMA" or "EMA"
    mar_standardize: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        validate_feature_config(self)


class FeatureCalculator:
    """Calculate technical features for clustering."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    @staticmethod
    def z_score(src: pd.Series, length: int) -> pd.Series:
        """Calculate rolling z-score standardization."""
        return calculate_zscore(src, length)

    @staticmethod
    def round_fisher(val: float) -> float:
        """Safe clamp for Fisher transform to avoid infinite values."""
        return 0.999 if val > 0.99 else (-0.999 if val < -0.99 else val)

    @staticmethod
    def fisher_transform(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
        """
        Calculate Fisher Transform applied to hl2 over length bars.

        Uses Numba JIT compilation for the core recursive calculation if available,
        falling back to pure Python if Numba is not installed.
        """
        return calculate_fisher_transform(high, low, close, length)

    @staticmethod
    def dmi_difference(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
        """Calculate simplified DMI difference (plus - minus)."""
        return calculate_dmi_difference(high, low, close, length)

    def compute_rsi(self, close: pd.Series, lookback: int) -> tuple[pd.Series, pd.Series]:
        """Compute RSI and optionally standardized version."""
        rsi = ta.rsi(close, length=self.config.rsi_len)
        if rsi is None:
            rsi = pd.Series(50.0, index=close.index)
        rsi = rsi.fillna(50.0)

        rsi_z = self.z_score(rsi, lookback)
        rsi_val = rsi_z if self.config.rsi_standardize else rsi
        return rsi, rsi_val

    def compute_cci(
        self, high: pd.Series, low: pd.Series, close: pd.Series, lookback: int
    ) -> tuple[pd.Series, pd.Series]:
        """Compute CCI and optionally standardized version."""
        cci = calculate_cci(high, low, close, self.config.cci_len)

        cci_z = self.z_score(cci, lookback)
        cci_val = cci_z if self.config.cci_standardize else cci
        return cci, cci_val

    def compute_fisher(
        self, high: pd.Series, low: pd.Series, close: pd.Series, lookback: int
    ) -> tuple[pd.Series, pd.Series]:
        """Compute Fisher Transform and optionally standardized version."""
        fisher = self.fisher_transform(high, low, close, self.config.fisher_len)
        fisher_z = self.z_score(fisher, lookback)
        fisher_val = fisher_z if self.config.fisher_standardize else fisher
        return fisher, fisher_val

    def compute_dmi(
        self, high: pd.Series, low: pd.Series, close: pd.Series, lookback: int
    ) -> tuple[pd.Series, pd.Series]:
        """Compute DMI difference and optionally standardized version."""
        dmi = self.dmi_difference(high, low, close, self.config.dmi_len)
        dmi_z = self.z_score(dmi, lookback)
        dmi_val = dmi_z if self.config.dmi_standardize else dmi
        return dmi, dmi_val

    def compute_zscore(self, close: pd.Series) -> pd.Series:
        """Compute z-score of price itself."""
        return self.z_score(close, self.config.zscore_len)

    def compute_mar(self, close: pd.Series, lookback: int) -> tuple[pd.Series, pd.Series]:
        """Compute MAR (Moving Average Ratio) and optionally standardized version."""
        mar = calculate_mar(close, self.config.mar_len, self.config.mar_type)
        mar_z = self.z_score(mar, lookback)
        mar_val = mar_z if self.config.mar_standardize else mar
        return mar, mar_val

    def compute_all(self, high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> dict[str, pd.Series]:
        """Compute all enabled features."""
        results = {}

        if self.config.use_rsi:
            rsi, rsi_val = self.compute_rsi(close, lookback)
            results["rsi"] = rsi
            results["rsi_val"] = rsi_val

        if self.config.use_cci:
            cci, cci_val = self.compute_cci(high, low, close, lookback)
            results["cci"] = cci
            results["cci_val"] = cci_val

        if self.config.use_fisher:
            fisher, fisher_val = self.compute_fisher(high, low, close, lookback)
            results["fisher"] = fisher
            results["fisher_val"] = fisher_val

        if self.config.use_dmi:
            dmi, dmi_val = self.compute_dmi(high, low, close, lookback)
            results["dmi"] = dmi
            results["dmi_val"] = dmi_val

        if self.config.use_zscore:
            zsc_val = self.compute_zscore(close)
            results["zsc_val"] = zsc_val

        if self.config.use_mar:
            mar, mar_val = self.compute_mar(close, lookback)
            results["mar"] = mar
            results["mar_val"] = mar_val

        return results


def compute_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    config: Optional[FeatureConfig] = None,
    lookback: int = 1000,
) -> dict[str, pd.Series]:
    """Convenience function to compute all features."""
    calculator = FeatureCalculator(config)
    return calculator.compute_all(high, low, close, lookback)


__all__ = ["FeatureConfig", "FeatureCalculator", "compute_features"]
