"""
Configuration for Random Forest Decision Matrix.

Stores configuration parameters for Random Forest classification.
Based on Pine Script input parameters (lines 19-33).
"""

from dataclasses import dataclass
from enum import Enum


class TargetType(Enum):
    """Target variable types for classification."""

    RED_GREEN_CANDLE = "Red/Green Candle"
    BULLISH_BEARISH_ATR = "Bullish / Bearish ATR"
    ATR_RANGE_BREAKOUT = "ATR Range Breakout"


class FeatureType(Enum):
    """Feature variable types for classification."""

    STOCHASTIC = "Stochastic"
    RSI = "RSI"
    Z_SCORE = "Z-Score"
    MFI = "MFI"
    VOLUME = "Volume"
    EMA = "EMA"
    SMA = "SMA"


@dataclass
class RandomForestConfig:
    """
    Configuration for Random Forest Decision Matrix.

    Based on Pine Script parameters:
    - training_length (line 19)
    - select_x1, select_x2 (lines 20-21)
    - select_y (line 22)
    """

    training_length: int = 850

    x1_type: FeatureType = FeatureType.STOCHASTIC
    x2_type: FeatureType = FeatureType.VOLUME
    target_type: TargetType = TargetType.RED_GREEN_CANDLE

    ema_length: int = 14
    ema_source: str = "close"

    rsi_length: int = 14
    rsi_source: str = "close"

    mfi_length: int = 14
    mfi_source: str = "hlc3"

    z_score_length: int = 14
    z_score_source: str = "close"

    atr_length: int = 14

    volume_std_length: int = 14

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            True if configuration is valid
        """
        if self.training_length <= 0:
            return False
        if self.ema_length <= 0:
            return False
        if self.rsi_length <= 0:
            return False
        if self.mfi_length <= 0:
            return False
        if self.z_score_length <= 0:
            return False
        if self.atr_length <= 0:
            return False
        if self.volume_std_length <= 0:
            return False
        return True

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "training_length": self.training_length,
            "x1_type": self.x1_type.value,
            "x2_type": self.x2_type.value,
            "target_type": self.target_type.value,
            "ema_length": self.ema_length,
            "ema_source": self.ema_source,
            "rsi_length": self.rsi_length,
            "rsi_source": self.rsi_source,
            "mfi_length": self.mfi_length,
            "mfi_source": self.mfi_source,
            "z_score_length": self.z_score_length,
            "z_score_source": self.z_score_source,
            "atr_length": self.atr_length,
            "volume_std_length": self.volume_std_length,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RandomForestConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            RandomForestConfig instance
        """
        return cls(
            training_length=config_dict.get("training_length", 850),
            x1_type=FeatureType(config_dict.get("x1_type", "Stochastic")),
            x2_type=FeatureType(config_dict.get("x2_type", "Volume")),
            target_type=TargetType(config_dict.get("target_type", "Red/Green Candle")),
            ema_length=config_dict.get("ema_length", 14),
            ema_source=config_dict.get("ema_source", "close"),
            rsi_length=config_dict.get("rsi_length", 14),
            rsi_source=config_dict.get("rsi_source", "close"),
            mfi_length=config_dict.get("mfi_length", 14),
            mfi_source=config_dict.get("mfi_source", "hlc3"),
            z_score_length=config_dict.get("z_score_length", 14),
            z_score_source=config_dict.get("z_score_source", "close"),
            atr_length=config_dict.get("atr_length", 14),
            volume_std_length=config_dict.get("volume_std_length", 14),
        )


# Constants for weighted impact capping
MAX_WEIGHT_CAP_N2 = 0.6
MAX_WEIGHT_CAP_N3_PLUS = 0.4
MAX_CAP_ITERATIONS = 10

__all__ = ["RandomForestConfig", "FeatureType", "TargetType", "MAX_WEIGHT_CAP_N2", "MAX_WEIGHT_CAP_N3_PLUS", "MAX_CAP_ITERATIONS"]
