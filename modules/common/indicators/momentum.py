"""Momentum indicator block."""

from __future__ import annotations

import logging

import pandas as pd
import pandas_ta as ta

from .base import IndicatorMetadata, IndicatorResult, collect_metadata

logger = logging.getLogger(__name__)


class MomentumIndicators:
    """Momentum oscillators and oscillation strength."""

    CATEGORY = "momentum"

    @staticmethod
    def apply(df: pd.DataFrame) -> IndicatorResult:
        result = df.copy()
        before = result.columns.tolist()

        def _fill_rsi(length: int) -> pd.Series:
            rsi = ta.rsi(result["close"], length=length)
            return (
                rsi.fillna(50.0)
                if rsi is not None
                else pd.Series(50.0, index=result.index)
            )

        result["RSI_9"] = _fill_rsi(9)
        result["RSI_14"] = _fill_rsi(14)
        result["RSI_25"] = _fill_rsi(25)

        macd = ta.macd(result["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            macd = macd.copy()
            for col in ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]:
                if col in macd.columns:
                    macd[col] = macd[col].fillna(0.0)
            result[macd.columns] = macd
        else:
            logger.warning("MACD calculation failed, using neutral values (0).")
            result["MACD_12_26_9"] = 0.0
            result["MACDh_12_26_9"] = 0.0
            result["MACDs_12_26_9"] = 0.0

        bbands = ta.bbands(result["close"], length=20, std=2.0)
        if bbands is not None and not bbands.empty:
            bbp_cols = [col for col in bbands.columns if col.startswith("BBP")]
            if bbp_cols:
                result["BBP_5_2.0"] = bbands[bbp_cols[0]].fillna(0.5)
            else:
                logger.warning(
                    "BBP column missing in Bollinger Bands output, falling back to 0.5."
                )
                result["BBP_5_2.0"] = 0.5
        else:
            logger.warning("Bollinger Bands calculation failed, defaulting BBP to 0.5.")
            result["BBP_5_2.0"] = 0.5

        stochrsi = ta.stochrsi(result["close"], length=14, rsi_length=14, k=3, d=3)
        if stochrsi is not None and not stochrsi.empty:
            stochrsi = stochrsi.copy()
            for col in ["STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"]:
                if col in stochrsi.columns:
                    stochrsi[col] = stochrsi[col].fillna(50.0)
            result[stochrsi.columns] = stochrsi
        else:
            logger.warning("Stochastic RSI failed, using neutral values (50).")
            result["STOCHRSIk_14_14_3_3"] = 50.0
            result["STOCHRSId_14_14_3_3"] = 50.0

        metadata = collect_metadata(
            before,
            result.columns,
            MomentumIndicators.CATEGORY,
        )
        return result, metadata


__all__ = ["MomentumIndicators"]
