from typing import List, cast

import pandas as pd

from ..models.order_block import OrderBlock
from .constants import BEARISH, BULLISH


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR for volatility filter."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr  # type: ignore[return-value]


def _apply_volatility_filter(df: pd.DataFrame, atr: pd.Series, idx: pd.Timestamp) -> tuple:
    """
    Apply volatility filter to determine parsedHigh and parsedLow.
    parsedHigh = low if (high-low) >= 2*atr else high
    parsedLow = high if (high-low) >= 2*atr else low
    """
    row = df.loc[idx]
    high = row["High"]
    low = row["Low"]
    atr_value = atr.loc[idx]

    range_size = high - low

    if range_size >= 2 * atr_value:
        parsed_high = low
        parsed_low = high
    else:
        parsed_high = high
        parsed_low = low

    return parsed_high, parsed_low


def _normalize_timestamp(ts: pd.Timestamp, target_tz) -> pd.Timestamp:
    """Normalize timestamp to match target timezone."""
    if pd.isna(ts):
        return ts
    if target_tz is not None and ts.tzinfo is not None:
        return ts.tz_convert(target_tz)
    elif target_tz is not None:
        return ts.tz_localize(target_tz)
    elif ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def _create_ob_from_structure_events(
    df: pd.DataFrame,
    structure_events: pd.DataFrame,
    bias: int,
) -> List[OrderBlock]:
    """
    Create OrderBlocks from structure break events (BOS/CHoCH).
    For bullish: find bar with min parsedLow in range [pivot_time → crossing_time]
    For bearish: find bar with max parsedHigh in range [pivot_time → crossing_time]
    """
    blocks: List[OrderBlock] = []

    if structure_events.empty:
        return blocks

    df_index_tz = df.index.tz  # type: ignore[union-attr]
    atr = _calculate_atr(df)

    for _, event in structure_events.iterrows():
        pivot_col = None
        for col in [
            "Pivot_bullishBos_Time",
            "Pivot_bearishBos_Time",
            "Pivot_bullishChoch_Time",
            "Pivot_bearishChoch_Time",
        ]:
            if col in event.index and pd.notna(event[col]):  # type: ignore[union-attr]
                pivot_col = col
                break

        if pivot_col is None:
            continue

        pivot_time = _normalize_timestamp(pd.Timestamp(event[pivot_col]), df_index_tz)  # type: ignore[arg-type]
        crossing_time = _normalize_timestamp(pd.Timestamp(event["Crossing_Time"]), df_index_tz)  # type: ignore[arg-type]

        if pd.isna(pivot_time) or pd.isna(crossing_time):
            continue

        if pivot_time >= crossing_time:
            continue

        df_range = df[(df.index >= pivot_time) & (df.index <= crossing_time)]

        if df_range.empty:  # type: ignore[union-attr]
            continue

        if bias == BULLISH:
            min_parsed_low = float("inf")
            min_bar: pd.Timestamp | None = None
            for idx in df_range.index:
                idx_ts = cast(pd.Timestamp, idx)
                parsed_high, parsed_low = _apply_volatility_filter(df, atr, idx_ts)
                if parsed_low < min_parsed_low:
                    min_parsed_low = parsed_low
                    min_bar = idx_ts

            if min_bar is not None:
                row = df.loc[min_bar]
                ob = OrderBlock(
                    start=min_bar,  # type: ignore[arg-type]
                    end=crossing_time,
                    level_y0=row["Low"],
                    level_y1=row["High"],
                    bias=bias,
                    bar_low=row["Low"],
                    bar_high=row["High"],
                )
                blocks.append(ob)

        else:  # BEARISH
            max_parsed_high = float("-inf")
            max_bar: pd.Timestamp | None = None
            for idx in df_range.index:
                idx_ts = cast(pd.Timestamp, idx)
                parsed_high, parsed_low = _apply_volatility_filter(df, atr, idx_ts)
                if parsed_high > max_parsed_high:
                    max_parsed_high = parsed_high
                    max_bar = idx_ts

            if max_bar is not None:
                row = df.loc[max_bar]
                ob = OrderBlock(
                    start=max_bar,  # type: ignore[arg-type]
                    end=crossing_time,
                    level_y0=row["Low"],
                    level_y1=row["High"],
                    bias=bias,
                    bar_low=row["Low"],
                    bar_high=row["High"],
                )
                blocks.append(ob)

    return blocks


def _filter_ob_mitigation(df: pd.DataFrame, blocks: List[OrderBlock]) -> List[OrderBlock]:
    """
    Filter OrderBlocks based on mitigation.
    Bullish OB removed when low < ob.bar_low
    Bearish OB removed when high > ob.bar_high
    """
    filtered: List[OrderBlock] = []

    for ob in blocks:
        if ob.end is None:
            continue
        df_slice = df.loc[ob.end :]  # type: ignore[index]
        remove = False

        if ob.bias == BULLISH:
            for idx, row in df_slice.iterrows():
                if row["Low"] < ob.bar_low:
                    remove = True
                    break
        elif ob.bias == BEARISH:
            for idx, row in df_slice.iterrows():
                if row["High"] > ob.bar_high:
                    remove = True
                    break

        if not remove:
            filtered.append(ob)

    return filtered


def _update_ob_end(df: pd.DataFrame, blocks: List[OrderBlock], last_candle_time) -> List[OrderBlock]:
    """
    Update end of OrderBlocks based on market conditions.
    """
    last_low = df["Low"].iloc[-1]
    last_high = df["High"].iloc[-1]

    for idx, ob in enumerate(blocks):
        if ob.bias == BULLISH and last_low > ob.level_y1:
            ob.end = last_candle_time
        elif ob.bias == BEARISH and last_high < ob.level_y0:
            ob.end = last_candle_time
        blocks[idx] = ob

    return blocks


def identify_order_blocks_from_structure(
    df: pd.DataFrame,
    bullish_events: pd.DataFrame,
    bearish_events: pd.DataFrame,
) -> List[OrderBlock]:
    """
    Create OrderBlocks from structure break events (BOS/CHoCH).

    Args:
        df: DataFrame with OHLC data
        bullish_events: DataFrame with bullish BOS/CHoCH events (must have 'Pivot_level' and 'Crossing_Time')
        bearish_events: DataFrame with bearish BOS/CHoCH events (must have 'Pivot_level' and 'Crossing_Time')

    Returns:
        List of OrderBlock objects
    """
    last_candle_time = df.index[-1]
    blocks: List[OrderBlock] = []

    bullish_blocks = _create_ob_from_structure_events(df, bullish_events, BULLISH)
    bearish_blocks = _create_ob_from_structure_events(df, bearish_events, BEARISH)

    blocks = bullish_blocks + bearish_blocks

    blocks = [ob for ob in blocks if ob.start != ob.end]
    blocks = _filter_ob_mitigation(df, blocks)
    blocks = _update_ob_end(df, blocks, last_candle_time)

    return blocks
