import pandas as pd
from typing import List

from ..models.pivot import Pivot
from ..models.order_block import OrderBlock

# Constants
BULLISH = 1
BEARISH = -1
NEUTRAL = 0

def process_swings(df: pd.DataFrame, swings: List[Pivot], bias: int) -> List[OrderBlock]:
    """
    Create OrderBlock from the list of swings (can be used for BULLISH, BEARISH, or NEUTRAL).
    Apply logic that prioritizes 3 swings if available, then process adjacent 2 swings.
    """
    blocks = []
    i = len(swings) - 1
    while i >= 1:
        if i >= 2:
            recent = swings[i]
            mid = swings[i - 1]
            prev = swings[i - 2]
            if bias == BULLISH and (recent.bar_time > prev.bar_time) and (mid.level > recent.level and mid.level > prev.level):
                s_time, e_time = sorted([recent.bar_time, prev.bar_time])
                dfrange = df[(df.index >= s_time) & (df.index <= e_time)]
                if not dfrange.empty:
                    index, row = dfrange['Low'].idxmin(), dfrange.loc[dfrange['Low'].idxmin()]
                    ob = OrderBlock(
                        start=index,
                        end=e_time,
                        level_y0=row["Low"],
                        level_y1=row["High"],
                        bias=bias
                    )
                    blocks.append(ob)
                i -= 2
                continue
            elif bias == BEARISH and (recent.bar_time > prev.bar_time) and (mid.level < recent.level and mid.level < prev.level):
                s_time, e_time = sorted([recent.bar_time, prev.bar_time])
                dfrange = df[(df.index >= s_time) & (df.index <= e_time)]
                if not dfrange.empty:
                    index, row = dfrange['High'].idxmax(), dfrange.loc[dfrange['High'].idxmax()]
                    ob = OrderBlock(
                        start=index,
                        end=e_time,
                        level_y0=row["Low"],
                        level_y1=row["High"],
                        bias=bias
                    )
                    blocks.append(ob)
                i -= 2
                continue

        # Process with 2 adjacent swings
        current = swings[i]
        adjacent = swings[i - 1]
        s_time, e_time = sorted([current.bar_time, adjacent.bar_time])
        dfrange = df[(df.index >= s_time) & (df.index <= e_time)]
        
        if not dfrange.empty:
            index_low, row_low = dfrange['Low'].idxmin(), dfrange.loc[dfrange['Low'].idxmin()]
            index_high, row_high = dfrange['High'].idxmax(), dfrange.loc[dfrange['High'].idxmax()]
            
            if bias == BULLISH:
                ob = OrderBlock(
                    start=index_low,
                    end=e_time,
                    level_y0=row_low["Low"],
                    level_y1=row_low["High"],
                    bias=bias
                )
                blocks.append(ob)
            elif bias == BEARISH:
                ob = OrderBlock(
                    start=index_high,
                    end=e_time,
                    level_y0=row_high["Low"],
                    level_y1=row_high["High"],
                    bias=bias
                )
                blocks.append(ob)
            else:
                ob = OrderBlock(
                    start=index_low,
                    end=e_time,
                    level_y0=row_low["Low"],
                    level_y1=row_low["High"],
                    bias=bias
                )
                blocks.append(ob)
                ob_high = OrderBlock(
                    start=index_high,
                    end=e_time,
                    level_y0=row_high["Low"],
                    level_y1=row_high["High"],
                    bias=bias
                )
                blocks.append(ob_high)
        i -= 1
    return blocks

def filter_order_blocks(df: pd.DataFrame, blocks: List[OrderBlock]) -> List[OrderBlock]:
    """
    Lọc bỏ các OrderBlock không đạt yêu cầu theo tiêu chí giá trị.
    """
    filtered = []
    for ob in blocks:
        df_slice = df.loc[ob.end:df.index[-1]].sort_index(ascending=False)
        remove = False
        if ob.bias == BULLISH:
            for idx, row in df_slice.iterrows():
                if row["Low"] < ob.level_y0 or row["Low"] < ob.level_y1:
                    remove = True
                    break
        elif ob.bias == BEARISH:
            for idx, row in df_slice.iterrows():
                if row["High"] > ob.level_y0 or row["High"] > ob.level_y1:
                    remove = True
                    break
        elif ob.bias == NEUTRAL:
            for idx, row in df_slice.iterrows():
                if (row["Low"] < ob.level_y0 or row["Low"] < ob.level_y1) and (row["High"] > ob.level_y0 or row["High"] > ob.level_y1):
                    remove = True
                    break
        if not remove:
            filtered.append(ob)
    return filtered

def update_order_blocks(df: pd.DataFrame, blocks: List[OrderBlock], last_candle_time) -> List[OrderBlock]:
    """
    Cập nhật lại end của các OrderBlock nếu điều kiện thị trường thay đổi.
    """
    last_low = df["Low"].iloc[-1]
    last_high = df["High"].iloc[-1]
    for idx, ob in enumerate(blocks):
        if ob.bias == BULLISH and last_low > ob.level_y1:
            ob.end = last_candle_time
        elif ob.bias == BEARISH and last_high < ob.level_y0:
            ob.end = last_candle_time
        elif ob.bias == NEUTRAL and (last_low < ob.level_y0 or last_high > ob.level_y1):
            ob.end = last_candle_time
        blocks[idx] = ob
    return blocks


def build_internal_order_blocks(
    df: pd.DataFrame,
    highs: List[Pivot],
    lows: List[Pivot],
    trend: int,
    last_candle_time,
) -> List[OrderBlock]:
    blocks: List[OrderBlock] = []

    if trend == BULLISH:
        if len(lows) < 2:
            return blocks
        blocks = process_swings(df, lows, BULLISH)
    elif trend == BEARISH:
        if len(highs) < 2:
            return blocks
        blocks = process_swings(df, highs, BEARISH)
    elif trend == NEUTRAL:
        blocks_lows = process_swings(df, lows, NEUTRAL) if len(lows) >= 2 else []
        blocks_highs = process_swings(df, highs, NEUTRAL) if len(highs) >= 2 else []
        blocks = blocks_lows + blocks_highs

    blocks = [ob for ob in blocks if ob.start != ob.end]
    blocks = filter_order_blocks(df, blocks)
    blocks = update_order_blocks(df, blocks, last_candle_time)
    return blocks


def build_swing_order_blocks(
    df: pd.DataFrame,
    highs: List[Pivot],
    lows: List[Pivot],
    trend: int,
    last_candle_time,
) -> List[OrderBlock]:
    return build_internal_order_blocks(df, highs, lows, trend, last_candle_time)

def identify_order_blocks(df: pd.DataFrame, highs: List[Pivot], lows: List[Pivot], trend: int) -> List[OrderBlock]:
    """
    Identify and build order blocks from the provided swings.
    """
    last_candle_time = df.index[-1]

    # Public API keeps generic name and delegates to internal builder.
    return build_internal_order_blocks(df, highs, lows, trend, last_candle_time)
