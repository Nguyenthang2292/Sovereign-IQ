import datetime
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from colorama import init
from scipy.signal import argrelextrema

# Initialize colorama for Windows
init(autoreset=True)

import warnings

warnings.filterwarnings("ignore", message="Boolean Series key will be reindexed to match DataFrame index")
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated."
)

current_dir = os.path.dirname(os.path.abspath(__file__))
strategies_dir = os.path.dirname(current_dir)
main_dir = os.path.dirname(strategies_dir)
if main_dir not in sys.path:
    sys.path.insert(0, main_dir)

from modules.smart_money_concept.models import OrderBlock, Pivot

# Global variables for OHLC and pivot data
opens: List[float] = []
highs: List[float] = []
lows: List[float] = []
closes: List[float] = []
times: List[datetime.datetime] = []

internal_swing_highs: List[Pivot] = []
internal_swing_lows: List[Pivot] = []
ChoCh_internal_bullish: List[Pivot] = []
ChoCh_internal_bearish: List[Pivot] = []
internal_order_blocks: List[OrderBlock] = []

swing_highs: List[Pivot] = []
swing_lows: List[Pivot] = []
ChoCh_swings_bullish: List[Pivot] = []
ChoCh_lows_bearish: List[Pivot] = []
swing_order_blocks: List[OrderBlock] = []

pivot_internalHigh_bos: pd.DataFrame = pd.DataFrame(columns=['Pivot_level', 'Pivot_bullishBos_Time', 'Crossing_Time'])
pivot_internalLow_bos: pd.DataFrame = pd.DataFrame(columns=['Pivot_level', 'Pivot_bullishBos_Time', 'Crossing_Time'])

pivot_swingHigh_bos: pd.DataFrame = pd.DataFrame(columns=['Pivot_level', 'Pivot_bullishBos_Time', 'Crossing_Time'])
pivot_swingLow_bos: pd.DataFrame = pd.DataFrame(columns=['Pivot_level', 'Pivot_bullishBos_Time', 'Crossing_Time'])

BULLISH = 1
NEUTRAL = 0
BEARISH = -1

def compute_atr(highs_arr: np.ndarray, lows_arr: np.ndarray, closes_arr: np.ndarray, period: int = 200) -> Optional[float]:
    """
    Compute the Average True Range (ATR) from highs, lows, and closes arrays.
    """
    if len(highs_arr) != len(lows_arr) or len(highs_arr) != len(closes_arr):
        raise ValueError("Lengths of highs, lows, and closes must be the same.")
    if len(highs_arr) < period + 1:
        print(f"Not enough data to compute ATR for period={period}.")
        return None

    tr = np.maximum(
        highs_arr[1:] - lows_arr[1:],
        np.maximum(
            np.abs(highs_arr[1:] - closes_arr[:-1]),
            np.abs(lows_arr[1:] - closes_arr[:-1])
        )
    )
    if len(tr) < period:
        print(f"Not enough TR values to compute ATR for period={period}.")
        return None

    atr = np.mean(tr[-period:])
    return float(atr) if not np.isnan(atr) else None

# ====================== DETECT TREND ======================
def detect_trend() -> int:
    """
    Returns BULLISH (1), BEARISH (-1), or NEUTRAL (0) based on the last two swing highs and swing lows,
    and prints the trend before returning.
    """
    global internal_swing_highs, internal_swing_lows, BULLISH, NEUTRAL, BEARISH

    if len(internal_swing_highs) < 2 or len(internal_swing_lows) < 2:
        # print(Back.BLUE + "Trend is - NEUTRAL")
        return NEUTRAL

    if (internal_swing_highs[-1].level is not None and internal_swing_highs[-2].level is not None and internal_swing_highs[-1].level > internal_swing_highs[-2].level
        and internal_swing_lows[-1].level is not None and internal_swing_lows[-2].level is not None and internal_swing_lows[-1].level > internal_swing_lows[-2].level):
        # print(Back.GREEN + "Trend is - BULLISH")
        return BULLISH
    elif (internal_swing_highs[-1].level is not None and internal_swing_highs[-2].level is not None and internal_swing_highs[-1].level < internal_swing_highs[-2].level
        and internal_swing_lows[-1].level is not None and internal_swing_lows[-2].level is not None and internal_swing_lows[-1].level < internal_swing_lows[-2].level):
        # print(Back.RED + "Trend is - BEARISH")
        return BEARISH
    else:
        # print("Trend is - NEUTRAL")
        return NEUTRAL

def detect_trend_export(internal_swing_highs, internal_swing_lows) -> int:
    """
    Returns BULLISH (1), BEARISH (-1), or NEUTRAL (0) based on the last two swing highs and swing lows,
    and prints the trend before returning.
    """
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1

    if len(internal_swing_highs) < 2 or len(internal_swing_lows) < 2:
        # print(Back.BLUE + "Trend is - NEUTRAL")
        return NEUTRAL

    if (internal_swing_highs[-1].level is not None and internal_swing_highs[-2].level is not None and internal_swing_highs[-1].level > internal_swing_highs[-2].level
        and internal_swing_lows[-1].level is not None and internal_swing_lows[-2].level is not None and internal_swing_lows[-1].level > internal_swing_lows[-2].level):
        # print(Back.GREEN + "Trend is - BULLISH")
        return BULLISH
    elif (internal_swing_highs[-1].level is not None and internal_swing_highs[-2].level is not None and internal_swing_highs[-1].level < internal_swing_highs[-2].level
        and internal_swing_lows[-1].level is not None and internal_swing_lows[-2].level is not None and internal_swing_lows[-1].level < internal_swing_lows[-2].level):
        # print(Back.RED + "Trend is - BEARISH")
        return BEARISH
    else:
        # print("Trend is - NEUTRAL")
        return NEUTRAL

# ====================== GET SWING HIGH LOW ======================
def get_swing_high_low(df: pd.DataFrame, internal=True, order: int = 5) -> None:
    """
    Identify swing highs and lows from the DataFrame and save the results in global variables.
    """
    global internal_swing_highs, internal_swing_lows, swing_highs, swing_lows, highs, lows

    # Compute local extrema indices
    swing_high_idx = argrelextrema(df['High'].values, np.greater_equal, order=order)[0]
    swing_low_idx = argrelextrema(df['Low'].values, np.less_equal, order=order)[0]

    # Add temporary columns for swing data
    df['Swing_High'] = np.nan
    df['Swing_Low'] = np.nan

    if swing_high_idx.size > 0:
        df.loc[df.index[swing_high_idx], 'Swing_High'] = df['High'].iloc[swing_high_idx]
    if swing_low_idx.size > 0:
        df.loc[df.index[swing_low_idx], 'Swing_Low'] = df['Low'].iloc[swing_low_idx]

    # Build lists of Pivot objects from non-NaN swing data
    swing_H = df.dropna(subset=['Swing_High'])
    swing_L = df.dropna(subset=['Swing_Low'])

    if internal:
        internal_swing_highs = [Pivot(level=row, bar_time=idx)
                                for idx, row in zip(swing_H.index, swing_H['Swing_High'])]
        internal_swing_lows = [Pivot(level=row, bar_time=idx)
                                for idx, row in zip(swing_L.index, swing_L['Swing_Low'])]
        # Remove the last pivot if it is a high and the previous one is also a high
        if internal_swing_highs and internal_swing_lows:
            if (internal_swing_highs[-1].bar_time is not None and internal_swing_lows[-1].bar_time is not None
                and internal_swing_highs[-1].bar_time > internal_swing_lows[-1].bar_time):
                internal_swing_highs = internal_swing_highs[:-1]
            else:
                internal_swing_lows = internal_swing_lows[:-1]
    else:
        # If not using internal, assign to globally defined swing_highs and swing_lows
        swing_highs = [Pivot(level=row, bar_time=idx)
                                for idx, row in zip(swing_H.index, swing_H['Swing_High'])]
        swing_lows = [Pivot(level=row, bar_time=idx)
                        for idx, row in zip(swing_L.index, swing_L['Swing_Low'])]

def get_swing_high_low_export(df: pd.DataFrame, internal=True, order: int = 5) -> Tuple[List[Pivot], List[Pivot], List[Pivot], List[Pivot]]:
    """
    Identify swing highs and lows from the DataFrame and save the results in global variables.
    """
    internal_swing_highs: List[Pivot] = []
    internal_swing_lows: List[Pivot] = []
    swing_highs: List[Pivot] = []
    swing_lows: List[Pivot] = []

    # Compute local extrema indices
    swing_high_idx = argrelextrema(df['High'].values, np.greater_equal, order=order)[0]
    swing_low_idx = argrelextrema(df['Low'].values, np.less_equal, order=order)[0]

    # Add temporary columns for swing data
    df['Swing_High'] = np.nan
    df['Swing_Low'] = np.nan

    if swing_high_idx.size > 0:
        df.loc[df.index[swing_high_idx], 'Swing_High'] = df['High'].iloc[swing_high_idx]
    if swing_low_idx.size > 0:
        df.loc[df.index[swing_low_idx], 'Swing_Low'] = df['Low'].iloc[swing_low_idx]

    # Build lists of Pivot objects from non-NaN swing data
    swing_H = df.dropna(subset=['Swing_High'])
    swing_L = df.dropna(subset=['Swing_Low'])

    if internal:
        internal_swing_highs = [Pivot(level=row, bar_time=idx)
                                for idx, row in zip(swing_H.index, swing_H['Swing_High'])]
        internal_swing_lows = [Pivot(level=row, bar_time=idx)
                                for idx, row in zip(swing_L.index, swing_L['Swing_Low'])]
        # Remove the last pivot if it is a high and the previous one is also a high
        if internal_swing_highs and internal_swing_lows:
            if (internal_swing_highs[-1].bar_time is not None and internal_swing_lows[-1].bar_time is not None
                and internal_swing_highs[-1].bar_time > internal_swing_lows[-1].bar_time):
                internal_swing_highs = internal_swing_highs[:-1]
            else:
                internal_swing_lows = internal_swing_lows[:-1]
    else:
        # If not using internal, assign to globally defined swing_highs and swing_lows
        swing_highs = [Pivot(level=row, bar_time=idx)
                                for idx, row in zip(swing_H.index, swing_H['Swing_High'])]
        swing_lows = [Pivot(level=row, bar_time=idx)
                        for idx, row in zip(swing_L.index, swing_L['Swing_Low'])]
    return internal_swing_highs, internal_swing_lows, swing_highs, swing_lows

# ====================== CLASSIFY SWING TYPES ======================
def classify_swing_types() -> Tuple[List[Tuple[Pivot, str]], List[Tuple[Pivot, str]]]:
    """
    Classify swing highs and lows into types:
    - For swing_highs: iterate from the end of the list, if swing_high[n].level > swing_high[n-1].level
        then swing_high[n] is assigned "HH" and swing_high[n-1] is assigned "HL".
    - For swing_lows: iterate from the end of the list, if swing_low[n].level > swing_low[n-1].level
        then swing_low[n] is assigned "LH" and swing_low[n-1] is assigned "LL".
    
    Returns:
        Tuple of 2 lists, each list containing tuples of (Pivot, classification)
    """
    global internal_swing_highs, internal_swing_lows, swing_highs, swing_lows

    highs_list = swing_highs
    lows_list = swing_lows

    # Initialize result lists with empty classification strings
    classified_highs = [(ph, "") for ph in highs_list]
    classified_lows = [(pl, "") for pl in lows_list]

    # Classify swing highs:
    for i in range(len(highs_list) - 1, 0, -1):
        if (highs_list[i].level is not None and highs_list[i - 1].level is not None
            and float(highs_list[i].level) > float(highs_list[i - 1].level)): # type: ignore
            # Current swing_high is Higher High, previous swing_high is Lower High
            classified_highs[i] = (highs_list[i], "HH")
            classified_highs[i - 1] = (highs_list[i - 1], "HL")
            # if classified_highs:
            #     print(Fore.GREEN + "classify_swing_types FUNCTION: Last classified high:", classified_highs[-1])
            #     print(Fore.GREEN + "classify_swing_types FUNCTION: Last classified low:", classified_highs[-2])
            # else:
            #     print(Fore.GREEN + "classify_swing_types FUNCTION: classified_highs is empty")

    # Classify swing lows:
    # print(Fore.RED + "classify_swing_types FUNCTION: lows_list:", lows_list)
    for i in range(len(lows_list) - 1, 0, -1):
        if (lows_list[i].level is not None and lows_list[i - 1].level is not None
            and float(lows_list[i].level) > float(lows_list[i - 1].level)): # type: ignore
            # For lows: current swing_low is assigned "LH" and previous swing_low is assigned "LL"
            classified_lows[i] = (lows_list[i], "LH")
            classified_lows[i - 1] = (lows_list[i - 1], "LL")
            # if classified_lows:
            #     print(Fore.RED + "classify_swing_types FUNCTION: Last classified low:", classified_lows[-1])
            #     print(Fore.RED + "classify_swing_types FUNCTION: Last classified low:", classified_lows[-2])
            # else:
            #     print(Fore.RED + "classify_swing_types FUNCTION: classified_lows is empty")

    # Cut off the first element of the result list before returning
    return classified_highs[1:], classified_lows[1:]

# ====================== DRAW SWING HIGH LOW ======================
def draw_swing_high_low(fig: go.Figure, internal=True) -> go.Figure:
    global internal_swing_highs, internal_swing_lows, swing_highs, swing_lows

    if internal:
        # Extract x and y values (bar_time and level) for internal swing highs and lows.
        swing_high_x = [p.bar_time for p in internal_swing_highs]
        swing_high_y = [p.level for p in internal_swing_highs]
        swing_low_x = [p.bar_time for p in internal_swing_lows]
        swing_low_y = [p.level for p in internal_swing_lows]

        # Add markers for swing highs.
        fig.add_trace(go.Scatter(
            x=swing_high_x,
            y=swing_high_y,
            mode='markers',
            marker=dict(color='red', symbol='triangle-up', size=10),
            name='Internal Swing High'
        ))
        # Add markers for swing lows.
        fig.add_trace(go.Scatter(
            x=swing_low_x,
            y=swing_low_y,
            mode='markers',
            marker=dict(color='blue', symbol='triangle-down', size=10),
            name='Internal Swing Low'
        ))
    else:
        # Get list of classified swing types for swing highs and lows
        classified_highs, classified_lows = classify_swing_types()

        # Extract x and y values based on classified swings.
        swing_high_x = [p.bar_time for (p, _) in classified_highs]
        swing_high_y = [p.level for (p, _) in classified_highs]
        swing_low_x = [p.bar_time for (p, _) in classified_lows]
        swing_low_y = [p.level for (p, _) in classified_lows]

        # Add markers for swing highs.
        fig.add_trace(go.Scatter(
            x=swing_high_x,
            y=swing_high_y,
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Swing High'
        ))
        # Add annotations based on classification (HH, HL, etc.) for each swing high.
        for (p, cls) in classified_highs:
            fig.add_annotation(
                x=p.bar_time,
                y=p.level,
                text=cls,
                showarrow=False,
                yanchor='bottom',
                font=dict(color='green')
            )

        # Add markers for swing lows.
        fig.add_trace(go.Scatter(
            x=swing_low_x,
            y=swing_low_y,
            mode='markers',
            marker=dict(color='purple', symbol='triangle-down', size=10),
            name='Swing Low'
        ))
        # Add annotations based on classification for each swing low.
        for (p, cls) in classified_lows:
            fig.add_annotation(
                x=p.bar_time,
                y=p.level,
                text=cls,
                showarrow=False,
                yanchor='top',
                font=dict(color='purple')
            )
    return fig

def draw_swing_high_low_export(fig: go.Figure, internal_swing_highs, internal_swing_lows) -> go.Figure:
    # Extract x and y values (bar_time and level) for internal swing highs and lows.
    swing_high_x = [p.bar_time for p in internal_swing_highs]
    swing_high_y = [p.level for p in internal_swing_highs]
    swing_low_x = [p.bar_time for p in internal_swing_lows]
    swing_low_y = [p.level for p in internal_swing_lows]

    # Add markers for swing highs.
    fig.add_trace(go.Scatter(
            x=swing_high_x,
            y=swing_high_y,
            mode='markers',
            marker=dict(color='red', symbol='triangle-up', size=10),
            name='Internal Swing High'
        ))
    # Add markers for swing lows.
    fig.add_trace(go.Scatter(
            x=swing_low_x,
            y=swing_low_y,
            mode='markers',
            marker=dict(color='blue', symbol='triangle-down', size=10),
            name='Internal Swing Low'
        ))
    return fig

# ====================== DRAW WEAK HIGH LOW ======================
def draw_weak_high_low(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """
    Draw only Weak High if the trend is BULLISH, only Weak Low if the trend is BEARISH,
    and both if the trend is NEUTRAL.
    """
    global internal_swing_highs, internal_swing_lows

    trend_direction = detect_trend()
    if not internal_swing_highs or not internal_swing_lows:
        return fig

    max_swing_high = max(internal_swing_highs, key=lambda x: x.level) # type: ignore
    max_swing_low = min(internal_swing_lows, key=lambda x: x.level) # type: ignore

    max_date = df.index[df['High'] == max_swing_high.level][0]
    min_date = df.index[df['Low'] == max_swing_low.level][0]

    # Show only Weak High if BULLISH or NEUTRAL
    if trend_direction in [BULLISH, NEUTRAL]:
        fig.add_shape(
            type='line',
            x0=max_date, y0=max_swing_high.level,
            x1=df.index[-1], y1=max_swing_high.level,
            line=dict(color='red', dash='dash'),
            name='Weak High'
        )
        fig.add_annotation(
            x=df.index[-1], y=max_swing_high.level,
            text="Weak High",
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
            font=dict(color='red')
        )

    # Show only Weak Low if BEARISH or NEUTRAL
    if trend_direction in [BEARISH, NEUTRAL]:
        fig.add_shape(
            type='line',
            x0=min_date, y0=max_swing_low.level,
            x1=df.index[-1], y1=max_swing_low.level,
            line=dict(color='blue', dash='dash'),
            name='Weak Low'
        )
        fig.add_annotation(
            x=df.index[-1], y=max_swing_low.level,
            text="Weak Low",
            showarrow=False,
            xanchor='left',
            yanchor='top',
            font=dict(color='blue')
        )

    return fig

# ====================== DRAW STRONG HIGH LOW ======================
def draw_strong_high_low(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """
    Draws a strong high/low marker based on the trend using classified pivots
    with switch-case (match-case) structure.
    """
    global BULLISH, BEARISH, NEUTRAL

    trend_direction = detect_trend()
    classified_highs, classified_lows = classify_swing_types()

    if trend_direction == BULLISH:
        # For bullish trend, iterate over all swing lows tagged "LH".
        strong_low_candidates = [p for p, label in classified_lows if label == "LH"]
        for strong_low in strong_low_candidates:
            strong_date = df.index[df['Low'] == strong_low.level][0]
            fig.add_shape(
                type='line',
                x0=strong_date, y0=strong_low.level,
                x1=df.index[-1], y1=strong_low.level,
                line=dict(color='blue', dash='dot', width=3),
                name='Strong Low'
            )
            fig.add_annotation(
                x=df.index[-1], y=strong_low.level,
                text="Strong Low",
                showarrow=False,
                xanchor='left',
                yanchor='top',
                font=dict(color='blue', size=12)
            )
    elif trend_direction == BEARISH:
        # For bearish trend, iterate over all swing highs tagged "HL".
        strong_high_candidates = [p for p, label in classified_highs if label == "HL"]
        for strong_high in strong_high_candidates:
            strong_date = df.index[df['High'] == strong_high.level][0]
            fig.add_shape(
                type='line',
                x0=strong_date, y0=strong_high.level,
                x1=df.index[-1], y1=strong_high.level,
                line=dict(color='red', dash='dot', width=3),
                name='Strong High'
            )
            fig.add_annotation(
                x=df.index[-1], y=strong_high.level,
                text="Strong High",
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
                font=dict(color='red', size=12)
            )
    return fig

# ====================== IDENTIFY BREAK OF STRUCTURE ======================
def identify_pivot_bos(df, internal =True) -> None:
    """
    Identify Break of Structure (BOS) pivot points from swing data by checking the range
    from one swing high to the next. Within that range, a breakout candle is selected as
    the first candle where any price (High, Open, Close, Low) exceeds the previous swing high's level.

    The resulting BOS DataFrame contains:
    - Pivot_level: the High of the previous swing high,
    - Pivot_bullishBos_Time: the bar_time of that previous swing high,
    - Crossing_Time: the bar_time of the breakout candle.
    """
    global pivot_internalHigh_bos, pivot_internalLow_bos, internal_swing_highs, internal_swing_lows
    global pivot_swingHigh_bos, pivot_swingLow_bos, swing_highs, swing_lows

    if internal:
        if not internal_swing_highs or not internal_swing_lows:
            return
        # Reset BOS DataFrames/lists.
        pivot_internalHigh_bos = pd.DataFrame(columns=['Pivot_level', 'Pivot_bullishBos_Time', 'Crossing_Time'])
        pivot_internalLow_bos = pd.DataFrame(columns=['Pivot_level', 'Pivot_bearishBos_Time', 'Crossing_Time'])

        # ------------------- Swing Highs (Bullish BOS) -------------------
        # For each swing high (except the last one), check candles between it and the next swing high.
        for idx in range(len(internal_swing_highs) - 1):
            current = internal_swing_highs[idx]
            next_swing = internal_swing_highs[idx + 1]

            # Filter df to only include candles between the current and next swing high.
            df_range = df[(df.index > current.bar_time) & (df.index <= next_swing.bar_time)]

            # Identify the breakout candle: first candle where any price is above current swing high's level.
            breakout = df_range[
                (df_range['High'] > current.level) | (df_range['Open'] > current.level) |
                (df_range['Close'] > current.level) | (df_range['Low'] > current.level)
            ]

            if not breakout.empty:
                # Earliest breakout candle
                breakout_candle_time = breakout.index.min()
                new_row = pd.DataFrame([{
                    'Pivot_level': current.level,
                    'Pivot_bullishBos_Time': current.bar_time,
                    'Crossing_Time': breakout_candle_time
                }])
                pivot_internalHigh_bos = pd.concat([pivot_internalHigh_bos, new_row], ignore_index=True)

        # print(Fore.GREEN + "identify_pivot_bos FUNCTION: pivot_internalHigh_bos:")
        # print(pivot_internalHigh_bos)

        # ------------------- Swing Lows (Bearish BOS) -------------------
        for idx in range(len(internal_swing_lows) - 1):
            current = internal_swing_lows[idx]
            next_swing = internal_swing_lows[idx + 1]

            # Filter df to only include candles between the current and next swing low.
            df_range = df[(df.index > current.bar_time) & (df.index <= next_swing.bar_time)]

            # Identify the breakout candle: first candle where any price is below current swing low's level.
            breakout = df_range[
                (df_range['Low'] < current.level) | (df_range['Open'] < current.level) |
                (df_range['Close'] < current.level) | (df_range['High'] < current.level)
            ]

            if not breakout.empty:
                breakout_candle_time = breakout.index.min()
                new_row = pd.DataFrame([{
                    'Pivot_level': current.level,
                    'Pivot_bearishBos_Time': current.bar_time,
                    'Crossing_Time': breakout_candle_time
                }])
                pivot_internalLow_bos = pd.concat([pivot_internalLow_bos, new_row], ignore_index=True)

        # print(Fore.RED + "identify_pivot_bos FUNCTION: pivot_internalLow_bos:")
        # print(pivot_internalLow_bos)
    else:
        if not swing_highs or not swing_lows:
            return
        # Reset BOS DataFrames/lists.
        pivot_swingHigh_bos = pd.DataFrame(columns=['Pivot_level', 'Pivot_bullishBos_Time', 'Crossing_Time'])
        pivot_swingLow_bos = pd.DataFrame(columns=['Pivot_level', 'Pivot_bearishBos_Time', 'Crossing_Time'])

        # ------------------- Swing Highs (Bullish BOS) -------------------
        # For each swing high (except the last one), check candles between it and the next swing high.
        for idx in range(len(swing_highs) - 1):
            current = swing_highs[idx]
            next_swing = swing_highs[idx + 1]

            # Filter df to only include candles between the current and next swing high.
            df_range = df[(df.index > current.bar_time) & (df.index <= next_swing.bar_time)]

            # Identify the breakout candle: first candle where any price is above current swing high's level.
            breakout = df_range[
                (df_range['High'] > current.level) | (df_range['Open'] > current.level) |
                (df_range['Close'] > current.level) | (df_range['Low'] > current.level)
            ]

            if not breakout.empty:
                # Earliest breakout candle
                breakout_candle_time = breakout.index.min()
                new_row = pd.DataFrame([{
                    'Pivot_level': current.level,
                    'Pivot_bullishBos_Time': current.bar_time,
                    'Crossing_Time': breakout_candle_time
                }])
                pivot_swingHigh_bos = pd.concat([pivot_swingHigh_bos, new_row], ignore_index=True)

        # print(Fore.GREEN + "identify_pivot_bos FUNCTION: pivot_swingHigh_bos:")
        # print(pivot_swingHigh_bos)

        # ------------------- Swing Lows (Bearish BOS) -------------------
        for idx in range(len(swing_lows) - 1):
            current = swing_lows[idx]
            next_swing = swing_lows[idx + 1]

            # Filter df to only include candles between the current and next swing low.
            df_range = df[(df.index > current.bar_time) & (df.index <= next_swing.bar_time)]

            # Identify the breakout candle: first candle where any price is below current swing low's level.
            breakout = df_range[
                (df_range['Low'] < current.level) | (df_range['Open'] < current.level) |
                (df_range['Close'] < current.level) | (df_range['High'] < current.level)
            ]

            if not breakout.empty:
                breakout_candle_time = breakout.index.min()
                new_row = pd.DataFrame([{
                    'Pivot_level': current.level,
                    'Pivot_bearishBos_Time': current.bar_time,
                    'Crossing_Time': breakout_candle_time
                }])
                pivot_swingLow_bos = pd.concat([pivot_swingLow_bos, new_row], ignore_index=True)

        # print(Fore.RED + "identify_pivot_bos FUNCTION: pivot_swingLow_bos:")
        # print(pivot_swingLow_bos)

# ====================== DRAW PIVOT BOS ======================
def draw_pivot_bos(fig: go.Figure, internal=True) -> go.Figure:
    """
    Draw BOS lines and annotations based on the identified bullish and bearish pivot points.
    """
    global pivot_internalHigh_bos, pivot_internalLow_bos, pivot_swingHigh_bos, pivot_swingLow_bos

    if internal:
        # Draw bullish BOS shapes and annotations
        for _, row in pivot_internalHigh_bos.iterrows():
            fig.add_shape(
                type='line',
                x0=row['Pivot_bullishBos_Time'], y0=row['Pivot_level'],
                x1=row['Crossing_Time'], y1=row['Pivot_level'],
                line=dict(color='green', dash='dash')
            )
            midpoint = row['Pivot_bullishBos_Time'] + (row['Crossing_Time'] - row['Pivot_bullishBos_Time']) / 2
            fig.add_annotation(
                x=midpoint, y=row['Pivot_level'],
                text="BOS",
                showarrow=False,
                xanchor='center',
                yanchor='bottom',
                font=dict(color='green')
            )

        # Draw bearish BOS shapes and annotations
        for _, row in pivot_internalLow_bos.iterrows():
            fig.add_shape(
                type='line',
                x0=row['Pivot_bearishBos_Time'], y0=row['Pivot_level'],
                x1=row['Crossing_Time'], y1=row['Pivot_level'],
                line=dict(color='red', dash='dash')
            )
            midpoint = row['Pivot_bearishBos_Time'] + (row['Crossing_Time'] - row['Pivot_bearishBos_Time']) / 2
            fig.add_annotation(
                x=midpoint, y=row['Pivot_level'],
                text="BOS",
                showarrow=False,
                xanchor='center',
                yanchor='top',
                font=dict(color='red')
            )
    else:
        # Draw bullish BOS shapes and annotations
        for _, row in pivot_swingHigh_bos.iterrows():
            fig.add_shape(
                type='line',
                x0=row['Pivot_bullishBos_Time'], y0=row['Pivot_level'],
                x1=row['Crossing_Time'], y1=row['Pivot_level'],
                line=dict(color='green', dash='dash')
            )
            midpoint = row['Pivot_bullishBos_Time'] + (row['Crossing_Time'] - row['Pivot_bullishBos_Time']) / 2
            fig.add_annotation(
                x=midpoint, y=row['Pivot_level'],
                text="Swing BOS",
                showarrow=False,
                xanchor='center',
                yanchor='bottom',
                font=dict(color='green')
            )

        # Draw bearish BOS shapes and annotations
        for _, row in pivot_swingLow_bos.iterrows():
            fig.add_shape(
                type='line',
                x0=row['Pivot_bearishBos_Time'], y0=row['Pivot_level'],
                x1=row['Crossing_Time'], y1=row['Pivot_level'],
                line=dict(color='red', dash='dash')
            )
            midpoint = row['Pivot_bearishBos_Time'] + (row['Crossing_Time'] - row['Pivot_bearishBos_Time']) / 2
            fig.add_annotation(
                x=midpoint, y=row['Pivot_level'],
                text="Swing BOS",
                showarrow=False,
                xanchor='center',
                yanchor='top',
                font=dict(color='red'))

    return fig

# ====================== IDENTIFY CHOCH PIVOT POINTS ======================
def identify_pivot_ChoCh(internal=True) -> None:
    """
    Identify ChoCh pivot points from BOS pivots based on time intervals.
    
    For bullish ChoCh:
        For each adjacent pair in pivot_*_bos (high), iterate through swing_lows 
        and if a swing.low has bar_time such that:
            t_prev < swing.bar_time < t_curr
        then append t_prev (the pivot immediately preceding swing.bar_time) into the corresponding list.
    
    For bearish ChoCh:
        For each adjacent pair in pivot_*_bos (low), iterate through swing_highs 
        and if a swing.high has bar_time such that:
            t_prev < swing.bar_time < t_curr
        then append t_prev (the pivot immediately preceding swing.bar_time) into the corresponding list.
    """
    global swing_highs, swing_lows, pivot_internalHigh_bos, pivot_internalLow_bos
    global ChoCh_internal_bullish, ChoCh_internal_bearish, ChoCh_swings_bullish, ChoCh_lows_bearish

    if internal:
        # --- Xử lý bullish ChoCh dùng internal pivots ---
        pivot_internalHigh_bos.sort_values(by="Pivot_bullishBos_Time", inplace=True)
        pivot_internalLow_bos.sort_values(by="Pivot_bearishBos_Time", inplace=True)
        ChoCh_internal_bullish.clear()
        ChoCh_internal_bearish.clear()

        for i in range(1, len(pivot_internalHigh_bos)):
            t_prev = pivot_internalHigh_bos.iloc[i-1]["Pivot_bullishBos_Time"]
            t_curr = pivot_internalHigh_bos.iloc[i]["Pivot_bullishBos_Time"]
            for swing in swing_lows:
                if t_prev < swing.bar_time < t_curr:
                    ChoCh_internal_bullish.append(t_prev)
                    break

        # print("Elements in ChoCh_internal_bullish:")
        # for element in ChoCh_internal_bullish:
        #     print(element)

        for i in range(1, len(pivot_internalLow_bos)):
            t_prev = pivot_internalLow_bos.iloc[i-1]["Pivot_bearishBos_Time"]
            t_curr = pivot_internalLow_bos.iloc[i]["Pivot_bearishBos_Time"]
            for swing in swing_highs:
                if t_prev < swing.bar_time < t_curr:
                    ChoCh_internal_bearish.append(t_prev)
                    break

        # print("Elements in ChoCh_internal_bearish:")
        # for element in ChoCh_internal_bearish:
        #     print(element)
        else:
        # --- Handling the external case (swing pivots) ---
        # Reset the corresponding lists
            ChoCh_swings_bullish.clear()
            ChoCh_lows_bearish.clear()

        # Processing Bullish ChoCh: using pivot_swingHigh_bos and swing_lows
        pivot_swingHigh_bos.sort_values(by="Pivot_bullishBos_Time", inplace=True)
        for i in range(1, len(pivot_swingHigh_bos)):
            t_prev = pivot_swingHigh_bos.iloc[i-1]["Pivot_bullishBos_Time"]
            t_curr = pivot_swingHigh_bos.iloc[i]["Pivot_bullishBos_Time"]
            for swing in swing_lows:
                if t_prev < swing.bar_time < t_curr:
                    ChoCh_swings_bullish.append(t_prev)
                    break

        # print("Elements in ChoCh_swings_bullish:")
        # for element in ChoCh_swings_bullish:
        #     print(element)

        # Processing Bearish ChoCh: using pivot_swingLow_bos and swing_highs
        pivot_swingLow_bos.sort_values(by="Pivot_bearishBos_Time", inplace=True)
        for i in range(1, len(pivot_swingLow_bos)):
            t_prev = pivot_swingLow_bos.iloc[i-1]["Pivot_bearishBos_Time"]
            t_curr = pivot_swingLow_bos.iloc[i]["Pivot_bearishBos_Time"]
            for swing in swing_highs:
                if t_prev < swing.bar_time < t_curr:
                    ChoCh_lows_bearish.append(t_prev)
                    break

        # print("Elements in ChoCh_lows_bearish:")
        # for element in ChoCh_lows_bearish:
        #     print(element)

# ====================== DRAW CHOCH PIVOT POINTS ======================
def draw_ChoCh(fig: go.Figure, df: pd.DataFrame, internal=True) -> go.Figure:
    global ChoCh_internal_bullish, ChoCh_internal_bearish, ChoCh_swings_bullish, ChoCh_lows_bearish

    # Xác định trend hiện tại
    trend_direction = detect_trend()  # trả về BULLISH, BEARISH or NEUTRAL

    # Tính toán tọa độ cho Bullish ChoCh và Swing ChoCh
    bullish_x, bullish_y = [], []
    bullish_swing_x, bullish_swing_y = [], []
    for t in ChoCh_internal_bullish:
        if t in df.index:
            bullish_x.append(t)
            bullish_y.append(df.loc[t, "High"]) # type: ignore
        else:
            nearest_time = min(df.index, key=lambda x: abs(x - t))
            bullish_x.append(nearest_time)
            bullish_y.append(df.loc[nearest_time, "Low"])
    for t in ChoCh_swings_bullish:
        if t in df.index:
            bullish_swing_x.append(t)
            bullish_swing_y.append(df.loc[t, "High"]) # type: ignore
        else:
            nearest_time = min(df.index, key=lambda x: abs(x - t))
            bullish_swing_x.append(nearest_time)
            bullish_swing_y.append(df.loc[nearest_time, "Low"])

    # Tính toán tọa độ cho Bearish ChoCh và Swing ChoCh
    bearish_x, bearish_y = [], []
    bearish_swing_x, bearish_swing_y = [], []
    for t in ChoCh_internal_bearish:
        if t in df.index:
            bearish_x.append(t)
            bearish_y.append(df.loc[t, "High"]) # type: ignore
        else:
            nearest_time = min(df.index, key=lambda x: abs(x - t))
            bearish_x.append(nearest_time)
            bearish_y.append(df.loc[nearest_time, "Low"])
    for t in ChoCh_lows_bearish:
        if t in df.index:
            bearish_x.append(t)
            bearish_y.append(df.loc[t, "High"]) # type: ignore
        else:
            nearest_time = min(df.index, key=lambda x: abs(x - t))
            bearish_x.append(nearest_time)
            bearish_y.append(df.loc[nearest_time, "Low"])

    if trend_direction == BULLISH:
        if internal:
            fig.add_trace(go.Scatter(
                x=bullish_x,
                y=bullish_y,
                mode='markers+text',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                text=["ChoCh"] * len(bullish_x),
                textposition="top center",
                textfont=dict(color='green', size=12, family="Arial Black"),
                name='ChoCh'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=bullish_swing_x,
                y=bullish_swing_y,
                mode='markers+text',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                text=["Swing ChoCh"] * len(bullish_swing_x),
                textposition="top center",
                textfont=dict(color='green', size=12, family="Arial Black"),
                name='Swing ChoCh'
            ))
    elif trend_direction == BEARISH:
        if internal:
            fig.add_trace(go.Scatter(
                x=bearish_x,
                y=bearish_y,
                mode='markers+text',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                text=["ChoCh"] * len(bearish_x),
                textposition="bottom center",
                textfont=dict(color='red', size=12, family="Arial Black"),
                name='ChoCh'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=bearish_swing_x,
                y=bearish_swing_y,
                mode='markers+text',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                text=["Swing ChoCh"] * len(bearish_swing_x),
                textposition="bottom center",
                textfont=dict(color='green', size=12, family="Arial Black"),
                name='Swing ChoCh'
            ))
    elif trend_direction == NEUTRAL:
        if internal:
            fig.add_trace(go.Scatter(
                x=bullish_x,
                y=bullish_y,
                mode='markers+text',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                text=["ChoCh"] * len(bullish_x),
                textposition="top center",
                textfont=dict(color='green', size=12, family="Arial Black"),
                name='ChoCh'
            ))
            fig.add_trace(go.Scatter(
                x=bearish_x,
                y=bearish_y,
                mode='markers+text',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                text=["ChoCh"] * len(bearish_x),
                textposition="bottom center",
                textfont=dict(color='red', size=12, family="Arial Black"),
                name='ChoCh'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=bullish_swing_x,
                y=bullish_swing_y,
                mode='markers+text',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                text=["Swing ChoCh"] * len(bullish_swing_x),
                textposition="top center",
                textfont=dict(color='green', size=12, family="Arial Black"),
                name='Swing ChoCh'
            ))
            fig.add_trace(go.Scatter(
                x=bearish_swing_x,
                y=bearish_swing_y,
                mode='markers+text',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                text=["Swing ChoCh"] * len(bearish_swing_x),
                textposition="bottom center",
                textfont=dict(color='green', size=12, family="Arial Black"),
                name='Swing ChoCh'
            ))
    return fig

# ====================== IDENTIFY EQUAL HIGHS LOWS ======================
def identify_equal_highs_lows(equalHighsLowsThresholdInput: float = 0.1, size: int = 1) -> tuple:
    """
    Identify Equal High and Equal Low groups.
    """

    # Calculate ATR
    atr = compute_atr(np.array(highs), np.array(lows), np.array(closes), period=200)
    if atr is None:
        print("Unable to compute ATR, returning two empty lists.")
        return [], []
    threshold_value = equalHighsLowsThresholdInput * atr

    equal_high_groups = []
    # Iterate over swing_highs, compare pivot[i] with pivot[i+size]
    for i in range(len(internal_swing_highs) - size):
        current = internal_swing_highs[i]
        compare_with = internal_swing_highs[i+size]
        if abs(current.level - compare_with.level) < threshold_value:  # type: ignore
            equal_high_groups.append([current, compare_with])
    # for group in equal_high_groups:
    #     print("Equal High Group: " + str(group))

    equal_low_groups = []
    # Iterate over swing_lows, compare pivot[i] with pivot[i+size]
    for i in range(len(internal_swing_lows) - size):
        current = internal_swing_lows[i]
        compare_with = internal_swing_lows[i+size]
        if abs(current.level - compare_with.level) < threshold_value:  # type: ignore
            equal_low_groups.append([current, compare_with])
    # for group in equal_low_groups:
    #     print("Equal Low Group: " + str(group))

    return equal_high_groups, equal_low_groups

# ====================== DRAW EQUAL HIGHS LOWS ======================
def draw_equal_highs_low(fig: go.Figure, equal_high_groups: list, equal_low_groups: list) -> go.Figure:
    """
    Draw dotted lines connecting the peaks of Equal High and Equal Low pairs,
    and annotate with "EQH" and "EQL" at the midpoint of each line.
    
    Parameters:
        fig (go.Figure): The Plotly Figure on which to draw.
        equal_high_groups (list): A list of Equal High groups (e.g., [[Pivot1, Pivot2], ...]).
        equal_low_groups (list): A list of Equal Low groups (e.g., [[Pivot1, Pivot2], ...]).
        
    Returns:
        go.Figure: The updated Figure.
    """
    # Draw connection lines for Equal High groups with the label "EQH".
    for group in equal_high_groups:
        start_pivot = group[0]
        end_pivot = group[1]

        # Calculate the midpoint based on time (bar_time)
        midpoint_time = start_pivot.bar_time + (end_pivot.bar_time - start_pivot.bar_time) / 2

        # Calculate the average price level between both pivots for label placement.
        midpoint_level = (start_pivot.level + end_pivot.level) / 2

        fig.add_shape(
            type="line",
            x0=start_pivot.bar_time, y0=start_pivot.level,
            x1=end_pivot.bar_time, y1=end_pivot.level,
            line=dict(color="blue", dash="dot", width=2),
            name="Equal High"
        )
        fig.add_annotation(
            x=midpoint_time,
            y=midpoint_level,
            text="EQH",
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font=dict(color="blue", size=12)
        )

    # Draw connection lines for Equal Low groups with the label "EQL".
    for group in equal_low_groups:
        start_pivot = group[0]
        end_pivot = group[1]

        midpoint_time = start_pivot.bar_time + (end_pivot.bar_time - start_pivot.bar_time) / 2
        midpoint_level = (start_pivot.level + end_pivot.level) / 2

        fig.add_shape(
            type="line",
            x0=start_pivot.bar_time, y0=start_pivot.level,
            x1=end_pivot.bar_time, y1=end_pivot.level,
            line=dict(color="red", dash="dot", width=2),
            name="Equal Low"
        )
        fig.add_annotation(
            x=midpoint_time,
            y=midpoint_level,
            text="EQL",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(color="orange", size=12)
        )
    return fig

# ====================== IDENTIFY ORDER BLOCKS ======================
def identify_orderblock(df, internal=True) -> None:
    global internal_order_blocks, swing_order_blocks
    current_trend = detect_trend()
    last_candle_time = df.index[-1]

    if internal:
        internal_order_blocks = build_internal_order_blocks(
            df, current_trend, last_candle_time
        )
        # print_order_blocks("Internal Order Blocks", internal_order_blocks)
    else:
        swing_order_blocks = build_swing_order_blocks(
            df, current_trend, last_candle_time
        )
        # print_order_blocks("External Swing Order Blocks", swing_order_blocks)

# ====================== IDENTIFY ORDER BLOCKS ======================
def build_internal_order_blocks(df, current_trend, last_candle_time):
    blocks = []
    if current_trend == BULLISH:
        # Xử lý dựa trên internal_swing_lows
        if len(internal_swing_lows) < 2:
            return blocks
        blocks = process_swings(df, internal_swing_lows, BULLISH)
    elif current_trend == BEARISH:
        if len(internal_swing_highs) < 2:
            return blocks
        blocks = process_swings(df, internal_swing_highs, BEARISH)
    elif current_trend == NEUTRAL:
        blocks_lows = process_swings(df, internal_swing_lows, NEUTRAL) if len(internal_swing_lows) >= 2 else []
        blocks_highs = process_swings(df, internal_swing_highs, NEUTRAL) if len(internal_swing_highs) >= 2 else []
        blocks = blocks_lows + blocks_highs

    blocks = [ob for ob in blocks if ob.start != ob.end]
    blocks = filter_order_blocks(df, blocks)
    blocks = update_order_blocks(df, blocks, last_candle_time)
    return blocks

def build_swing_order_blocks(df, current_trend, last_candle_time):
    blocks = []
    if current_trend == BULLISH:
        if len(swing_lows) < 2:
            return blocks
        blocks = process_swings(df, swing_lows, BULLISH)
    elif current_trend == BEARISH:
        if len(swing_highs) < 2:
            return blocks
        blocks = process_swings(df, swing_highs, BEARISH)
    elif current_trend == NEUTRAL:
        blocks_lows = process_swings(df, swing_lows, NEUTRAL) if len(swing_lows) >= 2 else []
        blocks_highs = process_swings(df, swing_highs, NEUTRAL) if len(swing_highs) >= 2 else []
        blocks = blocks_lows + blocks_highs

    blocks = [ob for ob in blocks if ob.start != ob.end]
    blocks = filter_order_blocks(df, blocks)
    blocks = update_order_blocks(df, blocks, last_candle_time)
    return blocks

def process_swings(df, swings, bias):
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
            # For NEUTRAL, only process with 2 swings (to avoid complexity)

        # Process with 2 adjacent swings
        current = swings[i]
        adjacent = swings[i - 1]
        s_time, e_time = sorted([current.bar_time, adjacent.bar_time])
        dfrange = df[(df.index >= s_time) & (df.index <= e_time)]
        index_low, row_low = dfrange['Low'].idxmin(), dfrange.loc[dfrange['Low'].idxmin()]
        index_high, row_high = dfrange['High'].idxmax(), dfrange.loc[dfrange['High'].idxmax()]
        if not dfrange.empty:
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
                ob = OrderBlock(
                    start=index_high,
                    end=e_time,
                    level_y0=row_high["Low"],
                    level_y1=row_high["High"],
                    bias=bias
                )
                blocks.append(ob)
        i -= 1
    return blocks

def filter_order_blocks(df, blocks):
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

def update_order_blocks(df, blocks, last_candle_time):
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

def print_order_blocks(title, blocks):
    print(title + ":")
    for ob in blocks:
        print(ob)

def draw_orderblock(fig: go.Figure, order_blocks: list, internal = True) -> go.Figure:
    for block in order_blocks:
        if internal:
            if block.bias == BULLISH:
                fig.add_shape(
                    type="rect",
                    x0=block.start, x1=block.end,
                    y0=block.level_y0, y1=block.level_y1,
                    fillcolor="green",
                    opacity=0.1,
                    layer="below",
                    line=dict(color="green")
                )
            elif block.bias == BEARISH:
                fig.add_shape(
                    type="rect",
                    x0=block.start, x1=block.end,
                    y0=block.level_y0, y1=block.level_y1,
                    fillcolor="red",
                    opacity=0.1,
                    layer="below",
                    line=dict(color="red")
                )
            elif block.bias == NEUTRAL:
                fig.add_shape(
                    type="rect",
                    x0=block.start, x1=block.end,
                    y0=block.level_y0, y1=block.level_y1,
                    fillcolor="gray",
                    opacity=0.1,
                    layer="below",
                    line=dict(color="gray"))
        else:
            if block.bias == BULLISH:
                fig.add_shape(
                    type="rect",
                    x0=block.start, x1=block.end,
                    y0=block.level_y0, y1=block.level_y1,
                    fillcolor="blue",
                    opacity=0.1,
                    layer="below",
                    line=dict(color="blue")
                )
            elif block.bias == BEARISH:
                fig.add_shape(
                    type="rect",
                    x0=block.start, x1=block.end,
                    y0=block.level_y0, y1=block.level_y1,
                    fillcolor="orange",
                    opacity=0.1,
                    layer="below",
                    line=dict(color="orange")
                )
            elif block.bias == NEUTRAL:
                fig.add_shape(
                    type="rect",
                    x0=block.start, x1=block.end,
                    y0=block.level_y0, y1=block.level_y1,
                    fillcolor="magenta",
                    opacity=0.1,
                    layer="below",
                    line=dict(color="magenta"))
    return fig

# ======================
def export_data(df: pd.DataFrame, order_of_swing=30, backtest=True):
    """
    Export processed data from a DataFrame with order_of_swing = 30.
    """
    # Prepare DataFrame (similar to df_filtered in main())
    if backtest:
        df_filtered = pd.DataFrame({
            "Date": df.index,
            "Open": df["Open"],
            "High": df["High"],
            "Low": df["Low"],
            "Close": df["Close"]
        })
    else:
        df_filtered = pd.DataFrame({
            "Date": df["Date"],
            "Open": df["Open"],
            "High": df["High"],
            "Low": df["Low"],
            "Close": df["Close"]
        })

    # (If necessary, remove the last candle as in main())
    # df_filtered = df_filtered.iloc[:-1]
    df_filtered.set_index("Date", inplace=True)

    # Update OHLC and time lists
    global opens, highs, lows, closes, times
    opens = df_filtered["Open"].to_list()
    highs = df_filtered["High"].to_list()
    lows = df_filtered["Low"].to_list()
    closes = df_filtered["Close"].to_list()
    times = df_filtered.index.tolist()

    # Calculate internal swings (using default values)
    get_swing_high_low(df_filtered, internal=True)
    # Calculate external swings with order_of_swing = 30
    get_swing_high_low(df_filtered, internal=False, order=order_of_swing)
    # Identify BOS pivots for both internal and external swings
    identify_pivot_bos(df_filtered, internal=True)
    identify_pivot_bos(df_filtered, internal=False)

    # Identify ChoCh for both internal and external swings
    identify_pivot_ChoCh(internal=True)
    identify_pivot_ChoCh(internal=False)

    # Identify OrderBlock for both internal and external swings
    identify_orderblock(df_filtered, internal=True)
    identify_orderblock(df_filtered, internal=False)

    # Determine the current trend
    trend = detect_trend()

    # Return the values in the required format
    return (opens, highs, lows, closes, times, trend,
            internal_swing_highs, ChoCh_internal_bullish, ChoCh_internal_bearish, internal_order_blocks,
            swing_highs, swing_lows, ChoCh_swings_bullish, ChoCh_lows_bearish, swing_order_blocks)

def main():
    global opens, highs, lows, closes, times

    # Prompt the user to enter a stock ticker symbol, default to "AAPL" if nothing is entered
    ticker = input("Enter stock ticker symbol: ") or "AAPL"

    # Download stock data from yfinance for the given ticker symbol
    df = yf.download(ticker, start="2024-01-01", end="2025-02-09", interval="1d")
    if df is None or df.empty:
        print("⚠️ Error: Could not download data for {} from yfinance.".format(ticker))
        return

    # Prepare the DataFrame with necessary columns
    df_filtered = pd.DataFrame({
        "Date": df.index,
        "Open": df["Open"].squeeze(),
        "High": df["High"].squeeze(),
        "Low": df["Low"].squeeze(),
        "Close": df["Close"].squeeze()
    })
    df_filtered = df_filtered.iloc[:-1]
    df_filtered.set_index("Date", inplace=True)

    # Populate OHLC lists globally
    opens = df_filtered["Open"].to_list()
    highs = df_filtered["High"].to_list()
    lows = df_filtered["Low"].to_list()
    closes = df_filtered["Close"].to_list()
    times = df_filtered.index.tolist()

    # Create a candlestick chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_filtered.index,
        open=df_filtered["Open"],
        high=df_filtered["High"],
        low=df_filtered["Low"],
        close=df_filtered["Close"],
        name="{} Candlestick".format(ticker)
    ))
    fig.update_xaxes(range=[df_filtered.index.min(), df_filtered.index.max()])

    # Compute and display ATR
    atr_value = compute_atr(df_filtered["High"].to_numpy(), df_filtered["Low"].to_numpy(), df_filtered["Close"].to_numpy(), 200)
    # if atr_value is not None:
    #     print(f"Computed ATR: {atr_value}")

    # Identify internal swing highs/lows
    get_swing_high_low(df_filtered)
    fig = draw_swing_high_low(fig)

    # Identify external swing highs/lows
    get_swing_high_low(df_filtered, internal=False, order=30)
    fig = draw_swing_high_low(fig, internal=False)

    # INTERNAL BOS (Break of Structure)
    identify_pivot_bos(df_filtered)
    fig = draw_pivot_bos(fig, internal=True)

    # SWING BOS (Break of Structure)
    identify_pivot_bos(df_filtered, internal=False)
    fig = draw_pivot_bos(fig, internal=False)

    # Change of Character (ChoCh)
    identify_pivot_ChoCh()
    fig = draw_ChoCh(fig, df_filtered)
    identify_pivot_ChoCh(internal=False)
    fig = draw_ChoCh(fig, df_filtered, internal=False)

    # Equal Highs and Equal Lows (EQH/EQL)
    equal_high_groups, equal_low_groups = identify_equal_highs_lows()
    fig = draw_equal_highs_low(fig, equal_high_groups, equal_low_groups)

    # ORDER BLOCK
    identify_orderblock(df_filtered)
    fig = draw_orderblock(fig, internal_order_blocks)
    identify_orderblock(df_filtered, internal=False)
    fig = draw_orderblock(fig, swing_order_blocks, internal=False)

    fig = draw_weak_high_low(fig, df_filtered)
    fig = draw_strong_high_low(fig, df_filtered)
    fig.show()

if __name__ == "__main__":
    main()
