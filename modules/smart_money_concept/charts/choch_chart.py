import pandas as pd
import plotly.graph_objects as go

from ..analyzer import SMCState
from ..core.constants import BEARISH, BULLISH, NEUTRAL
from modules.common.ui.logging import log_warn


def _get_coords(df: pd.DataFrame, times: list) -> tuple:
    x_coords, y_coords = [], []
    for t in times:
        if t in df.index:
            x_coords.append(t)
            y_coords.append(df.loc[t, "High"])
        else:
            try:
                nearest_time = min(df.index, key=lambda x: abs(x - t))
                x_coords.append(nearest_time)
                y_coords.append(df.loc[nearest_time, "Low"])
            except (KeyError, ValueError):
                pass
    return x_coords, y_coords


def draw_ChoCh(fig: go.Figure, df: pd.DataFrame, state: SMCState, internal: bool = True) -> go.Figure:
    trend_direction = state.trend

    if internal:
        bullish_times = state.internal_structure.bullish_choch.get(
            "Pivot_bullishChoch_Time", pd.Series(dtype="datetime64[ns]")
        )
        bearish_times = state.internal_structure.bearish_choch.get(
            "Pivot_bearishChoch_Time", pd.Series(dtype="datetime64[ns]")
        )
        bullish_x, bullish_y = _get_coords(df, list(bullish_times))
        bearish_x, bearish_y = _get_coords(df, list(bearish_times))
    else:
        bullish_swing_times = state.swing_structure.bullish_choch.get(
            "Pivot_bullishChoch_Time", pd.Series(dtype="datetime64[ns]")
        )
        bearish_swing_times = state.swing_structure.bearish_choch.get(
            "Pivot_bearishChoch_Time", pd.Series(dtype="datetime64[ns]")
        )
        bullish_swing_x, bullish_swing_y = _get_coords(df, list(bullish_swing_times))
        bearish_swing_x, bearish_swing_y = _get_coords(df, list(bearish_swing_times))

    if trend_direction == BULLISH:
        if internal:
            if bullish_x:
                fig.add_trace(
                    go.Scatter(
                        x=bullish_x,
                        y=bullish_y,
                        mode="markers+text",
                        marker=dict(color="green", size=10, symbol="triangle-up"),
                        text=["ChoCh"] * len(bullish_x),
                        textposition="top center",
                        textfont=dict(color="green", size=12, family="Arial Black"),
                        name="ChoCh",
                    )
                )
        else:
            if bullish_swing_x:
                fig.add_trace(
                    go.Scatter(
                        x=bullish_swing_x,
                        y=bullish_swing_y,
                        mode="markers+text",
                        marker=dict(color="green", size=10, symbol="triangle-up"),
                        text=["Swing ChoCh"] * len(bullish_swing_x),
                        textposition="top center",
                        textfont=dict(color="green", size=12, family="Arial Black"),
                        name="Swing ChoCh",
                    )
                )
    elif trend_direction == BEARISH:
        if internal:
            if bearish_x:
                fig.add_trace(
                    go.Scatter(
                        x=bearish_x,
                        y=bearish_y,
                        mode="markers+text",
                        marker=dict(color="red", size=10, symbol="triangle-down"),
                        text=["ChoCh"] * len(bearish_x),
                        textposition="bottom center",
                        textfont=dict(color="red", size=12, family="Arial Black"),
                        name="ChoCh",
                    )
                )
        else:
            if bearish_swing_x:
                fig.add_trace(
                    go.Scatter(
                        x=bearish_swing_x,
                        y=bearish_swing_y,
                        mode="markers+text",
                        marker=dict(color="green", size=10, symbol="triangle-up"),
                        text=["Swing ChoCh"] * len(bearish_swing_x),
                        textposition="bottom center",
                        textfont=dict(color="green", size=12, family="Arial Black"),
                        name="Swing ChoCh",
                    )
                )
    elif trend_direction == NEUTRAL:
        if internal:
            if bullish_x:
                fig.add_trace(
                    go.Scatter(
                        x=bullish_x,
                        y=bullish_y,
                        mode="markers+text",
                        marker=dict(color="green", size=10, symbol="triangle-up"),
                        text=["ChoCh"] * len(bullish_x),
                        textposition="top center",
                        textfont=dict(color="green", size=12, family="Arial Black"),
                        name="ChoCh",
                    )
                )
            if bearish_x:
                fig.add_trace(
                    go.Scatter(
                        x=bearish_x,
                        y=bearish_y,
                        mode="markers+text",
                        marker=dict(color="red", size=10, symbol="triangle-down"),
                        text=["ChoCh"] * len(bearish_x),
                        textposition="bottom center",
                        textfont=dict(color="red", size=12, family="Arial Black"),
                        name="ChoCh",
                    )
                )
        else:
            if bullish_swing_x:
                fig.add_trace(
                    go.Scatter(
                        x=bullish_swing_x,
                        y=bullish_swing_y,
                        mode="markers+text",
                        marker=dict(color="green", size=10, symbol="triangle-up"),
                        text=["Swing ChoCh"] * len(bullish_swing_x),
                        textposition="top center",
                        textfont=dict(color="green", size=12, family="Arial Black"),
                        name="Swing ChoCh",
                    )
                )
            if bearish_swing_x:
                fig.add_trace(
                    go.Scatter(
                        x=bearish_swing_x,
                        y=bearish_swing_y,
                        mode="markers+text",
                        marker=dict(color="green", size=10, symbol="triangle-up"),
                        text=["Swing ChoCh"] * len(bearish_swing_x),
                        textposition="bottom center",
                        textfont=dict(color="green", size=12, family="Arial Black"),
                        name="Swing ChoCh",
                    )
                )
    return fig
