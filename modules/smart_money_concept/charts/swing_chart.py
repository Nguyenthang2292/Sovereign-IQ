import pandas as pd
import plotly.graph_objects as go

from ..analyzer import SMCState
from ..core.constants import BEARISH, BULLISH, NEUTRAL
from ..core.swing import classify_swing_types


def draw_swing_high_low(fig: go.Figure, state: SMCState, internal: bool = True) -> go.Figure:
    if internal:
        swing_high_x = [p.bar_time for p in state.swings.internal_highs if p.bar_time is not None]
        swing_high_y = [p.level for p in state.swings.internal_highs if p.bar_time is not None]
        swing_low_x = [p.bar_time for p in state.swings.internal_lows if p.bar_time is not None]
        swing_low_y = [p.level for p in state.swings.internal_lows if p.bar_time is not None]

        fig.add_trace(go.Scatter(
            x=swing_high_x,
            y=swing_high_y,
            mode='markers',
            marker=dict(color='red', symbol='triangle-up', size=10),
            name='Internal Swing High'
        ))
        fig.add_trace(go.Scatter(
            x=swing_low_x,
            y=swing_low_y,
            mode='markers',
            marker=dict(color='blue', symbol='triangle-down', size=10),
            name='Internal Swing Low'
        ))
    else:
        classified_highs, classified_lows = classify_swing_types(state.swings.swing_highs, state.swings.swing_lows)

        swing_high_x = [p.bar_time for (p, _) in classified_highs if p.bar_time is not None]
        swing_high_y = [p.level for (p, _) in classified_highs if p.bar_time is not None]
        swing_low_x = [p.bar_time for (p, _) in classified_lows if p.bar_time is not None]
        swing_low_y = [p.level for (p, _) in classified_lows if p.bar_time is not None]

        fig.add_trace(go.Scatter(
            x=swing_high_x,
            y=swing_high_y,
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Swing High'
        ))

        for (p, cls) in classified_highs:
            if p.bar_time is not None:
                fig.add_annotation(
                    x=p.bar_time,
                    y=p.level,
                    text=cls,
                    showarrow=False,
                    yanchor='bottom',
                    font=dict(color='green')
                )

        fig.add_trace(go.Scatter(
            x=swing_low_x,
            y=swing_low_y,
            mode='markers',
            marker=dict(color='purple', symbol='triangle-down', size=10),
            name='Swing Low'
        ))

        for (p, cls) in classified_lows:
            if p.bar_time is not None:
                fig.add_annotation(
                    x=p.bar_time,
                    y=p.level,
                    text=cls,
                    showarrow=False,
                    yanchor='top',
                    font=dict(color='purple')
                )
    return fig

def draw_weak_high_low(fig: go.Figure, df: pd.DataFrame, state: SMCState) -> go.Figure:
    trend_direction = state.trend
    internal_highs = state.swings.internal_highs
    internal_lows = state.swings.internal_lows

    if not internal_highs or not internal_lows:
        return fig

    max_swing_high = max(internal_highs, key=lambda x: x.level)
    max_swing_low = min(internal_lows, key=lambda x: x.level)

    try:
        max_date = df.index[df['High'] == max_swing_high.level][0]
    except IndexError:
        max_date = max_swing_high.bar_time

    try:
        min_date = df.index[df['Low'] == max_swing_low.level][0]
    except IndexError:
        min_date = max_swing_low.bar_time

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

def draw_strong_high_low(fig: go.Figure, df: pd.DataFrame, state: SMCState) -> go.Figure:
    trend_direction = state.trend
    classified_highs, classified_lows = classify_swing_types(state.swings.swing_highs, state.swings.swing_lows)

    if trend_direction == BULLISH:
        strong_low_candidates = [p for p, label in classified_lows if label == "LH"]
        for strong_low in strong_low_candidates:
            try:
                strong_date = df.index[df['Low'] == strong_low.level][0]
            except IndexError:
                strong_date = strong_low.bar_time

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
        strong_high_candidates = [p for p, label in classified_highs if label == "HL"]
        for strong_high in strong_high_candidates:
            try:
                strong_date = df.index[df['High'] == strong_high.level][0]
            except IndexError:
                strong_date = strong_high.bar_time

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
