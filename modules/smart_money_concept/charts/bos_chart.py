import plotly.graph_objects as go

from ..analyzer import SMCState


def draw_pivot_bos(fig: go.Figure, state: SMCState, internal: bool = True) -> go.Figure:
    if internal:
        high_bos = state.internal_structure.bullish_bos
        low_bos = state.internal_structure.bearish_bos

        for _, row in high_bos.iterrows():
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

        for _, row in low_bos.iterrows():
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
        high_bos = state.swing_structure.bullish_bos
        low_bos = state.swing_structure.bearish_bos

        for _, row in high_bos.iterrows():
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

        for _, row in low_bos.iterrows():
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
                font=dict(color='red')
            )

    return fig
