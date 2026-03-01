import plotly.graph_objects as go

from ..analyzer import SMCState


def draw_equal_highs_low(fig: go.Figure, state: SMCState) -> go.Figure:
    equal_high_groups = state.equal_hl.equal_highs
    equal_low_groups = state.equal_hl.equal_lows

    for group in equal_high_groups:
        start_pivot = group[0]
        end_pivot = group[1]

        midpoint_time = start_pivot.bar_time + (end_pivot.bar_time - start_pivot.bar_time) / 2
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
