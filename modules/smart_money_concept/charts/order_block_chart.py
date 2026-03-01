import plotly.graph_objects as go

from ..analyzer import SMCState
from ..core.order_block import BEARISH, BULLISH, NEUTRAL


def draw_orderblock(fig: go.Figure, state: SMCState, internal: bool = True) -> go.Figure:
    order_blocks = state.ob_internal if internal else state.ob_swing
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
