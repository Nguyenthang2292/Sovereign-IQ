import plotly.graph_objects as go

from ..analyzer import SMCState
from ..core.constants import BEARISH, BULLISH, NEUTRAL

INTERNAL_COLORS = {
    BULLISH: "green",
    BEARISH: "red",
    NEUTRAL: "gray",
}

EXTERNAL_COLORS = {
    BULLISH: "blue",
    BEARISH: "orange",
    NEUTRAL: "magenta",
}


def draw_orderblock(fig: go.Figure, state: SMCState, internal: bool = True) -> go.Figure:
    order_blocks = state.ob_internal if internal else state.ob_swing
    colors = INTERNAL_COLORS if internal else EXTERNAL_COLORS

    for block in order_blocks:
        color = colors.get(block.bias, "gray")
        fig.add_shape(
            type="rect",
            x0=block.start,
            x1=block.end,
            y0=block.level_y0,
            y1=block.level_y1,
            fillcolor=color,
            opacity=0.1,
            layer="below",
            line=dict(color=color),
        )
    return fig
