import plotly.graph_objects as go

from ..analyzer import SMCState

BOS_COLORS = {
    "bullish": "green",
    "bearish": "red",
}

BOS_LABELS = {
    True: "BOS",
    False: "Swing BOS",
}

BOS_YANCHOR = {
    "bullish": "bottom",
    "bearish": "top",
}


def _draw_bos_lines(fig: go.Figure, bos_df, pivot_col: str, is_bullish: bool, label: str) -> None:
    color = BOS_COLORS["bullish" if is_bullish else "bearish"]
    yanchor = BOS_YANCHOR["bullish" if is_bullish else "bearish"]

    for _, row in bos_df.iterrows():
        fig.add_shape(
            type="line",
            x0=row[pivot_col],
            y0=row["Pivot_level"],
            x1=row["Crossing_Time"],
            y1=row["Pivot_level"],
            line=dict(color=color, dash="dash"),
        )
        midpoint = row[pivot_col] + (row["Crossing_Time"] - row[pivot_col]) / 2
        fig.add_annotation(
            x=midpoint,
            y=row["Pivot_level"],
            text=label,
            showarrow=False,
            xanchor="center",
            yanchor=yanchor,
            font=dict(color=color),
        )


def draw_pivot_bos(fig: go.Figure, state: SMCState, internal: bool = True) -> go.Figure:
    if internal:
        high_bos = state.internal_structure.bullish_bos
        low_bos = state.internal_structure.bearish_bos
    else:
        high_bos = state.swing_structure.bullish_bos
        low_bos = state.swing_structure.bearish_bos

    label = BOS_LABELS[internal]
    _draw_bos_lines(fig, high_bos, "Pivot_bullishBos_Time", is_bullish=True, label=label)
    _draw_bos_lines(fig, low_bos, "Pivot_bearishBos_Time", is_bullish=False, label=label)

    return fig
