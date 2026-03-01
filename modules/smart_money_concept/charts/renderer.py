import plotly.graph_objects as go

from ..analyzer import SMCState
from .bos_chart import draw_pivot_bos
from .choch_chart import draw_ChoCh
from .equal_hl_chart import draw_equal_highs_low
from .order_block_chart import draw_orderblock
from .swing_chart import draw_strong_high_low, draw_swing_high_low, draw_weak_high_low


class SMCChartRenderer:
    def __init__(self, title: str = "Smart Money Concepts"):
        self.title = title

    def render(self, state: SMCState, ticker: str) -> go.Figure:
        df = state.ohlcv
        opens = df["Open"].to_list()
        highs = df["High"].to_list()
        lows = df["Low"].to_list()
        closes = df["Close"].to_list()
        times = df.index.tolist()

        fig = go.Figure()

        # Base Candlestick
        fig.add_trace(go.Candlestick(
            x=times,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name='Candlestick'
        ))

        # Swings
        fig = draw_swing_high_low(fig, state, internal=True)
        fig = draw_swing_high_low(fig, state, internal=False)
        fig = draw_weak_high_low(fig, df, state)
        fig = draw_strong_high_low(fig, df, state)

        # BOS
        fig = draw_pivot_bos(fig, state, internal=True)
        fig = draw_pivot_bos(fig, state, internal=False)

        # ChoCh
        fig = draw_ChoCh(fig, df, state, internal=True)
        fig = draw_ChoCh(fig, df, state, internal=False)

        # Equal Highs / Lows
        fig = draw_equal_highs_low(fig, state)

        # Order Blocks
        fig = draw_orderblock(fig, state, internal=True)
        fig = draw_orderblock(fig, state, internal=False)

        fig.update_layout(
            title=f'{self.title} - {ticker}',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )

        return fig
