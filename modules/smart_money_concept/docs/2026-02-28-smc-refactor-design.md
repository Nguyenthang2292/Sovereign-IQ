# SMC v3.0 → Sub-Module Refactor Design

**Date:** 2026-02-28  
**Status:** Approved  
**Source file:** `modules/smart_money_concept/SMC_v3_0.py` (1430 lines)

---

## Goals

- Support **both** standalone CLI usage and integration into `auto_trade` pipeline
- Eliminate **global state** (`global internal_swing_highs`, etc.)
- Separate **visualization (Plotly)** completely from business logic
- Make every `core/` function independently **unit-testable**
- Maintain **backward compatibility** via `SMCAnalyzer.export()`

---

## Architecture Decisions

| Question | Decision |
|---|---|
| Usage mode | Both standalone + integrated |
| State management | Hybrid: pure stateless `core/` + orchestrating `SMCAnalyzer` class |
| Chart layer | Fully separated into `charts/` — `core/` never imports Plotly |
| Data classes | Moved into `models/` sub-package inside `smart_money_concept/` |

---

## Target Folder Structure

```
modules/smart_money_concept/
│
├── __init__.py                  # Re-exports: SMCAnalyzer, SMCState, Pivot, OrderBlock
│
├── models/                      # Data classes only — no logic, no Plotly
│   ├── __init__.py
│   ├── pivot.py                 # Moved from modules/smart_money_concept/pivot.py
│   └── order_block.py           # Ported from data_class.class_order_block
│
├── core/                        # Pure stateless functions — no global, no Plotly
│   ├── __init__.py
│   ├── trend.py                 # detect_trend(), compute_atr()
│   ├── swing.py                 # detect_swings(), classify_swing_types()
│   ├── bos.py                   # identify_bos()
│   ├── choch.py                 # identify_choch()
│   ├── equal_hl.py              # identify_equal_hl()
│   └── order_block.py           # build_*, process_swings(), filter_*, update_*
│
├── charts/                      # All Plotly draw_* functions — only layer importing plotly
│   ├── __init__.py
│   ├── renderer.py              # SMCChartRenderer — single entry point
│   ├── swing_chart.py           # draw_swing_high_low(), draw_weak/strong_high_low()
│   ├── bos_chart.py             # draw_pivot_bos()
│   ├── choch_chart.py           # draw_choch()
│   ├── equal_hl_chart.py        # draw_equal_highs_low()
│   └── order_block_chart.py     # draw_orderblock()
│
├── analyzer.py                  # SMCAnalyzer class (orchestrator)
├── cli.py                       # Standalone runner — replaces main()
└── docs/
    └── 2026-02-28-smc-refactor-design.md
```

---

## Dependency Rule (Strict)

```
cli.py
  └─► SMCChartRenderer (charts/)   ← imports plotly
  └─► SMCAnalyzer (analyzer.py)
        └─► core/*                  ← pure Python + pandas/numpy only
              └─► models/*          ← dataclasses only, zero imports
```

> **Core never imports charts. Models never import anything from the project.**

---

## Core Layer — Signature Changes

### `core/trend.py`

```python
# BEFORE (global):
def detect_trend() -> int:  # reads global vars

# AFTER (stateless):
BULLISH = 1
NEUTRAL = 0
BEARISH = -1

def detect_trend(swing_highs: list[Pivot], swing_lows: list[Pivot]) -> int:
    ...

def compute_atr(highs, lows, closes, period: int = 200) -> float | None:
    ...  # already stateless — no change needed
```

### `core/swing.py`

```python
@dataclass
class SwingResult:
    internal_highs: list[Pivot]
    internal_lows: list[Pivot]
    swing_highs: list[Pivot]    # external (order=30)
    swing_lows: list[Pivot]     # external (order=30)

def detect_swings(df: pd.DataFrame, internal_order: int = 5, external_order: int = 30) -> SwingResult:
    ...

def classify_swing_types(
    swing_highs: list[Pivot],
    swing_lows: list[Pivot],
) -> tuple[list[tuple[Pivot, str]], list[tuple[Pivot, str]]]:
    ...
```

### `core/bos.py`

```python
@dataclass
class BOSResult:
    high_bos: pd.DataFrame   # Bullish BOS records
    low_bos: pd.DataFrame    # Bearish BOS records

def identify_bos(
    df: pd.DataFrame,
    swing_highs: list[Pivot],
    swing_lows: list[Pivot],
) -> BOSResult:
    ...
```

### `core/choch.py`

```python
@dataclass
class ChochResult:
    bullish: list        # timestamps
    bearish: list        # timestamps

def identify_choch(
    bos: BOSResult,
    swing_highs: list[Pivot],
    swing_lows: list[Pivot],
) -> ChochResult:
    ...
```

### `core/equal_hl.py`

```python
@dataclass
class EqualHLResult:
    equal_highs: list[list[Pivot]]
    equal_lows: list[list[Pivot]]

def identify_equal_hl(
    internal_highs: list[Pivot],
    internal_lows: list[Pivot],
    highs_arr: np.ndarray,
    lows_arr: np.ndarray,
    closes_arr: np.ndarray,
    threshold_factor: float = 0.1,
    size: int = 1,
) -> EqualHLResult:
    ...
```

### `core/order_block.py`

```python
def identify_order_blocks(
    df: pd.DataFrame,
    swing_highs: list[Pivot],
    swing_lows: list[Pivot],
    trend: int,
) -> list[OrderBlock]:
    ...
```

---

## Orchestrator — `analyzer.py`

```python
@dataclass
class SMCState:
    """Complete snapshot after one analysis run."""
    swings: SwingResult
    trend: int
    bos_internal: BOSResult
    bos_swing: BOSResult
    choch_internal: ChochResult
    choch_swing: ChochResult
    equal_hl: EqualHLResult
    order_blocks_internal: list[OrderBlock]
    order_blocks_swing: list[OrderBlock]
    ohlcv: pd.DataFrame          # kept for chart layer

class SMCAnalyzer:
    """Orchestrator — no Plotly, no global state."""

    def run(self, df: pd.DataFrame, internal_order=5, external_order=30) -> SMCState:
        """Run full pipeline, return SMCState."""
        swings = detect_swings(df, internal_order, external_order)
        trend  = detect_trend(swings.internal_highs, swings.internal_lows)

        bos_internal = identify_bos(df, swings.internal_highs, swings.internal_lows)
        bos_swing    = identify_bos(df, swings.swing_highs, swings.swing_lows)

        choch_internal = identify_choch(bos_internal, swings.swing_highs, swings.swing_lows)
        choch_swing    = identify_choch(bos_swing,    swings.swing_highs, swings.swing_lows)

        highs_arr  = df["High"].to_numpy()
        lows_arr   = df["Low"].to_numpy()
        closes_arr = df["Close"].to_numpy()
        equal_hl   = identify_equal_hl(swings.internal_highs, swings.internal_lows,
                                       highs_arr, lows_arr, closes_arr)

        ob_int   = identify_order_blocks(df, swings.internal_highs, swings.internal_lows, trend)
        ob_swing = identify_order_blocks(df, swings.swing_highs, swings.swing_lows, trend)

        return SMCState(
            swings=swings, trend=trend,
            bos_internal=bos_internal, bos_swing=bos_swing,
            choch_internal=choch_internal, choch_swing=choch_swing,
            equal_hl=equal_hl,
            order_blocks_internal=ob_int, order_blocks_swing=ob_swing,
            ohlcv=df,
        )

    def export(self, df: pd.DataFrame) -> tuple:
        """Backward-compatible replacement for export_data()."""
        state = self.run(df)
        return (
            state.ohlcv["Open"].tolist(),
            state.ohlcv["High"].tolist(),
            state.ohlcv["Low"].tolist(),
            state.ohlcv["Close"].tolist(),
            state.ohlcv.index.tolist(),
            state.trend,
            state.swings.internal_highs,
            state.choch_internal.bullish,
            state.choch_internal.bearish,
            state.order_blocks_internal,
            state.swings.swing_highs,
            state.swings.swing_lows,
            state.choch_swing.bullish,
            state.choch_swing.bearish,
            state.order_blocks_swing,
        )
```

---

## Chart Layer — `charts/renderer.py`

```python
import plotly.graph_objects as go
from modules.smart_money_concept.analyzer import SMCState

class SMCChartRenderer:
    """Accepts SMCState, renders all layers onto a Figure."""

    def render(self, state: SMCState, ticker: str = "") -> go.Figure:
        fig = self._base_candlestick(state.ohlcv, ticker)
        fig = swing_chart.draw(fig, state)
        fig = bos_chart.draw(fig, state)
        fig = choch_chart.draw(fig, state)
        fig = equal_hl_chart.draw(fig, state)
        fig = order_block_chart.draw(fig, state)
        fig = swing_chart.draw_weak_strong(fig, state)
        return fig
```

---

## Standalone CLI — `cli.py`

```python
import yfinance as yf
from modules.smart_money_concept.analyzer import SMCAnalyzer
from modules.smart_money_concept.charts.renderer import SMCChartRenderer

def main():
    ticker = input("Enter stock ticker symbol: ") or "AAPL"
    df = yf.download(ticker, start="2024-01-01", end="2025-02-09", interval="1d")
    if df is None or df.empty:
        print(f"Error: No data for {ticker}")
        return

    df = df.squeeze()
    df = df.iloc[:-1]  # remove incomplete last candle

    state = SMCAnalyzer().run(df)
    fig   = SMCChartRenderer().render(state, ticker)
    fig.show()

if __name__ == "__main__":
    main()
```

---

## Integration Usage (auto_trade)

```python
from modules.smart_money_concept import SMCAnalyzer

analyzer = SMCAnalyzer()
state = analyzer.run(df)

# Access data cleanly — no globals
trend          = state.trend                   # BULLISH=1, BEARISH=-1, NEUTRAL=0
order_blocks   = state.order_blocks_internal   # list[OrderBlock]
swing_highs    = state.swings.internal_highs   # list[Pivot]
choch_bullish  = state.choch_internal.bullish  # list[timestamp]
```

---

## Test Structure

```
tests/smart_money_concept/
├── test_pivot.py           ← Existing ✅
├── test_order_block.py     ← New: models/order_block.py
├── test_swing.py           ← New: core/swing.py (pure functions)
├── test_trend.py           ← New: core/trend.py
├── test_bos.py             ← New: core/bos.py
├── test_choch.py           ← New: core/choch.py
├── test_equal_hl.py        ← New: core/equal_hl.py
├── test_order_block_core.py← New: core/order_block.py
└── test_analyzer.py        ← Integration: SMCAnalyzer.run()
```

---

## Layer Summary

| Layer | Purpose | Plotly | Global State |
|---|---|---|---|
| `models/` | Data classes | ❌ | ❌ |
| `core/` | Pure business logic | ❌ | ❌ |
| `analyzer.py` | Orchestrator + state container | ❌ | ❌ |
| `charts/` | Visualization | ✅ | ❌ |
| `cli.py` | Standalone entry point | ✅ | ❌ |
