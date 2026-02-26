# Gemini Gann Square Module — Design Document

**Date:** 2026-02-25  
**Status:** Approved for Implementation

---

## 1. Purpose

Combine Gann Square technical analysis with Google Gemini AI to produce trading signals (LONG/SHORT/SKIP) with Entry, Stop Loss, and Take Profit levels for a single crypto symbol.

---

## 2. Architecture

```
OHLCV Data (via common/core/data_fetcher)
    ↓
[swing_detector.py]  — Zigzag pivot detection → Swing High / Swing Low
    ↓
[gann_calculator.py] — Build Gann Square zones (4 zones), detect trend (UP/DOWN)
    ↓
[gann_chart_generator.py] — Draw candlestick chart + Gann zone overlay (PNG)
    ↓
[GeminiChartAnalyzer]    — Send chart image + structured prompt to Gemini
    ↓
[gann_signal_engine.py]  — Parse Gemini response → final recommendation
    ↓
CLI output (terminal)
```

---

## 3. Core Components

### 3.1 `swing_detector.py` — Pivot Zigzag Detection

- **Input:** OHLCV DataFrame
- **Algorithm:**
  - Swing High at index `i` if `high[i] == max(high[i-N : i+N+1])`
  - Swing Low at index `i` if `low[i] == min(low[i-N : i+N+1])`
  - Default lookback `N = 5` (configurable)
- **Output:**
  - `swing_highs: List[SwingPoint]` — all local pivot highs
  - `swing_lows: List[SwingPoint]` — all local pivot lows
  - `highest_swing: SwingPoint` — the highest pivot high
  - `lowest_swing: SwingPoint` — the lowest pivot low

```python
@dataclass
class SwingPoint:
    index: int          # candle index in DataFrame
    timestamp: datetime
    price: float        # high price (for swing high) or low price (for swing low)
    kind: Literal["high", "low"]
```

### 3.2 `gann_calculator.py` — Gann Square Zones

- **Inputs:** `highest_swing`, `lowest_swing`
- **Trend Detection:**
  - If `highest_swing.index < lowest_swing.index` → trend = **DOWN**
  - Else → trend = **UP**
- **Price Range:** `price_range = highest_swing.price - lowest_swing.price`
- **4 Zones (DOWN trend):**

| Zone | Upper Bound | Lower Bound | Signal |
|------|-------------|-------------|--------|
| 1    | swing_high  | swing_high - 0.25 × range | SHORT ✅ |
| 2    | swing_high - 0.25 × range | swing_high - 0.50 × range | SHORT ✅ |
| 3    | swing_high - 0.50 × range | swing_high - 0.75 × range | SKIP ⛔ |
| 4    | swing_high - 0.75 × range | swing_low  | SKIP ⛔ |

- **UP trend:** Zone 1,2 = LONG, Zone 3,4 = SKIP
- **Current Zone:** determined by current close price position

```python
@dataclass
class GannSquareResult:
    trend: Literal["UP", "DOWN"]
    swing_high: SwingPoint
    swing_low: SwingPoint
    price_range: float
    zones: List[GannZone]       # 4 zones
    current_zone: int           # 1-4
    signal_code: Literal["LONG", "SHORT", "SKIP"]
```

### 3.3 `gann_chart_generator.py` — Chart with Gann Overlay

Extends `ChartGenerator` pattern from `gemini_chart_analyzer`:

- Draw candlestick chart (dark background)
- Overlay 4 horizontal zones as colored bands:
  - Zone 1: semi-transparent red/green (active)
  - Zone 2: lighter red/green (active)
  - Zone 3: gray (skip)
  - Zone 4: darker gray (skip)
- Mark Swing High with `▼` marker + price label
- Mark Swing Low with `▲` marker + price label
- Horizontal dashed lines at zone boundaries
- Current price highlighted with yellow horizontal line
- Title: `"{SYMBOL} {TIMEFRAME} | Gann Square | Trend: {UP/DOWN} | Zone: {N} | Signal: {LONG/SHORT/SKIP}"`

### 3.4 `gann_signal_engine.py` — Orchestrator

```python
class GannSignalEngine:
    def analyze(symbol, timeframe, limit, lookback_n) -> GannAnalysisResult:
        1. Fetch OHLCV via DataFetcher
        2. Run swing_detector → SwingPoints
        3. Run gann_calculator → GannSquareResult
        4. Generate chart PNG via gann_chart_generator
        5. Build prompt (inject zone context + trend + current_zone)
        6. Call GeminiChartAnalyzer.analyze_chart()
        7. Parse Gemini response → GannAnalysisResult
        8. Return result
```

### 3.5 Gemini Prompt Template (`prompts/gann_analysis.txt`)

```
You are a professional crypto trader analyzing a {SYMBOL} {TIMEFRAME} chart using Gann Square methodology.

Context provided by the system:
- Detected Trend: {TREND}
- Swing High: {SWING_HIGH_PRICE} at {SWING_HIGH_TIME}
- Swing Low: {SWING_LOW_PRICE} at {SWING_LOW_TIME}
- Current Price: {CURRENT_PRICE}
- Current Zone: {CURRENT_ZONE} of 4
- Pre-calculated Signal: {PRECALC_SIGNAL}

The chart shows a Gann Square with 4 zones overlaid as colored bands.

Please:
1. CONFIRM or OVERRIDE the zone identification (Zone 1-4)
2. CONFIRM or OVERRIDE the trend direction (UP/DOWN)
3. Provide your FINAL recommendation:
   - Signal: LONG / SHORT / SKIP
   - Entry Price: [specific price]
   - Stop Loss: [specific price]
   - Take Profit 1: [specific price]
   - Take Profit 2: [specific price]
   - Confidence: [X%]
   - Reasoning: [2-3 sentences]

Format your response as JSON.
```

---

## 4. CLI Interface

### Command-line args mode

```bash
python -m modules.gemini_gann_square --symbol BTCUSDT --timeframe 4h --limit 200 --lookback 5
```

### Interactive menu mode (fallback)

```
=== GEMINI GANN SQUARE ===
1. Analyze symbol
2. Change settings
3. Exit
```

---

## 5. Output Format (Terminal)

```
══════════════════════════════════════════
  GEMINI GANN SQUARE ANALYSIS
  Symbol: BTC/USDT | Timeframe: 4h
══════════════════════════════════════════
  Swing High: $98,500 (2024-01-15 08:00)
  Swing Low:  $85,200 (2024-01-20 12:00)
  Price Range: $13,300 | Trend: DOWN

  Current Price: $91,000
  Gann Zone: 2 (SHORT zone)
  ──────────────────────────────
  🤖 GEMINI FINAL RECOMMENDATION
  ──────────────────────────────
  Signal:    SHORT ⬇
  Entry:     $91,000
  Stop Loss: $93,500
  TP1:       $88,000
  TP2:       $85,500
  Confidence: 78%
  Reasoning: Price is in Zone 2 of the Gann Square
             during a confirmed downtrend...
══════════════════════════════════════════
  Chart saved: charts/BTCUSDT_4h_gann_20260225.png
```

---

## 6. Dependencies

- `modules.gemini_chart_analyzer` — reuse `GeminiChartAnalyzer`, `plotting_utils`
- `modules.common.core.data_fetcher` — OHLCV data
- `modules.common.ui.logging` — log utilities
- `matplotlib` — chart rendering
- `pandas`, `numpy` — data processing

---

## 7. Testing Plan

```
tests/
├── test_swing_detector.py     # Unit: pivot detection on synthetic data
└── test_gann_calculator.py    # Unit: zone calculation, trend detection
```
