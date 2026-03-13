# Adaptive Trend Parity Contract (Source Of Truth)

## Scope
This document defines the exact behavior that `modules/adaptive_trend_LTS_mini` and `modules/adaptive_trend_LTS_serverless` must match from `modules/adaptive_trend`.

Source-of-truth implementation:
- `modules/adaptive_trend/core/compute_atc_signals.py`
- `modules/adaptive_trend/core/process_layer1.py`
- `modules/adaptive_trend/core/signal_detection.py`
- `modules/adaptive_trend/core/compute_moving_averages.py`
- `modules/adaptive_trend/core/compute_equity.py`
- `modules/adaptive_trend/utils/diflen.py`
- `modules/adaptive_trend/utils/exp_growth.py`
- `modules/adaptive_trend/utils/rate_of_change.py`

## Parameter Scaling Contract
- `La_scaled = La / 1000.0` (`compute_atc_signals`)
- `De_scaled = De / 100.0` (`compute_atc_signals`)
- Default thresholds:
  - `long_threshold = 0.1`
  - `short_threshold = -0.1`

## MA Set Contract (9-MA set per MA family)
For each MA family (`EMA`, `HMA`, `WMA`, `DEMA`, `LSMA`, `KAMA`), Layer 1 must use exactly 9 MA series:
- Base length: `length`
- 8 offsets from `diflen(length, robustness)`:
  - Narrow: `+1,+2,+3,+4,-1,-2,-3,-4`
  - Medium: `+1,+2,+4,+6,-1,-2,-4,-6`
  - Wide: `+1,+3,+5,+7,-1,-3,-5,-7`

Reference:
- `set_of_moving_averages` in `modules/adaptive_trend/core/compute_moving_averages.py`
- `diflen` in `modules/adaptive_trend/utils/diflen.py`

## MA Type Contract
- `EMA -> ta.ema`
- `HMA -> ta.hma` (not SMA)
- `WMA -> ta.wma`
- `DEMA -> ta.dema`
- `LSMA -> ta.linreg`
- `KAMA -> custom `calculate_kama_atc`:
  - `fast = 0.666`
  - `slow = 0.064`

Reference:
- `ma_calculation` and `calculate_kama_atc` in `modules/adaptive_trend/core/compute_moving_averages.py`

## Signal State Persistence Contract (crossover/crossunder)
Per MA variation signal generation must be event-based and stateful:

1. Event detection:
- `up_t = (price_t > ma_t) AND (price_{t-1} <= ma_{t-1})`
- `down_t = (price_t < ma_t) AND (price_{t-1} >= ma_{t-1})`

2. Event-to-signal:
- `raw_t = 1` if `up_t`
- `raw_t = -1` if `down_t`
- `raw_t = 0` otherwise

3. Persistent state (Pine `var` behavior):
- `sig_t = last_non_zero(raw)` forward-filled
- initial state `0`

Equivalent pandas behavior in source:
- `sig.replace(0, np.nan).ffill().fillna(0).astype("int8")`

Reference:
- `crossover`, `crossunder`, `generate_signal_from_ma` in `modules/adaptive_trend/core/signal_detection.py`

## Layer 1 Contract (equity-weighted signal over 9 MA variations)
For each MA family:

1. Build 9 variation signals: `s1..s9`
2. Build 9 variation equities: `e1..e9` using `equity_series(starting_equity=1.0, sig=sj, ...)`
3. Compute Layer 1 weighted signal:
- `layer1_t = round((sum_j(sj_t * ej_t)) / (sum_j(ej_t)), 2)`

Reference:
- `_layer1_signal_for_ma` and `weighted_signal` in `modules/adaptive_trend/core/process_layer1.py`

## Equity Curve Contract
`equity_series(starting_equity, sig, R, L, De, cutout)`:

1. `growth_t = exp(L * (bar_index_t - cutout))`, with `bar_index_0 = 1`
2. `r_t = R_t * growth_t`
3. Use previous signal (`sig.shift(1)`):
  - if previous signal > 0: `a_t = r_t`
  - if previous signal < 0: `a_t = -r_t`
  - else: `a_t = 0`
4. Recursive equity:
  - first valid bar after cutout: `e_t = starting_equity`
  - otherwise: `e_t = (e_{t-1} * (1 - De)) * (1 + a_t)`
5. Floor clamp: `e_t = max(e_t, 0.25)`

Reference:
- `equity_series` in `modules/adaptive_trend/core/compute_equity.py`
- `exp_growth` in `modules/adaptive_trend/utils/exp_growth.py`

## Layer 2 Contract (equity weighting of MA families)
For each MA family `M`:
- `M_S = equity_series(starting_equity=M_weight, sig=M_Layer1_signal, R, L=La_scaled, De=De_scaled, cutout)`

Reference:
- `calculate_layer2_equities` in `modules/adaptive_trend/core/compute_atc_signals.py`

## Final Average_Signal Contract
For each MA family `M`:
1. Discretize Layer 1 signal via `cut_signal`:
  - `cut_M_t = 1` if `M_Layer1_t > long_threshold`
  - `cut_M_t = -1` if `M_Layer1_t < short_threshold`
  - `cut_M_t = 0` otherwise
2. Aggregate:
  - `nom_t = sum_M(cut_M_t * M_S_t)`
  - `den_t = sum_M(M_S_t)`
  - `Average_Signal_t = nom_t / den_t` (if `den_t == 0` then `0`)

Reference:
- `cut_signal` in `modules/adaptive_trend/core/process_layer1.py`
- final aggregation in `modules/adaptive_trend/core/compute_atc_signals.py`

## Classification Contract (for parity checks)
Use latest `Average_Signal` sign:
- `LONG` if `Average_Signal[-1] > 0`
- `SHORT` if `Average_Signal[-1] < 0`
- `NEUTRAL` if `Average_Signal[-1] == 0` (or non-finite treated as neutral)

This matches `trend_sign` sign semantics used by source scanner/display flow.
