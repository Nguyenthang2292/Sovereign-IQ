# Raw vs Execution Signal Contract

## Scope

This contract applies to:

- `modules/adaptive_trend_LTS_mini`
- `modules/adaptive_trend_LTS_serverless`

## Output Layers

1. `Average_Signal` (required)

- This is the **raw causal signal** from the core engine.
- Contract: `Average_Signal[t]` depends only on data available up to bar `t`.
- Core computation must never apply execution shift.

2. `Average_Signal_Exec` (optional)

- This is an execution-view signal produced by adapter/consumer layers.
- Formula: `Average_Signal_Exec[t] = Average_Signal[t-1]`.
- Fill policy: first bar is `0.0`.

## Classification Policy

- Production scanning/classification should read from `Average_Signal` (raw).
- Backtest/order-execution adapters may read `Average_Signal_Exec`.

## Anti-Regression Rules

- No internal core shift in batch aggregation.
- No internal core shift in incremental updater.
- No double-shift: execution adapter may shift at most once.

## Migration Notes

- `strategy_mode` no longer mutates `Average_Signal` in core.
- Consumers that previously expected shifted output from `Average_Signal`
  must migrate to `Average_Signal_Exec`.
- During migration, both fields can be returned where supported.

## Serverless Snapshot Note

`adaptive_trend_LTS_serverless` returns snapshot-style symbol results (`score`, `signal_type`).
Optional `average_signal_raw` can be included for contract alignment. `average_signal_exec`
may be omitted when execution-shift cannot be derived from snapshot-only output.