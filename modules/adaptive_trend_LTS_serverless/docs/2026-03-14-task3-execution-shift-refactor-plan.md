# Task 3 Plan - Refactor Execution Shift Out Of Core

## Goal

Tach logic execution shift (strategy/non-repainting view) khoi core signal engine, dong bo giua:

- `modules/adaptive_trend_LTS_mini`
- `modules/adaptive_trend_LTS_serverless`

Core phai tra ve `Average_Signal` dang causal/raw. Shift 1 bar (neu can) chi duoc ap dung o execution layer (consumer/adapter/backtest pipeline).

## Current Risk Snapshot

### LTS mini

- Batch path: shift hien dang nam trong core average aggregation.
  - `modules/adaptive_trend_LTS_mini/core/compute_atc_signals/average_signal.py`
- Incremental path: strategy mode tra ve previous average ngay trong core updater.
  - `modules/adaptive_trend_LTS_mini/core/compute_atc_signals/incremental/core.py`
- Mot so helper/docs van mo ta strategy shift la thanh phan noi bo cua core.

### LTS serverless

- Python wrapper khong cho thay strategy shift ro rang trong core.
- Rust core can duoc chot contract de xuat signal raw la nguon su that.
  - Candidate check points:
    - `modules/adaptive_trend_LTS_serverless/src/signal_detection.rs`
    - `modules/adaptive_trend_LTS_serverless/src/lib.rs`
    - `modules/adaptive_trend_LTS_serverless/lambda/src/handler.rs`

## Target Contract

Moi implementation tra ve 2 lop output ro rang:

1. `average_signal_raw`: output causal tu core (bat buoc)
2. `average_signal_exec`: output da shift (tuy chon, chi ap dung o execution adapter)

Quy uoc:

- `average_signal_raw[t]` chi duoc phu thuoc du lieu toi da den bar `t`.
- `average_signal_exec[t] = average_signal_raw[t-1]` (fill gia tri dau tien theo policy thong nhat, de xuat `0.0`).
- Classification LONG/SHORT/NEUTRAL cho production scanner uu tien doc tu raw, trinh backtest co the doc exec.

## Refactor Tasks

- [x] **Task 1: Contract first**
  - Tao tai lieu contract output chung cho mini + serverless.
  - Chot naming field: `Average_Signal` (raw) va `Average_Signal_Exec` (optional).
  - Verify: ca 2 module co mot mo ta contract thong nhat trong docs.

- [x] **Task 2: Mini batch path**
  - Go bo shift khoi core aggregator.
  - File chinh: `modules/adaptive_trend_LTS_mini/core/compute_atc_signals/average_signal.py`.
  - Verify: khi `strategy_mode=True`, core van tra raw khong shift.

- [x] **Task 3: Mini incremental path**
  - Go bo logic tra previous average trong core incremental updater.
  - File chinh: `modules/adaptive_trend_LTS_mini/core/compute_atc_signals/incremental/core.py`.
  - Verify: raw incremental output trung voi batch raw o cung dataset.

- [x] **Task 4: Mini execution adapter**
  - Tao helper/applier rieng de shift output khi can cho backtest.
  - Co the dat o adapter layer hoac scanner/backtest integration layer.
  - Verify: `Average_Signal_Exec` khop cong thuc shift tu raw.

- [x] **Task 5: Serverless API sync**
  - Chot payload/response field de bieu dien raw va optional exec.
  - Neu can, them flag `apply_strategy_shift` tai request adapter, khong o core Rust.
  - Verify: Lambda response giu backward compatibility (co migration note ro rang).

- [x] **Task 6: Serverless core audit**
  - Kiem tra va dam bao Rust core khong shift noi bo.
  - Files uu tien: `src/signal_detection.rs`, `src/lib.rs`.
  - Verify: test khang dinh `Average_Signal` la raw.

- [x] **Task 7: Parity + regression tests**
  - Them test ngan double-shift cho mini va serverless.
  - Bo sung fixture so sanh raw vs exec.
  - Verify:
    - Raw parity voi source module dat tolerance da chot.
    - Exec = raw.shift(1) dung 100% theo policy fill.

- [x] **Task 8: Rollout and deprecation**
  - Giu compatibility tam thoi cho call-site cu.
  - Danh dau deprecated voi path cu co shift trong core.
  - Verify: changelog + migration guide cap nhat day du.

## Validation Matrix

- Unit tests:
  - raw output no-shift
  - exec output single-shift
  - no double-shift in strategy mode
- Parity tests:
  - source adaptive_trend vs mini raw
  - source adaptive_trend vs serverless raw
- Integration tests:
  - scanner classification from raw
  - backtest pipeline from exec

## Done When

- Khong con shift logic nam trong core signal computation.
- Tat ca module thong nhat contract raw/execution.
- Khong con risk double-shift o mini incremental hoac batch.
- CI co regression test chan viec tai dua shift vao core.
