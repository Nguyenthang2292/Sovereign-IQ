# Test Performance Optimization Plan (2026-03-10)

## 1. Understanding Summary
- Muc tieu: giam thoi gian test suite auto_trade, uu tien nhom test dang cham nhat.
- Pham vi: toi uu test integration theo huong giu y nghia E2E, khong mock qua muc.
- Pham vi: toi uu test non-integration toi da (unit/component), loai bo I/O that va sleep that.
- Nguyen nhan cham chinh: network/DB that, khoi tao dependency nang lap lai, sleep/timeout theo thoi gian that, walk-forward + model training lap trong backtest.
- Tieu chi thanh cong: PR test nhanh hon ro ret, flaky giam, khong mat tin hieu chat luong quan trong.
- Rang buoc: khong thay doi business logic de chi phuc vu test speed; phai giu duoc niem tin vao integration path.

## 2. Assumptions
- Co the chinh test code, fixtures, marker va test profile config.
- Co the cap nhat CI command de tach luong test (fast/smoke/slow).
- Integration full-fidelity van can, nhung co the chuyen sang nightly.
- Team chap nhan quy uoc moi: unit tests khong duoc goi network/DB/sleep that.

## 3. Open Questions (Resolved)
- Uu tien da chon: integration cai thien toc do; non-integration fix toi da.
- Huong xu ly da chon: mo hinh lai 3 lop test + rollout 2 pha.

## 4. Final Design

### 4.1 Test Layering Architecture
- `unit_fast`: tat ca test non-integration. Bat buoc mock I/O (ccxt, boto3, ticker, analyzer, repo) va mock clock/sleep.
- `integration_smoke`: giu luong tich hop chinh, nhung du lieu nho hon, profile nhe hon, runtime hop ly de chay tren PR.
- `integration_slow`: test full-fidelity (du lieu lon, indicator day du), chay nightly/cron.

### 4.2 Per-Test Optimization Strategy
1. `tests/auto_trade/integration/test_backtest_phase6.py`
- Giu la integration.
- Tao smoke profile:
  - giam `lookback_days` (vi du 12 -> 2 hoac 3),
  - dung indicator set nhe trong smoke (uu tien bo `xgboost`, `hmm`, `random_forest`),
  - giu full case sang `integration_slow`.
- Muc tieu: giam manh runtime nhung van cover duoc luong end-to-end.

2. `tests/auto_trade/gui/utils/test_data_service.py`
- Chuyen cac test non-integration sang mock triet de:
  - patch `RepositoryContext.from_env`,
  - patch `get_cached_tpsl`,
  - patch `client.exchange.fetch_ticker`.
- Sua fallback test dung dependency dung ten:
  - thay vi `service.database_manager = None`, dung `service.repo_context = None`.

3. `tests/auto_trade/core/test_health.py::test_check_timeout`
- Loai bo `time.sleep(5)` that.
- Dung fake future/executor hoac dong bo gia lap timeout de assert timeout path ma runtime < 200ms.

4. `tests/auto_trade/core/test_scan_cache.py` (TTL expiration tests)
- Loai bo `time.sleep(...)` that.
- Dung monkeypatch/freeze clock de nhay thoi gian logic TTL ngay lap tuc.

5. `tests/auto_trade/test_adaptive_close.py::test_fallback_no_data`
- Patch `_fetch_ohlcv` tra `None` truc tiep de tranh goi `ccxt.binance().fetch_ohlcv` that.

6. `tests/auto_trade/core/test_gemini_integration.py::test_init_with_defaults`
- Dung fixture patch `ChartGenerator` + `GeminiChartAnalyzer` theo class/module scope de tranh init nang lap lai.

### 4.3 CI and Execution Model
- PR pipeline:
  - Buoc 1: `unit_fast` (bat buoc)
  - Buoc 2: `integration_smoke` (bat buoc)
- Nightly pipeline:
  - `integration_slow` full-fidelity
- Bat buoc ghi `--durations=20` de tracking drift runtime.

### 4.4 Non-Functional Requirements
- Performance: giam 50-80% runtime nhom non-integration cham.
- Reliability: giam flaky do sleep/network.
- Security/Privacy: unit test khong dung secret/API that.
- Maintainability: them test moi theo convention marker + fixture mac dinh.

### 4.5 Validation and Metrics
- Baseline truoc/sau: tong runtime + top 20 test cham.
- KPI:
  - PR runtime,
  - flaky rate,
  - regression defects (khong tang).
- Neu smoke nhe qua: bo sung 1-2 smoke case trong diem thay vi bat lai full suite tren PR.

## 5. Risks and Mitigation
- Risk: mock qua muc mat realism.
  - Mitigation: giu `integration_smoke` bat buoc tren PR.
- Risk: smoke va slow diverge.
  - Mitigation: dung chung fixture/profile source.
- Risk: test moi vi pham quy uoc fast.
  - Mitigation: them guideline + check don gian phat hien `time.sleep`/network call trong unit tests.

## 6. Decision Log
1. Adopt hybrid architecture (`unit_fast`, `integration_smoke`, `integration_slow`).
- Alternatives: 2-layer split, minimal patching only.
- Why: can bang toc do va do tin cay.

2. Integration policy: optimize but keep meaningful realism.
- Alternatives: full mocking integration.
- Why: bao toan tin hieu chat luong E2E.

3. Non-integration policy: maximize speed via strict isolation.
- Alternatives: selective mocking only.
- Why: dat muc tieu giam runtime lon nhat va giam flaky.

4. Rollout in 2 phases.
- Phase A: non-integration fast-first.
- Phase B: integration smoke/slow tuning.
- Why: giam risk va nhan ket qua speed nhanh som.

## 7. Implementation Handoff Gate
Truoc khi vao implementation, neu can do tin cay cao, thuc hien review boi `multi-agent-brainstorming` tren tai lieu nay + Decision Log de xac nhan trade-off.
