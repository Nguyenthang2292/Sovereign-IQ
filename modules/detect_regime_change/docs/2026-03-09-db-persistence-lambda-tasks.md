# DB Persistence & Lambda Offloading — Implementation Tasks

## Goal

Triển khai 2 tính năng bổ sung cho hệ thống Adaptive Close Time:
- **Feature A:** Lưu metadata vào DB để GUI và audit trail biết nguồn gốc deadline sau khi restart
- **Feature B:** (Optional) Offload regime analysis lên AWS Lambda để giải phóng local CPU

> **Design doc:** [2026-03-09-db-persistence-lambda-design.md](./2026-03-09-db-persistence-lambda-design.md)
> **Phụ thuộc:** Phase 1 + 2 đã hoàn thành ✅

---

## Feature A — DB Persistence (Bắt buộc)

### A.1 Data Model: `AdaptiveCloseResult`

- [x] Thêm dataclass `AdaptiveCloseResult` vào `modules/auto_trade/execution/adaptive_close_calculator.py`:
  - Fields: `deadline_utc`, `source`, `duration_hours`, `pelt_hours`, `hmm_hours`
  - `source`: Literal["adaptive", "static", "adaptive_fallback"]
  - Verify: `python -c "from modules.auto_trade.execution.adaptive_close_calculator import AdaptiveCloseResult; r = AdaptiveCloseResult(deadline_utc=None, source='static', duration_hours=4.0, pelt_hours=None, hmm_hours=None); print(r)"`

### A.2 Method mới trong `AdaptiveCloseCalculator`

- [x] Thêm method `compute_adaptive_deadline_with_meta()` vào `AdaptiveCloseCalculator`:
  - Trả `AdaptiveCloseResult` thay vì `Optional[datetime]`
  - Gán `source = "adaptive"` khi analysis thành công
  - Gán `source = "adaptive_fallback"` khi analysis thất bại → dùng fallback
  - Gán `source = "static"` khi `adaptive.enabled = False`
  - Lưu `pelt_hours = analysis.pelt_avg_duration_hours`
  - Lưu `hmm_hours = analysis.hmm_next_state_duration_hours`
  - Verify: pytest mock analyzer, assert `result.source == "adaptive"` và `result.pelt_hours` đúng

### A.3 DynamoDB — Không cần migration

DynamoDB là schema-less — **không cần migration script**, không có `ALTER TABLE`.
Chỉ cần include 4 attributes mới vào `order_data` dict khi gọi `ctx.orders.create_order()`.
Các order items cũ trong table `AutoTrade` đơn giản là không có 4 attributes này — không lỗi.

- [x] Xác nhận 4 attribute keys sẽ dùng (chỉ cần document, không cần tạo schema):
  - `auto_close_deadline_source` — String
  - `adaptive_close_duration_hours` — Number (float)
  - `adaptive_close_pelt_hours` — Number (float), bỏ qua nếu None
  - `adaptive_close_hmm_hours` — Number (float), bỏ qua nếu None
  - Verify: Sau khi place 1 order test với `adaptive.enabled=true`, dùng AWS Console hoặc `boto3` để GetItem, xác nhận 4 attributes xuất hiện trong item `pk=ORDER#<id>, sk=METADATA`

### A.4 Integration vào Order Flow

- [x] Cập nhật điểm tích hợp trong `order_manager.py` hoặc `order_executor.py`:
  - Thay `compute_adaptive_deadline()` → `compute_adaptive_deadline_with_meta()`
  - Gán cả 5 fields vào `order_updates` dict trước khi save DB
  - Verify: `pytest tests/auto_trade/execution/test_adaptive_close_integration.py -v`, payload DB có đủ 5 fields

### A.5 Tests

- [x] Viết pytest `tests/auto_trade/test_adaptive_close_result.py`:
  - Test `source = "adaptive"` khi analysis valid
  - Test `source = "adaptive_fallback"` khi analysis.is_valid = False
  - Test `source = "static"` khi `adaptive.enabled = False`
  - Test `pelt_hours` và `hmm_hours` được gán đúng từ `RegimeDurationResult`
  - Verify: `pytest tests/auto_trade/test_adaptive_close_result.py -v` (tất cả pass)

- [x] Viết pytest `tests/auto_trade/execution/test_order_flow_metadata.py`:
  - Mock `AdaptiveCloseCalculator.compute_adaptive_deadline_with_meta()`
  - Assert DB update dict có 4 fields mới
  - Verify: `pytest tests/auto_trade/execution/test_order_flow_metadata.py -v`

---

## Feature B — AWS Lambda Offloading (Optional)

> Chỉ implement khi Feature A đã hoàn thành và có nhu cầu thực sự.
> Mặc định: `use_lambda: false` trong settings.yaml

### B.1 Settings Update

- [ ] Thêm vào `modules/auto_trade/settings.yaml` section `auto_close.adaptive`:
  ```yaml
  use_lambda: false
  lambda_endpoint: ""
  lambda_timeout_seconds: 3.0
  ```
  - Verify: YAML load, `settings_manager.get("auto_close.adaptive.use_lambda")` trả `False`

### B.2 `RegimeLambdaClient` — Local HTTP Client

- [ ] Tạo `modules/detect_regime_change/regime_lambda_client.py`:
  - Class `RegimeLambdaClient(endpoint, timeout_seconds=3.0)`
  - Method `invoke(ohlcv_df, symbol, config) -> Optional[RegimeDurationResult]`
  - Method `_serialize_ohlcv(df) -> dict` — chuyển DataFrame → JSON dict
  - Method `_deserialize_result(data) -> RegimeDurationResult`
  - Trả `None` (không raise) khi timeout hoặc lỗi HTTP
  - Verify: pytest mock `requests.post`, assert serialization/deserialization đúng

### B.3 Tích hợp `RegimeLambdaClient` vào `AdaptiveCloseCalculator`

- [ ] Cập nhật `compute_adaptive_deadline_with_meta()`:
  - Kiểm tra `cfg["use_lambda"]` và `cfg["lambda_endpoint"]`
  - Nếu True: gọi `RegimeLambdaClient.invoke()` trước
  - Nếu Lambda trả `None` → fallback về local `RegimeDurationAnalyzer`
  - Verify: pytest test fallback chain (Lambda fail → local succeed → static)

### B.4 Lambda Handler (Rust)

- [ ] Tạo `modules/detect_regime_change/regime_lambda/Cargo.toml`:
  - Dependency: `lambda_runtime`, `serde`, `serde_json`, `tokio`
  - Reuse Rust PELT logic từ `modules/detect_regime_change/rust_extensions/`
  - Verify: `cargo check` pass

- [ ] Tạo `modules/detect_regime_change/regime_lambda/src/models.rs`:
  - Struct `RegimeAnalysisRequest` (deserialize từ JSON)
  - Struct `RegimeAnalysisResponse` (serialize sang JSON)
  - Verify: `cargo test` pass cho serialization round-trip

- [ ] Tạo `modules/detect_regime_change/regime_lambda/src/handler.rs`:
  - Nhận request, parse OHLCV, chạy PELT + HMM logic
  - Trả `RegimeAnalysisResponse`
  - Verify: unit test với mock OHLCV data

- [ ] Tạo `modules/detect_regime_change/regime_lambda/src/main.rs`:
  - Lambda entry point với `lambda_runtime::run()`
  - Verify: `cargo lambda build --release` thành công

- [ ] Tạo `modules/detect_regime_change/regime_lambda/template.yaml`:
  - SAM template, follow pattern `adaptive_trend_LTS_serverless/template.yaml`
  - Memory: 512MB, Timeout: 30s
  - Verify: `sam validate` pass

### B.5 Tests

- [ ] Viết pytest `tests/detect_regime_change/test_regime_lambda_client.py`:
  - Test serialization OHLCV DataFrame → JSON
  - Test deserialization JSON → RegimeDurationResult
  - Test timeout handling (mock requests timeout)
  - Test HTTP error handling (mock 500 response)
  - Verify: `pytest tests/detect_regime_change/test_regime_lambda_client.py -v`

- [ ] Viết pytest `tests/auto_trade/test_adaptive_close_lambda_fallback.py`:
  - Test: Lambda thành công → dùng Lambda result, source="adaptive"
  - Test: Lambda timeout → fallback local, source="adaptive"
  - Test: Lambda + local fail → static fallback, source="adaptive_fallback"
  - Verify: `pytest tests/auto_trade/test_adaptive_close_lambda_fallback.py -v`

- [ ] Rust unit tests cho Lambda handler:
  - Test parse request JSON
  - Test PELT execution trên mock data
  - Test response serialization
  - Verify: `cargo test` trong `regime_lambda/`

### B.6 Deploy & Smoke Test

- [ ] Build Lambda package:
  ```bash
  cd modules/detect_regime_change/regime_lambda
  cargo lambda build --release --target x86_64-unknown-linux-gnu
  ```
- [ ] Deploy lên AWS Lambda:
  ```bash
  cargo lambda deploy --iam-role arn:aws:iam::ACCOUNT:role/ROLE regime-analysis
  ```
- [ ] Smoke test với real OHLCV data:
  ```bash
  python scripts/test_regime_lambda.py --endpoint FUNCTION_URL --symbol BTC/USDT
  ```
- [ ] Bật `use_lambda: true` trong settings, test end-to-end với order flow

---

## Done When

### Feature A
- [ ] 4 DB columns được thêm, không NULL error khi update
- [x] `compute_adaptive_deadline_with_meta()` trả đúng `source`, `pelt_hours`, `hmm_hours`
- [x] Order flow integration: 5 fields được save vào DB sau khi place order
- [ ] Sau restart: `auto_close_timer_job` vẫn hoạt động đúng (đọc `auto_close_deadline_utc`)
- [ ] Tất cả pytest pass

### Feature B (nếu implement)
- [ ] `use_lambda: false` mặc định → không ảnh hưởng flow hiện tại
- [ ] `use_lambda: true` → Lambda được gọi, fallback hoạt động đúng khi Lambda fail
- [ ] `cargo lambda build` thành công
- [ ] Smoke test pass với real endpoint

---

## Notes

- **Feature A không thay đổi** `auto_close_timer.py` hay `auto_close_timer_job.py` ✅
- **Restart recovery** đã hoạt động qua `auto_close_deadline_utc` — 4 fields mới chỉ là metadata cho GUI
- **Feature B** hoàn toàn opt-in, `use_lambda: false` = zero risk với flow hiện tại
- **Fallback order (Feature B):** Lambda → Local Rust/Python → Static 4h — không bao giờ để order "treo"
- **OHLCV fetch** luôn ở local, Lambda chỉ nhận data tính toán — không cần API key trên cloud
