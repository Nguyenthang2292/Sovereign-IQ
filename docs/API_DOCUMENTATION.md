# Gemini Chart Analyzer REST API – Documentation

Base URL: `http://<host>:8000/api/` (endpoints được gắn với FastAPI trong web/app.py).

Mô tả ngắn gọn về các endpoint hiện có và cách dùng.

---

## Health & Root (API phụ trợ)

- GET `/health` — Kiểm tra tình trạng dịch vụ.
- GET `/` — Dữ liệu trạng thái hoặc SPA Vue nếu đã build.

---

## Logs API

- GET `/api/logs/{session_id}`
  - Mô tả: Đọc nội dung log cho một session nhất định.
  - Tham số đường dẫn:
    - `session_id` (string): ID session.
  - Tham số query:
    - `offset` (int, mặc định 0, ≥0): Chỉ số byte bắt đầu đọc.
    - `command_type` (string, enum `scan` hoặc `analyze`, mặc định `scan`): Loại log cần đọc.
  - Trả về:
    - `success`, `logs`, `offset`, `has_more`, `file_size`.

  - Ví dụ
    - Curl:

      ```bash
      curl -s "http://localhost:8000/api/logs/{session_id}?offset=0&command_type=scan"
      ```

---

## Chart Analyzer API

- POST `/api/analyze/single`
  - Mô tả: Khởi động phân tích một khung thời gian đơn ở chế độ nền.
  - Tham số body (JSON):
    - `symbol` (str, bắt buộc)
    - `timeframe` (str, bắt buộc)
    - `indicators` (object, tùy chọn)
    - `prompt_type` (str, tùy chọn, mặc định `detailed`)
    - `custom_prompt` (str, tùy chọn)
    - `limit` (int, tùy chọn, mặc định `500`)
    - `chart_figsize` (mảng 2 phần tử int, tùy chọn, mặc định `[16, 10]`)
    - `chart_dpi` (int, tùy chọn, mặc định `150`)
    - `no_cleanup` (bool, tùy chọn, mặc định `false`)
  - Trả về:
    - `success`, `session_id`, `status`, `message`

  - Ví dụ
    - Curl:

      ```bash
      curl -X POST -H "Content-Type: application/json" -d '{
        "symbol": "BTC/USDT",
        "timeframe": "1h"
      }' http://localhost:8000/api/analyze/single
      ```

- POST `/api/analyze/multi`
  - Mô tả: Phân tích cho nhiều timeframe cùng lúc.
  - Tham số body (JSON):
    - `symbol` (str, bắt buộc)
    - `timeframes` (array of str, bắt buộc)
    - `indicators` (object, tùy chọn)
    - `prompt_type` (str, tùy chọn, mặc định `detailed`)
    - `custom_prompt` (str, tùy chọn)
    - `limit` (int, tùy chọn, mặc định `500`)
    - `chart_figsize` (mảng 2 phần tử int, tùy chọn, `[16, 10]`)
    - `chart_dpi` (int, tùy chọn, `150`)
    - `no_cleanup` (bool, tùy chọn, `false`)
  - Trả về: đúng cấu trúc tương tự single với trường `timeframes`.

  - Ví dụ
    - Curl:

      ```bash
      curl -X POST -H "Content-Type: application/json" -d '{"symbol":"BTC/USDT","timeframes":["1h","4h"]}' http://localhost:8000/api/analyze/multi
      ```

- GET `/api/analyze/{session_id}/status`
  - Mô tả: Theo dõi trạng thái phân tích và kết quả khi hoàn tất.
  - Tham số đường dẫn: `session_id`.
  - Trả về: trạng thái và kết quả/ lỗi (nếu có).

---

## Batch Scanner API

- POST `/api/batch/scan`
  - Mô tả: Khởi động quét thị trường hàng loạt (async) ở chế độ một hoặc nhiều timeframe.
  - Tham số body (JSON):
    - `timeframe` (str, tùy chọn)
    - `timeframes` (array of str, tùy chọn)
    - `max_symbols` (int, tùy chọn)
    - `limit` (int, tùy chọn, mặc định `500`)
    - `cooldown` (float, tùy chọn, mặc định `2.5`)
    - `charts_per_batch` (int, tùy chọn, mặc định `100`)
    - `quote_currency` (str, tùy chọn, mặc định `USDT`)
    - `exchange_name` (str, tùy chọn, mặc định `binance`)
  - Trả về: `session_id`, `status`, `message`

  - Ví dụ
    - Curl:

      ```bash
      curl -X POST -H "Content-Type: application/json" -d '{"timeframe":"1h"}' http://localhost:8000/api/batch/scan
      ```

- GET `/api/batch/scan/{session_id}/status`
  - Mô tả: Theo dõi trạng thái batch scan.
  - Trả về: trạng thái, thời gian bắt đầu/kết thúc và kết quả nếu có.

- POST `/api/batch/scan/{session_id}/cancel`
  - Mô tả: Hủy batch scan đang chạy.
  - Trả về: `success`, `session_id`, `status`, `message`

- GET `/api/batch/results/{filename:path}`
  - Mô tả: Lấy kết quả batch scan đã lưu (JSON).
  - Bảo mật: ngăn dir traversal; chỉ cho phép các file JSON.
  - Trả về: nội dung JSON của file lưu trữ.

  - Ví dụ
    - Curl:

      ```bash
      curl http://localhost:8000/api/batch/results/sample.json
      ```

- GET `/api/batch/list`
  - Mô tả: Liệt kê danh sách các kết quả batch có sẵn với metadata.
  - Tham số query: `skip`, `limit`, `metadata_only`.
  - Trả về: `count`, `results` (danh sách metadata và URL).

---

## OpenAPI – Machine-readable (Tự động sinh)

- OpenAPI YAML: `docs/openapi.yaml`
- OpenAPI JSON: `docs/openapi.json`
- Extended docs: `docs/openapi_extended.md`

Bạn có thể dùng OpenAPI để tự động sinh tài liệu, kiểm thử, hoặc generate client SDK.

---

## OpenAPI Extended (Schemas & Examples)

- File mở rộng: `docs/openapi_extended.md` chứa phần OpenAPI, schemas và ví dụ chi tiết cho từng endpoint.

---

## Endpoints Summary (Programmatic View)

- Logs
  - GET `/api/logs/{session_id}`
- Chart Analysis
  - POST `/api/analyze/single`
  - POST `/api/analyze/multi`
  - GET `/api/analyze/{session_id}/status`
- Batch Scanner
  - POST `/api/batch/scan`
  - GET `/api/batch/scan/{session_id}/status`
  - POST `/api/batch/scan/{session_id}/cancel`
  - GET `/api/batch/results/{filename:path}`
  - GET `/api/batch/list`

---

## Thông tin thêm

- OpenAPI YAML: `docs/openapi.yaml` | JSON: `docs/openapi.json`
- OpenAPI Extended: `docs/openapi_extended.md`

Bạn có muốn mình gom phần OpenAPIExtender này vào file OPENAPI chỉ với một khối duy nhất (như một phần bên dưới), hay vẫn giữ nguyên ở dạng mục lục như hiện tại không?
