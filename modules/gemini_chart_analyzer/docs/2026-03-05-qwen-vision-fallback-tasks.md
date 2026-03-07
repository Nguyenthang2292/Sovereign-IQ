# Qwen Vision Fallback — Implementation Tasks

## Goal

Thêm Qwen VL (DashScope) làm provider fallback khi tất cả Gemini models bị
quota/rate-limit, không thay đổi interface hiện có của `GeminiChartAnalyzer`.

**Design doc:** [`2026-03-05-qwen-vision-fallback-design.md`](./2026-03-05-qwen-vision-fallback-design.md)

---

## Tasks

### Step 1 — Kiểm tra dependency `openai` package

- [x] Chạy `pip show openai` để xác nhận package đã có
- [x] Nếu chưa: thêm `openai>=1.0.0` vào `requirements.txt` và chạy `pip install openai` *(đã có sẵn trong `requirements.txt`)*
- **Verify:** [x] `python -c "from openai import OpenAI; print('ok')"` không lỗi

---

### Step 2 — Thêm `get_dashscope_api_key()` vào `config/config_api.py`

- [x] Mở `config/config_api.py`
- [x] Thêm hàm:

  ```python
  def get_dashscope_api_key() -> str | None:
      """Get DashScope (Qwen VL) API key."""
      return os.getenv("DASHSCOPE_API_KEY")
  ```

- [x] Thêm `DASHSCOPE_API_KEY=` vào file `.env` (và `modules/auto_trade/.env`) — để trống nếu chưa có key
- **Verify:** [x] `from config.config_api import get_dashscope_api_key; print(get_dashscope_api_key())` không lỗi

---

### Step 3 — Tạo `vision_provider_protocol.py`

- [x] Tạo file: `modules/gemini_chart_analyzer/core/analyzers/vision_provider_protocol.py`
- [x] Định nghĩa `VisionProvider` Protocol với `analyze_chart()` và `is_available()`
- [x] Dùng `@runtime_checkable` để có thể dùng `isinstance()` check
- **Verify:** [x] `from modules.gemini_chart_analyzer.core.analyzers.vision_provider_protocol import VisionProvider` không lỗi

---

### Step 4 — Tạo `qwen_vision_provider.py`

- [x] Tạo file: `modules/gemini_chart_analyzer/core/analyzers/qwen_vision_provider.py`
- [x] Implement `QwenVisionProvider` lấy danh sách model từ list đã tải về
  - Không hardcode 3 model cố định trong code
  - Duyệt model theo list đã tải về, ưu tiên `qwen-vl-max` nếu có
- [x] Đọc image → base64 encode → gửi qua OpenAI-compatible client
  - base_url: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- [x] Retry logic: 429 → skip model ngay; 503 → exponential backoff tối đa 3 lần
- [x] Reuse prompt từ `gemini_chart_analyzer/core/prompts/` (cùng prompt files)
- **Verify:** [x] Unit test với mock client trả về response giả — xem `tests/` trong `gemini_chart_analyzer`

---

### Step 5 — Tạo `vision_analyzer_chain.py`

- [x] Tạo file: `modules/gemini_chart_analyzer/core/analyzers/vision_analyzer_chain.py`
- [x] Implement `VisionAnalyzerChain`:
  - `__init__(gemini_api_key, qwen_api_key, skip_unavailable=True)`
  - Build provider list: `[GeminiVisionProvider, QwenVisionProvider]` — bỏ qua nếu key không có
  - Nếu list rỗng: raise `VisionChainExhaustedError` ngay tại `__init__`
- [x] `analyze_chart()`: iterate providers, try each, log fallback, raise nếu tất cả fail
- [x] `is_available()`: `True` nếu ít nhất 1 provider available
- **Verify:** [x] Instantiate chain với chỉ Gemini key → `chain.is_available() == True`

---

### Step 6 — Cập nhật `__init__.py` exports

- [x] Thêm exports vào `modules/gemini_chart_analyzer/core/analyzers/__init__.py`:

  ```python
  from .vision_provider_protocol import VisionProvider
  from .qwen_vision_provider import QwenVisionProvider
  from .vision_analyzer_chain import VisionAnalyzerChain, VisionChainExhaustedError
  ```

- **Verify:** [x] `from modules.gemini_chart_analyzer.core.analyzers import VisionAnalyzerChain` không lỗi

---

### Step 7 — Cập nhật `gann_signal_engine.py`

- [x] Mở `modules/gemini_gann_square/core/gann_signal_engine.py`
- [x] Thay dòng 140:

  ```python
  # BEFORE
  self.gemini_analyzer = GeminiChartAnalyzer(api_key=gemini_api_key)
  # AFTER
  from modules.gemini_chart_analyzer.core.analyzers.vision_analyzer_chain import VisionAnalyzerChain
  self.gemini_analyzer = VisionAnalyzerChain(gemini_api_key=gemini_api_key)
  ```

- [x] Thêm param `qwen_api_key: Optional[str] = None` vào `GannSignalEngine.__init__()`
- **Verify:** [x] `GannSignalEngine()` khởi tạo không lỗi khi có `GEMINI_API_KEY`

---

### Step 8 — Cập nhật `gemini_integration.py`

- [x] Mở `modules/auto_trade/core/gemini_integration.py`
- [x] Thêm param `qwen_api_key: Optional[str] = None` vào `GeminiIntegration.__init__()`
- [x] Thay dòng 125:

  ```python
  # BEFORE
  self.analyzer = GeminiChartAnalyzer(api_key=self._api_key)
  # AFTER
  from modules.gemini_chart_analyzer.core.analyzers.vision_analyzer_chain import VisionAnalyzerChain
  self._qwen_api_key = qwen_api_key or os.getenv("DASHSCOPE_API_KEY")
  self.analyzer = VisionAnalyzerChain(
      gemini_api_key=self._api_key,
      qwen_api_key=self._qwen_api_key,
  )
  ```

- [x] Cập nhật `is_available()`:

  ```python
  def is_available(self) -> bool:
      return self.analyzer.is_available()
  ```

- [x] Cập nhật type hint `analyzer: GeminiChartAnalyzer` → `analyzer: VisionAnalyzerChain`
- **Verify:** [x] `GeminiIntegration(data_fetcher=...)` không lỗi; `.is_available()` đã delegate về `self.analyzer.is_available()`

---

### Step 9 — Viết tests

- [x] Tạo `modules/gemini_chart_analyzer/tests/test_vision_analyzer_chain.py`
- [x] Test cases (dùng pytest + mock):
  - `test_chain_uses_gemini_when_available` — Gemini trả về OK → không gọi Qwen
  - `test_chain_falls_back_to_qwen_on_gemini_failure` — Gemini raise → Qwen được gọi
  - `test_chain_raises_when_all_fail` — cả 2 fail → `VisionChainExhaustedError`
  - `test_chain_skips_unavailable_provider` — không có DASHSCOPE_API_KEY → chỉ 1 provider
  - `test_qwen_provider_is_available_with_key` — `is_available()` đúng
- [x] Tạo `modules/gemini_chart_analyzer/tests/test_qwen_vision_provider.py`
- [x] Test base64 encode, model fallback, retry logic (mock OpenAI client)
- **Verify:** [x] `pytest modules/gemini_chart_analyzer/tests/ -v` — `54 passed, 0 failed` (run ngày 2026-03-05)

---

### Step 10 — Smoke test end-to-end

- [x] Set `DASHSCOPE_API_KEY` thật trong `.env`
- [x] Chạy thử `GannSignalEngine` với 1 symbol — xem log có `[qwen]` không nếu Gemini fail
- [x] Kiểm tra log line: `"Falling back to Qwen vision provider..."` xuất hiện đúng lúc
- **Verify:** [x] Pipeline hoàn thành và trả về `GannAnalysisResult` hợp lệ

---

## Done When

- [x] `pytest modules/gemini_chart_analyzer/tests/` — `54 passed, 0 failed` (2026-03-05)
- [x] `GannSignalEngine` và `GeminiIntegration` dùng `VisionAnalyzerChain`
- [x] Khi Gemini quota hết, log hiện `"Trying next provider: qwen"` và Qwen trả về kết quả
- [x] Khi chỉ có `GEMINI_API_KEY` (không có `DASHSCOPE_API_KEY`), behavior giữ nguyên như cũ
- [x] Không có breaking change nào ở interface `analyze_chart()`

---

## Notes

- **Prompt reuse:** `QwenVisionProvider` dùng lại prompt files tại
  `gemini_chart_analyzer/core/prompts/` — không cần viết prompt mới
- **Image format:** DashScope chấp nhận base64 data URL (`data:image/png;base64,...`)
  thay vì PIL Image object như Gemini
- **Không sửa** `_call_model_with_retries()` trong `GeminiChartAnalyzer` — Gemini
  fallback chain bên trong vẫn giữ nguyên; chain mới chỉ kick in khi Gemini class throw
- **API key an toàn:** DashScope key không log ra — dùng cùng `_mask_api_key()` pattern
  như `GeminiIntegration` nếu cần log
