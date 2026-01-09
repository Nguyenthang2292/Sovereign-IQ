# Luồng Hoạt Động của Batch Scanner

## Tổng Quan

File `batch_scanner_main.py` là điểm vào chính cho chức năng batch scanning toàn bộ thị trường với Gemini. Luồng hoạt động được chia thành các giai đoạn chính:

1. **Khởi tạo và Setup**
2. **Thu thập Input từ User**
3. **Xác nhận và Khởi tạo Scanner**
4. **Thực thi Scan Market**
5. **Hiển thị Kết quả**

---

## Chi Tiết Luồng Hoạt Động

### 1. Entry Point: `main()`

**File:** `modules/gemini_chart_analyzer/cli/batch_scanner_main.py`

```python
def main():
    try:
        interactive_batch_scan()
    except KeyboardInterrupt:
        log_warn("\nExiting...")
        sys.exit(0)
    except Exception as e:
        log_error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
```

**Luồng:**
- Gọi `interactive_batch_scan()`
- Xử lý `KeyboardInterrupt` (Ctrl+C)
- Xử lý các exception khác

---

### 2. Hàm Chính: `interactive_batch_scan()`

#### 2.1. Khởi tạo và Hiển thị Menu

**Dòng 134-140:**
- In header "MARKET BATCH SCANNER"
- Chuẩn bị giao diện tương tác

#### 2.2. Thu thập Input từ User

**⚠️ ĐIỂM QUAN TRỌNG:** Tất cả các `input()` calls đều sử dụng `input()` trực tiếp, KHÔNG dùng `safe_input()` (trừ confirmation prompt).

**a) Chọn Mode (Dòng 142-148)**
```python
mode = input(color_text("Select mode (1/2) [2]: ", Fore.YELLOW)).strip()
```
- **Mode 1:** Single timeframe
- **Mode 2:** Multi-timeframe (recommended)
- **Default:** Mode 2

**b) Nhập Timeframes (Dòng 153-189)**
- **Nếu Mode 2 (Multi-TF):**
  - Dòng 159: `timeframes_input = input(...)` - ⚠️ **Có thể gây lỗi stdin**
  - Parse và normalize timeframes
- **Nếu Mode 1 (Single-TF):**
  - Dòng 175: `timeframe = input(...)` - ⚠️ **Có thể gây lỗi stdin**
  - Normalize timeframe

**c) Nhập Max Symbols (Dòng 191-202)**
```python
max_symbols_input = input(color_text("Max symbols to scan (press Enter for all): ", Fore.YELLOW)).strip()
```
- ⚠️ **Có thể gây lỗi stdin**

**d) Nhập Cooldown (Dòng 204-215)**
```python
cooldown_input = input(color_text("Cooldown between batches in seconds [2.5]: ", Fore.YELLOW)).strip()
```
- ⚠️ **Có thể gây lỗi stdin**

**e) Nhập Limit (Dòng 217-228)**
```python
limit_input = input(color_text("Number of candles per symbol [500]: ", Fore.YELLOW)).strip()
```
- ⚠️ **Có thể gây lỗi stdin**

**f) Nhập Pre-filter Percentage (Dòng 230-244)**
```python
pre_filter_input = input(color_text("Pre-filter percentage (0-100, press Enter to skip) [0]: ", Fore.YELLOW)).strip()
```
- ⚠️ **Có thể gây lỗi stdin**
- Validate: 0-100

#### 2.3. Hiển thị Configuration và Xác nhận

**Dòng 246-263:**
- In tất cả thông tin cấu hình
- Hiển thị pre-filter status

**Dòng 265-274: Xác nhận TRƯỚC KHI khởi tạo Scanner**
```python
# Flush stdout/stderr
sys.stdout.flush()
sys.stderr.flush()

confirm = safe_input(color_text("Start batch scan? (y/n) [y]: ", Fore.YELLOW), default='y').lower()
```
- ⚠️ **ĐIỂM QUAN TRỌNG:** Sử dụng `safe_input()` thay vì `input()` trực tiếp
- Flush stdout/stderr để đảm bảo clean state
- Nếu user chọn 'n' hoặc 'no', return và không tiếp tục

#### 2.4. Khởi tạo Scanner và Chạy Scan

**Dòng 280-292:**
```python
scanner = MarketBatchScanner(cooldown_seconds=cooldown)

results = scanner.scan_market(
    timeframe=timeframe,
    timeframes=timeframes,
    max_symbols=max_symbols,
    limit=limit,
    pre_filter_percentage=pre_filter_percentage if pre_filter_percentage > 0 else None
)
```

**⚠️ ĐIỂM CÓ THỂ XẢY RA LỖI:**
- **Dòng 283:** `MarketBatchScanner.__init__()` - Khởi tạo các components
  - `ExchangeManager()`
  - `PublicExchangeManager()`
  - `DataFetcher()`
  - `ChartBatchGenerator()`
  - `GeminiBatchChartAnalyzer` - **Lazy initialization** (không khởi tạo ở đây)
- **Dòng 286-292:** `scanner.scan_market()` - Bắt đầu quá trình scan

#### 2.5. Hiển thị Kết quả (Dòng 294-352)

- Hiển thị LONG signals với confidence
- Hiển thị SHORT signals với confidence
- Hiển thị summary statistics
- Hiển thị file path của results

---

### 3. Hàm `safe_input()`

**File:** `modules/gemini_chart_analyzer/cli/batch_scanner_main.py` (Dòng 34-131)

**Mục đích:** Đọc input từ stdin một cách an toàn, xử lý các lỗi I/O trên Windows.

**Luồng xử lý:**

1. **Kiểm tra stdin availability (Dòng 47-49)**
   - Nếu `sys.stdin is None` → return default

2. **Kiểm tra stdin closed (Dòng 52-63)**
   - Nếu `sys.stdin.closed == True`:
     - Trên Windows: Thử mở lại từ 'CON'
     - Nếu không được: return default

3. **Kiểm tra TTY (Dòng 66-68)**
   - Nếu không phải TTY → return default

4. **Double-check trên Windows (Dòng 72-94)**
   - Kiểm tra lại stdin status
   - Nếu không OK, thử mở lại từ 'CON'

5. **Gọi `input()` (Dòng 98)**
   ```python
   result = input(prompt).strip()
   ```
   - ⚠️ **ĐIỂM CÓ THỂ XẢY RA LỖI:** "I/O operation on closed file"

6. **Xử lý Exception (Dòng 100-131)**
   - `ValueError`, `OSError`, `IOError`: Thử mở lại stdin, return default
   - `EOFError`: Return default
   - `KeyboardInterrupt`: Re-raise
   - `AttributeError`: Return default

---

### 4. Hàm `MarketBatchScanner.scan_market()`

**File:** `modules/gemini_chart_analyzer/core/scanners/market_batch_scanner.py` (Dòng 201-...)

#### 4.1. Khởi tạo và Setup (Dòng 230-252)

- Xác định mode: Single-TF hoặc Multi-TF
- Normalize timeframes
- Khởi tạo generators và aggregators

#### 4.2. Step 0: Cleanup (Dòng 254-255)

```python
self._cleanup_old_results()
```

#### 4.3. Step 1: Lấy Tất cả Symbols (Dòng 257-276)

```python
all_symbols = self.get_all_symbols()
```

- Lấy danh sách symbols từ exchange
- Xử lý lỗi `SymbolFetchError`
- Lưu `original_symbol_count` để so sánh sau pre-filter

#### 4.4. Step 1.5: Pre-filter Symbols (Dòng 279-355)

**⚠️ ĐIỂM QUAN TRỌNG:** Đây là nơi có thể xảy ra lỗi "I/O operation on closed file".

**Luồng:**

1. **Kiểm tra điều kiện (Dòng 282)**
   ```python
   if pre_filter_percentage is not None and pre_filter_percentage > 0.0 and all_symbols:
   ```

2. **Lưu stdin state (Dòng 284-297)**
   ```python
   saved_stdin = None
   if sys.platform == 'win32' and hasattr(sys, 'stdin') and sys.stdin is not None:
       saved_stdin = sys.stdin
       if hasattr(sys.stdin, 'closed') and sys.stdin.closed:
           sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
   ```
   - Lưu reference đến stdin hiện tại
   - Nếu stdin đã closed, thử mở lại

3. **Xác định primary timeframe (Dòng 299-305)**
   - Multi-TF: Dùng timeframe đầu tiên
   - Single-TF: Dùng timeframe đã chọn
   - Default: '1h'

4. **Gọi Pre-filter (Dòng 308-314)**
   ```python
   all_symbols = self._pre_filter_symbols_with_voting(
       all_symbols=all_symbols,
       percentage=pre_filter_percentage,
       timeframe=primary_timeframe,
       limit=limit
   )
   ```
   - ⚠️ **ĐIỂM CÓ THỂ XẢY RA LỖI:** Exception có thể xảy ra ở đây

5. **Xử lý Exception (Dòng 315-324)**
   - Nếu pre-filter fails, log warning và tiếp tục với tất cả symbols
   - **KHÔNG re-raise** exception

6. **Restore stdin (Dòng 325-338) - Finally Block**
   ```python
   finally:
       if sys.platform == 'win32' and saved_stdin is not None:
           if sys.stdin is None or sys.stdin.closed:
               if saved_stdin is not None and not saved_stdin.closed:
                   sys.stdin = saved_stdin
               else:
                   sys.stdin = open('CON', 'r', encoding='utf-8', errors='replace')
   ```
   - Luôn restore stdin sau pre-filter

7. **Outer Exception Handler (Dòng 349-355)**
   - Bắt mọi exception từ toàn bộ pre-filter section
   - Log warning và tiếp tục với tất cả symbols

#### 4.5. Step 2: Apply max_symbols (Dòng 357-360)

```python
if max_symbols and all_symbols:
    all_symbols = all_symbols[:max_symbols]
```

- Áp dụng **SAU** pre-filter

#### 4.6. Step 3: Split into Batches (Dòng 364-368)

```python
batch_size = self.MULTI_TF_CHARTS_PER_BATCH if is_multi_tf else self.charts_per_batch
batches = self._split_into_batches(all_symbols, batch_size=batch_size)
```

#### 4.7. Step 4: Process Each Batch (Dòng 370-...)

- Vòng lặp qua từng batch
- Fetch data, generate charts, analyze với Gemini
- Aggregate results

---

### 5. Hàm `_pre_filter_symbols_with_voting()`

**File:** `modules/gemini_chart_analyzer/core/scanners/market_batch_scanner.py`

**Mục đích:** Lọc symbols dựa trên `weighted_score` từ `VotingAnalyzer`.

**Luồng:**

1. **Validate input (Dòng ~748)**
   ```python
   if not all_symbols or percentage <= 0.0:
       return all_symbols
   ```

2. **Tính toán target count (Dòng ~752)**
   ```python
   target_count = int(total_symbols * percentage / 100.0)
   ```

3. **Chạy trong Subprocess (Dòng ~767-938)**
   - Tạo subprocess với `prefilter_worker.py`
   - Truyền input qua stdin (JSON)
   - Nhận output từ stdout (JSON)
   - **Fallback:** Nếu subprocess fails, chạy in-process

4. **Fallback: In-process Execution (Dòng ~940-1100)**
   - Tạo `VotingAnalyzer` instance
   - Chạy `run_atc_scan()`
   - Chạy `calculate_and_vote()`
   - Lấy `long_signals_final` và `short_signals_final`
   - Sort theo `weighted_score` descending
   - Trả về top N% symbols

---

## Sơ Đồ Luồng Hoạt Động

```
main()
  └─> interactive_batch_scan()
       │
       ├─> [INPUT] Mode selection (input())
       ├─> [INPUT] Timeframes (input())
       ├─> [INPUT] Max symbols (input())
       ├─> [INPUT] Cooldown (input())
       ├─> [INPUT] Limit (input())
       ├─> [INPUT] Pre-filter percentage (input())
       │
       ├─> [CONFIRM] safe_input("Start batch scan?")
       │
       ├─> MarketBatchScanner.__init__()
       │    ├─> ExchangeManager()
       │    ├─> PublicExchangeManager()
       │    ├─> DataFetcher()
       │    ├─> ChartBatchGenerator()
       │    └─> GeminiBatchChartAnalyzer (lazy init - KHÔNG khởi tạo ở đây)
       │
       └─> scanner.scan_market()
            │
            ├─> Step 0: Cleanup old results
            │
            ├─> Step 1: Get all symbols
            │
            ├─> Step 1.5: Pre-filter (NẾU pre_filter_percentage > 0)
            │    │
            │    ├─> Save stdin state
            │    │
            │    ├─> _pre_filter_symbols_with_voting()
            │    │    │
            │    │    ├─> [SUBPROCESS] prefilter_worker.py
            │    │    │    ├─> Read input từ stdin (JSON)
            │    │    │    ├─> Create VotingAnalyzer
            │    │    │    ├─> run_atc_scan()
            │    │    │    ├─> calculate_and_vote()
            │    │    │    ├─> Extract signals (long/short)
            │    │    │    ├─> Sort by weighted_score
            │    │    │    └─> Write output ra stdout (JSON)
            │    │    │
            │    │    └─> [FALLBACK] In-process execution
            │    │         └─> (Tương tự subprocess nhưng chạy trong process hiện tại)
            │    │
            │    └─> Restore stdin state
            │
            ├─> Step 2: Apply max_symbols
            │
            ├─> Step 3: Split into batches
            │
            ├─> Step 4: Process each batch
            │    ├─> Fetch OHLCV data
            │    ├─> Generate batch chart
            │    ├─> Analyze with Gemini (batch_gemini_analyzer - lazy init ở đây)
            │    └─> Aggregate results
            │
            └─> Step 5: Return results
```

---

## Các Điểm Có Thể Xảy Ra Lỗi

### 1. Lỗi "I/O operation on closed file"

**Vị trí có thể xảy ra:**

1. **Trong `interactive_batch_scan()` - Các `input()` calls (Dòng 146, 159, 175, 192, 205, 218, 233)**
   - ⚠️ **VẤN ĐỀ:** Tất cả đều dùng `input()` trực tiếp, không dùng `safe_input()`
   - **Nguyên nhân:** Có thể stdin bị đóng bởi một thư viện hoặc process khác

2. **Trong `MarketBatchScanner.__init__()` (Dòng 283)**
   - Khởi tạo `GeminiBatchChartAnalyzer` (lazy init) có thể đóng stdin
   - **Giải pháp hiện tại:** Lazy initialization - chỉ khởi tạo khi cần

3. **Trong `_pre_filter_symbols_with_voting()` (Dòng 309)**
   - Subprocess có thể ảnh hưởng đến stdin của parent process
   - **Giải pháp hiện tại:** 
     - Chạy trong subprocess riêng
     - Save/restore stdin state
     - Fallback to in-process nếu subprocess fails

4. **Trong `safe_input()` (Dòng 98)**
   - Ngay cả khi đã kiểm tra, `input()` vẫn có thể fail
   - **Giải pháp:** Exception handler trả về default

### 2. Lỗi trong Pre-filter

**Vị trí:**
- `_pre_filter_symbols_with_voting()` (Dòng 309)
- Subprocess communication (Dòng ~800-900)
- In-process fallback (Dòng ~940-1100)

**Xử lý:**
- Exception được catch và không re-raise
- Tiếp tục với tất cả symbols nếu pre-filter fails

### 3. Lỗi trong Scanner Initialization

**Vị trí:**
- `MarketBatchScanner.__init__()` (Dòng 283)

**Xử lý:**
- Exception được catch ở outer handler (Dòng 353)
- Log error và exit

---

## Các Điểm Cần Lưu Ý Khi Debug

### 1. Thứ tự Input Calls

Tất cả các `input()` calls trong `interactive_batch_scan()` đều sử dụng `input()` trực tiếp, **TRỪ** confirmation prompt (dòng 274) sử dụng `safe_input()`.

**Các input() calls:**
- Dòng 146: Mode selection
- Dòng 159: Timeframes (multi-TF mode)
- Dòng 175: Timeframe (single-TF mode)
- Dòng 192: Max symbols
- Dòng 205: Cooldown
- Dòng 218: Limit
- Dòng 233: Pre-filter percentage

**Nếu lỗi xảy ra ở bất kỳ input() nào ở trên, có thể:**
- Stdin đã bị đóng bởi một thư viện trước đó
- Hoặc stdin bị đóng trong quá trình thực thi

### 2. Pre-filter Flow

Pre-filter được thực thi **SAU** khi user đã nhập tất cả input và xác nhận. Điều này đảm bảo:
- Tất cả user input đã được thu thập
- Stdin vẫn còn available cho confirmation prompt

Tuy nhiên, nếu lỗi xảy ra trong pre-filter:
- Exception được catch và không re-raise
- Scan tiếp tục với tất cả symbols
- Stdin được restore trong finally block

### 3. Lazy Initialization của GeminiBatchChartAnalyzer

`GeminiBatchChartAnalyzer` được khởi tạo **lazy** (chỉ khi cần):
- **KHÔNG** khởi tạo trong `__init__()`
- **KHÔNG** khởi tạo trong pre-filter
- **CHỈ** khởi tạo khi `analyze_batch_chart()` được gọi (trong Step 4)

Điều này tránh đóng stdin sớm trong quá trình thu thập input.

---

## Checklist Debug

Khi gặp lỗi "I/O operation on closed file", kiểm tra:

- [ ] Lỗi xảy ra ở input() call nào? (Dòng số?)
- [ ] Stdin có bị đóng trước đó không? (Kiểm tra `sys.stdin.closed`)
- [ ] Có thư viện nào đóng stdin không? (Kiểm tra imports và initialization)
- [ ] Pre-filter có chạy không? (Kiểm tra `pre_filter_percentage > 0`)
- [ ] Subprocess có chạy thành công không? (Kiểm tra return code)
- [ ] Stdin có được restore sau pre-filter không? (Kiểm tra finally block)

---

## Gợi Ý Fix

### 1. Thay tất cả `input()` bằng `safe_input()`

**Vị trí:** Tất cả các `input()` calls trong `interactive_batch_scan()` (trừ confirmation đã dùng `safe_input()`)

**Lý do:** Đảm bảo tất cả input calls đều xử lý lỗi stdin một cách an toàn.

### 2. Kiểm tra stdin state trước mỗi input() call

Thêm logging để kiểm tra stdin state trước mỗi input() call để xác định khi nào stdin bị đóng.

### 3. Đảm bảo stdin được restore sau mỗi operation

Kiểm tra xem stdin có được restore đúng cách sau pre-filter không.

---

## File Liên Quan

- `modules/gemini_chart_analyzer/cli/batch_scanner_main.py` - Entry point và interactive menu
- `modules/gemini_chart_analyzer/core/scanners/market_batch_scanner.py` - Core scanner logic
- `modules/gemini_chart_analyzer/core/prefilter_worker.py` - Pre-filter worker (subprocess)
- `modules/gemini_chart_analyzer/core/analyzers/gemini_batch_chart_analyzer.py` - Gemini analyzer (lazy init)
