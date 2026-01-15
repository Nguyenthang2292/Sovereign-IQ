# Memory Management Improvements

Tài liệu này mô tả các cải thiện về quản lý bộ nhớ và garbage collection đã được thực hiện trong backend.

## Tổng quan

Các cải thiện tập trung vào:
1. **TaskManager**: Cleanup thread references và result data
2. **LogFileManager**: Cleanup locks dictionary và auto-cleanup log files
3. **ExchangeManager**: Cleanup exchange connections
4. **MarketBatchScanner**: Cleanup resources sau khi scan
5. **Memory Utilities**: Công cụ monitoring và optimization

## Chi tiết cải thiện

### 1. TaskManager (`web/utils/task_manager.py`)

**Vấn đề:**
- Thread objects được lưu trong `_tasks` dict và không được giải phóng
- Result data có thể rất lớn và được giữ trong memory lâu
- Không có explicit garbage collection sau cleanup

**Giải pháp:**
- Thêm conditional `gc.collect()` sau khi cleanup tasks (chỉ khi có tasks được xóa)
- Explicitly clear `result`, `error`, và `thread` references trước khi xóa task
- Thêm method `clear_all_results()` để clear results từ tất cả completed tasks
- Xem phần "Garbage Collection Performance" bên dưới để biết chi tiết về khi nào nên force GC

**Sử dụng:**
```python
from web.utils.task_manager import get_task_manager

task_manager = get_task_manager()
# Cleanup một task cụ thể
task_manager.cleanup_task(session_id)

# Clear tất cả results để free memory
task_manager.clear_all_results()
```

### 2. LogFileManager (`web/utils/log_manager.py`)

**Vấn đề:**
- Locks dictionary `_locks` không bao giờ được cleanup, có thể tăng trưởng vô hạn
- Log files được tạo nhưng không tự động cleanup

**Giải pháp:**
- Thêm tracking `_lock_last_used` để theo dõi thời gian sử dụng lock
- Thêm background cleanup thread `_cleanup_locks_loop()` chạy mỗi 10 phút
- Thêm method `cleanup_lock()` để manually cleanup một lock cụ thể
- Auto-cleanup locks không được sử dụng trong 2 giờ
- Garbage collection có thể được trigger sau cleanup (xem "Garbage Collection Performance" bên dưới)

**Cấu hình timing values:**

| Giá trị | Mặc định | Có thể cấu hình? | Cách cấu hình |
|---------|----------|------------------|---------------|
| Cleanup loop interval | 10 phút | ❌ **Hard-coded** | Không thể thay đổi (được hard-code trong `_cleanup_locks_loop()`) |
| Lock threshold (max_age_hours) | 2 giờ | ✅ **Có thể** | Thay đổi trong code: `_cleanup_unused_locks(max_age_hours=X)` hoặc tạo instance với custom logic |

**Lưu ý về trade-offs:**
- **Cleanup loop interval (10 phút)**: 
  - Giá trị nhỏ hơn → cleanup thường xuyên hơn, tốn CPU hơn nhưng memory được giải phóng nhanh hơn
  - Giá trị lớn hơn → tiết kiệm CPU nhưng locks có thể tích tụ lâu hơn
  - 10 phút là cân bằng tốt cho hầu hết trường hợp
- **Lock threshold (2 giờ)**:
  - Giá trị nhỏ hơn → giải phóng memory sớm hơn nhưng có thể cleanup nhầm locks đang được sử dụng
  - Giá trị lớn hơn → giữ locks lâu hơn, an toàn hơn nhưng tốn memory hơn
  - 2 giờ phù hợp cho các session dài, nếu session ngắn có thể giảm xuống 30-60 phút

**Cách điều chỉnh cho operators:**
- Để thay đổi lock threshold, cần modify code trong `_cleanup_unused_locks()` method hoặc override method này trong subclass
- Cleanup loop interval hiện tại là hard-coded và không thể cấu hình mà không sửa code

**Sử dụng:**
```python
from web.utils.log_manager import get_log_manager

log_manager = get_log_manager()
# Cleanup một lock cụ thể
log_manager.cleanup_lock(session_id)

# Cleanup old log files (đã có sẵn)
log_manager.cleanup_old_logs(max_age_hours=24)
```

### 3. ExchangeManager (`modules/common/core/exchange_manager.py`)

**Vấn đề:**
- Exchange connections được cache trong `_authenticated_exchanges` và `_public_exchanges` không có giới hạn
- Connections không được đóng đúng cách

**Giải pháp:**
- Thêm method `cleanup_unused_exchanges()` để clear tất cả cached exchanges
- Thêm method `close_exchange()` để close và remove một exchange cụ thể
- Thêm logging để track cleanup operations
- Force garbage collection sau cleanup (tùy chọn, xem "Garbage Collection Performance" bên dưới)

**Sử dụng:**
```python
from modules.common.core.exchange_manager import ExchangeManager

exchange_manager = ExchangeManager()
# Cleanup tất cả unused exchanges
exchange_manager.cleanup_unused_exchanges(max_age_hours=24)

# Close một exchange cụ thể
exchange_manager.close_exchange('binance', testnet=False, contract_type='future')
```

### 4. MarketBatchScanner (`modules/gemini_chart_analyzer/core/scanners/market_batch_scanner.py`)

**Vấn đề:**
- Tạo nhiều objects (ExchangeManager, DataFetcher, ChartBatchGenerator, etc.) nhưng không có cleanup
- Exchange connections không được giải phóng sau khi scan

**Giải pháp:**
- Thêm method `cleanup()` để cleanup resources
- Cleanup exchange connections sau khi scan hoàn thành
- Force garbage collection sau cleanup (tùy chọn, xem "Garbage Collection Performance" bên dưới)

**Sử dụng:**
```python
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner

scanner = MarketBatchScanner()
try:
    results = scanner.scan_market(...)
finally:
    scanner.cleanup()  # Cleanup resources
```

**Tự động cleanup trong API:**
- `web/api/batch_scanner.py` đã được cập nhật để tự động gọi `scanner.cleanup()` sau khi task hoàn thành

### 5. Memory Utilities (`web/utils/memory_utils.py`)

**Tính năng:**
- `get_memory_usage()`: Lấy thông tin memory usage hiện tại
- `force_garbage_collection()`: Force GC và return statistics
- `start_memory_tracking()`: Bắt đầu tracking memory allocations với tracemalloc
- `get_memory_snapshot_diff()`: So sánh hai memory snapshots
- `log_memory_usage()`: Log memory usage với context
- `optimize_memory()`: Chạy memory optimization routines

**Sử dụng:**
```python
from web.utils.memory_utils import (
    get_memory_usage,
    force_garbage_collection,
    log_memory_usage,
    optimize_memory
)

# Log memory usage
log_memory_usage("before operation")

# Force garbage collection
gc_stats = force_garbage_collection(verbose=True)

# Get memory usage
mem_info = get_memory_usage()
print(f"RSS: {mem_info['rss_mb']:.2f}MB")

# Optimize memory
optimize_memory()
```

## Garbage Collection Performance

### Tổng quan về Forced GC

Việc force garbage collection (`gc.collect()`) có thể giúp giải phóng bộ nhớ ngay lập tức sau các cleanup operations lớn, nhưng có thể gây ra **pause times đáng kể** trong production, đặc biệt nếu được gọi thường xuyên.

### Khi nào Force GC là cần thiết?

**Force GC được khuyến nghị trong các trường hợp:**
1. **Sau cleanup operations lớn:**
   - Cleanup nhiều tasks cùng lúc (TaskManager: `cleanup_old_tasks()`)
   - Cleanup kết quả từ nhiều completed tasks (TaskManager: `clear_all_results()`)
   - Cleanup nhiều exchange connections (ExchangeManager: `cleanup_unused_exchanges()`)
   - Sau khi hoàn thành batch scan operations lớn (MarketBatchScanner)

2. **Khi có dấu hiệu memory pressure:**
   - Memory usage vượt quá threshold (ví dụ minh họa: >80% RSS - cần calibrate cho môi trường của bạn)
   - Nhiều objects chờ được collect (gc.get_count() cho thấy số lượng cao)
   - Performance bị ảnh hưởng do memory pressure

3. **Trong low-traffic windows:**
   - Khi không có requests đang xử lý
   - Trong background cleanup threads (scheduled cleanup)
   - Trong maintenance windows

### Khi nào KHÔNG nên Force GC?

**Tránh force GC trong các trường hợp:**
1. **Trong request handling path với high traffic:**
   - Force GC có thể gây pause times đáng kể (thường từ vài chục ms đến vài trăm ms, nhưng phụ thuộc vào môi trường)
   - ⚠️ **Lưu ý:** Pause times cụ thể phụ thuộc rất nhiều vào môi trường (số lượng objects, CPU, workload). Cần đo lường trong môi trường thực tế của bạn để xác định impact chính xác.
   - Ảnh hưởng đến response time và user experience
   - Trong production với nhiều concurrent requests

2. **Sau cleanup operations nhỏ:**
   - Cleanup một task đơn lẻ
   - Cleanup một lock đơn lẻ
   - Python's automatic GC thường đủ hiệu quả cho các trường hợp này

3. **Khi không có đủ garbage để collect:**
   - `gc.collect()` vẫn tốn CPU cycles ngay cả khi không có gì để collect
   - Sử dụng `gc.get_count()` để kiểm tra trước khi force GC

### Thực tế trong codebase

**TaskManager:**
- Force GC chỉ được gọi khi có tasks được cleanup (`cleanup_old_tasks()`)
- Force GC được gọi sau `clear_all_results()` nếu có results được cleared
- `cleanup_task()` đơn lẻ **không** force GC để tránh overhead

**ExchangeManager:**
- Force GC sau `cleanup_unused_exchanges()` - thường chạy trong background/scheduled
- Có thể tùy chọn dựa trên số lượng exchanges được cleanup

**MarketBatchScanner:**
- Force GC sau `cleanup()` - thường được gọi sau khi operation lớn hoàn thành
- Thường chạy trong background thread, không ảnh hưởng request handling

### Metrics và Thresholds

> **⚠️ QUAN TRỌNG:** Các thresholds và pause time estimates trong phần này là **minh họa và phụ thuộc vào môi trường**. Chúng chỉ nên được sử dụng như điểm khởi đầu thô. **Bạn PHẢI validate và calibrate các giá trị này cho môi trường deployment cụ thể của mình** thông qua profiling và đo lường thực tế.

**Tài liệu tham khảo:**
- [Python `gc` module documentation](https://docs.python.org/3/library/gc.html)
- [CPython Garbage Collection](https://devguide.python.org/internals/garbage-collector/)
- [Python Memory Management](https://docs.python.org/3/c-api/memory.html)

**Sử dụng các metrics sau để quyết định khi nào force GC (các giá trị chỉ mang tính minh họa):**

```python
from web.utils.memory_utils import get_memory_usage, force_garbage_collection
import gc

# Kiểm tra memory pressure
# ⚠️ LƯU Ý: Threshold 1000MB chỉ là ví dụ minh họa
# Cần calibrate dựa trên workload và resources thực tế của bạn
mem_info = get_memory_usage()
if mem_info['rss_mb'] > 1000:  # Ví dụ minh họa: >1GB (điểm khởi đầu thô)
    # Memory pressure cao, có thể cần GC
    
# Kiểm tra số lượng objects chờ collect
# ⚠️ LƯU Ý: Threshold 1000 chỉ là ví dụ minh họa
# Giá trị thực tế phụ thuộc vào application workload
gc_counts = gc.get_count()
if sum(gc_counts) > 1000:  # Ví dụ minh họa: nhiều objects chờ collect (điểm khởi đầu thô)
    # Có thể có lợi từ forced GC
    
# Force GC với monitoring
gc_stats = force_garbage_collection(verbose=True)
if gc_stats['collected'] > 100:  # Ví dụ minh họa: threshold hiệu quả (điểm khởi đầu thô)
    # GC đã thu thập nhiều objects, có hiệu quả
```

**Đo lường và Calibration cho môi trường của bạn:**

Để xác định thresholds phù hợp, bạn có thể chạy các benchmark đơn giản:

```python
import gc
import time
import psutil
import os

def benchmark_gc_impact():
    """Đo lường pause time và memory impact của forced GC trong môi trường của bạn."""
    process = psutil.Process(os.getpid())
    
    # Memory trước GC
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    gc_counts_before = gc.get_count()
    
    # Đo pause time của GC
    start_time = time.perf_counter()
    collected = gc.collect()
    pause_time_ms = (time.perf_counter() - start_time) * 1000
    
    # Memory sau GC
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    gc_counts_after = gc.get_count()
    
    print(f"GC pause time: {pause_time_ms:.2f} ms")
    print(f"Objects collected: {collected}")
    print(f"Memory before: {mem_before:.2f} MB, after: {mem_after:.2f} MB")
    print(f"GC counts before: {gc_counts_before}, after: {gc_counts_after}")
    
    return {
        'pause_time_ms': pause_time_ms,
        'collected': collected,
        'memory_freed_mb': mem_before - mem_after,
        'gc_counts_before': gc_counts_before,
        'gc_counts_after': gc_counts_after
    }

# Chạy benchmark trong các điều kiện khác nhau:
# 1. Trong idle state
# 2. Sau khi tạo nhiều objects (simulate workload)
# 3. Trong các thời điểm khác nhau của application lifecycle

# Metrics để theo dõi:
# - Pause time: nếu > 100ms có thể ảnh hưởng response time
# - Objects collected: giúp xác định threshold có ý nghĩa
# - Memory freed: đánh giá hiệu quả của GC
# - GC counts: hiểu pattern của automatic GC
```

### Best Practices cho Forced GC

1. **Conditional GC:**
   - Chỉ force GC khi cleanup một số lượng lớn objects
   - Sử dụng metrics để quyết định

2. **Async/Scheduled GC:**
   - Chạy forced GC trong background threads
   - Tránh trong request handling path

3. **Monitoring:**
   - Log GC statistics để theo dõi hiệu quả
   - Monitor pause times nếu có thể (sử dụng `time.perf_counter()` để đo chính xác)
   - Calibrate thresholds dựa trên metrics thực tế từ production

4. **Gradual cleanup:**
   - Cleanup theo batch nhỏ thay vì cleanup tất cả cùng lúc
   - Phân tán GC load theo thời gian

### Ví dụ: Conditional GC

```python
def cleanup_with_smart_gc(items_cleaned: int, min_for_gc: int = 10):
    """
    Cleanup items và chỉ force GC nếu cleanup đủ nhiều items.
    
    ⚠️ LƯU Ý: Giá trị `min_for_gc=10` chỉ là ví dụ minh họa (điểm khởi đầu thô).
    Cần calibrate dựa trên profiling trong môi trường thực tế của bạn.
    
    Args:
        items_cleaned: Số lượng items đã được cleanup
        min_for_gc: Số lượng tối thiểu để trigger GC (default: 10, chỉ là ví dụ minh họa)
    """
    if items_cleaned >= min_for_gc:
        gc.collect()
        log_debug(f"Garbage collected after cleaning {items_cleaned} items")
    # Nếu items_cleaned < min_for_gc, để Python tự động GC
```

## Best Practices

1. **Sau khi task hoàn thành:**
   - Gọi `scanner.cleanup()` nếu sử dụng MarketBatchScanner
   - Gọi `task_manager.cleanup_task(session_id)` nếu không cần giữ task metadata
   - Gọi `log_manager.cleanup_lock(session_id)` để cleanup lock

2. **Periodic cleanup:**
   - TaskManager tự động cleanup tasks cũ hơn 1 giờ (có thể cấu hình qua constructor parameter `cleanup_after_hours`)
   - LogFileManager tự động cleanup locks không dùng trong 2 giờ (có thể cấu hình bằng cách subclass và override `_cleanup_unused_locks()`)
   - Có thể gọi `cleanup_old_logs()` để cleanup log files cũ

   **Cấu hình timing values cho periodic cleanup:**

   | Component            | Giá trị                | Mặc định | Có thể cấu hình?     | Cách cấu hình                                                                                    |
   |----------------------|------------------------|----------|---------------------|--------------------------------------------------------------------------------------------------|
   | **TaskManager**      | Cleanup loop interval  | 5 phút   | ❌ **Không thể**     | Hard-coded trong `_cleanup_loop()`. Chỉ có thể thay đổi bằng cách sửa code.                     |
   | **TaskManager**      | Task threshold         | 1 giờ    | ✅ **Có thể**        | Tạo instance với `TaskManager(cleanup_after_hours=X)` hoặc modify global instance                |
   | **LogFileManager**   | Cleanup loop interval  | 10 phút  | ❌ **Không thể**     | Hard-coded trong `_cleanup_locks_loop()` method với `time.sleep(600)`. Chỉ có thể thay đổi bằng cách sửa code. |
   | **LogFileManager**   | Lock threshold        | 2 giờ    | ✅ **Có thể**        | Subclass `LogFileManager` và override `_cleanup_unused_locks()`, hoặc monkey-patch method này.  |

   **Lưu ý quan trọng về LogFileManager:**
   - **Cleanup loop interval (10 phút)**: Giá trị này được hard-code trong method `_cleanup_locks_loop()` của `web/utils/log_manager.py` với `time.sleep(600)`. **Không thể cấu hình** mà không sửa đổi code nguồn.
   - **Lock threshold (2 giờ)**: Giá trị này có thể được thay đổi bằng cách subclass `LogFileManager` và override method `_cleanup_unused_locks()`, hoặc bằng cách monkey-patch method này. Xem ví dụ cụ thể bên dưới.

   **Trade-offs khi điều chỉnh:**

   - **TaskManager cleanup_after_hours (mặc định: 1 giờ)**:
     - Giá trị nhỏ hơn (30 phút): Giải phóng memory sớm hơn, phù hợp khi có nhiều tasks ngắn. Nhưng có thể xóa tasks trước khi user kịp lấy kết quả.
     - Giá trị lớn hơn (2-4 giờ): Giữ task metadata lâu hơn, tiện cho debugging và monitoring. Nhưng tốn memory hơn.
     - **Khuyến nghị**: 1 giờ phù hợp cho hầu hết trường hợp. Tăng lên 2-3 giờ nếu cần giữ task history lâu hơn.

   - **LogFileManager lock threshold (mặc định: 2 giờ)**:
     - Giá trị nhỏ hơn (30-60 phút): Phù hợp khi session ngắn, giải phóng memory nhanh. Nhưng có thể cleanup nhầm locks đang active.
     - Giá trị lớn hơn (4-6 giờ): An toàn hơn cho session dài. Nhưng locks tích tụ lâu hơn.
     - **Khuyến nghị**: 2 giờ là cân bằng tốt. Giảm xuống 1 giờ nếu session thường < 30 phút.

   **Cách điều chỉnh cho operators:**

   1. **TaskManager task threshold**:
      ```python
      # Tạo custom instance với threshold khác
      from web.utils.task_manager import TaskManager
      custom_task_manager = TaskManager(cleanup_after_hours=2)  # 2 giờ thay vì 1 giờ
      ```

   2. **LogFileManager lock threshold** (và cleanup loop interval nếu cần):

      **Cách 1: Subclass LogFileManager (khuyến nghị)**

      Tạo một subclass và override method `_cleanup_unused_locks()` để thay đổi lock threshold. Nếu cần thay đổi cleanup loop interval, cũng override method `_cleanup_locks_loop()`:

      ```python
      from web.utils.log_manager import LogFileManager
      from datetime import datetime, timedelta
      import time
      import threading
      from modules.common.ui.logging import log_debug
      
      class CustomLogFileManager(LogFileManager):
          """
          Custom LogFileManager với lock threshold và cleanup interval tùy chỉnh.
          """
          
          def __init__(self, logs_dir=None, auto_cleanup_before_new=None, 
                      max_log_age_hours=None, start_cleanup_thread=True,
                      lock_threshold_hours=1, cleanup_interval_seconds=300):
              """
              Args:
                  lock_threshold_hours: Thời gian (giờ) trước khi cleanup lock không dùng (mặc định: 1 giờ)
                  cleanup_interval_seconds: Khoảng thời gian (giây) giữa các lần cleanup (mặc định: 5 phút = 300 giây)
              """
              # Lưu các giá trị tùy chỉnh
              self.lock_threshold_hours = lock_threshold_hours
              self.cleanup_interval_seconds = cleanup_interval_seconds
              
              # Gọi constructor của parent class
              super().__init__(logs_dir, auto_cleanup_before_new, max_log_age_hours, 
                             start_cleanup_thread=False)  # Tạm thời không start thread
              
              # Nếu cần start cleanup thread với interval tùy chỉnh
              if start_cleanup_thread:
                  self._cleanup_thread = threading.Thread(target=self._cleanup_locks_loop, daemon=True)
                  self._cleanup_thread.start()
          
          def _cleanup_locks_loop(self):
              """Override để sử dụng cleanup interval tùy chỉnh."""
              while True:
                  try:
                      time.sleep(self.cleanup_interval_seconds)  # Sử dụng interval tùy chỉnh
                      self._cleanup_unused_locks()
                  except Exception as e:
                      from modules.common.ui.logging import log_error
                      log_error(f"Error in lock cleanup loop: {e}")
          
          def _cleanup_unused_locks(self, max_age_hours=None):
              """
              Override để sử dụng lock threshold tùy chỉnh.
              
              Args:
                  max_age_hours: Nếu None, sử dụng self.lock_threshold_hours
              """
              if max_age_hours is None:
                  max_age_hours = self.lock_threshold_hours
              
              cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
              
              with self._locks_lock:
                  to_remove = []
                  for session_id, last_used in self._lock_last_used.items():
                      if last_used < cutoff_time:
                          to_remove.append(session_id)
                  
                  for session_id in to_remove:
                      self._locks.pop(session_id, None)
                      self._lock_last_used.pop(session_id, None)
                  
                  if to_remove:
                      log_debug(f"Cleaned up {len(to_remove)} unused locks (threshold: {max_age_hours}h)")
      
      # Sử dụng custom manager
      custom_log_manager = CustomLogFileManager(
          lock_threshold_hours=1,      # Cleanup locks sau 1 giờ (thay vì 2 giờ)
          cleanup_interval_seconds=300  # Cleanup mỗi 5 phút (thay vì 10 phút)
      )
      ```

      **Các bước cần thiết khi subclass:**
      1. Override `__init__()` để nhận các tham số tùy chỉnh (`lock_threshold_hours`, `cleanup_interval_seconds`)
      2. Lưu các giá trị tùy chỉnh vào instance variables (`self.lock_threshold_hours`, `self.cleanup_interval_seconds`)
      3. Gọi `super().__init__()` với `start_cleanup_thread=False` để tránh start thread với interval mặc định
      4. Override `_cleanup_locks_loop()` để sử dụng `self.cleanup_interval_seconds` thay vì hard-code 600 giây
      5. Override `_cleanup_unused_locks()` để sử dụng `self.lock_threshold_hours` thay vì default 2 giờ
      6. Tự start cleanup thread sau khi gọi `super().__init__()` nếu `start_cleanup_thread=True`

      **Cách 2: Monkey-patch (nhanh nhưng không khuyến nghị cho production)**
      
      ```python
      from web.utils.log_manager import get_log_manager
      
      # Lấy instance hiện tại
      log_manager = get_log_manager()
      
      # Monkey-patch method để thay đổi threshold
      original_cleanup = log_manager._cleanup_unused_locks
      def custom_cleanup(max_age_hours=1):  # 1 giờ thay vì 2 giờ
          return original_cleanup(max_age_hours)
      log_manager._cleanup_unused_locks = custom_cleanup
      ```
      
      **Lưu ý**: Monkey-patch chỉ thay đổi được lock threshold, không thể thay đổi cleanup loop interval (10 phút) vì nó được hard-code trong `_cleanup_locks_loop()`.

3. **Memory monitoring:**
   - Sử dụng `memory_utils` để monitor memory usage
   - Log memory usage tại các điểm quan trọng trong code
   - Sử dụng tracemalloc để track memory leaks

4. **Exchange connections:**
   - Cleanup exchange connections sau khi không cần dùng nữa
   - Sử dụng `cleanup_unused_exchanges()` định kỳ để free memory

## Thread-Safety và Error Handling

### Thread-Safety Mechanisms

Các cleanup operations sử dụng các cơ chế đồng bộ hóa sau để đảm bảo thread-safety:

#### 1. TaskManager (`web/utils/task_manager.py`)

**Synchronization primitives:**
- `threading.Lock()` (`self._lock`) - Bảo vệ dictionary `self._tasks`, được khởi tạo trong `__init__()` method
- `threading.Lock()` (`_task_manager_lock`) - Bảo vệ global instance creation trong `get_task_manager()` function

**Cleanup methods với thread-safety:**
- `cleanup_task()`: Sử dụng `with self._lock:` để bảo vệ access vào `self._tasks`
- `_cleanup_old_tasks()`: Sử dụng `with self._lock:` để đọc và xóa tasks
- `clear_all_results()`: Sử dụng `with self._lock:` để iterate và clear results

**Error handling:**
- `_cleanup_loop()`: Bắt `Exception` và log với `log_error()` trong exception handler
  - Log message: `"Error in cleanup loop: {e}"`
  - Failure type: **Non-fatal** - cleanup loop tiếp tục chạy sau khi catch exception
  - Errors caught: Mọi exceptions từ `_cleanup_old_tasks()` (bao gồm OOM, KeyError khi dict bị thay đổi concurrently, etc.)

#### 2. LogFileManager (`web/utils/log_manager.py`)

**Synchronization primitives:**
- `threading.Lock()` (`self._locks_lock`) - Bảo vệ dictionaries `self._locks` và `self._lock_last_used`, được khởi tạo trong `__init__()` method
- `threading.Lock()` (`self._cleanup_lock`) - Bảo vệ cleanup operations, được khởi tạo trong `__init__()` method
- Per-session locks (`self._locks[session_id]`) - `threading.Lock()` objects để bảo vệ file I/O operations, được tạo bởi `_get_lock()` method

**Cleanup methods với thread-safety:**
- `cleanup_lock()`: Sử dụng `with self._locks_lock:` để remove lock entries
- `_cleanup_unused_locks()`: Sử dụng `with self._locks_lock:` để iterate và remove locks
- `cleanup_old_logs()`: Không có lock riêng nhưng được gọi từ `_cleanup_old_logs_before_new_request()` có `_cleanup_lock`
- `_cleanup_old_logs_before_new_request()`: Sử dụng `with self._cleanup_lock:` để serialize cleanup
- File operations (`write_log()`, `read_log()`, `delete_log()`): Sử dụng per-session locks (`with self._get_lock(session_id):`)

**Error handling:**
- `write_log()`: Bắt `Exception` và log với `log_error()`, không throw trong exception handler
  - Log message: `"Error writing to log file {log_path}: {e}"`
  - Failure type: **Non-fatal** - operation fails silently, file write không thành công
  - Errors caught: IOError, OSError (disk full, permission denied), UnicodeEncodeError, etc.
  
- `read_log()`: Bắt `Exception` và log với `log_error()`, trả về empty string trong exception handler
  - Log message: `"Error reading log file {log_path}: {e}"`
  - Failure type: **Non-fatal** - trả về `("", offset)` khi có lỗi
  - Errors caught: IOError, FileNotFoundError, PermissionError, etc.

- `delete_log()`: Bắt `Exception` và log với `log_error()`, trả về False trong exception handler
  - Log message: `"Error deleting log file {log_path}: {e}"`
  - Failure type: **Non-fatal** - trả về `False` khi có lỗi
  - Errors caught: IOError, PermissionError, FileNotFoundError, etc.

- `cleanup_old_logs()`: Bắt `Exception` cho từng file và toàn bộ operation trong exception handlers
  - Log messages: 
    - Per-file: `"Error cleaning up log file {log_file}: {e}"`
    - Overall: `"Error during log cleanup: {e}"`
  - Failure type: **Non-fatal** - tiếp tục cleanup các files khác khi một file fail
  - Errors caught: IOError, PermissionError, OSError (disk errors), etc.

- `_cleanup_old_logs_before_new_request()`: Bắt `Exception` và log với `log_warn()` trong exception handler
  - Log message: `"Error during auto-cleanup before new request: {e}"`
  - Failure type: **Non-fatal** - log creation tiếp tục ngay cả khi cleanup fails (comment: "Don't fail log creation if cleanup fails")
  - Errors caught: Exceptions từ `cleanup_old_logs()`

- `_cleanup_locks_loop()`: Bắt `Exception` và log với `log_error()` trong exception handler
  - Log message: `"Error in lock cleanup loop: {e}"`
  - Failure type: **Non-fatal** - cleanup loop tiếp tục chạy sau khi catch exception
  - Errors caught: Mọi exceptions từ `_cleanup_unused_locks()`

#### 3. ExchangeManager (`modules/common/core/exchange_manager.py`)

**Synchronization primitives:**
- `threading.Lock()` (`self._request_lock`) - Bảo vệ dictionaries `_authenticated_exchanges` và `_public_exchanges`, được khởi tạo trong `__init__()` method của cả `AuthenticatedExchangeManager` và `PublicExchangeManager`
- Không có atomic flags hay concurrent queues - chỉ sử dụng simple locks

**Cleanup methods với thread-safety:**
- `AuthenticatedExchangeManager.cleanup_unused_exchanges()`: Sử dụng `with self._request_lock:` để clear cache
- `PublicExchangeManager.cleanup_unused_exchanges()`: Sử dụng `with self._request_lock:` để clear cache
- `AuthenticatedExchangeManager.close_exchange()`: Sử dụng `with self._request_lock:` để remove exchange
- `PublicExchangeManager.close_exchange()`: Sử dụng `with self._request_lock:` để remove exchange

**Error handling:**
- `close_exchange()` (authenticated và public): Bắt `Exception` khi gọi `exchange.close()` và log với `logger.warning()` trong exception handler
  - Log message: `"Error closing exchange {exchange_id}: {e}"` (authenticated) hoặc `"Error closing public exchange {exchange_id}: {e}"` (public)
  - Failure type: **Non-fatal** - exchange vẫn được remove khỏi cache ngay cả khi close() fails
  - Errors caught: Exceptions từ exchange.close() method (có thể là network errors, connection errors, etc.)
  - Note: `cleanup_unused_exchanges()` không có try-catch riêng, chỉ sử dụng lock để serialize access

#### 4. MarketBatchScanner (`modules/gemini_chart_analyzer/core/scanners/market_batch_scanner.py`)

**Synchronization primitives:**
- **Không có thread-safety primitives** - cleanup method không sử dụng locks, mutexes, hay atomic flags
- Method này được thiết kế để chạy trong single-threaded context (sau khi scan operation hoàn thành)

**Cleanup methods:**
- `cleanup()`: Không có synchronization primitives

**Error handling:**
- Exchange manager cleanup trong `cleanup()` method: Bắt `Exception` và log với `log_warn()` trong exception handler
  - Log message: `"Error cleaning up exchange managers: {e}"`
  - Failure type: **Non-fatal** - cleanup tiếp tục với các operations khác
  - Errors caught: Exceptions từ `cleanup_unused_exchanges()` hoặc `clear()` methods (AttributeError nếu method không tồn tại, exceptions từ exchange cleanup, etc.)

- Cache attribute cleanup trong `cleanup()` method: Bắt `Exception` cho mỗi cache attribute và log với `log_warn()` trong exception handler
  - Log message: `"Error clearing cache attribute {attr_name}: {e}"`
  - Failure type: **Non-fatal** - tiếp tục cleanup các attributes khác
  - Errors caught: Exceptions từ `clear()` method hoặc `setattr()` (AttributeError, TypeError, etc.)

---

**⚠️ Maintenance Note:** Các tham chiếu code trong section này sử dụng tên method/function và file paths thay vì số dòng để tránh trở nên lỗi thời khi code thay đổi. Maintainers nên re-verify các tham chiếu này trong các PR reviews hoặc theo lịch định kỳ (ví dụ: mỗi quý).

**Last Verified:** [Cần cập nhật khi verify lại các tham chiếu]

### Tổng kết Error Handling

**Failure types:**
- Tất cả cleanup failures đều là **non-fatal** - application tiếp tục hoạt động
- Cleanup operations được thiết kế để fail gracefully và không ảnh hưởng đến các operations khác

**Log levels:**
- `log_error()`: Sử dụng cho các errors trong background loops (TaskManager, LogFileManager cleanup loops) và file I/O errors
- `log_warn()`: Sử dụng cho các errors không critical (auto-cleanup failures, exchange close errors, cache cleanup errors)
- `logger.warning()`: Sử dụng trong ExchangeManager cho exchange close errors
- `logger.info()`: Sử dụng cho successful cleanup operations

**Sample log messages:**
- `"Error in cleanup loop: {e}"` (TaskManager)
- `"Error in lock cleanup loop: {e}"` (LogFileManager)
- `"Error writing to log file {log_path}: {e}"` (LogFileManager)
- `"Error cleaning up exchange managers: {e}"` (MarketBatchScanner)
- `"Error closing exchange {exchange_id}: {e}"` (ExchangeManager)

**Metrics:**
- Hiện tại không có metrics được expose (không có metric counters cho cleanup failures)
- Cleanup operations chỉ được track qua log messages

**Common exceptions handled:**
- IOError/OSError: File operations (disk full, permission denied, file locked)
- KeyError: Dictionary access trong concurrent scenarios (mitigated bằng locks)
- AttributeError: Method/attribute không tồn tại (trong MarketBatchScanner cleanup)
- Network errors: Exchange connection errors (trong ExchangeManager.close_exchange())
- UnicodeEncodeError/UnicodeDecodeError: File encoding issues (trong LogFileManager)

## Deployment & Rollout

### Backward Compatibility

**Tính tương thích ngược:**
- ✅ **Additive changes**: Tất cả các thay đổi là additive - không có breaking changes
- ✅ **Existing APIs không thay đổi**: Các method cleanup mới được thêm vào, không modify signatures của methods hiện tại
- ✅ **Default behavior không đổi**: Cleanup operations chỉ chạy khi được gọi explicitly hoặc trong background threads; không ảnh hưởng đến flow hiện tại
- ✅ **Không cần feature flags**: Code mới được tích hợp sẵn và tự động hoạt động
- ✅ **No config migration cần thiết**: Không có thay đổi cấu hình; sử dụng giá trị mặc định hợp lý

**Lưu ý:**
- Background cleanup threads (`TaskManager._cleanup_loop`, `LogFileManager._cleanup_locks_loop`) tự động start khi instance được tạo
- Nếu muốn disable auto-cleanup, có thể tạo instance với `start_cleanup_thread=False` (chỉ áp dụng cho LogFileManager, TaskManager luôn start thread)

### Staged Deployment Path

**Giai đoạn 1: Staging (24-48 giờ)**
1. Deploy code mới lên staging environment
2. Verify các cleanup operations hoạt động đúng:
   - TaskManager cleanup tasks cũ
   - LogFileManager cleanup locks không dùng
   - ExchangeManager cleanup connections
   - MarketBatchScanner cleanup resources
3. Monitor memory usage và verify giảm RSS sau cleanup
4. Check logs để đảm bảo không có errors trong cleanup loops

**Giai đoạn 2: Canary/Smoke Test (1-2 giờ)**
- Deploy đến 5-10% production traffic
- **Traffic percentage**: 5-10%
- **Duration**: Tối thiểu 1 giờ, lý tưởng 2 giờ để capture cleanup cycles (TaskManager: 5 phút, LogFileManager: 10 phút)
- **Verification gates:**
  - ✅ Memory RSS stable hoặc giảm
  - ✅ Không có spike trong error rates
  - ✅ GC pause times trong phạm vi bình thường (<100ms)
  - ✅ Response times không bị ảnh hưởng
  - ✅ Cleanup threads chạy đúng chu kỳ (check logs)

**Giai đoạn 3: Full Rollout (Gradual - 4-6 giờ)**
- Tăng traffic lên 50% → 100% trong 2-3 steps
- **Timing**: Mỗi step cách nhau 1-2 giờ để monitor ổn định
- **Verification tại mỗi step:**
  - Memory metrics stable
  - No new error patterns
  - Cleanup frequency matches expected (logs)

### Monitoring & Alerting

**Key Metrics cần theo dõi:**

1. **Memory Metrics:**
   - `memory_rss_mb`: RSS memory usage (MB) - **Alert nếu >80% limit hoặc tăng đột ngột >20% trong 15 phút**
   - `memory_heap_mb`: Heap memory (nếu available) - **Alert nếu >1GB và tiếp tục tăng**
   - `gc_collections`: Số lần GC runs - **Monitor để detect excessive GC (>10/giờ có thể là dấu hiệu memory pressure)**

2. **GC Performance:**
   - `gc_pause_time_ms`: GC pause duration (nếu measurable) - **Alert nếu >200ms thường xuyên**
   - `gc_collected_objects`: Số objects được collect - **Log để verify cleanup hiệu quả**

3. **Cleanup Operations:**
   - Cleanup frequency: TaskManager (mỗi 5 phút), LogFileManager (mỗi 10 phút) - **Alert nếu skip >2 cycles liên tiếp**
   - Cleanup success rate: Check logs cho errors trong cleanup loops - **Alert nếu error rate >10%**
   - Tasks/locks/exchanges cleaned: Số lượng items được cleanup - **Monitor trends để verify effectiveness**

4. **Error Rates:**
   - `cleanup_errors_total`: Tổng số errors trong cleanup operations - **Alert nếu >5 errors/giờ**
   - Application error rate - **Alert nếu tăng >20% so với baseline**

**Dashboard Panels (suggested):**
- Panel 1: Memory Usage Over Time (RSS, Heap)
- Panel 2: GC Statistics (collections, pause times, objects collected)
- Panel 3: Cleanup Operations (frequency, items cleaned, success rate)
- Panel 4: Error Rates (cleanup errors, application errors)

**Alert Thresholds (examples):**
```
- Memory RSS > 2GB (hoặc >80% container limit) trong 10 phút → WARNING
- Memory RSS > 2.5GB (hoặc >90% container limit) → CRITICAL
- GC pause time >200ms trong 5 lần liên tiếp → WARNING
- Cleanup loop skip >2 cycles → WARNING
- Cleanup error rate >10% trong 1 giờ → WARNING
- Application error rate tăng >30% so với baseline → CRITICAL
```

**Log Queries (ví dụ):**
```bash
# Check cleanup loop errors
grep "Error in cleanup loop" /path/to/logs | tail -20

# Verify cleanup frequency
grep "Cleaned up.*tasks" /path/to/logs | tail -10
grep "Cleaned up.*unused locks" /path/to/logs | tail -10

# Monitor memory cleanup effectiveness
grep "Garbage collected" /path/to/logs | tail -20
```

### Migration Notes

**Steps để deploy:**
1. Backup current code và configs (standard deployment procedure)
2. Deploy code mới (không cần downtime - additive changes)
3. Monitor logs để verify cleanup threads start đúng cách
4. Verify memory usage trends sau 1-2 cleanup cycles

**Fallback Plan:**
- **Rollback command**: Revert về commit trước đó (không có config changes cần rollback)
- **Rollback impact**: Cleanup operations sẽ dừng, nhưng không ảnh hưởng existing functionality (code cũ vẫn hoạt động bình thường)
- **Rollback time**: Standard deployment rollback time (~5-10 phút)
- **Data safety**: Không có data changes; cleanup chỉ remove in-memory objects

**Config Changes:**
- ✅ **Không có config changes cần thiết**: Sử dụng default values (TaskManager: 1 giờ, LogFileManager: 2 giờ, cleanup intervals hard-coded)
- ✅ **Không có data migration**: Tất cả operations là in-memory
- ⚠️ **Custom timing**: Nếu cần customize cleanup thresholds, xem phần "Best Practices" trên

**Verification Commands:**
```bash
# Check cleanup threads đang chạy
ps aux | grep python | grep -E "(task_manager|log_manager)"

# Monitor memory usage (nếu có memory_utils)
python -c "from web.utils.memory_utils import get_memory_usage; print(get_memory_usage())"

# Check logs cho cleanup activity
tail -f /path/to/logs | grep -E "(Cleaned up|cleanup loop|Garbage collected)"
```

## Lưu ý

- Garbage collection được force có điều kiện sau các cleanup operations lớn (xem "Garbage Collection Performance")
- Logging được thêm vào để track cleanup operations và errors
- **Quan trọng:** Force GC có thể gây pause times, chỉ sử dụng khi cần thiết và trong các điều kiện phù hợp

## Testing

Để test memory management:

```python
from web.utils.memory_utils import log_memory_usage, optimize_memory

# Before operation
log_memory_usage("before")

# Run your operation
# ...

# After operation
log_memory_usage("after")
optimize_memory()
log_memory_usage("after optimization")
```
