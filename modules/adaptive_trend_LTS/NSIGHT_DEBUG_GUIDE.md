# Hướng dẫn Debug CUDA với Nsight Copilot trong VS Code

## Bước 1: Chuẩn bị môi trường

### 1.1 Kiểm tra Nsight Copilot đã cài đặt
- Mở VS Code Extensions (Ctrl+Shift+X)
- Tìm "NVIDIA Nsight" hoặc "Nsight Copilot"
- Đảm bảo extension đã được enable

### 1.2 Kiểm tra CUDA Toolkit
```powershell
nvcc --version
```
Phải có CUDA 12.8 hoặc tương thích.

---

## Bước 2: Tạo Launch Configuration (✅ Đã hoàn thành)

File `.vscode/launch.json` đã được tạo với 2 configurations:
1. **"CUDA Debug: Python Test"** - Launch Python script với CUDA debugging
2. **"CUDA Attach: Python Process"** - Attach vào Python process đang chạy

---

## Bước 3: Set Breakpoint trong CUDA Kernel

### 3.1 Mở file CUDA kernel
```
modules/adaptive_trend_LTS/core/gpu_backend/batch_signal_kernels.cu
```

### 3.2 Đặt breakpoint tại dòng quan trọng
- **Line 65**: Trước khi tính crossover/crossunder
- **Line 68**: Sau khi update current_sig
- **Line 72**: Khi gán signals[i]

**Cách đặt breakpoint**:
- Click vào lề trái (margin) bên cạnh số dòng
- Hoặc đặt con trỏ tại dòng và nhấn F9
- Breakpoint sẽ hiện dấu chấm đỏ

### 3.3 Điều kiện breakpoint (Conditional Breakpoint)
Để chỉ dừng tại bar 31 của symbol 0:
- Right-click vào breakpoint đỏ
- Chọn "Edit Breakpoint..."
- Nhập điều kiện: `symbol_idx == 0 && i == 31`

---

## Bước 4: Chạy Debug Session

### Phương pháp 1: Launch Mode (Đề xuất cho lần đầu)

1. **Mở file test**:
   ```
   modules/adaptive_trend_LTS/benchmarks/simple_cuda_test.py
   ```

2. **Bắt đầu debug**:
   - Nhấn F5 hoặc
   - Vào menu "Run" → "Start Debugging" hoặc
   - Click nút "Run and Debug" ở sidebar trái

3. **Chọn configuration**:
   - Chọn "CUDA Debug: Python Test"

4. **Chờ breakpoint kích hoạt**:
   - Chương trình sẽ chạy Python
   - Khi gọi CUDA kernel, nó sẽ dừng tại breakpoint
   - VS Code sẽ highlight dòng code hiện tại

### Phương pháp 2: Attach Mode (Nâng cao)

1. **Chạy Python script trong terminal riêng**:
   ```powershell
   $env:CUDA_LAUNCH_BLOCKING = "1"
   python modules/adaptive_trend_LTS/benchmarks/simple_cuda_test.py
   ```

2. **Attach debugger**:
   - Nhấn F5
   - Chọn "CUDA Attach: Python Process"
   - Chọn process `python.exe` từ danh sách

---

## Bước 5: Step Through Code

Khi debugger dừng tại breakpoint, bạn có thể:

### 5.1 Xem giá trị biến
- **Variables panel** (bên trái): Hiển thị tất cả biến local
- **Watch panel**: Thêm biến muốn theo dõi
- **Hover**: Di chuột lên biến để xem giá trị

**Ví dụ biến quan trọng**:
- `symbol_idx`: Index của symbol (phải = 0 cho TEST)
- `i`: Bar index (28-35 là vùng freeze)
- `p_curr`, `p_prev`: Giá price hiện tại và trước đó
- `m_curr`, `m_prev`: Giá MA hiện tại và trước đó
- `crossover`, `crossunder`: Boolean flags
- `current_sig`: Signal hiện tại (-1.0, 0.0, hoặc 1.0)

### 5.2 Điều khiển execution
- **F10 (Step Over)**: Chạy dòng hiện tại, không vào hàm con
- **F11 (Step Into)**: Vào bên trong hàm được gọi
- **Shift+F11 (Step Out)**: Thoát khỏi hàm hiện tại
- **F5 (Continue)**: Chạy tiếp đến breakpoint tiếp theo
- **Shift+F5 (Stop)**: Dừng debug session

### 5.3 Debug Console
- Mở Debug Console (Ctrl+Shift+Y)
- Gõ lệnh GDB trực tiếp:
  ```gdb
  print current_sig
  print crossover
  print p_curr - m_curr
  ```

---

## Bước 6: Phân tích vấn đề Freeze

### Kịch bản debug cho bar 31:

1. **Set conditional breakpoint**:
   ```
   symbol_idx == 0 && i == 31
   ```

2. **Khi dừng tại bar 31, kiểm tra**:
   - `p_curr` vs `m_curr`: Giá có cross MA không?
   - `p_prev` vs `m_prev`: Giá trước đó ở đâu?
   - `crossover`: Có phải = true không?
   - `crossunder`: Có phải = true không?
   - `current_sig` trước và sau if-else

3. **Nếu crossover/crossunder = false**:
   - Kiểm tra logic: `(p_prev <= m_prev) && (p_curr > m_curr)`
   - In ra: `p_prev - m_prev` và `p_curr - m_curr`
   - Xem có phải do floating point precision?

4. **Nếu current_sig không update**:
   - Kiểm tra có vào if (crossover) hay if (crossunder) không
   - Có thể logic sai hoặc NaN issue

---

## Bước 7: Lưu kết quả Debug

### 7.1 Chụp màn hình
- Variables panel với giá trị tại bar 31
- Code highlight tại dòng freeze

### 7.2 Ghi log
Tạo file `debug_findings.txt`:
```
Bar 31 Debug Results:
- p_curr = 98.xxxx
- m_curr = 99.xxxx
- p_prev = 97.xxxx
- m_prev = 100.xxxx
- crossover = false (expected: true?)
- crossunder = false
- current_sig = -1.0 (stuck from bar 30)
```

---

## Troubleshooting

### Vấn đề 1: Breakpoint không kích hoạt
**Nguyên nhân**: CUDA kernel được compile runtime, VS Code không nhận ra source.

**Giải pháp**:
1. Đảm bảo file `.cu` đã được save
2. Rebuild Rust extensions: `build_cuda.ps1`
3. Restart VS Code
4. Thử attach mode thay vì launch mode

### Vấn đề 2: "No symbol table" error
**Nguyên nhân**: Thiếu debug symbols.

**Giải pháp**:
- Cudarc 0.19 không hỗ trợ compile options
- Cần upgrade lên cudarc 0.20+ hoặc dùng Nsight Compute

### Vấn đề 3: Debugger quá chậm
**Nguyên nhân**: CUDA_LAUNCH_BLOCKING = 1 làm chậm.

**Giải pháp**:
- Chỉ dùng cho debug, không dùng cho production
- Giảm số symbols test (1 thay vì 10)
- Giảm số bars (50 thay vì 500)

---

## Kết luận

Sau khi debug với Nsight Copilot, bạn sẽ biết chính xác:
1. Giá trị của mọi biến tại bar 31
2. Tại sao crossover/crossunder không trigger
3. Logic nào trong kernel bị sai

Từ đó có thể fix triệt để bug freeze!
