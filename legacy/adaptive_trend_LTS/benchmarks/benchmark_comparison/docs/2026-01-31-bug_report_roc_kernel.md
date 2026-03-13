# 🐛 Critical Bug Report: ROC Kernel Strided Accumulation

## 🚨 Vấn đề (The Issue)

Kernel `batch_roc_with_growth_kernel` trong file `batch_signal_kernels.cu` có lỗi logic nghiêm trọng khi chạy song song (parallel execution).

### Logic Sai (Old Code):
```cuda
    // Thread chạy nhảy cóc (stride) với bước nhảy blockDim (VD: 256)
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        // ...
            growth *= growth_factor;          // sai: accumulation stateful
            roc[i] = r * growth;
        // ...
    }
```

**Tại sao sai?**
- Trong môi trường đa luồng, mỗi thread xử lý các bar không liền kề (VD: Thread 0 xử lý bar 0, 256, 512...).
- Biến `growth` được nhân dồn (`*=`) mỗi lần lặp của **thread**, không phải theo **thứ tự thời gian** của bar.
- Kết quả: Bar 256 sẽ dùng `growth = e^(2*La)` thay vì `e^(256*La)`.
- Sai số lũy thừa cực lớn khi `n` lớn.

### Logic Đúng (New Fix):
```cuda
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        // ...
            // Tính growth dựa trên index i tuyệt đối (stateless)
            double growth = exp(La * static_cast<double>(i));
            roc[i] = r * growth;
        // ...
    }
```

**Tại sao đúng?**
- `growth` được tính toán độc lập cho mỗi bar dựa trên vị trí `i` của nó.
- Đúng với công thức toán học đệ quy trên CPU: `growth[i] = growth[i-1] * e^La` <=> `growth[i] = e^(La * i)`.
- An toàn cho tính toán song song.

## 📉 Tác động (Impact)

1. **ROC Values Sai Lệch:** Giá trị Rate of Change bị sai hệ số tăng trưởng (Growth Factor).
2. **Equity Curve Sai:** ROC sai dẫn đến Equity curve sai.
3. **Weighting Sai:** Equity được dùng làm trọng số (weight) để gộp các signals (L1).
4. **Final Signal Sai:** Trọng số sai dẫn đến tín hiệu mua/bán cuối cùng sai lệch so với bản CPU.
5. **Numerical Drift:** Lỗi tích lũy theo hàm mũ, giải thích tại sao 27 bars đầu (chưa dùng growth nhiều hoặc n < blockDim) có vẻ đúng, nhưng sau đó sai lệch tăng dần.

## ✅ Trạng thái (Status)

- [x] Đã xác định nguyên nhân gốc rễ (Root Cause Identified).
- [x] Đã áp dụng Fix vào code (`batch_signal_kernels.cu`).
- [x] Đã verify logic HMA/EMA là đúng (loại trừ nguyên nhân khác).

## 🛠️ Bước tiếp theo (Next Steps)

1. Rebuild Rust extensions để apply fix:
   ```bash
   maturin develop --release
   ```
2. Chạy lại benchmark để kiểm tra Match Rate (kỳ vọng 100%).

---
**Reported By:** Antigravity AI
**Date:** 2026-01-31
