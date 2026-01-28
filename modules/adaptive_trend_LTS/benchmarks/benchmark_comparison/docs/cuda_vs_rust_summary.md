# Tóm tắt: Vì sao CUDA chậm hơn Rust thuần trong benchmark ATC

## 1. Số liệu benchmark

- **Rust**: 70.95s  
- **CUDA**: 147.16s  
- **Rust nhanh hơn CUDA ~2.07×** (CUDA vs Rust = 0.48×)

Workload: **518 symbols × 1500 bars**, so sánh Original / Enhanced / Rust / CUDA.

---

## 2. Nguyên nhân chính

### a) Chi phí truyền dữ liệu (PCIe)

- **Rust**: Toàn bộ tính trên CPU, không copy qua GPU.
- **CUDA**: Phải copy **input** (prices, …) CPU → GPU, rồi copy **output** (MAs, signals, equities, average signal) GPU → CPU.

Với 518 symbol × 1500 bars × float64, cộng rất nhiều mảng trung gian (54 MAs, 6 layer-1 signals, 6 layer-2 equities, …), tổng lượng data qua PCIe rất lớn. Ở quy mô “vừa” này, **thời gian transfer có thể lớn hơn cả thời gian tính trên GPU**.

### b) Overhead khởi chạy kernel

- Pipeline CUDA gồm nhiều kernel: batch EMA, WMA, DEMA, LSMA, KAMA, HMA, rồi signal, equity, …
- Mỗi lần `launch` kernel có độ trễ vài µs. Nhiều kernel × nhiều lần gọi → tổng overhead đáng kể.
- **Rust**: Không có kernel launch, chỉ gọi hàm Rust liên tục.

### c) GPU bị underutilized

Trong `batch_processing.rs`, ví dụ batch EMA:

```
grid_dim: (num_symbols as u32, 1, 1)   // 518 blocks
block_dim: (1, 1, 1)                    // 1 thread per block
```

→ Chỉ **518 thread** cho cả GPU. GPU thiết kế cho hàng nghìn–triệu thread; 518 thread thì occupancy rất thấp, GPU gần như “đói” việc.

### d) Đặc điểm workload không “hợp” GPU

- Nhiều bước có **phụ thuộc tuần tự** (equity bar `i` phụ thuộc bar `i-1`, KAMA có vòng lặp lồng nhau, …). Khó song song hóa triệt để theo bars.
- GPU tận dụng tốt khi: **ít batch, mỗi batch rất lớn**, ít phụ thuộc tuần tự.
- Ở đây: **518 bài toán độc lập**, mỗi bài ~1500 bars, vừa phải. Rust xử lý từng symbol trên CPU, data nằm trong cache, SIMD tốt → rất phù hợp.

### e) Rust được tối ưu tốt cho quy mô này

- **Không** copy PCIe, **không** kernel launch.
- SIMD, cache-friendly, có thể dùng thêm parallelism (rayon, …) trên CPU.
- 518 × 1500 bars và các mảng trung gian vẫn nằm gọn trong bộ nhớ (và cache) CPU → Rust tận dụng rất tốt.

---

## 3. Bảng tóm tắt

| Yếu tố | Rust | CUDA |
|--------|------|------|
| PCIe transfer | Không | Có, khối lượng lớn |
| Kernel launch | Không | Nhiều kernel |
| Số thread / mức độ song song | Vừa phải, phù hợp CPU | 518 → vài nghìn thread, GPU underutilized |
| Cache / locality | Tốt (data trên CPU) | Chủ yếu trên GPU, phải qua PCIe |
| Dạng workload | Nhiều symbol, vừa phải mỗi symbol | Cùng workload, nhưng tốn thêm transfer + launch |

---

## 4. Kết luận

Với **518 symbols × 1500 bars**, chi phí **transfer + launch** và **underutilization** GPU làm CUDA chậm hơn Rust. Rust thuần CPU tránh được những chi phí đó và tận dụng tốt SIMD/cache.

GPU thường chỉ vượt CPU khi:

- Số symbol hoặc số bars **rất lớn** (hàng chục nghìn symbol hoặc bars rất dài), hoặc  
- Batch **rất lớn**, ít kernel, và kernel được viết để dùng **rất nhiều thread** (high occupancy).

Trong setup benchmark hiện tại, **Rust thuần là lựa chọn nhanh hơn CUDA** là điều dễ xảy ra.

---

*Nguồn: Phân tích từ `benchmark_results.txt` và code trong `modules/adaptive_trend_LTS`.*
