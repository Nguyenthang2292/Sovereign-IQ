# Phase 4: Advanced GPU Optimizations - Implementation Task Plan

> **Scope**: Custom CUDA Kernels (2.1) and GPU Streams for Overlapping (2.2)  
> **Expected Performance Gain**: 3-10x combined speedup  
> **Timeline**: 2-3 weeks

---

## 2.1 Custom CUDA Kernels

### Overview
Replace CuPy high-level operations with custom CUDA kernels optimized for ATC-specific calculations.

**Expected Gain**: 2-5x faster than CuPy for complex operations

---

### Task Breakdown

#### 📋 Task 2.1.1: Setup CUDA Development Environment

- [ ] **Install CUDA Toolkit**
  - Download and install CUDA Toolkit 12.x from NVIDIA
  - Verify installation: `nvcc --version`
  - Set up environment variables (`CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH`)

- [ ] **Install Development Tools**
  - Install `pycuda` package: `pip install pycuda`
  - Install `cupy` (already should be installed): verify with `python -c "import cupy; print(cupy.__version__)"`
  - Install CUDA samples for reference

- [ ] **Verify GPU Compatibility**
  - Check GPU compute capability: `nvidia-smi`
  - Ensure compute capability >= 6.0 for optimal performance
  - Test basic CUDA kernel compilation

**Deliverable**: Working CUDA development environment

---

#### 📋 Task 2.1.2: Design Custom CUDA Kernels

Analyze current CuPy operations and design custom kernels for:

##### A. Equity Calculation Kernel

- [ ] **Review Current Implementation**
  - Locate equity calculation in [adaptive_trend_enhance_v2/rust_extensions/src/equity.rs](file:///c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/adaptive_trend_enhance_v2/rust_extensions/src/equity.rs) or Python version
  - Document current algorithm and data flow
  - Identify parallelization opportunities

- [ ] **Design CUDA Kernel Specification**
  ```cuda
  __global__ void equity_kernel(
      const float* r_values,      // Returns array
      const float* sig_prev,      // Previous signals
      float* equity,              // Output equity curve
      float starting_equity,
      float decay,
      int cutout,
      int n                       // Array length
  )
  ```
  - Define thread block size (e.g., 256 threads)
  - Define grid size calculation
  - Plan memory access patterns for coalescing

- [ ] **Write Kernel Implementation**
  - Implement parallel equity calculation logic
  - Handle edge cases (cutout period, array bounds)
  - Add bounds checking
  
- [ ] **Write Python Wrapper**
  - Create Python function to invoke CUDA kernel
  - Handle memory transfer (CPU → GPU → CPU)
  - Add error handling

**Deliverable**: `equity_cuda_kernel.cu` and `equity_cuda.py`

---

##### B. Moving Average Kernels

- [ ] **EMA (Exponential Moving Average) Kernel**
  - Review current implementation in [core/moving_averages.py](file:///c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/adaptive_trend_enhance_v2/core/moving_averages.py)
  - Design parallel scan algorithm or use sequential with shared memory
  - Implement CUDA kernel
  - Write Python wrapper

- [ ] **KAMA (Kaufman Adaptive Moving Average) Kernel**
  - Review KAMA calculation logic
  - Design kernel for volatility calculation (ER computation)
  - Design kernel for adaptive smoothing
  - Implement dual-pass kernel or fused kernel
  - Write Python wrapper

- [ ] **WMA/HMA (Weighted/Hull Moving Average) Kernels**
  - Design convolution-based approach for weighted sums
  - Implement using shared memory for coefficients
  - Write Python wrapper

**Deliverable**: `ma_cuda_kernels.cu` and `ma_cuda.py`

---

##### C. Signal Classification Kernel

- [ ] **Average Signal Calculation Kernel**
  - Review weighted average logic from current implementation
  - Design kernel for parallel weighted sum
  - Implement with reduction for final averaging
  - Write Python wrapper

- [ ] **Trend Classification Kernel**
  - Implement threshold-based classification (long/short/neutral)
  - Fuse with signal calculation if beneficial
  - Write Python wrapper

**Deliverable**: `signal_cuda_kernels.cu` and `signal_cuda.py`

---

#### 📋 Task 2.1.3: Optimize Kernel Performance

- [ ] **Memory Access Optimization**
  - Analyze memory access patterns with NVIDIA Nsight Compute
  - Ensure coalesced global memory access
  - Use shared memory for frequently accessed data
  - Minimize bank conflicts in shared memory

- [ ] **Occupancy Optimization**
  - Profile kernel occupancy
  - Adjust block size for maximum occupancy
  - Balance register usage vs parallelism
  - Use `__launch_bounds__` if needed

- [ ] **Instruction-Level Optimization**
  - Use intrinsics where appropriate (`__fmul_rn`, `__fadd_rn`)
  - Minimize divergent branches
  - Use warp-level primitives for reductions

- [ ] **Loop Unrolling**
  - Apply `#pragma unroll` for small fixed-size loops
  - Benchmark different unroll factors

**Deliverable**: Optimized CUDA kernels with profiling reports

---

#### 📋 Task 2.1.4: Integration with Python Module

- [ ] **Create GPU Backend Module**
  - Create [adaptive_trend_enhance_v2/core/gpu_backend/](file:///c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/adaptive_trend_enhance_v2/core/gpu_backend/) directory
  - Organize CUDA kernels: `equity_kernel.cu`, `ma_kernels.cu`, `signal_kernels.cu`
  - Create Python wrappers: `cuda_equity.py`, `cuda_ma.py`, `cuda_signal.py`

- [ ] **Modify compute_atc_signals Function**
  - Add `use_cuda_kernels` parameter to config
  - Add conditional logic to use CUDA kernels when enabled
  - Ensure fallback to CuPy/NumPy if CUDA kernels unavailable

- [ ] **Build System Integration**
  - Create `setup.py` or extend existing build system to compile CUDA kernels
  - Add CUDA compilation flags
  - Handle different CUDA architectures (sm_60, sm_70, sm_80, sm_86)

**Deliverable**: Integrated CUDA kernel backend in Python module

---

#### 📋 Task 2.1.5: Testing and Validation

- [ ] **Unit Tests for Each Kernel**
  - Test equity kernel with known inputs/outputs
  - Test MA kernels against NumPy reference implementations
  - Test signal kernels for correctness
  - Add tests to [tests/adaptive_trend_enhance_v2/](file:///c:/Users/Admin/Desktop/i-ching/crypto-probability/tests/) directory

- [ ] **Numerical Accuracy Testing**
  - Compare CUDA kernel outputs with CuPy/NumPy (tolerance: 1e-6)
  - Test with edge cases (NaN, Inf, very small/large values)
  - Validate across different data sizes

- [ ] **Performance Benchmarking**
  - Benchmark individual kernels vs CuPy equivalents
  - Benchmark full ATC pipeline with CUDA kernels
  - Test on different data sizes (100, 1000, 10000 bars)
  - Compare with baseline in [benchmarks/benchmark_comparison.py](file:///c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/adaptive_trend_enhance_v2/benchmarks/benchmark_comparison.py)

**Deliverable**: Passing tests and benchmark comparison report

---

## 2.2 GPU Streams for Overlapping

### Overview
Use CUDA streams to overlap CPU-GPU data transfers with GPU computation, improving throughput for batch processing.

**Expected Gain**: 1.5-2x faster for large batch processing

---

### Task Breakdown

#### 📋 Task 2.2.1: Understand CUDA Streams Fundamentals

- [ ] **Research CUDA Streams**
  - Study CUDA stream API: `cuStreamCreate`, `cuStreamSynchronize`
  - Learn CuPy stream interface: `cp.cuda.Stream()`
  - Understand asynchronous data transfers: `cp.asarray(..., stream=...)`
  - Review stream synchronization patterns

- [ ] **Analyze Current Data Flow**
  - Map out current batch processing flow in [compute_atc_signals.py](file:///c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/adaptive_trend_enhance_v2/core/compute_atc_signals.py)
  - Identify transfer bottlenecks (H2D, D2H)
  - Identify computation phases that can overlap

**Deliverable**: Design document for stream-based pipeline

---

#### 📋 Task 2.2.2: Implement Multi-Stream Architecture

- [ ] **Create Stream Pool**
  ```python
  class CUDAStreamPool:
      def __init__(self, num_streams=4):
          self.streams = [cp.cuda.Stream() for _ in range(num_streams)]
          self.current_idx = 0
      
      def get_stream(self):
          stream = self.streams[self.current_idx]
          self.current_idx = (self.current_idx + 1) % len(self.streams)
          return stream
  ```

- [ ] **Implement Pipelined Batch Processing**
  - Design 3-stage pipeline: Transfer In → Compute → Transfer Out
  - Implement double/triple buffering for data
  - Use different streams for different stages

- [ ] **Implement Symbol Batch Processing**
  ```python
  def process_symbols_with_streams(symbols_data, config, num_streams=4):
      stream_pool = CUDAStreamPool(num_streams)
      results = {}
      
      for symbol, prices in symbols_data.items():
          stream = stream_pool.get_stream()
          with stream:
              # Transfer to GPU asynchronously
              gpu_prices = cp.asarray(prices.values, stream=stream)
              # Compute on GPU
              result = compute_atc_gpu(gpu_prices, config)
              # Transfer back asynchronously
              results[symbol] = cp.asnumpy(result, stream=stream)
      
      # Synchronize all streams
      for stream in stream_pool.streams:
          stream.synchronize()
      
      return results
  ```

**Deliverable**: `cuda_stream_manager.py` with multi-stream processing

---

#### 📋 Task 2.2.3: Optimize Stream Scheduling

- [ ] **Profile Stream Utilization**
  - Use NVIDIA Nsight Systems to visualize stream timeline
  - Identify idle periods and transfer bottlenecks
  - Measure overlap percentage

- [ ] **Optimize Number of Streams**
  - Test with 2, 4, 8, 16 concurrent streams
  - Find optimal number based on GPU concurrency capabilities
  - Consider memory constraints (each stream needs buffer space)

- [ ] **Implement Stream Priorities**
  - Use high-priority streams for time-critical computations
  - Use low-priority streams for prefetching

- [ ] **Pinned Memory Optimization**
  - Allocate pinned (page-locked) host memory for faster transfers
  ```python
  pinned_pool = cp.cuda.PinnedMemoryPool()
  cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)
  ```

**Deliverable**: Optimized stream configuration with profiling report

---

#### 📋 Task 2.2.4: Integration with Batch Processing

- [ ] **Modify Batch Processing Functions**
  - Update [compute_atc_signals.py](file:///c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/adaptive_trend_enhance_v2/core/compute_atc_signals.py) to support stream-based processing
  - Add `use_streams` parameter to configuration
  - Add `num_streams` parameter (default: 4)

- [ ] **Update Benchmark Script**
  - Modify [benchmark_comparison.py](file:///c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/adaptive_trend_enhance_v2/benchmarks/benchmark_comparison.py)
  - Add comparison: CuPy baseline vs CUDA kernels vs CUDA kernels + streams
  - Measure throughput (symbols/second)

- [ ] **Error Handling and Resource Management**
  - Ensure streams are properly destroyed on exit
  - Handle out-of-memory errors gracefully
  - Add timeout for stream synchronization

**Deliverable**: Integrated stream-based batch processing

---

#### 📋 Task 2.2.5: Testing and Validation

- [ ] **Correctness Testing**
  - Verify stream-based results match single-stream results
  - Test with various batch sizes (10, 100, 1000 symbols)
  - Test edge cases (empty batches, single symbol)

- [ ] **Performance Testing**
  - Benchmark streams vs non-streams on different batch sizes
  - Measure speedup: 1x (baseline) → 1.5-2x (with streams)
  - Test on different GPU models (if available)

- [ ] **Stress Testing**
  - Test with maximum concurrent streams
  - Test with very large batches (10,000+ symbols)
  - Monitor GPU memory usage

**Deliverable**: Passing tests and performance comparison report

---

## Verification Plan

### Automated Tests

#### Test 1: CUDA Kernel Correctness
```bash
# Location: tests/adaptive_trend_enhance_v2/test_cuda_kernels.py
pytest tests/adaptive_trend_enhance_v2/test_cuda_kernels.py -v
```
**Coverage**:
- Equity kernel numerical accuracy (tolerance: 1e-6)
- MA kernel outputs vs NumPy reference
- Signal kernel classification correctness

#### Test 2: Stream Processing Correctness
```bash
# Location: tests/adaptive_trend_enhance_v2/test_cuda_streams.py
pytest tests/adaptive_trend_enhance_v2/test_cuda_streams.py -v
```
**Coverage**:
- Multi-stream results match single-stream results
- Stream synchronization works correctly
- No data corruption in concurrent processing

#### Test 3: Integration Tests
```bash
# Run full benchmark comparison
python modules/adaptive_trend_enhance_v2/benchmarks/benchmark_comparison.py --symbols 100 --bars 1000
```
**Expected Results**:
- Custom CUDA kernels: 2-5x faster than CuPy baseline
- CUDA kernels + streams: 3-10x faster than CuPy baseline
- 100% signal matching with baseline (differences < 1e-6)

---

### Performance Benchmarks

#### Benchmark 1: Individual Kernel Performance
```bash
# Create new benchmark script
python modules/adaptive_trend_enhance_v2/benchmarks/benchmark_cuda_kernels.py
```
**Metrics**:
- Equity kernel: speedup vs CuPy
- MA kernels: speedup vs CuPy
- Signal kernel: speedup vs CuPy

#### Benchmark 2: End-to-End Performance
```bash
# Existing benchmark with new configurations
python modules/adaptive_trend_enhance_v2/benchmarks/benchmark_comparison.py \
  --symbols 1000 --bars 1000 --use-cuda-kernels --use-streams
```
**Metrics**:
- Total execution time
- Throughput (symbols/second)
- GPU utilization (via `nvidia-smi`)

#### Benchmark 3: Scaling Test
```bash
# Test with increasing batch sizes
for size in 10 100 500 1000 5000; do
  python benchmarks/benchmark_comparison.py --symbols $size --bars 1000
done
```
**Metrics**:
- Scaling efficiency
- Stream overlap percentage (via Nsight Systems)

---

### Manual Verification

#### Verification 1: Visual Profiling
1. Install NVIDIA Nsight Systems
2. Run profiling:
   ```bash
   nsys profile -o phase4_profile python benchmarks/benchmark_comparison.py --symbols 100
   ```
3. Open `phase4_profile.qdrep` in Nsight Systems GUI
4. Verify:
   - Multiple streams are active concurrently
   - Data transfer overlaps with computation
   - No large idle gaps in GPU timeline

#### Verification 2: Memory Usage Check
1. Run benchmark with monitoring:
   ```bash
   watch -n 0.1 nvidia-smi
   ```
2. Verify:
   - GPU memory stays within limits
   - No memory leaks over time
   - Pinned memory is being used (check with `nvidia-smi`)

---

## Success Criteria

### Phase 4 Completion Checklist

- [ ] **Custom CUDA Kernels (2.1)**
  - [ ] All kernels implemented and optimized
  - [ ] Numerical accuracy validated (< 1e-6 difference)
  - [ ] 2-5x speedup achieved vs CuPy baseline
  - [ ] Integrated into main module with fallback

- [ ] **GPU Streams (2.2)**
  - [ ] Multi-stream processing implemented
  - [ ] Stream overlap visualized in profiler
  - [ ] 1.5-2x speedup achieved for batch processing
  - [ ] Memory-efficient and stable

- [ ] **Combined Performance**
  - [ ] Total speedup: 3-10x vs CuPy baseline
  - [ ] All tests passing
  - [ ] Benchmark report generated
  - [ ] Code documented and reviewed

---

## Dependencies and Prerequisites

### Required Software
- CUDA Toolkit 12.x
- Python 3.9+
- CuPy >= 12.0
- PyCUDA >= 2023.1
- NVIDIA GPU with compute capability >= 6.0

### Required Knowledge
- CUDA C/C++ programming
- GPU memory management
- Performance profiling with Nsight tools
- Python-CUDA interop (CuPy, PyCUDA)

### Existing Code to Review
- [adaptive_trend_enhance_v2/core/compute_atc_signals.py](file:///c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/adaptive_trend_enhance_v2/core/compute_atc_signals.py) - Main computation entry point
- [adaptive_trend_enhance_v2/core/moving_averages.py](file:///c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/adaptive_trend_enhance_v2/core/moving_averages.py) - MA implementations
- [adaptive_trend_enhance_v2/rust_extensions/](file:///c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/adaptive_trend_enhance_v2/rust_extensions/) - Rust implementation for reference
- [benchmarks/benchmark_comparison.py](file:///c:/Users/Admin/Desktop/i-ching/crypto-probability/modules/adaptive_trend_enhance_v2/benchmarks/benchmark_comparison.py) - Benchmark framework

---

## Timeline Estimate

| Task | Estimated Time | Priority |
|------|---------------|----------|
| 2.1.1 Setup Environment | 0.5 days | High |
| 2.1.2 Design Kernels | 2 days | High |
| 2.1.3 Optimize Kernels | 3 days | High |
| 2.1.4 Integration | 2 days | High |
| 2.1.5 Testing | 2 days | High |
| **2.1 Total** | **9.5 days** | **High** |
| 2.2.1 Study Streams | 1 day | Medium |
| 2.2.2 Implement Streams | 2 days | Medium |
| 2.2.3 Optimize Streams | 2 days | Medium |
| 2.2.4 Integration | 1 day | Medium |
| 2.2.5 Testing | 1 day | Medium |
| **2.2 Total** | **7 days** | **Medium** |
| **Phase 4 Total** | **16.5 days (~3 weeks)** | |

---

## Notes and Recommendations

### Development Approach
1. **Iterative Development**: Implement one kernel at a time, test, optimize, then move to next
2. **Benchmark Early**: Compare against CuPy baseline after each kernel implementation
3. **Profile Continuously**: Use Nsight Compute and Nsight Systems throughout development
4. **Fallback Safety**: Always maintain fallback to CuPy/NumPy for compatibility

### Common Pitfalls to Avoid
- **Memory Alignment**: Ensure data is properly aligned for coalesced access
- **Kernel Launch Overhead**: Batch operations to reduce launch overhead
- **Stream Synchronization**: Don't over-synchronize; let streams run independently
- **Pinned Memory**: Allocate pinned memory once, reuse for transfers

### Future Enhancements (Post-Phase 4)
- Implement Tensor Core support for RTX GPUs (mixed precision)
- Add multi-GPU support for very large batches
- Implement persistent kernels for lower latency
- Add CUDA graph API for reduced CPU overhead
