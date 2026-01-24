# Phase 4 – Equity CUDA Kernel Design (Task 2.1.2)

## 1. Review of Current Implementation

### 1.1 Locations

- **Rust**: `modules/adaptive_trend_LTS/rust_extensions/src/equity.rs`
  - `calculate_equity_internal`, `calculate_equity_rust`
- **Python**: `modules/adaptive_trend_LTS/core/compute_equity/core.py`
  - `_calculate_equity_core_impl`, `_calculate_equity_core` (Numba JIT)

### 1.2 Algorithm and Data Flow

1. **Inputs**
   - `r_values`: array of returns `r[i]`, length `n`
   - `sig_prev`: array of previous signals `sig[i]` ∈ {−1, 0, +1} (or NaN)
   - `starting_equity`: initial equity
   - `decay_multiplier`: decay factor (1 − De)
   - `cutout`: number of leading bars to skip

2. **Output**
   - `equity[0..n)`:
     - `equity[i] = NaN` for `i < cutout`
     - for `i ≥ cutout`, `equity[i]` = equity at bar `i`

3. **Recurrence (single curve)**
   - For `i = cutout .. n-1`:
     - `a = 0` if `NaN(sig[i])` or `NaN(r[i])` or `sig[i] == 0`
     - else `a = r[i]` if `sig[i] > 0`, else `a = -r[i]`
   - `e_curr = starting_equity` if `prev_e` uninitialized, else  
     `e_curr = (prev_e * decay_multiplier) * (1 + a)`
   - `e_curr = max(e_curr, 0.25)`
   - `equity[i] = e_curr`, `prev_e = e_curr`

4. **Data flow**
   - Single sequence: `r`, `sig` → scalar `prev_e` updated sequentially → `equity`.

### 1.3 Parallelization

- **Within one curve**: the recurrence is strictly sequential (`e[i]` depends on `e[i-1]`). No fine‑grained parallelism over `i` for a single equity curve.
- **Across curves**: multiple symbols (multiple equity curves) can be processed in parallel (e.g. one thread per curve) when batching. The current CUDA kernel implements **one curve per launch** (single-thread loop).

## 2. CUDA Kernel Specification

### 2.1 Signature

```cuda
__global__ void equity_kernel(
    const double* r_values,
    const double* sig_prev,
    double* equity,
    double starting_equity,
    double decay_multiplier,
    int cutout,
    int n
);
```

### 2.2 Launch

- **Block/grid**: `block=(1,1,1)`, `grid=(1,1)` (one thread).
- **Memory**:
  - `r_values`, `sig_prev`: read-only, contiguous.
  - `equity`: write-only, contiguous.
  - No shared memory.

### 2.3 Edge Cases

- `cutout = 0`: no leading NaNs; loop starts at `i = 0`.
- `cutout > 0`: `equity[0..cutout]` set to NaN; loop `i = cutout .. n-1`.
- `n = 0`: kernel no-op.
- NaN in `r` or `sig`: `a = 0` as above.

## 3. Deliverables (Task 2.1.2 – Equity)

- [x] **Review**: algorithm and data flow documented above.
- [x] **Design**: kernel spec and launch configuration as above.
- [x] **Implementation**: `core/gpu_backend/equity_kernel.cu`.
- [x] **Python wrapper**: `core/gpu_backend/equity_cuda.py` (`calculate_equity_cuda`) - PyCUDA fallback.
- [x] **Rust wrapper**: `rust_extensions/src/equity_cuda.rs` - Primary orchestration using `cudarc`.
  - Compiles `.cu` via NVRTC at runtime.
  - Manages H2D/D2H transfers with Rust type safety.
  - Transparently integrated via `atc_rust` Python module.

## 4. References

- `rust_extensions/src/equity_cuda.rs`
- `rust_extensions/src/equity.rs`
- `compute_equity/core.py`
- Phase 4 task plan: `docs/phase4_task.md`
