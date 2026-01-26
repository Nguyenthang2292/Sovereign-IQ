# Finalize CUDA Compilation Fixes

## Goal
Apply remaining fixes to `ma_kernels.cu` (legacy non-batch kernels) and verify all CUDA files match the corrected source code from the review.

## Tasks

- [x] Task 1: Fix `ma_kernels.cu` → Add `<cmath>` and `<cuda_runtime.h>`, replace `F64_NAN` macro with `constexpr double`, add `__ldg` hints with `#if __CUDA_ARCH__ >= 600` guards
- [x] Task 2: Verify `batch_ma_kernels.cu` matches review → Check headers, `F64_NAN` constexpr, `__ldg` guards are present
- [x] Task 3: Verify `batch_signal_kernels.cu` matches review → Check `<cstdio>`, `<cmath>`, `<cuda_runtime.h>`, `F64_NAN` uses `ULL`, all `__ldg` hints present
- [x] Task 4: Verify `equity_kernel.cu` matches review → Check headers, `F64_NAN` constexpr, `__ldg` guards
- [x] Task 5: Verify `signal_kernels.cu` matches review → Check headers, `F64_NAN` constexpr, shared memory size documented
- [x] Task 6: Test compilation → Run Rust build with `maturin develop --release` to verify NVRTC compilation succeeds

## Done When

- [x] All 5 CUDA files have correct headers (`<cmath>`, `<cuda_runtime.h>`, `<cstdio>` where needed)
- [x] All files use `constexpr double F64_NAN = __longlong_as_double(0x7ff8000000000000ULL)` (not macro, uses `ULL`)
- [x] All `__ldg` calls are guarded with `#if __CUDA_ARCH__ >= 600` for backward compatibility
- [x] Rust build succeeds without CUDA compilation errors
- [x] Benchmark runs successfully with both batch and legacy CUDA kernels

## Notes

- `ma_kernels.cu` is NOT a duplicate - it contains non-batch kernels (`ema_kernel`, `kama_noise_kernel`) used by legacy code in `ma_cuda.rs`
- `batch_ma_kernels.cu` contains batch versions (`batch_ema_kernel`, `batch_kama_noise_kernel`) used by `batch_processing.rs`
- Both files are needed and should be fixed separately
- The review provides complete corrected source code - verify each file matches exactly
- Shared memory size for `weighted_average_and_classify_kernel` must be `2 * blockDim.x * sizeof(double)` (numerator + denominator)
