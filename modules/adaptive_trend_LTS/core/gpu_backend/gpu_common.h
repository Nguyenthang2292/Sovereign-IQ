#pragma once

// NVRTC-compatible header (no cuda_runtime.h dependency)
// For runtime compilation, we use built-in CUDA device types only
#ifndef __NVRTC__
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
// NVRTC-compatible NaN - use CUDART_NAN which is a built-in macro
#ifndef CUDART_NAN
#define CUDART_NAN __longlong_as_double(0xfff8000000000000ULL)
#endif
#define F64_NAN CUDART_NAN

static constexpr double EPSILON = 1e-10;
static constexpr double DEFAULT_EQUITY_FLOOR = 0.25;

// Layer counts for batch processing
static constexpr int L1_SIGNAL_COUNT = 9;  // Number of L1 signal variations
static constexpr int L2_EQUITY_COUNT = 6;  // Number of L2 equity variations (MA types)

#ifndef __NVRTC__
// ---------------------------------------------------------------------------
// Error-check macro (for host-side use after kernel launches)
//
// Usage example:
//   weighted_average_and_classify_kernel<<<grid, block, shmem>>>(...);
//   CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
//   CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
//
// Note: Use this in test harnesses and production code, NOT inside kernels.
// ---------------------------------------------------------------------------
#define CUDA_CHECK_AND_EXIT(call)                                          \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d – %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)
#endif

// ---------------------------------------------------------------------------
// Helper: fast validity test (device-side)
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool is_valid(double x) {
    return !isnan(x);
}
