/**
 * Weighted-average signal + trend classification.
 * One block processes a *single* time-bar (bar_idx).  All threads in the block
 * cooperate to reduce the per-MA contributions.
 */

#include "gpu_common.h"

// ---------------------------------------------------------------------------
// Helper: discretize a raw signal value
// ---------------------------------------------------------------------------
__device__ __forceinline__ double discretize_signal(double s, double long_thr, double short_thr) {
    if (isnan(s)) return 0.0;
    if (s > long_thr)  return 1.0;
    if (s < short_thr) return -1.0;
    return 0.0;
}

// ---------------------------------------------------------------------------
// Main kernel – weighted average + trend classification
//
// Launch configuration (REQUIRED):
//   dim3 grid(n_bars);                    // One block per bar
//   dim3 block(128);                      // 128 threads per block (must be power-of-2)
//   size_t shmem = 2 * block.x * sizeof(double);  // Shared memory for reduction
//
// Note: blockDim.x should be a power-of-2 (e.g., 32, 64, 128, 256) for optimal
//       reduction performance. The kernel handles non-power-of-2 sizes but may be slower.
//
// Example:
//   weighted_average_and_classify_kernel<<<grid, block, shmem>>>(
//       d_signals, d_equities, d_avg, d_trend,
//       n_mas, n_bars, cutout, long_thr, short_thr);
// ---------------------------------------------------------------------------
extern "C" __global__
void weighted_average_and_classify_kernel(
    const double* __restrict__ signals,   // [n_mas * n_bars] row-major
    const double* __restrict__ equities,  // same layout
    double*       __restrict__ avg_signal, // [n_bars]
    int*          __restrict__ trends,     // [n_bars]
    int n_mas,
    int n_bars,
    int cutout,
    double long_threshold,
    double short_threshold
) {
    // 1. Thread / block indexing
    const int bar_idx = blockIdx.x;
    if (bar_idx >= n_bars) return;

    // 2. Warm-up (cutout) handling
    if (bar_idx < cutout) {
        avg_signal[bar_idx] = 0.0;
        trends[bar_idx]     = 0;
        return;
    }

    // 3. Each thread processes a *chunk* of the MA components.
    double num_part = 0.0;
    double den_part = 0.0;

    // Note: No #pragma unroll here - n_mas can be large, unrolling would explode code size
    for (int ma = threadIdx.x; ma < n_mas; ma += blockDim.x) {
        const int idx = ma * n_bars + bar_idx;   // row-major lookup

#if __CUDA_ARCH__ >= 600
        const double sig = __ldg(&signals[idx]);
        const double eq  = __ldg(&equities[idx]);
#else
        const double sig = signals[idx];
        const double eq  = equities[idx];
#endif
        const double sig_disc = discretize_signal(sig, long_threshold, short_threshold);

        if (!isnan(eq) && eq != 0.0) {
            num_part += sig_disc * eq;
            den_part += eq;
        }
    }

    // 4. Intra-block reduction (numerator & denominator)
    extern __shared__ double shmem[];
    double* sh_num = shmem;                     // size = blockDim.x
    double* sh_den = shmem + blockDim.x;        // size = blockDim.x

    sh_num[threadIdx.x] = num_part;
    sh_den[threadIdx.x] = den_part;
    __syncthreads();

    // Reduce within the block
    if (blockDim.x > 32) {
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sh_num[threadIdx.x] += sh_num[threadIdx.x + stride];
                sh_den[threadIdx.x] += sh_den[threadIdx.x + stride];
            }
            __syncthreads();
        }
    } else {
        // warp-shuffle reduction
        double val_num = sh_num[threadIdx.x];
        double val_den = sh_den[threadIdx.x];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val_num += __shfl_down_sync(0xffffffff, val_num, offset);
            val_den += __shfl_down_sync(0xffffffff, val_den, offset);
        }
        sh_num[0] = val_num;
        sh_den[0] = val_den;
    }

    // 5. Thread 0 writes the final result for this bar
    if (threadIdx.x == 0) {
        double avg = 0.0;
        const double den = sh_den[0];
        if (den != 0.0 && isfinite(den)) {
            avg = sh_num[0] / den;
        }
        avg_signal[bar_idx] = avg;

        int trend = 0;
        if (avg > long_threshold)      trend = 1;
        else if (avg < short_threshold) trend = -1;
        trends[bar_idx] = trend;
    }
}

// ---------------------------------------------------------------------------
// Standalone weighted average kernel (Legacy / Modular use)
// ---------------------------------------------------------------------------
extern "C" __global__ void weighted_average_signal_kernel(
    const double* __restrict__ signals,
    const double* __restrict__ equities,
    double* __restrict__ avg_signal,
    int n_mas,
    int n_bars,
    int cutout,
    double long_threshold,
    double short_threshold
) {
    // Reusing the optimized logic would require shared memory setup. 
    // For now, keeping the simple strided loop for backward compatibility or simple launches.
    
    for (int bar_idx = blockIdx.x * blockDim.x + threadIdx.x; bar_idx < n_bars; bar_idx += blockDim.x * gridDim.x) {
        if (bar_idx < cutout) {
            avg_signal[bar_idx] = 0.0;
            continue;
        }
        double numerator = 0.0;
        double denominator = 0.0;
        
        #pragma unroll 4
        for (int ma_idx = 0; ma_idx < n_mas; ma_idx++) {
            int idx = ma_idx * n_bars + bar_idx;
            
#if __CUDA_ARCH__ >= 600
            const double signal = __ldg(&signals[idx]);
            const double equity = __ldg(&equities[idx]);
#else
            const double signal = signals[idx];
            const double equity = equities[idx];
#endif
            
            double signal_discrete = discretize_signal(signal, long_threshold, short_threshold);
            
            if (!isnan(equity) && equity != 0.0) {
                numerator += signal_discrete * equity;
                denominator += equity;
            }
        }
        
        if (denominator != 0.0 && isfinite(denominator)) {
            avg_signal[bar_idx] = numerator / denominator;
        } else {
            avg_signal[bar_idx] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone classification kernel
// ---------------------------------------------------------------------------
extern "C" __global__ void classify_trend_kernel(
    const double* __restrict__ signals,
    int* __restrict__ trends,
    int n,
    double long_threshold,
    double short_threshold
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        double signal = signals[i];
        int trend = 0;
        if (!isnan(signal)) {
            if (signal > long_threshold) trend = 1;
            else if (signal < short_threshold) trend = -1;
        }
        trends[i] = trend;
    }
}
