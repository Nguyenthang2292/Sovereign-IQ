/**
 * Phase 4 - Task 2.1.2B: Custom CUDA kernels for Moving Average calculations.
 *
 * Implements:
 * - EMA (Exponential Moving Average): Sequential recurrence
 * - KAMA (Kaufman Adaptive Moving Average): Dual-pass (noise + smoothing)
 * - WMA (Weighted Moving Average): Convolution with shared memory
 *
 * All kernels use double precision (f64) for numerical accuracy.
 */

#define F64_NAN __longlong_as_double(0x7ff8000000000000LL)

// ============================================================================
// EMA Kernel
// ============================================================================

/**
 * Calculate EMA with SMA initialization.
 * 
 * Algorithm:
 * 1. Initialize with SMA of first 'length' values
 * 2. Apply recursive EMA: ema[i] = alpha * price[i] + (1-alpha) * ema[i-1]
 * 
 * Launch: block=(1,1,1), grid=(1,1) - single thread sequential
 */
extern "C" __global__ void ema_kernel(
    const double* __restrict__ prices,
    double* __restrict__ ema,
    int length,
    int n
) {
    // Grid-stride loop (though this specific logic is sequential per curve)
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx == 0; idx += blockDim.x * gridDim.x) {
        int start_idx = 0;
        while (start_idx < n && isnan(prices[start_idx])) {
            start_idx++;
        }
        
        if (start_idx >= n || start_idx + length > n) {
            for (int i = 0; i < n; i++) ema[i] = F64_NAN;
            return;
        }
        
        for (int i = 0; i < start_idx + length - 1; i++) ema[i] = F64_NAN;
        
        double sum = 0.0;
        for (int i = 0; i < length; i++) sum += prices[start_idx + i];
        ema[start_idx + length - 1] = sum / (double)length;
        
        double alpha = 2.0 / ((double)length + 1.0);
        double one_minus_alpha = 1.0 - alpha;
        
        for (int i = start_idx + length; i < n; i++) {
            ema[i] = alpha * prices[i] + one_minus_alpha * ema[i - 1];
        }
    }
}

extern "C" __global__ void kama_noise_kernel(
    const double* __restrict__ prices,
    double* __restrict__ noise,
    int length,
    int n
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (i < length) {
            noise[i] = F64_NAN;
        } else {
            double sum = 0.0;
            int start_idx = (i - length + 1) > 1 ? (i - length + 1) : 1;
            #pragma unroll 4
            for (int j = start_idx; j <= i; j++) {
                sum += fabs(prices[j] - prices[j - 1]);
            }
            noise[i] = sum;
        }
    }
}

extern "C" __global__ void kama_smooth_kernel(
    const double* __restrict__ prices,
    const double* __restrict__ noise,
    double* __restrict__ kama,
    int length,
    int n
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx == 0; idx += blockDim.x * gridDim.x) {
        const double fast = 0.666;
        const double slow = 0.064;
        
        if (n < 1) return;
        kama[0] = prices[0];
        
        for (int i = 1; i < n; i++) {
            if (i < length) {
                kama[i] = kama[i - 1];
                continue;
            }
            
            double signal = fabs(prices[i] - prices[i - length]);
            double ratio = (noise[i] == 0.0) ? 0.0 : signal / noise[i];
            double smooth = (ratio * (fast - slow) + slow);
            smooth *= smooth;
            
            double prev_kama = isnan(kama[i - 1]) ? prices[i] : kama[i - 1];
            kama[i] = prev_kama + smooth * (prices[i] - prev_kama);
        }
    }
}

#define WMA_TILE_SIZE 256
extern "C" __global__ void wma_kernel(
    const double* __restrict__ prices,
    double* __restrict__ wma,
    int length,
    int n
) {
    // Shared memory to store prices for this block and the overlap needed for the window
    // Max window length we support in shared memory tile can be limited, but let's try simple tiling.
    // However, since window size 'length' can be large, we'll use a safer approach.
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (i < length - 1) {
            wma[i] = F64_NAN;
        } else {
            double weighted_sum = 0.0;
            #pragma unroll 4
            for (int j = 0; j < length; j++) {
                double weight = (double)(length - j);
                weighted_sum += prices[i - j] * weight;
            }
            double denominator = (double)(length * (length + 1)) / 2.0;
            wma[i] = weighted_sum / denominator;
        }
    }
}

extern "C" __global__ void sma_kernel(
    const double* __restrict__ prices,
    double* __restrict__ sma,
    int length,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (idx < length - 1) {
        sma[idx] = F64_NAN;
    } else {
        double sum = 0.0;
        // Basic optimization: unroll small loops if length is known at compile time, 
        // but here length is dynamic. However, the compiler can still optimize the loop body.
        for (int j = 0; j < length; j++) {
            sum += prices[idx - j];
        }
        sma[idx] = sum / (double)length;
    }
}

extern "C" __global__ void hma_diff_kernel(
    const double* __restrict__ wma_half,
    const double* __restrict__ wma_full,
    double* __restrict__ diff,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // HMA intermediate step: 2 * WMA(n/2) - WMA(n)
        double val_half = wma_half[idx];
        double val_full = wma_full[idx];
        if (isnan(val_half) || isnan(val_full)) {
            diff[idx] = F64_NAN;
        } else {
            diff[idx] = 2.0 * val_half - val_full;
        }
    }
}
