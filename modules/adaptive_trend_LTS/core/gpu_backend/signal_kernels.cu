/**
 * Phase 4 - Task 2.1.2C: CUDA kernels for Signal Classification.
 *
 * Implements:
 * - Weighted Average Signal: Parallel reduction for sum(signal * equity) / sum(equity)
 * - Trend Classification: Threshold-based classification (long/short/neutral)
 *
 * All kernels use double precision (f64) for numerical accuracy.
 */

#define F64_NAN __longlong_as_double(0x7ff8000000000000LL)

// ============================================================================
// Weighted Average Signal Kernel
// ============================================================================

/**
 * Calculate weighted average signal using parallel reduction.
 * 
 * Algorithm:
 * 1. Discretize signals: S > long_threshold → 1.0, S < short_threshold → -1.0, else → 0.0
 * 2. Calculate: sum(signal_discrete * equity) / sum(equity)
 * 
 * This kernel processes one bar (time index) at a time across multiple MA components.
 * Launch: block=(256,1,1), grid=((n_bars+255)/256, 1, 1)
 * 
 * Inputs:
 *   - signals: 2D array [n_mas x n_bars] of Layer 1 signals
 *   - equities: 2D array [n_mas x n_bars] of Layer 2 equity curves
 *   - n_mas: Number of MA components
 *   - n_bars: Number of time bars
 *   - long_threshold: Threshold for LONG classification
 *   - short_threshold: Threshold for SHORT classification
 * 
 * Output:
 *   - avg_signal: 1D array [n_bars] of weighted average signals
 */
extern "C" __global__ void weighted_average_signal_kernel(
    const double* __restrict__ signals,      // [n_mas x n_bars] flattened
    const double* __restrict__ equities,     // [n_mas x n_bars] flattened
    double* __restrict__ avg_signal,         // [n_bars]
    int n_mas,
    int n_bars,
    int cutout,
    double long_threshold,
    double short_threshold
) {
    for (int bar_idx = blockIdx.x * blockDim.x + threadIdx.x; bar_idx < n_bars; bar_idx += blockDim.x * gridDim.x) {
        if (bar_idx < cutout) {
            avg_signal[bar_idx] = 0.0;
            continue;
        }
        double numerator = 0.0;
        double denominator = 0.0;
        
        // Accumulate across all MA components for this bar
        #pragma unroll 4
        for (int ma_idx = 0; ma_idx < n_mas; ma_idx++) {
            int idx = ma_idx * n_bars + bar_idx;  // Row-major indexing
            
            double signal = signals[idx];
            double equity = equities[idx];
            
            // Discretize signal
            double signal_discrete = 0.0;
            if (!isnan(signal)) {
                if (signal > long_threshold) signal_discrete = 1.0;
                else if (signal < short_threshold) signal_discrete = -1.0;
            }
            
            // Skip if equity is NaN or zero
            if (!isnan(equity) && equity != 0.0) {
                numerator += signal_discrete * equity;
                denominator += equity;
            }
        }
        
        // Calculate weighted average
        if (denominator != 0.0 && isfinite(denominator)) {
            avg_signal[bar_idx] = numerator / denominator;
        } else {
            avg_signal[bar_idx] = 0.0;
        }
    }
}

extern "C" __global__ void classify_trend_kernel(
    const double* __restrict__ signals,
    int* __restrict__ trends,  // Output: 1 (LONG), -1 (SHORT), 0 (NEUTRAL)
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

extern "C" __global__ void weighted_average_and_classify_kernel(
    const double* __restrict__ signals,      // [n_mas x n_bars]
    const double* __restrict__ equities,     // [n_mas x n_bars]
    double* __restrict__ avg_signal,         // [n_bars]
    int* __restrict__ trends,                // [n_bars]
    int n_mas,
    int n_bars,
    int cutout,
    double long_threshold,
    double short_threshold
) {
    for (int bar_idx = blockIdx.x * blockDim.x + threadIdx.x; bar_idx < n_bars; bar_idx += blockDim.x * gridDim.x) {
        if (bar_idx < cutout) {
            avg_signal[bar_idx] = 0.0;
            trends[bar_idx] = 0;
            continue;
        }
        double numerator = 0.0;
        double denominator = 0.0;
        
        #pragma unroll 4
        for (int ma_idx = 0; ma_idx < n_mas; ma_idx++) {
            int idx = ma_idx * n_bars + bar_idx;
            
            double signal = signals[idx];
            double equity = equities[idx];
            
            double signal_discrete = 0.0;
            if (!isnan(signal)) {
                if (signal > long_threshold) signal_discrete = 1.0;
                else if (signal < short_threshold) signal_discrete = -1.0;
            }
            
            if (!isnan(equity) && equity != 0.0) {
                numerator += signal_discrete * equity;
                denominator += equity;
            }
        }
        
        double avg = 0.0;
        if (denominator != 0.0 && isfinite(denominator)) {
            avg = numerator / denominator;
        }
        
        avg_signal[bar_idx] = avg;
        
        int trend = 0;
        if (avg > long_threshold) trend = 1;
        else if (avg < short_threshold) trend = -1;
        
        trends[bar_idx] = trend;
    }
}
