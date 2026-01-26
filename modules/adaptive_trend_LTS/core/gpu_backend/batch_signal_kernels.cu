/**
 * Phase 4 - True Batch CUDA Processing: Batch Equity & Signal Kernels
 */

#include "gpu_common.h"

// ---------------------------------------------------------------------------
// DEBUG flag - set to 1 for verbose output, 0 for production builds
// Production builds MUST have DEBUG_PERSIST = 0 for performance.
// ---------------------------------------------------------------------------
#ifndef DEBUG_PERSIST
#define DEBUG_PERSIST 0
#endif

// Production build guard: prevent accidental debug builds
#ifdef NDEBUG
#if DEBUG_PERSIST != 0
#error "DEBUG_PERSIST must be 0 for release builds"
#endif
#endif

__device__ __forceinline__ bool safe_le(double a, double b) {
    return (a < b) || (fabs(a - b) < EPSILON);
}
__device__ __forceinline__ bool safe_gt(double a, double b) {
    return (a > b) && (fabs(a - b) >= EPSILON);
}
__device__ __forceinline__ bool safe_ge(double a, double b) {
    return (a > b) || (fabs(a - b) < EPSILON);
}
__device__ __forceinline__ bool safe_lt(double a, double b) {
    return (a < b) && (fabs(a - b) >= EPSILON);
}

// ---------------------------------------------------------------------------
// Crossover / cross-under helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool compute_crossover(double p_prev, double m_prev, double p_curr, double m_curr) {
    // Crossover: price was below/at MA, now above MA
    return safe_le(p_prev, m_prev) && safe_gt(p_curr, m_curr);
}

__device__ __forceinline__ bool compute_crossunder(double p_prev, double m_prev, double p_curr, double m_curr) {
    // Crossunder: price was above/at MA, now below MA
    return safe_ge(p_prev, m_prev) && safe_lt(p_curr, m_curr);
}

// ============================================================================
// BATCH SHIFT KERNEL (Equivalent to shift(1))
// ============================================================================

extern "C" __global__ void batch_shift_kernel(
    const double* __restrict__ input,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ output,
    double fill_val,
    int num_symbols
) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
#if __CUDA_ARCH__ >= 600
    int start = __ldg(&offsets[symbol_idx]);
    int n = __ldg(&lengths[symbol_idx]);
#else
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
#endif
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i == 0) output[start + i] = fill_val;
        else {
#if __CUDA_ARCH__ >= 600
            output[start + i] = __ldg(&input[start + i - 1]);
#else
            output[start + i] = input[start + i - 1];
#endif
        }
    }
}

// ============================================================================
// BATCH SIGNAL PERSISTENCE KERNEL (Stateful)
// ============================================================================

extern "C" __global__ void batch_signal_persistence_kernel(
    const double* __restrict__ all_prices,
    const double* __restrict__ all_ma,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_signals,
    int num_symbols
) {
    const int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;

#if __CUDA_ARCH__ >= 600
    const int start = __ldg(&offsets[symbol_idx]);
    const int n = __ldg(&lengths[symbol_idx]);
#else
    const int start = offsets[symbol_idx];
    const int n = lengths[symbol_idx];
#endif

    const double* __restrict__ prices = all_prices + start;
    const double* __restrict__ ma = all_ma + start;
    double* __restrict__ sig = all_signals + start;

    // Pine Script / Python starts with 0.0
    double current_sig = 0.0;

    for (int i = 0; i < n; ++i) {
        if (i > 0) {
#if __CUDA_ARCH__ >= 600
            const double p_curr = __ldg(&prices[i]);
            const double p_prev = __ldg(&prices[i-1]);
            const double m_curr = __ldg(&ma[i]);
            const double m_prev = __ldg(&ma[i-1]);
#else
            const double p_curr = prices[i];
            const double p_prev = prices[i-1];
            const double m_curr = ma[i];
            const double m_prev = ma[i-1];
#endif

            const bool valid = !isnan(p_curr) && !isnan(p_prev) && 
                               !isnan(m_curr) && !isnan(m_prev);

            if (valid) {
                const bool crossover = compute_crossover(p_prev, m_prev, p_curr, m_curr);
                const bool crossunder = compute_crossunder(p_prev, m_prev, p_curr, m_curr);

#if DEBUG_PERSIST
                if (symbol_idx == 0 && i >= 28 && i <= 35) {
                    printf("[PERSIST] Bar %d: p=%.4f m=%.4f p_prev=%.4f m_prev=%.4f cross_over=%d cross_under=%d sig_before=%.1f\n",
                           i, p_curr, m_curr, p_prev, m_prev, crossover, crossunder, current_sig);
                }
#endif

                if (crossover) current_sig = 1.0;
                else if (crossunder) current_sig = -1.0;

#if DEBUG_PERSIST
                if (symbol_idx == 0 && i >= 28 && i <= 35) {
                    printf(" sig_after=%.1f\n", current_sig);
                }
#endif
            }
#if DEBUG_PERSIST
            else if (symbol_idx == 0 && i >= 28 && i <= 35) {
                printf("[PERSIST] Bar %d: INVALID (NaN detected)\n", i);
            }
#endif
        }
        sig[i] = current_sig;
    }
}

// ============================================================================
// BATCH ROC WITH GROWTH KERNEL
// ============================================================================

// batch_signal_kernels.cu
extern "C" __global__ void batch_roc_with_growth_kernel(
    const double* __restrict__ all_prices,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_roc,
    double La,
    int num_symbols)
{
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;

#if __CUDA_ARCH__ >= 600
    int start = __ldg(&offsets[symbol_idx]);
    int n     = __ldg(&lengths[symbol_idx]);
#else
    int start = offsets[symbol_idx];
    int n     = lengths[symbol_idx];
#endif

    const double* prices = all_prices + start;
    double*       roc    = all_roc    + start;

    double growth = 1.0;                     // exp(La*0)
    const double growth_factor = exp(La);    // constant multiplier

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i == 0) {
            roc[i] = 0.0;
        } else {
            double p      = __ldg(&prices[i]);
            double p_prev = __ldg(&prices[i-1]);
            double r = (p_prev != 0.0 && !isnan(p) && !isnan(p_prev))
                       ? (p - p_prev) / p_prev : 0.0;
            growth *= growth_factor;          // exp(La*i)
            roc[i] = r * growth;
        }
    }
}

// ============================================================================
// BATCH EQUITY KERNEL
// ============================================================================

extern "C" __global__ void batch_equity_kernel(
    const double* __restrict__ all_r_growth,
    const double* __restrict__ all_sig_prev,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_equity,
    double starting_equity,
    double decay_multiplier,
    int cutout,
    int num_symbols
) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
#if __CUDA_ARCH__ >= 600
    int start = __ldg(&offsets[symbol_idx]);
    int n = __ldg(&lengths[symbol_idx]);
#else
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
#endif
    
    const double* r_values = all_r_growth + start;
    const double* sig_prev = all_sig_prev + start;
    double* equity = all_equity + start;
    
    double prev_e = -1.0;
    
    for (int i = 0; i < cutout && i < n; i++) equity[i] = F64_NAN;
    
    for (int i = cutout; i < n; i++) {
#if __CUDA_ARCH__ >= 600
        double r_i = __ldg(&r_values[i]);
        double s_p = __ldg(&sig_prev[i]);
#else
        double r_i = r_values[i];
        double s_p = sig_prev[i];
#endif
        double a = 0.0;
        if (!isnan(s_p) && !isnan(r_i)) {
            if (s_p > 0.0) a = r_i;
            else if (s_p < 0.0) a = -r_i;
        }
        
        double e_curr = (prev_e < 0.0) ? starting_equity : (prev_e * decay_multiplier) * (1.0 + a);
        
        if (e_curr < DEFAULT_EQUITY_FLOOR) e_curr = DEFAULT_EQUITY_FLOOR;
        
        equity[i] = e_curr;
        prev_e = e_curr;
    }
}

// ===================================
// BATCH WEIGHTED AVERAGE L1 KERNEL
// ===================================

extern "C" __global__ void batch_weighted_average_l1_kernel(
    const double* __restrict__ all_signals,
    const double* __restrict__ all_equities,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_avg_signals,
    int total_bars,
    int num_symbols
) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
#if __CUDA_ARCH__ >= 600
    int start = __ldg(&offsets[symbol_idx]);
    int n = __ldg(&lengths[symbol_idx]);
#else
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
#endif
    
    for (int bar = threadIdx.x; bar < n; bar += blockDim.x) {
        int idx = start + bar;
        double weighted_sum = 0.0;
        double weight_sum = 0.0;
        
        for (int i = 0; i < L1_SIGNAL_COUNT; i++) {
#if __CUDA_ARCH__ >= 600
            double s = __ldg(&all_signals[i * total_bars + idx]);
            double e = __ldg(&all_equities[i * total_bars + idx]);
#else
            double s = all_signals[i * total_bars + idx];
            double e = all_equities[i * total_bars + idx];
#endif
            if (!isnan(s) && !isnan(e)) {
                weighted_sum += s * e;
                weight_sum += e;
            }
        }
        
        if (weight_sum > 0.0) {
            all_avg_signals[idx] = weighted_sum / weight_sum;
        } else {
            all_avg_signals[idx] = 0.0;
        }
    }
}

// ===================================
// BATCH FINAL AVERAGE SIGNAL KERNEL
// ===================================

extern "C" __global__ void batch_final_average_signal_kernel(
    const double* __restrict__ all_l1_signals,
    const double* __restrict__ all_l2_equities,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_avg_signals,
    double long_threshold,
    double short_threshold,
    int cutout,
    int total_bars,
    int num_symbols
) {
    const int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;

#if __CUDA_ARCH__ >= 600
    const int start = __ldg(&offsets[symbol_idx]);
    const int n = __ldg(&lengths[symbol_idx]);
#else
    const int start = offsets[symbol_idx];
    const int n = lengths[symbol_idx];
#endif

    for (int bar = threadIdx.x; bar < n; bar += blockDim.x) {
        const int idx = start + bar;

        // Warm-up period -> zero output
        if (bar < cutout) {
            all_avg_signals[idx] = 0.0;
            continue;
        }

        double nom = 0.0;
        double den = 0.0;

        for (int i = 0; i < L2_EQUITY_COUNT; ++i) {
#if __CUDA_ARCH__ >= 600
            const double s = __ldg(&all_l1_signals[i * total_bars + idx]);
            const double e = __ldg(&all_l2_equities[i * total_bars + idx]);
#else
            const double s = all_l1_signals[i * total_bars + idx];
            const double e = all_l2_equities[i * total_bars + idx];
#endif

            if (!isnan(s) && !isnan(e)) {
                double c = 0.0;
                if (s > long_threshold) c = 1.0;
                else if (s < short_threshold) c = -1.0;

#if DEBUG_PERSIST
                if (symbol_idx == 0 && bar == 31) {
                    // Simpler debug print without arrays to avoid potential issues
                    printf("[FINAL] Bar 31, i=%d: s=%.6f, e=%.6f, c=%.1f\n", i, s, e, c);
                }
#endif
                nom += c * e;
                den += e;
            }
        }

        all_avg_signals[idx] = (den > 0.0) ? (nom / den) : 0.0;
    }
}

// ===================================
// BATCH RATE OF CHANGE KERNEL
// ===================================

extern "C" __global__ void batch_roc_kernel(
    const double* __restrict__ all_prices,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_roc,
    int num_symbols
) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
#if __CUDA_ARCH__ >= 600
    int start = __ldg(&offsets[symbol_idx]);
    int n = __ldg(&lengths[symbol_idx]);
#else
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
#endif
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i == 0) all_roc[start + i] = 0.0;
        else {
#if __CUDA_ARCH__ >= 600
            double p = __ldg(&all_prices[start + i]);
            double prev = __ldg(&all_prices[start + i - 1]);
#else
            double p = all_prices[start + i];
            double prev = all_prices[start + i - 1];
#endif
            all_roc[start + i] = (prev != 0.0 && !isnan(p) && !isnan(prev)) ? (p - prev)/prev : 0.0;
        }
    }
}
