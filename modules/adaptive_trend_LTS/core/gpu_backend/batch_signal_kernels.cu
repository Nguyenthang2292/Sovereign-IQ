/**
 * Phase 4 - True Batch CUDA Processing: Batch Equity & Signal Kernels
 */

#define F64_NAN __longlong_as_double(0x7ff8000000000000LL)

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
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i == 0) output[start + i] = fill_val;
        else output[start + i] = input[start + i - 1];
    }
}

// ============================================================================
// BATCH SIGNAL PERSISTENCE KERNEL (Stateful)
// ============================================================================

// ---------------------------------------------------------------------------
// Helper: Safe floating-point comparisons with epsilon tolerance
// ---------------------------------------------------------------------------
#define EPSILON 1e-10

__device__ __forceinline__
bool safe_le(double a, double b) {
    // a <= b with tolerance
    return (a < b) || (fabs(a - b) < EPSILON);
}

__device__ __forceinline__
bool safe_gt(double a, double b) {
    // a > b with tolerance
    return (a > b) && (fabs(a - b) >= EPSILON);
}

__device__ __forceinline__
bool safe_ge(double a, double b) {
    // a >= b with tolerance
    return (a > b) || (fabs(a - b) < EPSILON);
}

__device__ __forceinline__
bool safe_lt(double a, double b) {
    // a < b with tolerance
    return (a < b) && (fabs(a - b) >= EPSILON);
}

__device__ __forceinline__
bool compute_crossover(double p_prev, double m_prev, double p_curr, double m_curr) {
    // Crossover: price was below/at MA, now above MA
    // (p_prev <= m_prev) && (p_curr > m_curr)
    return safe_le(p_prev, m_prev) && safe_gt(p_curr, m_curr);
}

__device__ __forceinline__
bool compute_crossunder(double p_prev, double m_prev, double p_curr, double m_curr) {
    // Crossunder: price was above/at MA, now below MA
    // (p_prev >= m_prev) && (p_curr < m_curr)
    return safe_ge(p_prev, m_prev) && safe_lt(p_curr, m_curr);
}


extern "C" __global__ void batch_signal_persistence_kernel(
    const double* __restrict__ all_prices,
    const double* __restrict__ all_ma,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_signals,
    int num_symbols
) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    
    const double* prices = all_prices + start;
    const double* ma = all_ma + start;
    double* signals = all_signals + start;
    
    // Pine Script / Python starts with 0.0
    // var int sig = 0
    double current_sig = 0.0;
    
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            double p_curr = prices[i];
            double p_prev = prices[i-1];
            double m_curr = ma[i];
            double m_prev = ma[i-1];
            
            // Handle NaNs: crossover/under only valid if both current and prev are valid
            bool valid = !isnan(p_curr) && !isnan(p_prev) && !isnan(m_curr) && !isnan(m_prev);
            
            if (valid) {
                // Use safe comparison helpers to avoid floating-point precision issues
                bool crossover = compute_crossover(p_prev, m_prev, p_curr, m_curr);
                bool crossunder = compute_crossunder(p_prev, m_prev, p_curr, m_curr);
                
                // DEBUG: Print for first symbol, bars 28-35
                if (symbol_idx == 0 && i >= 28 && i <= 35) {
                    printf("[PERSIST] Bar %d: p=%.4f m=%.4f p_prev=%.4f m_prev=%.4f cross_over=%d cross_under=%d sig_before=%.1f",
                           i, p_curr, m_curr, p_prev, m_prev, crossover, crossunder, current_sig);
                }
                
                if (crossover) current_sig = 1.0;
                else if (crossunder) current_sig = -1.0;
                
                // DEBUG: Print after update
                if (symbol_idx == 0 && i >= 28 && i <= 35) {
                    printf(" sig_after=%.1f\n", current_sig);
                }
            } else {
                // DEBUG: Print NaN case
                if (symbol_idx == 0 && i >= 28 && i <= 35) {
                    printf("[PERSIST] Bar %d: INVALID (NaN detected)\n", i);
                }`r`n        }
        signals[i] = current_sig;
    }
}

// ============================================================================
// BATCH ROC WITH GROWTH KERNEL
// ============================================================================

/**
 * Calculates ROC scaled by exp growth.
 * Out[i] = ((p[i] - p[i-1]) / p[i-1]) * exp(La * bars)
 * bars = i == 0 ? 1 : i; (Matching exp_growth.py and Pine Script)
 */
extern "C" __global__ void batch_roc_with_growth_kernel(
    const double* __restrict__ all_prices,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_roc,
    double La,
    int num_symbols
) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    
    const double* prices = all_prices + start;
    double* roc = all_roc + start;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i == 0) {
            roc[i] = 0.0;
        } else {
            double p = prices[i];
            double p_prev = prices[i - 1];
            double r = (p_prev != 0.0 && !isnan(p) && !isnan(p_prev)) ? (p - p_prev) / p_prev : 0.0;
            
            // Python: bars = i == 0 ? 1 : i
            // Although i=0 is handled above, let's match the math strictly
            double growth = exp(La * (double)i);
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
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    
    const double* r_values = all_r_growth + start;
    const double* sig_prev = all_sig_prev + start;
    double* equity = all_equity + start;
    
    double prev_e = -1.0;
    
    for (int i = 0; i < cutout && i < n; i++) equity[i] = F64_NAN;
    
    for (int i = cutout; i < n; i++) {
        double r_i = r_values[i];
        double s_p = sig_prev[i];
        
        double a = 0.0;
        // Python logic treats NaN as Neutral (0)
        if (!isnan(s_p) && !isnan(r_i)) {
            if (s_p > 0.0) a = r_i;
            else if (s_p < 0.0) a = -r_i;
        }
        
        double e_curr = (prev_e < 0.0) ? starting_equity : (prev_e * decay_multiplier) * (1.0 + a);
        
        if (e_curr < 0.25) e_curr = 0.25;
        
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
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    
    for (int bar = threadIdx.x; bar < n; bar += blockDim.x) {
        int idx = start + bar;
        double weighted_sum = 0.0;
        double weight_sum = 0.0;
        
        for (int i = 0; i < 9; i++) {
            double s = all_signals[i * total_bars + idx];
            double e = all_equities[i * total_bars + idx];
            if (!isnan(s) && !isnan(e)) {
                weighted_sum += s * e;
                weight_sum += e;
            }
        }
        
        if (weight_sum > 0.0) {
            double res = weighted_sum / weight_sum;
            // NO ROUNDING - Python reference doesn't round Layer 1 signals
            all_avg_signals[idx] = res;
        } else {
            // Python weighted_signal returns NaN for den=0, but compute_average handles it
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
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    
    for (int bar = threadIdx.x; bar < n; bar += blockDim.x) {
        int idx = start + bar;
        
        // Apply cutout: zero out warmup period
        if (bar < cutout) {
            all_avg_signals[idx] = 0.0;
            continue;
        }
        
        double nom = 0.0, den = 0.0;
        
        for (int i = 0; i < 6; i++) {
            double s = all_l1_signals[i * total_bars + idx];
            double e = all_l2_equities[i * total_bars + idx];
            
            // DEBUG: Print for symbol 0, bar 31, all MA types
            if (symbol_idx == 0 && bar == 31) {
                const char* ma_names[] = {"EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"};
                printf("[FINAL] Bar 31, %s (i=%d): s=%.6f, e=%.6f", ma_names[i], i, s, e);
            }
            
            if (!isnan(s) && !isnan(e)) {
                double c = 0.0;
                if (s > long_threshold) c = 1.0;
                else if (s < short_threshold) c = -1.0;
                
                // DEBUG: Print classification
                if (symbol_idx == 0 && bar == 31) {
                    printf(", c=%.1f, nom+=%.6f, den+=%.6f\n", c, c*e, e);
                }
                
                nom += c * e;
                den += e;
            } else {
                // DEBUG: Print NaN case
                if (symbol_idx == 0 && bar == 31) {
                    printf(" - SKIPPED (NaN)\n");
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
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i == 0) all_roc[start + i] = 0.0;
        else {
            double p = all_prices[start + i];
            double prev = all_prices[start + i - 1];
            all_roc[start + i] = (prev != 0.0 && !isnan(p) && !isnan(prev)) ? (p - prev)/prev : 0.0;
        }
    }
}

