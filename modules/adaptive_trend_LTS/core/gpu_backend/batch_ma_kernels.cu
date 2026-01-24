/**
 * Phase 4 - True Batch CUDA Processing: Batch Moving Average Kernels
 */

#define F64_NAN __longlong_as_double(0x7ff8000000000000LL)

// ============================================================================
// BATCH EMA KERNEL (Standard SMA Init)
// ============================================================================

extern "C" __global__ void batch_ema_kernel(
    const double* __restrict__ all_prices,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_results,
    int ema_length,
    int num_symbols
) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    const double* prices = all_prices + start;
    double* ema = all_results + start;
    
    int start_idx = 0;
    while (start_idx < n && isnan(prices[start_idx])) start_idx++;
    if (start_idx >= n || start_idx + ema_length > n) {
        for (int i = 0; i < n; i++) ema[i] = F64_NAN;
        return;
    }
    for (int i = 0; i < start_idx + ema_length - 1; i++) ema[i] = F64_NAN;
    
    double sum = 0.0;
    for (int i = 0; i < ema_length; i++) sum += prices[start_idx + i];
    ema[start_idx + ema_length - 1] = sum / (double)ema_length;
    
    double alpha = 2.0 / ((double)ema_length + 1.0);
    double one_minus_alpha = 1.0 - alpha;
    for (int i = start_idx + ema_length; i < n; i++) {
        ema[i] = alpha * prices[i] + one_minus_alpha * ema[i - 1];
    }
}

// ============================================================================
// BATCH EMA SIMPLE KERNEL (Value Init)
// ============================================================================

/**
 * EMA with simple initialization (starts directly from first valid price).
 * Used for Pass 2 of DEMA to avoid double SMA warmup.
 */
extern "C" __global__ void batch_ema_simple_kernel(
    const double* __restrict__ all_prices,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_results,
    int ema_length,
    int num_symbols
) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    const double* prices = all_prices + start;
    double* ema = all_results + start;
    
    int start_idx = 0;
    while (start_idx < n && isnan(prices[start_idx])) start_idx++;
    if (start_idx >= n) {
        for (int i = 0; i < n; i++) ema[i] = F64_NAN;
        return;
    }
    
    for (int i = 0; i < start_idx; i++) ema[i] = F64_NAN;
    
    ema[start_idx] = prices[start_idx];
    double alpha = 2.0 / ((double)ema_length + 1.0);
    double one_minus_alpha = 1.0 - alpha;
    for (int i = start_idx + 1; i < n; i++) {
        ema[i] = alpha * prices[i] + one_minus_alpha * ema[i - 1];
    }
}

// ============================================================================
// BATCH WMA KERNEL
// ============================================================================

extern "C" __global__ void batch_wma_kernel(
    const double* __restrict__ all_prices,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_results,
    int wma_length,
    int num_symbols
) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    const double* prices = all_prices + start;
    double* wma = all_results + start;
    
    double denominator = (double)(wma_length * (wma_length + 1)) / 2.0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        // Find if we have enough bars before i (including i)
        bool enough = true;
        if (i < wma_length - 1) enough = false;
        else {
            // Check for NaNs in the window (for safety and matching)
            for(int j=0; j<wma_length; j++) {
                if (isnan(prices[i-j])) { enough = false; break; }
            }
        }
        
        if (!enough) {
            wma[i] = F64_NAN;
        } else {
            double weighted_sum = 0.0;
            for (int j = 0; j < wma_length; j++) {
                weighted_sum += prices[i - j] * (double)(wma_length - j);
            }
            wma[i] = weighted_sum / denominator;
        }
    }
}

// ============================================================================
// BATCH KAMA KERNEL
// ============================================================================

extern "C" __global__ void batch_kama_noise_kernel(
    const double* __restrict__ all_prices,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_noise,
    int kama_length,
    int num_symbols
) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    const double* prices = all_prices + start;
    double* noise = all_noise + start;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i < kama_length) noise[i] = F64_NAN;
        else {
            double sum = 0.0;
            bool valid = true;
            for (int j = 0; j < kama_length; j++) {
                double diff = prices[i - j] - prices[i - j - 1];
                if (isnan(diff)) { valid = false; break; }
                sum += fabs(diff);
            }
            noise[i] = valid ? sum : F64_NAN;
        }
    }
}

extern "C" __global__ void batch_kama_smooth_kernel(
    const double* __restrict__ all_prices,
    const double* __restrict__ all_noise,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_kama,
    int kama_length,
    int num_symbols
) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    const double* prices = all_prices + start;
    const double* noise = all_noise + start;
    double* kama = all_kama + start;
    
    const double fast = 0.666;
    const double slow = 0.064;
    
    if (n < 1) return;
    kama[0] = prices[0];
    for (int i = 1; i < n; i++) {
        if (i < kama_length || isnan(noise[i])) {
            kama[i] = isnan(kama[i-1]) ? prices[i] : kama[i-1];
            continue;
        }
        double signal = fabs(prices[i] - prices[i - kama_length]);
        double ratio = (noise[i] == 0.0) ? 0.0 : signal / noise[i];
        double sc = pow(ratio * (fast - slow) + slow, 2);
        double prev = isnan(kama[i-1]) ? prices[i] : kama[i-1];
        kama[i] = prev + sc * (prices[i] - prev);
    }
}

// ============================================================================
// BATCH LSMA KERNEL
// ============================================================================

extern "C" __global__ void batch_lsma_kernel(
    const double* __restrict__ all_prices,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ all_results,
    int length,
    int num_symbols
) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    const double* prices = all_prices + start;
    double* lsma = all_results + start;
    
    double sum_x = (double)(length * (length - 1)) / 2.0;
    double sum_x2 = (double)(length * (length - 1) * (2 * length - 1)) / 6.0;
    double den = (double)length * sum_x2 - sum_x * sum_x;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i < length - 1) lsma[i] = F64_NAN;
        else {
            double sum_y = 0.0, sum_xy = 0.0;
            bool valid = true;
            for (int j = 0; j < length; j++) {
                double val = prices[i - (length - 1) + j];
                if (isnan(val)) { valid = false; break; }
                sum_y += val; sum_xy += (double)j * val;
            }
            if (!valid) lsma[i] = F64_NAN;
            else {
                double m = ((double)length * sum_xy - sum_x * sum_y) / den;
                double c = (sum_y - m * sum_x) / (double)length;
                lsma[i] = m * (double)(length - 1) + c;
            }
        }
    }
}

// ============================================================================
// BATCH LINEAR COMBINE KERNEL
// ============================================================================

extern "C" __global__ void batch_linear_combine_kernel(
    const double* __restrict__ input_a,
    const double* __restrict__ input_b,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    double* __restrict__ output,
    double mult_a,
    double mult_b,
    int num_symbols
) {
    int symbol_idx = blockIdx.x;
    if (symbol_idx >= num_symbols) return;
    int start = offsets[symbol_idx];
    int n = lengths[symbol_idx];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double va = input_a[start + i], vb = input_b[start + i];
        if (isnan(va) || isnan(vb)) output[start + i] = F64_NAN;
        else output[start + i] = mult_a * va - mult_b * vb;
    }
}
