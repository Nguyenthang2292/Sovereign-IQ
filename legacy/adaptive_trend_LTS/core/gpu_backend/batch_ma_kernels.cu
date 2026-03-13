/**
 * Batch Moving-Average kernels (EMA, EMA-simple, WMA, KAMA, LSMA, Linear-combine)
 * All kernels accept a flat [num_symbols x n] layout and an `offsets` array.
 * Double-precision (`double`) is used throughout for numerical stability.
 */

#include "gpu_common.h"

// ---------------------------------------------------------------------------
// EMA - SMA init (sequential recurrence, one thread per symbol)
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(1)
void batch_ema_kernel(
    const double* __restrict__ all_prices,
    const int*    __restrict__ offsets,
    const int*    __restrict__ lengths,
    double*       __restrict__ all_results,
    int ema_length,
    int num_symbols)
{
    const int sym = blockIdx.x;
    if (sym >= num_symbols) return;

#if __CUDA_ARCH__ >= 600
    const int start = __ldg(&offsets[sym]);
    const int n     = __ldg(&lengths[sym]);
#else
    const int start = offsets[sym];
    const int n     = lengths[sym];
#endif
    const double* __restrict__ p = all_prices + start;
    double*       __restrict__ e = all_results + start;

    // Warm-up: skip leading NaNs
    int i0 = 0;
    while (i0 < n && isnan(p[i0])) ++i0;

    if (i0 >= n || i0 + ema_length > n) {
        for (int i = 0; i < n; ++i) e[i] = F64_NAN;
        return;
    }

    // NaNs before the first valid EMA value
    for (int i = 0; i < i0 + ema_length - 1; ++i) e[i] = F64_NAN;

    // SMA init
    double sum = 0.0;
    for (int i = 0; i < ema_length; ++i) {
#if __CUDA_ARCH__ >= 600
        sum += __ldg(&p[i0 + i]);
#else
        sum += p[i0 + i];
#endif
    }
    e[i0 + ema_length - 1] = sum / static_cast<double>(ema_length);

    // EMA recurrence (still serial)
    const double alpha = 2.0 / (static_cast<double>(ema_length) + 1.0);
    const double one_m_alpha = 1.0 - alpha;
    for (int i = i0 + ema_length; i < n; ++i) {
#if __CUDA_ARCH__ >= 600
        e[i] = alpha * __ldg(&p[i]) + one_m_alpha * e[i - 1];
#else
        e[i] = alpha * p[i] + one_m_alpha * e[i - 1];
#endif
    }
}

// ---------------------------------------------------------------------------
// EMA - simple init (starts from first valid price)
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(1)
void batch_ema_simple_kernel(
    const double* __restrict__ all_prices,
    const int*    __restrict__ offsets,
    const int*    __restrict__ lengths,
    double*       __restrict__ all_results,
    int ema_length,
    int num_symbols)
{
    const int sym = blockIdx.x;
    if (sym >= num_symbols) return;

#if __CUDA_ARCH__ >= 600
    const int start = __ldg(&offsets[sym]);
    const int n     = __ldg(&lengths[sym]);
#else
    const int start = offsets[sym];
    const int n     = lengths[sym];
#endif
    const double* __restrict__ p = all_prices + start;
    double*       __restrict__ e = all_results + start;

    int i0 = 0;
    while (i0 < n && isnan(p[i0])) ++i0;
    if (i0 >= n) {
        for (int i = 0; i < n; ++i) e[i] = F64_NAN;
        return;
    }

    for (int i = 0; i < i0; ++i) e[i] = F64_NAN;
    e[i0] = p[i0];

    const double alpha = 2.0 / (static_cast<double>(ema_length) + 1.0);
    const double one_m_alpha = 1.0 - alpha;
    for (int i = i0 + 1; i < n; ++i) {
#if __CUDA_ARCH__ >= 600
        e[i] = alpha * __ldg(&p[i]) + one_m_alpha * e[i - 1];
#else
        e[i] = alpha * p[i] + one_m_alpha * e[i - 1];
#endif
    }
}

// ---------------------------------------------------------------------------
// WMA - naive O(N*L) version (kept for correctness)
// ---------------------------------------------------------------------------
extern "C" __global__
void batch_wma_kernel(
    const double* __restrict__ all_prices,
    const int*    __restrict__ offsets,
    const int*    __restrict__ lengths,
    double*       __restrict__ all_results,
    int wma_length,
    int num_symbols)
{
    const int sym = blockIdx.x;
    if (sym >= num_symbols) return;

#if __CUDA_ARCH__ >= 600
    const int start = __ldg(&offsets[sym]);
    const int n     = __ldg(&lengths[sym]);
#else
    const int start = offsets[sym];
    const int n     = lengths[sym];
#endif
    const double* __restrict__ p = all_prices + start;
    double*       __restrict__ w = all_results + start;

    const double denom = static_cast<double>(wma_length) * (wma_length + 1) / 2.0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i < wma_length - 1) {
            w[i] = F64_NAN;
            continue;
        }

        // NaN guard for the whole window
        bool ok = true;
        for (int j = 0; j < wma_length; ++j) {
            if (isnan(p[i - j])) { ok = false; break; }
        }
        if (!ok) {
            w[i] = F64_NAN;
            continue;
        }

        double weighted_sum = 0.0;
        for (int j = 0; j < wma_length; ++j) {
#if __CUDA_ARCH__ >= 600
            weighted_sum += __ldg(&p[i - j]) * static_cast<double>(wma_length - j);
#else
            weighted_sum += p[i - j] * static_cast<double>(wma_length - j);
#endif
        }
        w[i] = weighted_sum / denom;
    }
}

// batch_ma_kernels.cu (optional tiled WMA)
extern "C" __global__
void batch_wma_tiled_kernel(
    const double* __restrict__ all_prices,
    const int*    __restrict__ offsets,
    const int*    __restrict__ lengths,
    double* __restrict__ all_results,
    int wma_length,
    int num_symbols)
{
    const int sym = blockIdx.x;
    if (sym >= num_symbols) return;

#if __CUDA_ARCH__ >= 600
    const int start = __ldg(&offsets[sym]);
    const int n     = __ldg(&lengths[sym]);
#else
    const int start = offsets[sym];
    const int n     = lengths[sym];
#endif
    const double* p  = all_prices + start;
    double*       w  = all_results + start;

    const int TILE = 256;                     // must be >= wma_length
    extern __shared__ double sdata[];         // size = TILE + wma_length

    for (int base = threadIdx.x; base < n; base += blockDim.x) {
        // Load a tile that covers [base - wma_length + 1, base]
        int first = base - wma_length + 1;
        if (first < 0) first = 0;
        int load_end = min(first + TILE + wma_length - 1, n);

        // cooperative loading
        for (int i = first + threadIdx.x; i < load_end; i += blockDim.x) {
            sdata[i - first] = __ldg(&p[i]);
        }
        __syncthreads();

        // Compute WMA for the current index if we have a full window
        if (base >= wma_length - 1) {
            bool ok = true;
            double weighted_sum = 0.0;
            for (int j = 0; j < wma_length; ++j) {
                double val = sdata[base - first - j];
                if (isnan(val)) { ok = false; break; }
                weighted_sum += val * static_cast<double>(wma_length - j);
            }
            w[base] = ok ? weighted_sum / ((double)wma_length * (wma_length + 1) / 2.0)
                         : F64_NAN;
        }
        __syncthreads();   // ensure next tile sees correct data
    }
}

// ---------------------------------------------------------------------------
// KAMA - two-pass (noise + smoothing)
// ---------------------------------------------------------------------------
extern "C" __global__
void batch_kama_noise_kernel(
    const double* __restrict__ all_prices,
    const int*    __restrict__ offsets,
    const int*    __restrict__ lengths,
    double*       __restrict__ all_noise,
    int kama_length,
    int num_symbols)
{
    const int sym = blockIdx.x;
    if (sym >= num_symbols) return;

#if __CUDA_ARCH__ >= 600
    const int start = __ldg(&offsets[sym]);
    const int n     = __ldg(&lengths[sym]);
#else
    const int start = offsets[sym];
    const int n     = lengths[sym];
#endif
    const double* __restrict__ p = all_prices + start;
    double*       __restrict__ nout = all_noise + start;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i < kama_length) {
            nout[i] = F64_NAN;
            continue;
        }

        double sum = 0.0;
        bool valid = true;
        for (int j = i - kama_length + 1; j <= i; ++j) {
#if __CUDA_ARCH__ >= 600
            double diff = __ldg(&p[j]) - __ldg(&p[j - 1]);
#else
            double diff = p[j] - p[j - 1];
#endif
            if (isnan(diff)) { valid = false; break; }
            sum += fabs(diff);
        }
        nout[i] = valid ? sum : F64_NAN;
    }
}

extern "C" __global__ __launch_bounds__(1)
void batch_kama_smooth_kernel(
    const double* __restrict__ all_prices,
    const double* __restrict__ all_noise,
    const int*    __restrict__ offsets,
    const int*    __restrict__ lengths,
    double*       __restrict__ all_kama,
    int kama_length,
    int num_symbols)
{
    const int sym = blockIdx.x;
    if (sym >= num_symbols) return;

#if __CUDA_ARCH__ >= 600
    const int start = __ldg(&offsets[sym]);
    const int n     = __ldg(&lengths[sym]);
#else
    const int start = offsets[sym];
    const int n     = lengths[sym];
#endif
    const double* __restrict__ p = all_prices + start;
    const double* __restrict__ nout = all_noise + start;
    double*       __restrict__ k = all_kama + start;

    constexpr double fast = 0.666;
    constexpr double slow = 0.064;

    if (n == 0) return;
    k[0] = p[0];

    for (int i = 1; i < n; ++i) {
        if (i < kama_length || isnan(nout[i])) {
            k[i] = isnan(k[i - 1]) ? p[i] : k[i - 1];
            continue;
        }

#if __CUDA_ARCH__ >= 600
        double signal = fabs(__ldg(&p[i]) - __ldg(&p[i - kama_length]));
#else
        double signal = fabs(p[i] - p[i - kama_length]);
#endif
        double ratio  = (nout[i] == 0.0) ? 0.0 : signal / nout[i];
        double sc = pow(ratio * (fast - slow) + slow, 2.0);
        double prev = isnan(k[i - 1]) ? p[i] : k[i - 1];
#if __CUDA_ARCH__ >= 600
        k[i] = prev + sc * (__ldg(&p[i]) - prev);
#else
        k[i] = prev + sc * (p[i] - prev);
#endif
    }
}

// ---------------------------------------------------------------------------
// LSMA - linear regression moving average
// ---------------------------------------------------------------------------
extern "C" __global__
void batch_lsma_kernel(
    const double* __restrict__ all_prices,
    const int*    __restrict__ offsets,
    const int*    __restrict__ lengths,
    double*       __restrict__ all_results,
    int length,
    int num_symbols)
{
    const int sym = blockIdx.x;
    if (sym >= num_symbols) return;

#if __CUDA_ARCH__ >= 600
    const int start = __ldg(&offsets[sym]);
    const int n     = __ldg(&lengths[sym]);
#else
    const int start = offsets[sym];
    const int n     = lengths[sym];
#endif
    const double* __restrict__ p = all_prices + start;
    double*       __restrict__ lsma = all_results + start;

    const double sum_x  = static_cast<double>(length) * (length - 1) / 2.0;
    const double sum_x2 = static_cast<double>(length) * (length - 1) * (2 * length - 1) / 6.0;
    const double den    = static_cast<double>(length) * sum_x2 - sum_x * sum_x;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (i < length - 1) {
            lsma[i] = F64_NAN;
            continue;
        }

        double sum_y  = 0.0;
        double sum_xy = 0.0;
        bool   valid  = true;
        for (int j = 0; j < length; ++j) {
#if __CUDA_ARCH__ >= 600
            double v = __ldg(&p[i - (length - 1) + j]);
#else
            double v = p[i - (length - 1) + j];
#endif
            if (isnan(v)) { valid = false; break; }
            sum_y  += v;
            sum_xy += static_cast<double>(j) * v;
        }

        if (!valid) {
            lsma[i] = F64_NAN;
        } else {
            double m = (static_cast<double>(length) * sum_xy - sum_x * sum_y) / den;
            double c = (sum_y - m * sum_x) / static_cast<double>(length);
            lsma[i] = m * static_cast<double>(length - 1) + c;
        }
    }
}

// ---------------------------------------------------------------------------
// Linear combine (a*A - b*B)
// ---------------------------------------------------------------------------
extern "C" __global__
void batch_linear_combine_kernel(
    const double* __restrict__ input_a,
    const double* __restrict__ input_b,
    const int*    __restrict__ offsets,
    const int*    __restrict__ lengths,
    double*       __restrict__ output,
    double mult_a,
    double mult_b,
    int num_symbols)
{
    const int sym = blockIdx.x;
    if (sym >= num_symbols) return;

#if __CUDA_ARCH__ >= 600
    const int start = __ldg(&offsets[sym]);
    const int n     = __ldg(&lengths[sym]);
#else
    const int start = offsets[sym];
    const int n     = lengths[sym];
#endif

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
#if __CUDA_ARCH__ >= 600
        double a = __ldg(&input_a[start + i]);
        double b = __ldg(&input_b[start + i]);
#else
        double a = input_a[start + i];
        double b = input_b[start + i];
#endif
        if (isnan(a) || isnan(b))
            output[start + i] = F64_NAN;
        else
            output[start + i] = mult_a * a - mult_b * b;
    }
}
