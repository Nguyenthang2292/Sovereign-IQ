/**
 * Equity kernel – one thread processes a whole time-series.
 *
 * Parameters
 * ----------
 * r_values          : per-symbol returns (size = total_bars * num_symbols)
 * sig_prev          : per-symbol signal values (same layout as r_values)
 * equity            : output buffer (same layout)
 * starting_equity   : equity value before the first valid bar
 * decay_multiplier  : multiplicative decay applied each step
 * cutout            : number of warm-up bars to set to NaN
 * total_bars        : length of each series (n)
 * offsets           : start index of each symbol inside the flat arrays
 *
 * Launch configuration
 * --------------------
 * dim3 grid(num_symbols);
 * dim3 block(1);   // one thread per symbol (can be >1 if you want intra-symbol parallelism)
 */

#include "gpu_common.h"

extern "C" __global__ __launch_bounds__(1)
void equity_kernel(
    const double* __restrict__ r_values,
    const double* __restrict__ sig_prev,
    double*       __restrict__ equity,
    double starting_equity,
    double decay_multiplier,
    int cutout,
    int total_bars,
    const int*   __restrict__ offsets,          // added offsets for batched layout
    int num_symbols)                            // explicit number of symbols
{
    const int symbol_idx = blockIdx.x;          // one block per symbol
    if (symbol_idx >= num_symbols) return;

    // Sanity check: ensure starting_equity is finite
    double safe_starting_equity = starting_equity;
    if (!isfinite(safe_starting_equity)) {
        safe_starting_equity = 1.0;
    }

    const int start = offsets[symbol_idx];
    const double* __restrict__ r = r_values + start;
    const double* __restrict__ s = sig_prev   + start;
    double*       __restrict__ e = equity      + start;

    // --------------------------------------------------------------------
    // Warm‑up period → NaN
    // --------------------------------------------------------------------
    int cut = cutout;
    if (cut < 0) cut = 0;
    if (cut > total_bars) cut = total_bars;

    for (int i = 0; i < cut; ++i) {
        e[i] = F64_NAN;
    }

    // --------------------------------------------------------------------
    // Serial recurrence (still per‑symbol, but now many symbols run in parallel)
    // --------------------------------------------------------------------
    double prev_e = -1.0;               // sentinel for “first valid bar”

    for (int i = cut; i < total_bars; ++i) {
#if __CUDA_ARCH__ >= 600
        const double r_i = __ldg(&r[i]);
        const double s_i = __ldg(&s[i]);
#else
        const double r_i = r[i];
        const double s_i = s[i];
#endif
        // a = r_i * sign(s_i)  (0 if NaN or s_i == 0)
        double a = 0.0;
        if (!isnan(r_i) && !isnan(s_i) && s_i != 0.0) {
            a = (s_i > 0.0) ? r_i : -r_i;
        }

        double e_curr = (prev_e < 0.0) ? safe_starting_equity
                                      : (prev_e * decay_multiplier) * (1.0 + a);

        // floor at DEFAULT_EQUITY_FLOOR
        if (e_curr < DEFAULT_EQUITY_FLOOR) e_curr = DEFAULT_EQUITY_FLOOR;

        e[i] = e_curr;
        prev_e = e_curr;
    }
}
