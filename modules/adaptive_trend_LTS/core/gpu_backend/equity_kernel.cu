/**
 * Phase 4 - Task 2.1.2: Custom CUDA kernel for equity calculation.
 *
 * Algorithm (matches Rust eq. and compute_equity/core.py):
 * - For i in [cutout, n): a = r[i] * sign(sig[i]) or 0; NaN/0 -> 0.
 * - e[i] = prev * decay * (1 + a), prev init = starting_equity; floor 0.25.
 * - Sequential recurrence → single-thread kernel.
 *
 * Block/grid: (1, 1), 1 thread.
 */

#define F64_NAN __longlong_as_double(0x7ff8000000000000LL)

extern "C" __global__ void equity_kernel(
    const double* __restrict__ r_values,
    const double* __restrict__ sig_prev,
    double* __restrict__ equity,
    double starting_equity,
    double decay_multiplier,
    int cutout,
    int n
) {
    double prev_e = -1.0;

    if (cutout > 0 && cutout <= n) {
        for (int i = 0; i < cutout; i++)
            equity[i] = F64_NAN;
    }

    for (int i = cutout; i < n; i++) {
        double r_i = r_values[i];
        double s_prev = sig_prev[i];

        double a = 0.0;
        if (!isnan(s_prev) && !isnan(r_i) && s_prev != 0.0) {
            if (s_prev > 0.0)
                a = r_i;
            else
                a = -r_i;
        }

        double e_curr;
        if (prev_e < 0.0)
            e_curr = starting_equity;
        else
            e_curr = (prev_e * decay_multiplier) * (1.0 + a);

        if (e_curr < 0.25)
            e_curr = 0.25;

        equity[i] = e_curr;
        prev_e = e_curr;
    }
}
