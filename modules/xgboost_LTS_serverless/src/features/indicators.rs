use std::collections::VecDeque;

pub fn compute_gains_losses(close: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if close.len() < 2 {
        return (vec![], vec![]);
    }
    let mut gains = Vec::with_capacity(close.len() - 1);
    let mut losses = Vec::with_capacity(close.len() - 1);
    for i in 1..close.len() {
        let change = close[i] - close[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }
    (gains, losses)
}

fn push_tail(tail: &mut VecDeque<f64>, value: f64, max_len: usize) {
    if max_len == 0 {
        return;
    }
    if tail.len() == max_len {
        tail.pop_front();
    }
    tail.push_back(value);
}

fn rsi_value(avg_gain: f64, avg_loss: f64) -> f64 {
    if avg_loss == 0.0 {
        100.0
    } else {
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
}

pub fn rsi_tail_streaming(close: &[f64], period: usize, tail_len: usize) -> Vec<f64> {
    if tail_len == 0 {
        return vec![];
    }

    if period == 0 || close.len() <= period {
        return vec![f64::NAN; tail_len];
    }

    let mut tail = VecDeque::with_capacity(tail_len);
    for _ in 0..period {
        push_tail(&mut tail, 50.0, tail_len);
    }

    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    for i in 1..=period {
        let change = close[i] - close[i - 1];
        if change > 0.0 {
            avg_gain += change;
        } else {
            avg_loss -= change;
        }
    }

    avg_gain /= period as f64;
    avg_loss /= period as f64;
    push_tail(&mut tail, rsi_value(avg_gain, avg_loss), tail_len);

    for i in (period + 1)..close.len() {
        let change = close[i] - close[i - 1];
        let (gain, loss) = if change > 0.0 {
            (change, 0.0)
        } else {
            (0.0, -change)
        };

        avg_gain = (avg_gain * (period - 1) as f64 + gain) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + loss) / period as f64;
        push_tail(&mut tail, rsi_value(avg_gain, avg_loss), tail_len);
    }

    let mut result: Vec<f64> = tail.into_iter().collect();
    if result.len() < tail_len {
        let mut padded = vec![f64::NAN; tail_len - result.len()];
        padded.extend(result);
        result = padded;
    }

    result
}

#[inline]
pub fn rsi_last_streaming(close: &[f64], period: usize) -> f64 {
    let tail = rsi_tail_streaming(close, period, 1);
    tail.first().copied().unwrap_or(f64::NAN)
}

pub fn rsi_tail_from_gains_losses(
    gains: &[f64],
    losses: &[f64],
    period: usize,
    tail_len: usize,
) -> Vec<f64> {
    if tail_len == 0 {
        return vec![];
    }

    if gains.len() < period || period == 0 {
        return vec![f64::NAN; tail_len];
    }

    let mut tail = VecDeque::with_capacity(tail_len);
    for _ in 0..period {
        push_tail(&mut tail, 50.0, tail_len);
    }

    let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;
    push_tail(&mut tail, rsi_value(avg_gain, avg_loss), tail_len);

    for i in period..gains.len() {
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
        push_tail(&mut tail, rsi_value(avg_gain, avg_loss), tail_len);
    }

    let mut result: Vec<f64> = tail.into_iter().collect();
    if result.len() < tail_len {
        let mut padded = vec![f64::NAN; tail_len - result.len()];
        padded.extend(result);
        result = padded;
    }

    result
}

pub fn rsi_last_from_gains_losses(gains: &[f64], losses: &[f64], period: usize) -> f64 {
    let tail = rsi_tail_from_gains_losses(gains, losses, period, 1);
    tail.first().copied().unwrap_or(f64::NAN)
}

pub fn rsi_from_gains_losses(gains: &[f64], losses: &[f64], period: usize) -> Vec<f64> {
    if gains.len() < period {
        return vec![];
    }
    let mut rsi_values = Vec::with_capacity(gains.len() + 1);
    rsi_values.extend(vec![50.0; period]);

    let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

    if avg_loss == 0.0 {
        rsi_values.push(100.0);
    } else {
        let rs = avg_gain / avg_loss;
        rsi_values.push(100.0 - (100.0 / (1.0 + rs)));
    }

    for i in period..gains.len() {
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

        if avg_loss == 0.0 {
            rsi_values.push(100.0);
        } else {
            let rs = avg_gain / avg_loss;
            rsi_values.push(100.0 - (100.0 / (1.0 + rs)));
        }
    }

    rsi_values
}

pub fn rsi(close: &[f64], period: usize) -> Vec<f64> {
    let (gains, losses) = compute_gains_losses(close);
    rsi_from_gains_losses(&gains, &losses, period)
}

pub fn macd(
    close: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ema_fast = super::moving_averages::ema(close, fast);
    let ema_slow = super::moving_averages::ema(close, slow);

    let macd_line: Vec<f64> = ema_fast
        .iter()
        .zip(ema_slow.iter())
        .map(|(f, s)| f - s)
        .collect();

    let signal_line = super::moving_averages::ema(&macd_line, signal);

    let histogram: Vec<f64> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(m, s)| m - s)
        .collect();

    (macd_line, signal_line, histogram)
}

#[allow(clippy::needless_range_loop)]
pub fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    if high.len() != low.len() || low.len() != close.len() || close.is_empty() {
        return vec![];
    }

    let mut tr = vec![f64::NAN; close.len()];
    tr[0] = high[0] - low[0];
    for i in 1..close.len() {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    let mut atr = vec![f64::NAN; close.len()];
    if close.len() < period {
        return atr;
    }

    let mut sum = 0.0;
    for i in 0..period {
        sum += tr[i];
    }
    atr[period - 1] = sum / period as f64;

    for i in period..close.len() {
        atr[i] = (atr[i - 1] * (period - 1) as f64 + tr[i]) / period as f64;
    }
    atr
}

pub fn atr_last(high: &[f64], low: &[f64], close: &[f64], period: usize) -> f64 {
    if high.len() != low.len() || low.len() != close.len() || close.is_empty() || period == 0 {
        return f64::NAN;
    }

    if close.len() < period {
        return f64::NAN;
    }

    let mut tr = Vec::with_capacity(close.len());
    tr.push(high[0] - low[0]);
    for i in 1..close.len() {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr.push(hl.max(hc).max(lc));
    }

    let mut atr_value = tr[..period].iter().sum::<f64>() / period as f64;
    for tr_value in tr.iter().skip(period) {
        atr_value = (atr_value * (period - 1) as f64 + tr_value) / period as f64;
    }

    atr_value
}

#[allow(clippy::needless_range_loop)]
pub fn macd_tail(
    close: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
    tail_len: usize,
) -> (Vec<f64>, f64, f64) {
    if close.len() < fast || close.len() < slow || fast == 0 || slow == 0 || signal == 0 {
        return (vec![], f64::NAN, f64::NAN);
    }

    let n = close.len();
    let fast_mult = 2.0 / (fast as f64 + 1.0);
    let slow_mult = 2.0 / (slow as f64 + 1.0);

    let fast_sma: f64 = close[..fast].iter().sum::<f64>() / fast as f64;
    let slow_sma: f64 = close[..slow].iter().sum::<f64>() / slow as f64;

    let mut fast_prev = fast_sma;
    let mut slow_prev = slow_sma;

    let mut signal_window: Vec<f64> = Vec::with_capacity(signal);
    let signal_mult = 2.0 / (signal as f64 + 1.0);
    let mut signal_prev = f64::NAN;

    let mut line_tail = VecDeque::with_capacity(tail_len);
    let mut last_signal = f64::NAN;

    for i in 0..n {
        let fast_val = if i < fast {
            fast_sma
        } else {
            fast_prev = (close[i] - fast_prev) * fast_mult + fast_prev;
            fast_prev
        };

        let slow_val = if i < slow {
            slow_sma
        } else {
            slow_prev = (close[i] - slow_prev) * slow_mult + slow_prev;
            slow_prev
        };

        let macd_line = fast_val - slow_val;
        push_tail(&mut line_tail, macd_line, tail_len);

        if signal_window.len() < signal {
            signal_window.push(macd_line);
            if signal_window.len() == signal {
                signal_prev = signal_window.iter().sum::<f64>() / signal as f64;
                last_signal = signal_prev;
            }
            continue;
        }

        signal_prev = (macd_line - signal_prev) * signal_mult + signal_prev;
        last_signal = signal_prev;
    }

    let last_line = line_tail.back().copied().unwrap_or(f64::NAN);
    let last_hist = if last_signal.is_nan() {
        f64::NAN
    } else {
        last_line - last_signal
    };

    (line_tail.into_iter().collect(), last_signal, last_hist)
}

pub fn macd_last(close: &[f64], fast: usize, slow: usize, signal: usize) -> (f64, f64, f64) {
    let (line_tail, signal_last, hist_last) = macd_tail(close, fast, slow, signal, 1);
    (
        line_tail.first().copied().unwrap_or(f64::NAN),
        signal_last,
        hist_last,
    )
}

pub fn stochastic_rsi(
    close: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    smooth_k: usize,
    smooth_d: usize,
) -> (Vec<f64>, Vec<f64>) {
    let rsi_vals = rsi(close, rsi_period);
    if rsi_vals.is_empty() {
        return (vec![], vec![]);
    }

    let mut k_un = vec![f64::NAN; rsi_vals.len()];
    for i in (stoch_period - 1)..rsi_vals.len() {
        let window = &rsi_vals[(i + 1 - stoch_period)..=i];
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for &v in window {
            if !v.is_nan() {
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }
        }
        if min_val == max_val || max_val == f64::NEG_INFINITY {
            k_un[i] = 100.0;
        } else {
            k_un[i] = (rsi_vals[i] - min_val) / (max_val - min_val) * 100.0;
        }
    }

    let k = super::moving_averages::sma(&k_un, smooth_k);
    let d = super::moving_averages::sma(&k, smooth_d);

    (k, d)
}

pub fn stochastic_rsi_last(
    close: &[f64],
    rsi_period: usize,
    stoch_period: usize,
    smooth_k: usize,
    smooth_d: usize,
) -> (f64, f64) {
    if stoch_period == 0 || smooth_k == 0 || smooth_d == 0 {
        return (f64::NAN, f64::NAN);
    }

    let required = stoch_period + smooth_k + smooth_d - 2;
    let rsi_vals = rsi_tail_streaming(close, rsi_period, required.max(1));

    if rsi_vals.len() < stoch_period + smooth_k + smooth_d - 2 {
        return (f64::NAN, f64::NAN);
    }

    let mut k_un = Vec::with_capacity(rsi_vals.len());
    for i in 0..rsi_vals.len() {
        if i + 1 < stoch_period {
            k_un.push(f64::NAN);
            continue;
        }

        let window = &rsi_vals[(i + 1 - stoch_period)..=i];
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for &v in window {
            if !v.is_nan() {
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }
        }
        if min_val == max_val || max_val == f64::NEG_INFINITY {
            k_un.push(100.0);
        } else {
            k_un.push((rsi_vals[i] - min_val) / (max_val - min_val) * 100.0);
        }
    }

    let mut k_tail_for_d = Vec::with_capacity(smooth_d);
    for window in k_un.windows(smooth_k).rev().take(smooth_d) {
        k_tail_for_d.push(super::moving_averages::sma_last(window, smooth_k));
    }
    k_tail_for_d.reverse();

    let k_last = k_tail_for_d.last().copied().unwrap_or(f64::NAN);
    let d_last = super::moving_averages::sma_last(&k_tail_for_d, smooth_d);

    (k_last, d_last)
}

pub fn on_balance_volume(close: &[f64], volume: &[f64]) -> Vec<f64> {
    let mut obv = vec![0.0; close.len()];
    if close.is_empty() {
        return obv;
    }
    obv[0] = volume[0]; // Or 0.0 depending on convention
    for i in 1..close.len() {
        if close[i] > close[i - 1] {
            obv[i] = obv[i - 1] + volume[i];
        } else if close[i] < close[i - 1] {
            obv[i] = obv[i - 1] - volume[i];
        } else {
            obv[i] = obv[i - 1];
        }
    }
    obv
}

pub fn on_balance_volume_last(close: &[f64], volume: &[f64]) -> f64 {
    if close.is_empty() || volume.is_empty() {
        return f64::NAN;
    }

    let mut obv = volume[0];
    for i in 1..close.len() {
        if close[i] > close[i - 1] {
            obv += volume[i];
        } else if close[i] < close[i - 1] {
            obv -= volume[i];
        }
    }
    obv
}

pub fn bollinger_band_percent(close: &[f64], period: usize, num_std: f64) -> Vec<f64> {
    let n = close.len();
    let mut bbp = vec![f64::NAN; n];
    if n < period {
        return bbp;
    }

    let sma_vals = super::moving_averages::sma(close, period);

    for i in (period - 1)..n {
        let window = &close[(i + 1 - period)..=i];
        let mean = sma_vals[i];
        let variance = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std_dev = variance.sqrt();

        let upper = mean + num_std * std_dev;
        let lower = mean - num_std * std_dev;

        if upper - lower == 0.0 {
            bbp[i] = f64::NAN;
        } else {
            bbp[i] = (close[i] - lower) / (upper - lower);
        }
    }

    bbp
}

pub fn bollinger_band_percent_last(close: &[f64], period: usize, num_std: f64) -> f64 {
    let n = close.len();
    if n < period || period == 0 {
        return f64::NAN;
    }

    let window = &close[(n - period)..n];
    let mean = window.iter().sum::<f64>() / period as f64;
    let variance = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
    let std_dev = variance.sqrt();

    let upper = mean + num_std * std_dev;
    let lower = mean - num_std * std_dev;

    if upper - lower == 0.0 {
        f64::NAN
    } else {
        (close[n - 1] - lower) / (upper - lower)
    }
}
