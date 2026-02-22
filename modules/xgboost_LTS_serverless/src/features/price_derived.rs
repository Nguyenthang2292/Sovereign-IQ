pub fn returns_n(close: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![f64::NAN; close.len()];
    for i in n..close.len() {
        if close[i - n] == 0.0 {
            result[i] = f64::NAN;
        } else {
            result[i] = (close[i] - close[i - n]) / close[i - n];
        }
    }
    result
}

#[inline]
pub fn returns_n_last(close: &[f64], n: usize) -> f64 {
    if close.len() <= n {
        return f64::NAN;
    }
    let i = close.len() - 1;
    if close[i - n] == 0.0 {
        f64::NAN
    } else {
        (close[i] - close[i - n]) / close[i - n]
    }
}

pub fn log_returns(close: &[f64]) -> Vec<f64> {
    let mut result = vec![f64::NAN; close.len()];
    for i in 1..close.len() {
        if close[i - 1] == 0.0 {
            result[i] = f64::NAN;
        } else {
            result[i] = (close[i] / close[i - 1]).ln();
        }
    }
    result
}

pub fn high_low_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let mut result = vec![f64::NAN; close.len()];
    for i in 0..close.len() {
        if close[i] != 0.0 {
            result[i] = (high[i] - low[i]) / close[i];
        } else {
            result[i] = f64::NAN;
        }
    }
    result
}

#[inline]
pub fn high_low_range_last(high: &[f64], low: &[f64], close: &[f64]) -> f64 {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return f64::NAN;
    }

    let i = close.len() - 1;
    if close[i] == 0.0 {
        return f64::NAN;
    }

    (high[i] - low[i]) / close[i]
}

pub fn close_open_diff(open: &[f64], close: &[f64]) -> Vec<f64> {
    let mut result = vec![f64::NAN; close.len()];
    for i in 0..close.len() {
        if open[i] != 0.0 {
            result[i] = (close[i] - open[i]) / open[i];
        } else {
            result[i] = f64::NAN;
        }
    }
    result
}

#[inline]
pub fn close_open_diff_last(open: &[f64], close: &[f64]) -> f64 {
    if open.is_empty() || close.is_empty() {
        return f64::NAN;
    }

    let i = close.len() - 1;
    if open[i] == 0.0 {
        return f64::NAN;
    }

    (close[i] - open[i]) / open[i]
}

pub fn log_volume(volume: &[f64]) -> Vec<f64> {
    let mut result = vec![f64::NAN; volume.len()];
    for i in 0..volume.len() {
        if volume[i] > 0.0 {
            result[i] = volume[i].ln();
        } else {
            result[i] = 0.0;
        }
    }
    result
}

#[inline]
pub fn log_volume_last(volume: &[f64]) -> f64 {
    if volume.is_empty() {
        return f64::NAN;
    }

    let last = volume[volume.len() - 1];
    if last > 0.0 {
        last.ln()
    } else {
        0.0
    }
}
