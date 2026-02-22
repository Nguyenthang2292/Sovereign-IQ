#[allow(clippy::needless_range_loop)]
pub fn sma(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![];
    }

    let mut sma_values = Vec::with_capacity(data.len());
    sma_values.extend(vec![f64::NAN; period - 1]);

    let mut current_sum: f64 = data[..period].iter().sum();
    sma_values.push(current_sum / period as f64);

    for i in period..data.len() {
        current_sum = current_sum + data[i] - data[i - period];
        sma_values.push(current_sum / period as f64);
    }

    sma_values
}

#[inline]
pub fn sma_last(data: &[f64], period: usize) -> f64 {
    if data.len() < period || period == 0 {
        return f64::NAN;
    }

    let start = data.len() - period;
    let sum: f64 = data[start..].iter().sum();
    sum / period as f64
}

#[allow(clippy::needless_range_loop)]
pub fn ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![];
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema_values = Vec::with_capacity(data.len());

    let sma: f64 = data[..period].iter().sum::<f64>() / period as f64;
    ema_values.extend(vec![sma; period]);

    let mut prev_ema = sma;
    for i in period..data.len() {
        let current_ema = (data[i] - prev_ema) * multiplier + prev_ema;
        ema_values.push(current_ema);
        prev_ema = current_ema;
    }

    ema_values
}

#[allow(clippy::needless_range_loop)]
pub fn ema_last(data: &[f64], period: usize) -> f64 {
    if data.len() < period || period == 0 {
        return f64::NAN;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let sma: f64 = data[..period].iter().sum::<f64>() / period as f64;

    let mut prev_ema = sma;
    for i in period..data.len() {
        prev_ema = (data[i] - prev_ema) * multiplier + prev_ema;
    }

    prev_ema
}

#[allow(clippy::needless_range_loop)]
pub fn wma(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![];
    }

    let mut wma_values = Vec::with_capacity(data.len());
    wma_values.extend(vec![f64::NAN; period - 1]);

    let denominator = (period * (period + 1)) / 2;

    let mut current_sum: f64 = 0.0;
    let mut weighted_sum: f64 = 0.0;

    for i in 0..period {
        current_sum += data[i];
        weighted_sum += data[i] * (i + 1) as f64;
    }
    wma_values.push(weighted_sum / denominator as f64);

    for i in period..data.len() {
        weighted_sum = weighted_sum + (period as f64) * data[i] - current_sum;
        current_sum = current_sum - data[i - period] + data[i];
        wma_values.push(weighted_sum / denominator as f64);
    }

    wma_values
}

pub fn wma_last(data: &[f64], period: usize) -> f64 {
    if data.len() < period || period == 0 {
        return f64::NAN;
    }

    let denominator = (period * (period + 1)) as f64 / 2.0;
    let start = data.len() - period;
    let mut weighted_sum = 0.0;

    for (idx, value) in data[start..].iter().enumerate() {
        weighted_sum += *value * (idx as f64 + 1.0);
    }

    weighted_sum / denominator
}
