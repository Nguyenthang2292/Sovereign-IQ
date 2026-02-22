pub fn roc(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![];
    }

    let mut roc_values = Vec::with_capacity(data.len());
    roc_values.extend(vec![f64::NAN; period]);

    for i in period..data.len() {
        if data[i - period] == 0.0 {
            roc_values.push(f64::NAN);
        } else {
            roc_values.push(((data[i] - data[i - period]) / data[i - period]) * 100.0);
        }
    }

    roc_values
}

#[inline]
pub fn roc_last(data: &[f64], period: usize) -> f64 {
    if data.len() < period + 1 || period == 0 {
        return f64::NAN;
    }
    let i = data.len() - 1;
    if data[i - period] == 0.0 {
        return f64::NAN;
    }
    ((data[i] - data[i - period]) / data[i - period]) * 100.0
}

pub fn rolling_std(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![];
    }

    let mut std_values = Vec::with_capacity(data.len());
    std_values.extend(vec![f64::NAN; period - 1]);

    for window in data.windows(period) {
        let mean = window.iter().sum::<f64>() / period as f64;
        let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        std_values.push(variance.sqrt());
    }

    std_values
}

#[inline]
pub fn rolling_std_last(data: &[f64], period: usize) -> f64 {
    if data.len() < period || period == 0 {
        return f64::NAN;
    }

    let window = &data[(data.len() - period)..];
    let mean = window.iter().sum::<f64>() / period as f64;
    let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
    variance.sqrt()
}

pub fn rolling_skewness(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < period {
        return vec![];
    }

    let mut skew_values = Vec::with_capacity(data.len());
    skew_values.extend(vec![f64::NAN; period - 1]);

    for window in data.windows(period) {
        let mean = window.iter().sum::<f64>() / period as f64;
        let std = (window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64).sqrt();

        if std > 0.0 {
            let skew = window
                .iter()
                .map(|x| ((x - mean) / std).powi(3))
                .sum::<f64>()
                / period as f64;
            skew_values.push(skew);
        } else {
            skew_values.push(0.0);
        }
    }

    skew_values
}

#[inline]
pub fn rolling_skewness_last(data: &[f64], period: usize) -> f64 {
    if data.len() < period || period == 0 {
        return f64::NAN;
    }

    let window = &data[(data.len() - period)..];
    let mean = window.iter().sum::<f64>() / period as f64;
    let std = (window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64).sqrt();

    if std > 0.0 {
        window
            .iter()
            .map(|x| ((x - mean) / std).powi(3))
            .sum::<f64>()
            / period as f64
    } else {
        0.0
    }
}

pub fn bollinger_bands(
    data: &[f64],
    period: usize,
    num_std: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let sma = super::moving_averages::sma(data, period);
    let std = rolling_std(data, period);

    let upper = sma
        .iter()
        .zip(std.iter())
        .map(|(m, s)| m + num_std * s)
        .collect();

    let lower = sma
        .iter()
        .zip(std.iter())
        .map(|(m, s)| m - num_std * s)
        .collect();

    (sma, upper, lower)
}

pub fn volatility(data: &[f64], period: usize) -> Vec<f64> {
    let returns: Vec<f64> = data
        .windows(2)
        .map(|w| {
            if w[0] == 0.0 {
                f64::NAN
            } else {
                (w[1] - w[0]) / w[0]
            }
        })
        .collect();

    rolling_std(&returns, period)
}
