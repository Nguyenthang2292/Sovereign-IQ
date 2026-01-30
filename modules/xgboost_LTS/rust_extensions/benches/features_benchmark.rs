use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;

fn benchmark_price_derived(c: &mut Criterion) {
    let n = 10000;
    let close = Array1::from(vec![100.0; n]);
    let high = Array1::from(vec![105.0; n]);
    let low = Array1::from(vec![95.0; n]);
    let open = Array1::from(vec![100.0; n]);
    let volume = Array1::from(vec![1000.0; n]);

    c.bench_function("add_price_derived_rust_logic", |b| {
        b.iter(|| {
            let mut returns_1 = Array1::<f64>::from_elem(n, f64::NAN);
            let mut returns_5 = Array1::<f64>::from_elem(n, f64::NAN);
            let mut log_volume = Array1::<f64>::zeros(n);
            let mut high_low_range = Array1::<f64>::zeros(n);
            let mut close_open_diff = Array1::<f64>::zeros(n);

            for i in 0..n {
                if i >= 1 && close[i - 1] != 0.0 {
                    returns_1[i] = (close[i] - close[i - 1]) / close[i - 1];
                }
                if i >= 5 && close[i - 5] != 0.0 {
                    returns_5[i] = (close[i] - close[i - 5]) / close[i - 5];
                }
                log_volume[i] = f64::ln(volume[i] + 1.0);
                if close[i] != 0.0 {
                    high_low_range[i] = (high[i] - low[i]) / close[i];
                    close_open_diff[i] = (close[i] - open[i]) / close[i];
                }
            }
            black_box((
                returns_1,
                returns_5,
                log_volume,
                high_low_range,
                close_open_diff,
            ))
        })
    });
}

fn benchmark_advanced_features(c: &mut Criterion) {
    let n = 10000;
    let returns_1 = Array1::from(vec![0.01; n]);
    let window = 20;

    c.bench_function("rolling_std_skew_20", |b| {
        b.iter(|| {
            let mut roll_std = Array1::<f64>::from_elem(n, f64::NAN);
            let mut roll_skew = Array1::<f64>::from_elem(n, f64::NAN);

            for i in 0..n {
                if i >= window - 1 {
                    let start = i - window + 1;
                    let slice = returns_1.slice(ndarray::s![start..=i]);

                    let mean = slice.mean().unwrap_or(0.0);

                    let mut m2 = 0.0;
                    let mut m3 = 0.0;
                    for &x in slice {
                        let diff = x - mean;
                        m2 += diff * diff;
                        m3 += diff * diff * diff;
                    }

                    // Std Dev
                    let variance = m2 / (window - 1) as f64;
                    let std_dev = if variance > 0.0 { variance.sqrt() } else { 0.0 };
                    roll_std[i] = std_dev;

                    // Skew
                    if m2 > 0.0 && window >= 3 {
                        let n_f = window as f64;
                        let skew = (n_f * m3) / ((n_f - 1.0) * (n_f - 2.0) * std_dev.powi(3));
                        roll_skew[i] = skew;
                    } else if window >= 3 {
                        roll_skew[i] = 0.0;
                    }
                }
            }
            black_box((roll_std, roll_skew))
        })
    });
}

criterion_group!(
    benches,
    benchmark_price_derived,
    benchmark_advanced_features
);
criterion_main!(benches);
