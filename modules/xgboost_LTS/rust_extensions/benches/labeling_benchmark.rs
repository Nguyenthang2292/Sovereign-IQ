use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use xgboost_rust::labeling::{rolling_mean_rust, rolling_quantile_rust};

fn benchmark_rolling_mean(c: &mut Criterion) {
    let data: Vec<f64> = (0..10000).map(|i| (i as f64).sin()).collect();
    let arr = Array1::from(data);

    c.bench_function("rolling_mean_1000", |b| {
        b.iter(|| {
            // Benchmark rolling mean with window=1000
            let window = 1000;
            let mut result = Array1::<f64>::from_elem(arr.len(), f64::NAN);
            let mut current_sum = 0.0;

            for i in 0..arr.len() {
                current_sum += arr[i];
                if i >= window {
                    current_sum -= arr[i - window];
                }

                if i >= window - 1 {
                    result[i] = current_sum / window as f64;
                }
            }
            black_box(result)
        })
    });
}

fn benchmark_rolling_quantile(c: &mut Criterion) {
    let data: Vec<f64> = (0..10000).map(|i| (i as f64).sin()).collect();
    let arr = Array1::from(data);

    c.bench_function("rolling_quantile_50", |b| {
        b.iter(|| {
            let window = 50;
            let q = 0.5;
            let mut result = Array1::<f64>::from_elem(arr.len(), f64::NAN);

            for i in 0..arr.len() {
                if i >= window - 1 {
                    let start = i - window + 1;
                    let mut window_slice: Vec<f64> = arr.slice(ndarray::s![start..=i]).to_vec();
                    window_slice.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let idx = ((window_slice.len() - 1) as f64 * q) as usize;
                    result[i] = window_slice[idx];
                }
            }
            black_box(result)
        })
    });
}

criterion_group!(benches, benchmark_rolling_mean, benchmark_rolling_quantile);
criterion_main!(benches);
