use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use atc_rust::ma_calculations;

fn ma_benchmark(c: &mut Criterion) {
    let n = 10000;
    let prices = Array1::from_iter((0..n).map(|i| 100.0 + (i as f64) * 0.1));
    let length = 20;

    let mut group = c.benchmark_group("ma_calculations");
    group.sample_size(100);

    group.bench_function("ema", |b| {
        b.iter(|| {
            ma_calculations::calculate_ema_internal(
                black_box(prices.view()),
                black_box(length),
            )
        })
    });

    group.bench_function("wma", |b| {
        b.iter(|| {
            ma_calculations::calculate_wma_internal(
                black_box(prices.view()),
                black_box(length),
            )
        })
    });

    group.bench_function("dema", |b| {
        b.iter(|| {
            ma_calculations::calculate_dema_internal(
                black_box(prices.view()),
                black_box(length),
            )
        })
    });

    group.bench_function("lsma", |b| {
        b.iter(|| {
            ma_calculations::calculate_lsma_internal(
                black_box(prices.view()),
                black_box(length),
            )
        })
    });

    group.bench_function("hma", |b| {
        b.iter(|| {
            ma_calculations::calculate_hma_internal(
                black_box(prices.view()),
                black_box(length),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, ma_benchmark);
criterion_main!(benches);
