use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use atc_rust::kama::calculate_kama_internal;

fn kama_benchmark(c: &mut Criterion) {
    let n = 10000;
    let prices = Array1::from_elem(n, 100.0);
    
    c.bench_function("kama_calculation", |b| {
        b.iter(|| {
            calculate_kama_internal(
                black_box(prices.view()),
                black_box(28),
            )
        })
    });
}

criterion_group!(benches, kama_benchmark);
criterion_main!(benches);
