use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use atc_rust::equity::calculate_equity_internal;

fn equity_benchmark(c: &mut Criterion) {
    let n = 10000;
    let r = Array1::from_elem(n, 0.01);
    let sig = Array1::from_elem(n, 1.0);
    
    c.bench_function("equity_calculation", |b| {
        b.iter(|| {
            calculate_equity_internal(
                black_box(r.view()),
                black_box(sig.view()),
                black_box(100.0),
                black_box(0.99),
                black_box(100),
            )
        })
    });
}

criterion_group!(benches, equity_benchmark);
criterion_main!(benches);
