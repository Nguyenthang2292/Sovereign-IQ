use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use atc_rust::signal_persistence::process_signal_persistence_internal;

fn persistence_benchmark(c: &mut Criterion) {
    let n = 10000;
    let up = Array1::from_elem(n, false);
    let down = Array1::from_elem(n, false);
    
    c.bench_function("persistence_calculation", |b| {
        b.iter(|| {
            process_signal_persistence_internal(
                black_box(up.view()),
                black_box(down.view()),
            )
        })
    });
}

criterion_group!(benches, persistence_benchmark);
criterion_main!(benches);
