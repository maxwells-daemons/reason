use criterion::*;
use pprof::criterion::{Output, PProfProfiler};

use solver::test_utils;

fn criterion_perft(c: &mut Criterion) {
    let mut group = c.benchmark_group("perft");
    group.sample_size(50);

    for depth in 1..6 {
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            b.iter(|| test_utils::perft::run_perft(black_box(depth)))
        });
    }

    group.finish();
}

criterion_group! {
    name = perft;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = criterion_perft
}
criterion_main!(perft);
