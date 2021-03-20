use criterion::*;

#[cfg(unix)]
use pprof::criterion::{Output, PProfProfiler};

use reason_othello::test_utils::perft;

fn criterion_perft(c: &mut Criterion) {
    let mut group = c.benchmark_group("perft");
    group.sample_size(50);

    for depth in 1..6 {
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            b.iter(|| perft::run_perft(black_box(depth)))
        });
    }

    group.finish();
}

#[cfg(unix)]
criterion_group! {
    name = perft;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = criterion_perft
}

#[cfg(not(unix))]
criterion_group! {
    name = perft;
    config = Criterion::default();
    targets = criterion_perft
}

criterion_main!(perft);
