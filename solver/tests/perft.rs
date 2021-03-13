use solver::test_utils::perft::run_perft;

#[test]
fn perft_1() {
    assert_eq!(run_perft(1), 4);
}

#[test]
fn perft_2() {
    assert_eq!(run_perft(2), 12);
}

#[test]
fn perft_3() {
    assert_eq!(run_perft(3), 56);
}

#[test]
fn perft_4() {
    assert_eq!(run_perft(4), 244);
}

#[test]
fn perft_5() {
    assert_eq!(run_perft(5), 1396);
}

#[test]
fn perft_6() {
    assert_eq!(run_perft(6), 8200);
}

#[test]
fn perft_7() {
    assert_eq!(run_perft(7), 55092);
}

#[test]
fn perft_8() {
    assert_eq!(run_perft(8), 390216);
}

// Passing moves begin here.
#[test]
fn perft_9() {
    assert_eq!(run_perft(9), 3005288);
}

#[test]
fn perft_10() {
    assert_eq!(run_perft(10), 24571284);
}

// Ending moves begin here.
#[test]
fn perft_11() {
    assert_eq!(run_perft(11), 212258800);
}

#[test]
fn perft_12() {
    assert_eq!(run_perft(12), 1939886636);
}

#[test]
fn perft_13() {
    assert_eq!(run_perft(13), 18429641748);
}

#[test]
fn perft_14() {
    assert_eq!(run_perft(14), 184042084512);
}
