//! "Perft" performance test: count the number of leaves at a given depth.
//! Useful for tuning bitboard.
//! See: http://www.aartbik.com/MISC/reversi.html

use crate::board::Board;

pub fn run_perft(depth: u64) -> u64 {
    leaves_below(Board::new(), depth, false)
}

fn leaves_below(board: Board, depth: u64, passed: bool) -> u64 {
    // Leaf node for this depth
    if depth == 0 {
        return 1;
    }

    let all_moves = board.get_moves();
    if all_moves.is_empty() {
        // Both players passed: game is over
        if passed {
            return 1;
        }

        return leaves_below(board.pass(), depth - 1, true);
    }

    all_moves
        .map(|mv| leaves_below(board.make_move(mv), depth - 1, false))
        .sum()
}

#[test]
fn perft_01() {
    assert_eq!(run_perft(1), 4);
}

#[test]
fn perft_02() {
    assert_eq!(run_perft(2), 12);
}

#[test]
fn perft_03() {
    assert_eq!(run_perft(3), 56);
}

#[test]
fn perft_04() {
    assert_eq!(run_perft(4), 244);
}

#[test]
fn perft_05() {
    assert_eq!(run_perft(5), 1396);
}

#[test]
fn perft_06() {
    assert_eq!(run_perft(6), 8200);
}

#[test]
fn perft_07() {
    assert_eq!(run_perft(7), 55092);
}

#[test]
fn perft_08() {
    assert_eq!(run_perft(8), 390216);
}

// Passing moves begin here.
#[test]
fn perft_09() {
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
