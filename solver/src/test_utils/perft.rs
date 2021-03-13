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
