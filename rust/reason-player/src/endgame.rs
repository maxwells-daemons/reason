//! Internal search functions for the endgame solver.

use arrayvec::ArrayVec;
use reason_othello::Board;

/// TODO: document
pub fn window(board: Board, alpha: i8, beta: i8) -> i8 {
    window_fastest_first(board, board.occupied_mask().count_empty(), alpha, beta)
}

/// Window search, using "fastest first" move ordering which first
/// explores moves where the opponent has the fewest legal moves.
fn window_fastest_first(board: Board, empties: u8, mut alpha: i8, beta: i8) -> i8 {
    // Below this depth, stop sorting moves.
    const MAX_SORT_DEPTH: u8 = 6;

    if empties < MAX_SORT_DEPTH {
        return window_unsorted(board, empties, alpha, beta);
    }

    let moves = board.get_moves();
    if moves.is_empty() {
        // Both players pass: game ends
        if board.just_passed {
            return board.score_absolute_difference();
        }

        // I pass, but my opponent may have moves
        return -window_fastest_first(board.pass(), empties, -beta, -alpha);
    }

    // Precompute all next states and their moves
    let mut next_states: ArrayVec<[_; 32]> = moves.map(|loc| board.apply_move(loc)).collect();
    next_states.sort_unstable_by_key(|s| s.get_moves().len());

    // Visit states by lowest-mobility first
    for next_state in next_states {
        let score = -window_fastest_first(next_state, empties - 1, -beta, -alpha);

        // Fail high: this branch has a line so good for me my opponent won't allow it
        if score >= beta {
            return beta;
        }

        // This branch is better than any line I could force before: update current lower bound
        if score > alpha {
            alpha = score
        }
    }

    alpha
}

/// Window search without move ordering, which is faster for shallow trees.
fn window_unsorted(board: Board, empties: u8, mut alpha: i8, beta: i8) -> i8 {
    if empties == 0 {
        return board.score_absolute_difference();
    }

    let moves = board.get_moves();
    if moves.is_empty() {
        if board.just_passed {
            return board.score_absolute_difference();
        }

        return -window_unsorted(board.pass(), empties, -beta, -alpha);
    }

    for mv in moves {
        let score = -window_unsorted(board.apply_move(mv), empties - 1, -beta, -alpha);

        if score >= beta {
            return beta;
        }

        if score > alpha {
            alpha = score
        }
    }

    alpha
}
