//! Internal search functions.

use arrayvec::ArrayVec;
use reason_othello::game::GameState;

/// TODO: document
pub fn window(state: GameState, alpha: i8, beta: i8) -> i8 {
    window_fastest_first(state, state.board.count_empties(), alpha, beta)
}

/// Window search, using "fastest first" move ordering which first
/// explores moves where the opponent has the fewest legal moves.
fn window_fastest_first(state: GameState, empties: u8, mut alpha: i8, beta: i8) -> i8 {
    // Below this depth, stop sorting moves.
    const MAX_SORT_DEPTH: u8 = 6;

    if empties < MAX_SORT_DEPTH {
        return window_unsorted(state, empties, alpha, beta);
    }

    let moves = state.board.get_moves();
    if moves.is_empty() {
        // Both players pass: game ends
        if state.just_passed {
            return state.board.score_absolute_difference();
        }

        // I pass, but my opponent may have moves
        return -window_fastest_first(state.pass(), empties, -beta, -alpha);
    }

    // Precompute all next states and their moves
    let mut next_states: ArrayVec<[_; 32]> = moves.map(|mv| state.apply_move(mv)).collect();
    next_states.sort_unstable_by_key(|s| s.board.get_moves().num_moves());

    // Visit states by lowest-mobility first
    for next_state in next_states {
        let score = -window_fastest_first(next_state, empties - 1, -beta, -alpha);

        // Fail high: this branch has a line so good for me my opponent won't allow it.
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
fn window_unsorted(state: GameState, empties: u8, mut alpha: i8, beta: i8) -> i8 {
    let moves = state.board.get_moves();
    if moves.is_empty() || empties == 0 {
        if state.just_passed || empties == 0 {
            return state.board.score_absolute_difference();
        }

        return -window_unsorted(state.pass(), empties, -beta, -alpha);
    }

    for mv in moves {
        let score = -window_unsorted(state.apply_move(mv), empties - 1, -beta, -alpha);

        if score >= beta {
            return beta;
        }

        if score > alpha {
            alpha = score
        }
    }

    alpha
}
