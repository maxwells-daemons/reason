#![feature(exact_size_is_empty)]

pub mod search;

use reason_othello::{Board, NUM_SPACES};

// Maximum achievable absolute-difference score.
const MAX_SCORE: i8 = NUM_SPACES as i8;

/// Solve the game, trying to determine the exact score.
/// Takes longer, but can be valuable for debugging or winning by a margin.
pub fn solve_exact(board: Board) -> i8 {
    search::window(board, -MAX_SCORE, MAX_SCORE)
}

// Solve the game, caring only about solving for a win, loss, or draw.
// Faster, but provides less information.
pub fn solve_win_loss_draw(board: Board) -> i8 {
    search::window(board, -1, 1)
}
