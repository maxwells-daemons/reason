pub mod search;

use reason_othello::game::{self, GameState};

/// Solve the game, trying to determine the exact score.
/// Takes longer, but can be valuable for debugging or winning by a margin.
pub fn solve_exact(state: GameState) -> i8 {
    search::window(state, -game::MAX_SCORE, game::MAX_SCORE)
}

// Solve the game, caring only about solving for a win, loss, or draw.
// Faster, but provides less information.
pub fn solve_win_loss_draw(state: GameState) -> i8 {
    search::window(state, -1, 1)
}
