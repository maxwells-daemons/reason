use async_trait::async_trait;
use reason_othello::{Move, Player};

/// A player-facing abstract interface to a way of playing Othello with "the outside world."
#[async_trait]
pub trait Connector {
    /// Alert the [`Connector`] that we're finished with setup.
    /// This must be called before the game can progress.
    fn set_ready(&mut self);

    /// Find out what color we're playing.
    fn get_player_color(&self) -> Player;

    /// Handle a move from the active player.
    fn make_move(&mut self, mv: Move);

    /// Get the opponent's move. If we're playing first and haven't sent a move, or if the opponent
    /// failed, this may hang forever.
    async fn get_opponent_move(&mut self) -> Move;
}
