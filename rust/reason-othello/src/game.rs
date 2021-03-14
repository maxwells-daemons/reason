//! Implements game-level Othello logic.
//!
//! For correctness, this higher-level interface is preferred, but for
//! performance you may use [`board.rs`] for raw bitboard access.

use crate::board::{Bitboard, Board, Location, MoveList};
use std::fmt;

/// One of the two players in a game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Player {
    Black,
    White,
}

impl Default for Player {
    /// Gets the starting player (black).
    fn default() -> Self {
        Self::Black
    }
}

impl std::ops::Not for Player {
    type Output = Self;

    /// Gets the other player.
    fn not(self) -> Self {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }
}

/// An action in an Othello game: pass or select a location.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    MakeMove(Location),
    Pass,
}

impl From<Location> for Action {
    fn from(mv: Location) -> Self {
        Self::MakeMove(mv)
    }
}

/// The complete state of an Othello game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameState {
    pub board: Board,
    pub just_passed: bool,
}

impl Default for GameState {
    /// Gets the starting board from the starting player's perspective.
    fn default() -> Self {
        Self::starting_board(Player::default())
    }
}

impl GameState {
    /// Construct the game state as seen by one player from a set of color bitboards.
    const fn for_player(
        player: Player,
        black_bitboard: Bitboard,
        white_bitboard: Bitboard,
        just_passed: bool,
    ) -> Self {
        match player {
            Player::Black => Self {
                board: Board {
                    player_bitboard: black_bitboard,
                    opponent_bitboard: white_bitboard,
                },
                just_passed,
            },
            Player::White => Self {
                board: Board {
                    player_bitboard: white_bitboard,
                    opponent_bitboard: black_bitboard,
                },
                just_passed,
            },
        }
    }

    /// Construct the starting game state as seen by one player.
    const fn starting_board(player: Player) -> Self {
        const BLACK_BITBOARD: u64 = 0x0000000810000000;
        const WHITE_BITBOARD: u64 = 0x0000001008000000;
        Self::for_player(player, BLACK_BITBOARD, WHITE_BITBOARD, false)
    }

    /// Get the list of actions available for the active player.
    #[inline]
    pub fn get_moves(self) -> MoveList {
        self.board.get_moves()
    }

    /// Make a pass move for the active player.
    #[inline]
    pub fn pass(self) -> Self {
        Self {
            board: self.board.pass(),
            just_passed: true,
        }
    }

    /// Make a placement move for the active player.
    #[inline]
    pub fn make_move(self, loc: Location) -> Self {
        Self {
            board: self.board.make_move(loc),
            just_passed: false,
        }
    }

    /// Make an action as the active player.
    #[inline]
    pub fn act(self, action: Action) -> Self {
        match action {
            Action::Pass => self.pass(),
            Action::MakeMove(mv) => self.make_move(mv),
        }
    }
}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.board.to_string())?;
        if self.just_passed {
            f.write_str("(Last move was a pass)\n")?;
        }
        Ok(())
    }
}
