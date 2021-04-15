//! Code for working with Othello boards at a medium level of abstraction.
use crate::bitboard::{self, Bitboard};
use crate::{utils, Location, LocationList, NUM_SPACES};
use derive_more::Error;
use std::fmt;

/// The complete state of an Othello game, seen from one player's perspective.
///
/// Acts as a convenient thin wrapper for [`bitboard`] operations, at a medium
/// level of abstraction ideal for engines.
///
/// [`Board`] operations are guaranteed to preserve validity contracts, but cannot
/// check them on creation; may cause undefined behavior if invalid data is passed
/// or if [`Board::pass`] and [`Board::apply_move`] are applied when they are not allowed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Board {
    pub active_bitboard: Bitboard,
    pub opponent_bitboard: Bitboard,
    pub just_passed: bool,
}

impl Board {
    /// Compute a list of all legal moves out of this position by the active player.
    #[inline]
    pub fn get_moves(self) -> LocationList {
        LocationList::from(bitboard::get_move_mask(
            self.active_bitboard,
            self.opponent_bitboard,
        ))
    }

    /// Compute the board after a non-pass move, without checking that this is valid.
    /// Undefined behavior if the location is occupied or not a valid move.
    #[inline]
    pub fn apply_move(self, loc: Location) -> Self {
        let (new_active, new_opponent) =
            bitboard::apply_move(self.active_bitboard, self.opponent_bitboard, loc.into());
        Self {
            active_bitboard: new_opponent,
            opponent_bitboard: new_active,
            just_passed: false,
        }
    }

    /// Compute the board after a pass.
    /// Results in inconsistent state if the active player has legal moves.
    #[inline]
    pub fn pass(self) -> Self {
        Self {
            active_bitboard: self.opponent_bitboard,
            opponent_bitboard: self.active_bitboard,
            just_passed: true,
        }
    }

    /// Compute the board from the opponent's perspective.
    #[inline]
    pub fn swap_players(self) -> Self {
        Self {
            active_bitboard: self.opponent_bitboard,
            opponent_bitboard: self.active_bitboard,
            ..self
        }
    }

    /// Score the board for the active player as the absolute piece count difference.
    /// Faster than [`Self::score_winner_gets_empties()`], but less common.
    #[inline]
    pub fn score_absolute_difference(self) -> i8 {
        bitboard::score_absolute_difference(self.active_bitboard, self.opponent_bitboard)
    }

    /// Score the board for the active player as as the piece count difference
    /// where empty spaces are scored for the winner.
    #[inline]
    pub fn score_winner_gets_empties(self) -> i8 {
        bitboard::score_winner_gets_empties(self.active_bitboard, self.opponent_bitboard)
    }

    /// Get a mask of occupied spaces on the board.
    #[inline]
    pub fn occupied_mask(self) -> Bitboard {
        self.active_bitboard | self.opponent_bitboard
    }

    /// Get a mask of unoccupied spaces on the board.
    #[inline]
    pub fn empty_mask(self) -> Bitboard {
        !self.occupied_mask()
    }
}

/// Get the starting board for the starting player.
impl Default for Board {
    fn default() -> Self {
        Self {
            active_bitboard: bitboard::BLACK_START,
            opponent_bitboard: bitboard::WHITE_START,
            just_passed: false,
        }
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let move_bitboard: Bitboard = self.get_moves().into();

        utils::format_grid(
            self.active_bitboard
                .into_iter()
                .zip(self.opponent_bitboard.into_iter())
                .zip(move_bitboard.into_iter())
                .map(|pos| match pos {
                    ((true, false), false) => 'X',  // Active space
                    ((false, true), false) => 'O',  // Opponent space
                    ((false, false), true) => '-',  // Legal move
                    ((false, false), false) => '.', // Empty space
                    _ => panic!("formatting an invalid bitboard"),
                }),
            f,
        )
    }
}

#[derive(Debug, PartialEq, Error)]
pub struct ParseBoardError;

impl fmt::Display for ParseBoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "could not parse board; please use FFO format (O/X/-)")
    }
}

impl std::str::FromStr for Board {
    type Err = ParseBoardError;

    /// Parse a board in the FFO format, assuming Black is going first.
    fn from_str(board_str: &str) -> Result<Self, Self::Err> {
        if board_str.len() != (NUM_SPACES as usize) {
            return Err(ParseBoardError);
        }

        let mut active_bitboard: u64 = 0;
        let mut opponent_bitboard: u64 = 0;

        for char in board_str.chars() {
            active_bitboard <<= 1;
            opponent_bitboard <<= 1;

            match char {
                'X' => (active_bitboard |= 1),
                'O' => (opponent_bitboard |= 1),
                '-' => (),
                _ => return Err(ParseBoardError),
            };
        }

        Ok(Board {
            active_bitboard: Bitboard::from(active_bitboard),
            opponent_bitboard: Bitboard::from(opponent_bitboard),
            just_passed: false,
        })
    }
}
