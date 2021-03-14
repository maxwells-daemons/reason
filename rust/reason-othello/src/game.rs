//! Implements high-level Othello logic.
//!
//! This interface acts as a safe wrapper for fast bitboard code. It should be
//! preferred in almost all cases to using [`bitboard.rs`] directly.

use crate::bitboard::{self, Bitboard};

use std::fmt::{self, Write};

pub const MAX_SCORE: i8 = 64;

/// The complete state of an Othello game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameState {
    pub board: Board,
    pub just_passed: bool,
}

/// All of the pieces currently on the board, as viewed from one player's perspective.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Board {
    active_bitboard: Bitboard,
    opponent_bitboard: Bitboard,
}

/// A location on the board or a move, including "Pass".
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Move(Bitboard);

/// A list of legal moves out of a position.
/// Acts as an iterator over move locations.
#[derive(Clone, Copy, Default, Debug, Eq, PartialEq)]
pub struct MoveList(Bitboard);

/// One of the two players in a game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Player {
    Black,
    White,
}

#[derive(Debug, PartialEq)]
pub struct ParseMoveError;

impl GameState {
    /// Compute the game state after a pass move.
    #[inline]
    pub fn pass(self) -> Self {
        Self {
            board: self.board.pass(),
            just_passed: true,
        }
    }

    /// Compute the game state after a non-pass move.
    #[inline]
    pub fn apply_move(self, mv: Move) -> Self {
        Self {
            board: self.board.apply_move(mv),
            just_passed: false,
        }
    }
}

impl Default for GameState {
    /// Gets the starting board from the starting player's perspective.
    fn default() -> Self {
        Self {
            board: Board::starting_board(Player::default()),
            just_passed: false,
        }
    }
}

impl Board {
    pub const EDGE_LENGTH: u8 = 8;
    pub const NUM_SPACES: u8 = 64;

    /// Construct a Board from bitboards for the active player and the opponent.
    pub const fn from_perspective_bitboards(
        active_bitboard: Bitboard,
        opponent_bitboard: Bitboard,
    ) -> Self {
        Self {
            active_bitboard,
            opponent_bitboard,
        }
    }

    /// Construct a Board from a player's perspective, given the color bitboards.
    pub const fn from_color_bitboards(
        black_bitboard: Bitboard,
        white_bitboard: Bitboard,
        player: Player,
    ) -> Self {
        match player {
            Player::Black => Self {
                active_bitboard: black_bitboard,
                opponent_bitboard: white_bitboard,
            },
            Player::White => Self {
                active_bitboard: white_bitboard,
                opponent_bitboard: black_bitboard,
            },
        }
    }

    /// Get the starting board from a player's perspective.
    pub const fn starting_board(player: Player) -> Self {
        const BLACK_BITBOARD: Bitboard = Bitboard(0x0000000810000000);
        const WHITE_BITBOARD: Bitboard = Bitboard(0x0000001008000000);
        Self::from_color_bitboards(BLACK_BITBOARD, WHITE_BITBOARD, player)
    }

    /// Compute a MoveList of all legal moves out of this position by the active player.
    #[inline]
    pub fn get_moves(self) -> MoveList {
        MoveList(bitboard::get_move_mask(
            self.active_bitboard,
            self.opponent_bitboard,
        ))
    }

    /// Compute the board after a pass move.
    #[inline]
    pub fn pass(self) -> Self {
        Self {
            active_bitboard: self.opponent_bitboard,
            opponent_bitboard: self.active_bitboard,
        }
    }

    /// Compute the board after a non-pass move.
    #[inline]
    pub fn apply_move(self, mv: Move) -> Self {
        let (new_active, new_opponent) =
            bitboard::apply_move(self.active_bitboard, self.opponent_bitboard, mv.0);
        Self {
            active_bitboard: new_opponent,
            opponent_bitboard: new_active,
        }
    }

    /// Score the board for the active player as: # my pieces - # opponent pieces.
    /// Faster than [`score_winner_gets_empties()`], but less common.
    #[inline]
    pub fn score_absolute_difference(self) -> i8 {
        bitboard::score_absolute_difference(self.active_bitboard, self.opponent_bitboard)
    }

    /// Score the board for the active player as as:
    /// # my spaces - # opponent spaces, where empty spaces are scored for the winner.
    #[inline]
    pub fn score_winner_gets_empties(self) -> i8 {
        bitboard::score_winner_gets_empties(self.active_bitboard, self.opponent_bitboard)
    }

    /// Count the number of empty spaces on the board.
    #[inline]
    pub fn count_empties(self) -> i8 {
        bitboard::count_empties(self.active_bitboard, self.opponent_bitboard)
    }
}

impl Move {
    pub const PASS: Self = Self(Bitboard(0));

    #[inline]
    pub fn is_pass(self) -> bool {
        self.0 .0 == 0
    }

    /// Construct a Move from a "move index": 0 for the bottom right, 63 for the top left.
    #[inline]
    pub fn from_index(index: u8) -> Self {
        Self(Bitboard(1 << index))
    }

    /// Convert a Move to a "move index": 0 for the bottom right, 63 for the top left.
    /// Undefined for PASS moves.
    #[inline]
    pub fn to_index(self) -> u8 {
        self.0 .0.trailing_zeros() as u8
    }

    /// Construct a Move from row and column coordinates.
    /// Returns None if the coordinates provided are not valid.
    pub fn from_coords(row: u8, col: u8) -> Option<Self> {
        if row > 7 || col > 7 {
            None
        } else {
            Some(Self::from_index(
                (7 - row) + ((7 - col) * Board::EDGE_LENGTH),
            ))
        }
    }

    /// Get the row and column a Move represents.
    /// Undefined for PASS moves.
    pub fn to_coords(self) -> (u8, u8) {
        let index = self.to_index();
        let row = 7 - (index % Board::EDGE_LENGTH);
        let col = 7 - (index.wrapping_div(Board::EDGE_LENGTH));
        (row, col)
    }
}

/// Build a Move from a 1-indexed string notation ("A4"; "PASS").
/// Returns None if the string is not valid notation.
/// Same behavior as FromString.
impl std::str::FromStr for Move {
    type Err = ParseMoveError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "PASS" {
            return Ok(Self::PASS);
        }

        let mut chars = s.chars();
        let col_str = chars.next().ok_or(ParseMoveError)?.to_ascii_uppercase();
        let col = "ABCDEFGH".find(col_str).ok_or(ParseMoveError)? as u8;
        let row = chars
            .next()
            .ok_or(ParseMoveError)?
            .to_digit(10)
            .ok_or(ParseMoveError)? as u8;

        if chars.next() != None {
            return Err(ParseMoveError);
        }

        Self::from_coords(row - 1, col).ok_or(ParseMoveError)
    }
}

/// Convert this Move into string notation ("A4" / "PASS").
impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_pass() {
            f.write_str("PASS")
        } else {
            let (row, col) = self.to_coords();
            let row_str = "12345678".chars().nth(row as usize).unwrap();
            let col_str = "ABCDEFGH".chars().nth(col as usize).unwrap();
            f.write_char(col_str)?;
            f.write_char(row_str)
        }
    }
}

impl MoveList {
    /// Returns whether the move list is empty.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 .0 == 0
    }
}

impl Iterator for MoveList {
    type Item = Move;

    fn next(&mut self) -> Option<Move> {
        if self.is_empty() {
            return None;
        }

        let next_move = 1 << self.0 .0.trailing_zeros();
        self.0 .0 ^= next_move;
        Some(Move(Bitboard(next_move)))
    }
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

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Selects the highest bit for both players
        const PIECE_MASK: u64 = 1 << 63;

        let mut player_bitboard = self.active_bitboard.0;
        let mut opponent_bitboard = self.opponent_bitboard.0;
        let mut my_piece: u64;
        let mut opponent_piece: u64;

        writeln!(f, "\n  A B C D E F G H")?;
        for row in 1..Self::EDGE_LENGTH + 1 {
            write!(f, "{} ", row)?;
            for _ in 1..Self::EDGE_LENGTH + 1 {
                my_piece = player_bitboard & PIECE_MASK;
                opponent_piece = opponent_bitboard & PIECE_MASK;

                match (my_piece, opponent_piece) {
                    (0, 0) => write!(f, ". "),
                    (PIECE_MASK, 0) => write!(f, "# "),
                    (0, PIECE_MASK) => write!(f, "O "),
                    _ => Err(fmt::Error),
                }?;

                player_bitboard <<= 1;
                opponent_bitboard <<= 1;
            }
            writeln!(f)?;
        }
        Ok(())
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

impl fmt::Display for MoveList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let string = self
            .into_iter()
            .map(|mv| mv.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        f.write_fmt(format_args!("[{}]", string))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn location_from_index() {
        assert_eq!(Move::from_index(0), Move(Bitboard(1)));
        assert_eq!(Move::from_index(63), Move(Bitboard(1 << 63)));
    }

    #[test]
    fn location_to_index() {
        assert_eq!(Move(Bitboard(1)).to_index(), 0);
        assert_eq!(Move(Bitboard(1 << 63)).to_index(), 63);
    }

    #[test]
    fn location_from_coords() {
        assert_eq!(Move::from_coords(0, 0), Some(Move(Bitboard(1 << 63))));
        assert_eq!(Move::from_coords(7, 7), Some(Move(Bitboard(1))));
        assert_eq!(Move::from_coords(0, 8), None);
        assert_eq!(Move::from_coords(8, 0), None);
    }

    #[test]
    fn location_to_coords() {
        assert_eq!(Move(Bitboard(1 << 63)).to_coords(), (0, 0));
        assert_eq!(Move(Bitboard(1)).to_coords(), (7, 7));
    }

    #[test]
    fn location_from_str_success() {
        assert_eq!(Move::from_str("A1"), Ok(Move(Bitboard(1 << 63))));
        assert_eq!(Move::from_str("h8"), Ok(Move(Bitboard(1))));
        assert_eq!(Move::from_str("D7"), Ok(Move::from_coords(6, 3).unwrap()));
    }

    #[test]
    fn location_from_str_fail() {
        assert_eq!(Move::from_str(""), Err(ParseMoveError));
        assert_eq!(Move::from_str("A12"), Err(ParseMoveError));
        assert_eq!(Move::from_str("AA"), Err(ParseMoveError));
        assert_eq!(Move::from_str("A9"), Err(ParseMoveError));
        assert_eq!(Move::from_str("I5"), Err(ParseMoveError));
    }

    #[test]
    fn location_to_str() {
        assert_eq!(Move(Bitboard(1)).to_string(), "H8");
        assert_eq!(Move(Bitboard(1 << 63)).to_string(), "A1");
        assert_eq!(Move::from_str("E2").unwrap().to_string(), "E2");
        assert_eq!(Move::from_str("F6").unwrap().to_string(), "F6");
    }
}
