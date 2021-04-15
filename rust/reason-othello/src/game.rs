//! Implements high-level Othello logic, as a completely safe wrapper for
//! bitboard code.

use crate::utils;
use crate::{
    bitboard::Bitboard, Board, Location, LocationList, ParseBoardError, ParseLocationError,
};
use derive_more::{Display, Error, From};
use std::fmt;
use std::ops::Not;

/// A safe interface to game logic. Slower than [`Board`], but implements a safer, checked interface.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Game {
    pub board: Board,
    pub active_player: Player,
    legal_moves: LocationList,
}

impl Game {
    /// Construct a game with an arbitrary [`Board`]. Results in invalid state
    /// if `board` is not a valid board.
    pub fn new(board: Board, active_player: Player) -> Self {
        Self {
            board,
            active_player,
            legal_moves: board.get_moves(),
        }
    }

    /// Compute the game state after a move.
    pub fn apply_move(self, mv: Move) -> Result<Self, InvalidMoveError> {
        match mv {
            Move::Pass => {
                if self.legal_moves.is_empty() {
                    Ok(Self::new(self.board.pass(), !self.active_player))
                } else {
                    Err(InvalidMoveError)
                }
            }
            Move::Piece(loc) => {
                if self.legal_moves.contains(loc) {
                    Ok(Self::new(self.board.apply_move(loc), !self.active_player))
                } else {
                    Err(InvalidMoveError)
                }
            }
        }
    }

    /// Get the legal moves out of this game state.
    pub fn get_moves(self) -> LocationList {
        self.legal_moves
    }

    /// Whether this board represents a finished game.
    pub fn is_finished(self) -> bool {
        self.legal_moves.is_empty() && self.board.just_passed
    }

    /// Compute the winner of the game, or None if the game is not finished or the result is a draw.
    pub fn winner(self) -> Option<Player> {
        if !self.is_finished() {
            return None;
        }

        let score = self.board.score_absolute_difference();
        if score == 0 {
            None
        } else if score > 0 {
            Some(self.active_player)
        } else {
            Some(!self.active_player)
        }
    }

    /// Score the game for `player` as the absolute piece count difference.
    /// Faster than [`Self::score_winner_gets_empties()`], but less common.
    pub fn score_absolute_difference(self, player: Player) -> i8 {
        if player == self.active_player {
            self.board.score_absolute_difference()
        } else {
            -1 * self.board.score_absolute_difference()
        }
    }

    /// Score the board for `player` as as the piece count difference
    /// where empty spaces are scored for the winner.
    pub fn score_winner_gets_empties(self, player: Player) -> i8 {
        if player == self.active_player {
            self.board.score_winner_gets_empties()
        } else {
            self.board.pass().score_winner_gets_empties()
        }
    }
}
#[derive(Debug, PartialEq, Error, Display)]
pub struct InvalidMoveError;

#[derive(Debug, PartialEq, Error, From, Display)]
pub enum ParseGameError {
    ParseBoardError(ParseBoardError),
    ParsePlayerError(ParsePlayerError),
}

impl fmt::Display for Game {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (black_bitboard, white_bitboard) = if self.active_player == Player::Black {
            (self.board.active_bitboard, self.board.opponent_bitboard)
        } else {
            (self.board.opponent_bitboard, self.board.active_bitboard)
        };
        let move_bitboard: Bitboard = self.legal_moves.into();

        utils::format_grid(
            black_bitboard
                .into_iter()
                .zip(white_bitboard.into_iter())
                .zip(move_bitboard.into_iter())
                .map(|pos| match pos {
                    ((true, false), false) => Player::Black.piece_char(),
                    ((false, true), false) => Player::White.piece_char(),
                    ((false, false), true) => '-',
                    ((false, false), false) => '.',
                    _ => panic!("formatting an invalid game",),
                }),
            f,
        )?;

        write!(
            f,
            "\nActive player: {} ({})",
            self.active_player,
            self.active_player.piece_char()
        )
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new(Board::default(), Player::default())
    }
}

impl std::str::FromStr for Game {
    type Err = ParseGameError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut segments = s.split(' ');
        let mut board: Board = segments
            .next()
            .ok_or(ParseGameError::from(ParseBoardError))?
            .parse()?;
        let active_player: Player = segments
            .next()
            .ok_or(ParseGameError::from(ParsePlayerError))?
            .parse()?;

        if active_player != Player::default() {
            board = board.swap_players();
        };

        Ok(Self::new(board, active_player))
    }
}

/// An Othello move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    Pass,
    Piece(Location),
}

#[derive(Debug, PartialEq, Error)]
pub struct ParseMoveError;

impl Default for Move {
    fn default() -> Self {
        Self::Pass
    }
}

impl From<Location> for Move {
    fn from(loc: Location) -> Self {
        Self::Piece(loc)
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            &Self::Pass => write!(f, "Pass"),
            &Self::Piece(loc) => write!(f, "{}", loc),
        }
    }
}

impl std::str::FromStr for Move {
    type Err = ParseMoveError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "PASS" => Ok(Self::Pass),
            other => Ok(Self::Piece(other.parse()?)),
        }
    }
}

impl fmt::Display for ParseMoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "failed to parse move; use 'pass' or notation like 'A1'")
    }
}

impl From<ParseLocationError> for ParseMoveError {
    fn from(_: ParseLocationError) -> Self {
        Self
    }
}

/// One of the two players in a game.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Player {
    Black,
    White,
}

#[derive(Debug, PartialEq, Error)]
pub struct ParsePlayerError;

impl Player {
    /// The piece type indicating this player.
    pub fn piece_char(self) -> char {
        match self {
            Player::Black => 'X',
            Player::White => 'O',
        }
    }
}

/// Gets the starting player (black).
impl Default for Player {
    fn default() -> Self {
        Self::Black
    }
}

/// Gets the other player.
impl Not for Player {
    type Output = Self;

    fn not(self) -> Self {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }
}

/// Parse "Black" or "White" into a Player object.
impl std::str::FromStr for Player {
    type Err = ParsePlayerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "BLACK" => Ok(Player::Black),
            "WHITE" => Ok(Player::White),
            _ => Err(ParsePlayerError),
        }
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Black => write!(f, "Black"),
            Self::White => write!(f, "White"),
        }
    }
}

impl fmt::Display for ParsePlayerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "failed to parse player; pass 'black' or 'white'")
    }
}
