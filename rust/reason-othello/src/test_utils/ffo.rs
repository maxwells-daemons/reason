//! Utilities for loading positions from the
//! [FFO endgame test suite](http://www.radagast.se/othello/ffotest.html).

use crate::game::{Game, Move, ParsePlayerError, Player};
use crate::{Board, Location, ParseBoardError};
use derive_more::{Display, Error};
use std::fs::File;
use std::io;
use std::io::prelude::*;

#[derive(Clone, Copy)]
/// A single FFO endgame test position.
pub struct FFOPosition {
    pub game: Game,
    pub best_move: Move,
    pub score: i8,
}

#[derive(Debug, PartialEq, Error, Display)]
pub enum LoadFFOError {
    MissingBoard,
    CannotParseBoard,
    CannotParsePlayer,
    CannotParseMove,
    CannotParseScore,
    CannotReadFile,
}

impl From<ParsePlayerError> for LoadFFOError {
    fn from(_: ParsePlayerError) -> Self {
        LoadFFOError::CannotParsePlayer
    }
}

impl From<ParseBoardError> for LoadFFOError {
    fn from(_: ParseBoardError) -> Self {
        LoadFFOError::CannotParseBoard
    }
}

/// Load all of the [`FFOPosition`]s in the file at `path`.
pub fn load_ffo_positions(path: &str) -> Result<Vec<FFOPosition>, LoadFFOError> {
    let file = File::open(path).unwrap();
    let reader = io::BufReader::new(file);

    reader
        .lines()
        .map(|line| line.or(Err(LoadFFOError::CannotReadFile))?.parse())
        .collect()
}

impl std::str::FromStr for FFOPosition {
    type Err = LoadFFOError;

    fn from_str(ffo_string: &str) -> Result<Self, Self::Err> {
        let mut sections = ffo_string.split_whitespace();

        let board_str = sections.next().ok_or(LoadFFOError::MissingBoard)?;

        let player: Player = sections
            .next()
            .ok_or(LoadFFOError::CannotParsePlayer)?
            .to_string()
            .parse()?;

        let mv = match sections.next().ok_or(LoadFFOError::CannotParseMove)? {
            "-1" => Move::Pass,
            n => Move::Piece(Location::from_index(
                n.parse().or(Err(LoadFFOError::CannotParseMove))?,
            )),
        };
        let mut score = sections
            .next()
            .ok_or(LoadFFOError::CannotParseScore)?
            .parse()
            .or(Err(LoadFFOError::CannotParseScore))?;

        let mut board: Board = board_str.parse()?;
        if player == Player::White {
            score *= -1;
            board = board.swap_players();
        };

        let position = FFOPosition {
            game: Game::new(board, player),
            best_move: mv,
            score,
        };

        Ok(position)
    }
}
