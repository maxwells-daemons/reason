//! `reason-othello` is a fast, full-featured Othello library for engines and UIs.
//!
//! This package implements three levels of abstraction:
//!
//!  - [`bitboard`] contains the raw, unchecked operations for working with Othello boards.
//!    These are fast, but may result in inconsistent state if their contracts are not manually checked.
//!    Bitboard operations are also provided through a C FFI.
//!  - [`Board`] implements the core game logic in the same fast, unchecked way as [`bitboard`].
//!    This is suitable for use with engines.
//!  - [`Game`] is a high-level, safe interface to all of the Othello game logic.
//!    It is slower but safer and more complete than [`Board`].
#![feature(iter_intersperse, exact_size_is_empty)]

pub mod bitboard;
pub mod ffi;
pub mod test_utils;

mod board;
mod game;
mod location;
mod utils;

pub use board::*;
pub use game::*;
pub use location::*;

/// The number of spaces on one edge of an Othello board.
pub const EDGE_LENGTH: usize = 8;

/// The number of spaces on an Othello board.
pub const NUM_SPACES: usize = 64;
