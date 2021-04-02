//! Code for working with [`Location`]s on the Othello board.

use crate::bitboard::Bitboard;
use crate::EDGE_LENGTH;
use derive_more::{From, Into};
use std::fmt::{self, Display, Formatter, Write};

/// A location on the Othello board.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Into)]
pub struct Location(Bitboard);

/// A list of locations on the Othello board, which can be iterated to retrieve them.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, PartialOrd, Ord, From, Into)]
pub struct LocationList(Bitboard);

impl Location {
    /// Convert from a one-hot [`Bitboard`].
    #[inline]
    pub fn from_onehot(bitboard: Bitboard) -> Self {
        assert_eq!(bitboard.count_occupied(), 1);
        Self::from_onehot_unchecked(bitboard)
    }

    /// Convert from a one-hot [`Bitboard`] without checking this invariant.
    /// Results in inconsistent state if `bitboard` has more than one location set.
    #[inline]
    pub fn from_onehot_unchecked(bitboard: Bitboard) -> Self {
        Self(bitboard)
    }

    /// Convert from a row-major square index.
    #[inline]
    pub fn from_index(index: u8) -> Self {
        Self(Bitboard::from(1 << index))
    }

    /// Convert into a row-major square index.
    #[inline]
    pub fn to_index(self) -> u8 {
        let bitboard: u64 = self.0.into();
        bitboard.trailing_zeros() as u8
    }

    /// Convert from row and column coordinates.
    pub fn from_coords(row: usize, col: usize) -> Self {
        assert!(row <= 7 && col <= 7);
        let index = (7 - col) + ((7 - row) * EDGE_LENGTH);
        Self::from_index(index as u8)
    }

    /// Get the row and column coordinates.
    pub fn to_coords(self) -> (usize, usize) {
        let index = self.to_index() as usize;
        let row = 7 - (index % EDGE_LENGTH);
        let col = 7 - (index.wrapping_div(EDGE_LENGTH));
        (col, row)
    }
}

/// Convert this [`Location`] into string notation ("A4").
impl Display for Location {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let (row, col) = self.to_coords();
        let row_str = "12345678".chars().nth(row as usize).ok_or(fmt::Error)?;
        let col_str = "ABCDEFGH".chars().nth(col as usize).ok_or(fmt::Error)?;
        f.write_char(col_str)?;
        f.write_char(row_str)
    }
}

impl LocationList {
    /// Returns whether `loc` is in this list.
    pub fn contains(self, loc: Location) -> bool {
        let loc_bitboard: Bitboard = loc.into();
        !(loc_bitboard & self.0).is_empty()
    }
}

#[derive(Debug, PartialEq)]
pub struct ParseLocationError;

impl Display for ParseLocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid location string")
    }
}

impl std::error::Error for ParseLocationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

/// Build a [`Location`] from a 1-indexed string notation ("A4"; "PASS").
/// Returns None if the string is not valid notation.
impl std::str::FromStr for Location {
    type Err = ParseLocationError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut chars = s.chars();
        let col_str = chars.next().ok_or(ParseLocationError)?.to_ascii_uppercase();
        let col = "ABCDEFGH".find(col_str).ok_or(ParseLocationError)? as usize;
        let row = chars
            .next()
            .ok_or(ParseLocationError)?
            .to_digit(10)
            .ok_or(ParseLocationError)? as usize;

        if (row > 8) || (col > 7) || chars.next() != None {
            return Err(ParseLocationError);
        }

        Ok(Self::from_coords(row - 1, col))
    }
}

impl ExactSizeIterator for LocationList {
    fn len(&self) -> usize {
        self.0.count_occupied() as usize
    }
}

impl Iterator for LocationList {
    type Item = Location;

    fn next(&mut self) -> Option<Location> {
        if self.is_empty() {
            return None;
        }

        let bitboard: u64 = self.0.into();
        let next_move: Bitboard = (1 << bitboard.trailing_zeros()).into();
        self.0 ^= next_move;

        Some(Location::from_onehot_unchecked(next_move))
    }
}

impl Display for LocationList {
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
        assert_eq!(Location::from_index(0), Location(Bitboard::from(1)));
        assert_eq!(Location::from_index(63), Location(Bitboard::from(1 << 63)));
    }

    #[test]
    fn location_to_index() {
        assert_eq!(Location(Bitboard::from(1)).to_index(), 0);
        assert_eq!(Location(Bitboard::from(1 << 63)).to_index(), 63);
    }

    #[test]
    fn location_from_coords() {
        assert_eq!(
            Location::from_coords(0, 0),
            Location(Bitboard::from(1 << 63))
        );
        assert_eq!(Location::from_coords(7, 7), Location(Bitboard::from(1)));
    }

    #[test]
    #[should_panic]
    fn location_from_coords_fail() {
        Location::from_coords(0, 8);
    }

    #[test]
    fn location_to_coords() {
        assert_eq!(Location(Bitboard::from(1 << 63)).to_coords(), (0, 0));
        assert_eq!(Location(Bitboard::from(1)).to_coords(), (7, 7));
    }

    #[test]
    fn location_from_str_success() {
        assert_eq!(
            Location::from_str("A1"),
            Ok(Location(Bitboard::from(1 << 63)))
        );
        assert_eq!(Location::from_str("h8"), Ok(Location(Bitboard::from(1))));
        assert_eq!(Location::from_str("D7"), Ok(Location::from_coords(6, 3)));
    }

    #[test]
    fn location_from_str_fail() {
        assert_eq!(Location::from_str(""), Err(ParseLocationError));
        assert_eq!(Location::from_str("A12"), Err(ParseLocationError));
        assert_eq!(Location::from_str("AA"), Err(ParseLocationError));
        assert_eq!(Location::from_str("A9"), Err(ParseLocationError));
        assert_eq!(Location::from_str("I5"), Err(ParseLocationError));
    }

    #[test]
    fn location_to_str() {
        assert_eq!(Location(Bitboard::from(1)).to_string(), "H8");
        assert_eq!(Location(Bitboard::from(1 << 63)).to_string(), "A1");
        assert_eq!(Location::from_str("E2").unwrap().to_string(), "E2");
        assert_eq!(Location::from_str("F6").unwrap().to_string(), "F6");
    }
}
