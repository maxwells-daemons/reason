//! A highly-efficient low-level implementation of Othello board dynamics.
//!
//! Under the hood, all types are u64 bitboards. This is intended to be used
//! through [`game.rs`], but hot-loop code can use bitboards directly.
//! By convention, the MSB is the upper-left of the board, and uses row-major order.

use packed_simd::{u64x4, u64x8};
use std::fmt::{self, Write};

/// Holds a single player's pieces on the board in a packed format.
pub type Bitboard = u64;

/// A pair of bitboards storing the complete game state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Board {
    pub player_bitboard: Bitboard,
    pub opponent_bitboard: Bitboard,
}

/// Stores a single location on the board as a one-hot vector, or all zeros for a Pass move.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Location(Bitboard);

/// Stores a list of legal moves out of a position as a bitboard mask.
/// Can be iterated to retrieve a list of move locations.
#[derive(Clone, Copy, Default, Debug, Eq, PartialEq)]
pub struct MoveList(Bitboard);

impl Board {
    pub const EDGE_LENGTH: u8 = 8;
    pub const NUM_SPACES: u8 = 64;

    #[inline]
    pub fn pass(self) -> Self {
        Self {
            player_bitboard: self.opponent_bitboard,
            opponent_bitboard: self.player_bitboard,
        }
    }

    /// Get a mask of the legal moves for the active player.
    // Algorithm adapted from Sam Blazes' Coin, released under the Apache 2.0 license:
    // https://github.com/Tenebryo/coin/blob/master/bitboard/src/find_moves_fast.rs
    #[inline]
    pub fn get_moves(self) -> MoveList {
        // Shifts for each vectorized direction: E/W, N/S, NW/SE, NE/SW.
        // The first direction is handled by SHL, the second by SHR.
        const SHIFTS_1: u64x4 = u64x4::new(1, 8, 7, 9);
        const SHIFTS_2: u64x4 = u64x4::new(2, 16, 14, 18);
        const SHIFTS_4: u64x4 = u64x4::new(4, 32, 28, 36);

        // Mask to clip off the invalid wraparound pieces on the edge.
        const EDGE_MASK: Bitboard = 0x7E7E7E7E7E7E7E7Eu64;

        let opponent_edge_mask = EDGE_MASK & self.opponent_bitboard;

        // Masks used for each shift direction. The same for SHL and SHR.
        let masks: u64x4 = u64x4::new(
            opponent_edge_mask,
            self.opponent_bitboard,
            opponent_edge_mask,
            opponent_edge_mask,
        );

        // Pieces we flip while shifting along each direction.
        let mut flip_l = u64x4::splat(self.player_bitboard);
        let mut flip_r = flip_l;

        // Accumulated mask when shifting along each direction.
        let mut masks_l = masks & (masks << SHIFTS_1);
        let mut masks_r = masks & (masks >> SHIFTS_1);

        // Smear our pieces in each direction while masking invalid flips.
        flip_l |= masks & (flip_l << SHIFTS_1);
        flip_l |= masks_l & (flip_l << SHIFTS_2);
        masks_l &= masks_l << SHIFTS_2;
        flip_l |= masks_l & (flip_l << SHIFTS_4);

        flip_r |= masks & (flip_r >> SHIFTS_1);
        flip_r |= masks_r & (flip_r >> SHIFTS_2);
        masks_r &= masks_r >> SHIFTS_2;
        flip_r |= masks_r & (flip_r >> SHIFTS_4);

        // Moves are the union of empties and one extra shift of all flipped locations.
        let empties = !(self.player_bitboard | self.opponent_bitboard);
        let captures_l = (flip_l & masks) << SHIFTS_1;
        let captures_r = (flip_r & masks) >> SHIFTS_1;

        MoveList(empties & (captures_l | captures_r).or())
    }

    /// Make a move for the active player, given a move mask.
    #[inline]
    pub fn make_move(self, loc: Location) -> Self {
        // Masks selecting everything except the far-left and far-right columns.
        const NOT_A_FILE: Bitboard = 0xfefefefefefefefe;
        const NOT_H_FILE: Bitboard = 0x7f7f7f7f7f7f7f7f;
        const FULL_MASK: Bitboard = 0xffffffffffffffff;

        // Masks applied when left-shifting or right-shifting.
        const LEFT_MASKS: u64x8 = u64x8::new(
            NOT_A_FILE, FULL_MASK, NOT_H_FILE, NOT_A_FILE, NOT_A_FILE, FULL_MASK, NOT_H_FILE,
            NOT_A_FILE,
        );
        const RIGHT_MASKS: u64x8 = u64x8::new(
            NOT_H_FILE, FULL_MASK, NOT_A_FILE, NOT_H_FILE, NOT_H_FILE, FULL_MASK, NOT_A_FILE,
            NOT_H_FILE,
        );

        // Shifts for each vectorized direction: E/W, N/S, NW/SE, NE/SW.
        // The first direction is handled by SHL, the second by SHR.
        const SHIFTS_1: u64x8 = u64x8::new(1, 8, 7, 9, 1, 8, 7, 9);
        const SHIFTS_2: u64x8 = u64x8::new(2, 16, 14, 18, 2, 16, 14, 18);
        const SHIFTS_4: u64x8 = u64x8::new(4, 32, 28, 36, 4, 32, 28, 36);

        // Generators and propagators for all LSH directions: E, N, NW, NE.
        // Two concatenated vectors: [new + opponent interactions, self + opponent interactions].
        let mut gen_left = u64x8::new(
            loc.0,
            loc.0,
            loc.0,
            loc.0,
            self.player_bitboard,
            self.player_bitboard,
            self.player_bitboard,
            self.player_bitboard,
        );
        let mut pro_left = u64x8::splat(self.opponent_bitboard);

        // Generators and propagators for all RSH directions: W, S, SE, SW.
        // Two concatenated vectors: [self + opponent interactions, new + opponent interactions].
        let mut gen_right = u64x8::new(
            self.player_bitboard,
            self.player_bitboard,
            self.player_bitboard,
            self.player_bitboard,
            loc.0,
            loc.0,
            loc.0,
            loc.0,
        );
        let mut pro_right = pro_left;

        // Compute all LSH propagations.
        pro_left &= LEFT_MASKS;
        gen_left |= pro_left & (gen_left << SHIFTS_1);
        pro_left &= pro_left << SHIFTS_1;
        gen_left |= pro_left & (gen_left << SHIFTS_2);
        pro_left &= pro_left << SHIFTS_2;
        gen_left |= pro_left & (gen_left << SHIFTS_4);

        // Compute all RSH propagations.
        pro_right &= RIGHT_MASKS;
        gen_right |= pro_right & (gen_right >> SHIFTS_1);
        pro_right &= pro_right >> SHIFTS_1;
        gen_right |= pro_right & (gen_right >> SHIFTS_2);
        pro_right &= pro_right >> SHIFTS_2;
        gen_right |= pro_right & (gen_right >> SHIFTS_4);

        // gen_left : [Egp, Ngp, NWgp, NEgp, Eop, Nop, NWop, NEop]
        // gen_right: [Wop, Sop, SEop, SWop, Wgp, Sgp, SEgp, SWgp]
        let flip_mask = (gen_left & gen_right).or();

        Self {
            player_bitboard: self.opponent_bitboard ^ flip_mask,
            opponent_bitboard: (self.player_bitboard ^ flip_mask) | loc.0,
        }
    }

    /// Score a board as: # my pieces - # opponent pieces.
    /// Faster than [`score_winner_gets_empties()`], but less common.
    #[inline]
    pub fn score_absolute_difference(self) -> i8 {
        (self.player_bitboard.count_ones() as i8) - (self.opponent_bitboard.count_ones() as i8)
    }

    /// Score a board as: # my spaces - # opponent spaces, where empty spaces are scored for the winner.
    #[inline]
    pub fn score_winner_gets_empties(self) -> i8 {
        let absolute_difference = self.score_absolute_difference();
        if absolute_difference.is_positive() {
            absolute_difference + self.empty_mask() as i8
        } else if absolute_difference.is_negative() {
            absolute_difference - self.empty_mask() as i8
        } else {
            0
        }
    }

    /// Get a mask indicating where the occupied spaces are.
    #[inline]
    pub fn occupied_mask(self) -> Bitboard {
        self.player_bitboard | self.opponent_bitboard
    }

    /// Get a mask indicating where the empty spaces are.
    #[inline]
    pub fn empty_mask(self) -> Bitboard {
        !self.occupied_mask()
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Selects the highest bit for both players
        const PIECE_MASK: Bitboard = 1 << 63;

        let mut player_bitboard = self.player_bitboard;
        let mut opponent_bitboard = self.opponent_bitboard;
        let mut my_piece: Bitboard;
        let mut opponent_piece: Bitboard;

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

impl Location {
    pub const PASS: Self = Self(0);

    #[inline]
    pub fn is_pass(self) -> bool {
        self.0 == 0
    }

    /// Construct a Location from a "move index": 0 for the bottom right, 63 for the top left.
    #[inline]
    pub fn from_index(index: u8) -> Self {
        Self(1 << index)
    }

    /// Convert a Location to a "move index": 0 for the bottom right, 63 for the top left.
    /// Undefined for PASS moves.
    #[inline]
    pub fn to_index(self) -> u8 {
        self.0.trailing_zeros() as u8
    }

    /// Construct a Location from row and column coordinates.
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

    /// Get the row and column a Location represents.
    /// Undefined for PASS moves.
    pub fn to_coords(self) -> (u8, u8) {
        let index = self.to_index();
        let row = 7 - (index % Board::EDGE_LENGTH);
        let col = 7 - (index.wrapping_div(Board::EDGE_LENGTH));
        (row, col)
    }
}

#[derive(Debug, PartialEq)]
pub struct ParseLocationError;

/// Build a Location from a 1-indexed string notation ("A4"; "PASS").
/// Returns None if the string is not valid notation.
/// Same behavior as FromString.
impl std::str::FromStr for Location {
    type Err = ParseLocationError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "PASS" {
            return Ok(Self::PASS);
        }

        let mut chars = s.chars();
        let col_str = chars.next().ok_or(ParseLocationError)?.to_ascii_uppercase();
        let col = "ABCDEFGH".find(col_str).ok_or(ParseLocationError)? as u8;
        let row = chars
            .next()
            .ok_or(ParseLocationError)?
            .to_digit(10)
            .ok_or(ParseLocationError)? as u8;

        if chars.next() != None {
            return Err(ParseLocationError);
        }

        Self::from_coords(row - 1, col).ok_or(ParseLocationError)
    }
}

/// Convert this Location into string notation ("A4" / "PASS").
impl fmt::Display for Location {
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
        self.0 == 0
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

impl Iterator for MoveList {
    type Item = Location;

    fn next(&mut self) -> Option<Location> {
        if self.is_empty() {
            return None;
        }

        let next_move = 1 << self.0.trailing_zeros();
        self.0 ^= next_move;
        Some(Location(next_move))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn location_from_index() {
        assert_eq!(Location::from_index(0), Location(1));
        assert_eq!(Location::from_index(63), Location(1 << 63));
    }

    #[test]
    fn location_to_index() {
        assert_eq!(Location(1).to_index(), 0);
        assert_eq!(Location(1 << 63).to_index(), 63);
    }

    #[test]
    fn location_from_coords() {
        assert_eq!(Location::from_coords(0, 0), Some(Location(1 << 63)));
        assert_eq!(Location::from_coords(7, 7), Some(Location(1)));
        assert_eq!(Location::from_coords(0, 8), None);
        assert_eq!(Location::from_coords(8, 0), None);
    }

    #[test]
    fn location_to_coords() {
        assert_eq!(Location(1 << 63).to_coords(), (0, 0));
        assert_eq!(Location(1).to_coords(), (7, 7));
    }

    #[test]
    fn location_from_str_success() {
        assert_eq!(Location::from_str("A1"), Ok(Location(1 << 63)));
        assert_eq!(Location::from_str("h8"), Ok(Location(1)));
        assert_eq!(
            Location::from_str("D7"),
            Ok(Location::from_coords(6, 3).unwrap())
        );
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
        assert_eq!(Location(1).to_string(), "H8");
        assert_eq!(Location(1 << 63).to_string(), "A1");
        assert_eq!(Location::from_str("E2").unwrap().to_string(), "E2");
        assert_eq!(Location::from_str("F6").unwrap().to_string(), "F6");
    }
}
