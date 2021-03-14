use packed_simd::{u64x4, u64x8};
use std::fmt;

/// Stores the board state as [my pieces, opponent pieces].
/// By convention, the MSB is the upper-left of the board, and proceeds in row-major order.
#[derive(Clone, Copy)]
pub struct Board {
    pub player_bitboard: u64,
    pub opponent_bitboard: u64,
}

/// Stores a list of legal moves out of a position as a bitboard mask.
/// Can be iterated to retrieve a list of move locations.
#[derive(Clone, Copy)]
pub struct MoveList(u64);

/// Stores a single location on the board as a bitboard mask.
#[derive(Clone, Copy)]
pub struct Location(u64);

impl Board {
    pub const EDGE_LENGTH: u8 = 8;
    pub const BOARD_SQUARES: u8 = 64;

    const BLACK_START: u64 = 0x0000000810000000;
    const WHITE_START: u64 = 0x0000001008000000;

    /// The Othello starting position from black's perspective.
    pub const fn new() -> Self {
        Self {
            player_bitboard: Self::BLACK_START,
            opponent_bitboard: Self::WHITE_START,
        }
    }

    pub fn pass(self) -> Self {
        Self {
            player_bitboard: self.opponent_bitboard,
            opponent_bitboard: self.player_bitboard,
        }
    }

    /// Get a mask of the legal moves for the active player.
    // Algorithm adapted from Sam Blazes' Coin, released under the Apache 2.0 license:
    // https://github.com/Tenebryo/coin/blob/master/bitboard/src/find_moves_fast.rs
    pub fn get_moves(self) -> MoveList {
        // Shifts for each vectorized direction: E/W, N/S, NW/SE, NE/SW.
        // The first direction is handled by SHL, the second by SHR.
        const SHIFTS_1: u64x4 = u64x4::new(1, 8, 7, 9);
        const SHIFTS_2: u64x4 = u64x4::new(2, 16, 14, 18);
        const SHIFTS_4: u64x4 = u64x4::new(4, 32, 28, 36);

        // Mask to clip off the invalid wraparound pieces on the edge.
        const EDGE_MASK: u64 = 0x7E7E7E7E7E7E7E7Eu64;

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
    // wonky-kong-vectorized
    // TODO: try a ray-lookup approach
    pub fn make_move(self, loc: Location) -> Self {
        // Masks selecting everything except the far-left and far-right columns.
        const NOT_A_FILE: u64 = 0xfefefefefefefefe;
        const NOT_H_FILE: u64 = 0x7f7f7f7f7f7f7f7f;
        const FULL_MASK: u64 = 0xffffffffffffffff;

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
    pub fn score_absolute_difference(self) -> i8 {
        (self.player_bitboard.count_ones() as i8) - (self.player_bitboard.count_ones() as i8)
    }

    /// Score a board as: # my spaces - # opponent spaces, where empty spaces are scored for the winner.
    pub fn score_winner_gets_empties(self) -> i8 {
        let absolute_difference = self.score_absolute_difference();
        if absolute_difference.is_positive() {
            absolute_difference + self.empties() as i8
        } else if absolute_difference.is_negative() {
            absolute_difference - self.empties() as i8
        } else {
            0
        }
    }

    fn empties(self) -> u64 {
        !(self.player_bitboard | self.opponent_bitboard)
    }
}

impl MoveList {
    pub fn is_empty(self) -> bool {
        self.0 == 0
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

impl Location {
    pub fn from_bitboard(bitboard: u64) -> Self {
        Self(bitboard)
    }

    pub fn from_index(index: u8) -> Self {
        Self(1 << index)
    }
}

// Board formatted like:
//   abcdefgh
// 1 ........
// 2 ........
// 3 ........
// 4 ...OX...
// 5 ...XO...
// 6 ........
// 7 ........
// 8 ........
impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Selects the highest bit for both players
        const PIECE_MASK: u64 = 1 << 63;
        println!("{}", PIECE_MASK);

        let mut player_bitboard = self.player_bitboard;
        let mut opponent_bitboard = self.opponent_bitboard;
        let mut my_piece: u64;
        let mut opponent_piece: u64;

        writeln!(f, "\n  a b c d e f g h")?;
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
