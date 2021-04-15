//! Low-level bitboard operations.
//!
//! For efficiency, [`Bitboard`] operations are unchecked and may cause undefined
//! behavior if invalid data is passed.
//!
//! Under the hood, all these operations work on u64 bitboards. By convention,
//! the MSB is the upper-left of the board, and uses row-major order.

use crate::{utils, NUM_SPACES};
use derive_more::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, From, Into, Not,
};
use packed_simd::{u64x4, u64x8};
use std::fmt::{self, Display, Formatter};

/// Holds a single bit per location on an Othello board.
/// Wraps [`u64`] for efficient bit-twiddling, but avoids mixing with numerics.
#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    PartialEq,
    PartialOrd,
    Ord,
    Default,
    From,
    Into,
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
    BitXor,
    BitXorAssign,
    Not,
)]
pub struct Bitboard(u64);

/// Starting bitboard for Black.
pub const BLACK_START: Bitboard = Bitboard(0x0000000810000000);

/// Starting bitboard for White.
pub const WHITE_START: Bitboard = Bitboard(0x0000001008000000);

impl Display for Bitboard {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        utils::format_grid(
            self.into_iter().map(|bit| match bit {
                false => '.',
                true => '#',
            }),
            f,
        )
    }
}

impl Bitboard {
    /// Count the number of occupied spaces in the bitboard.
    #[inline]
    pub fn count_occupied(self) -> u8 {
        self.0.count_ones() as u8
    }

    /// Count the number of empty spaces in the bitboard.
    #[inline]
    pub fn count_empty(self) -> u8 {
        self.0.count_zeros() as u8
    }

    /// Return true if this bitboard is empty.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Split the bits in this bitboard into an array.
    #[inline]
    pub fn unpack(self) -> [bool; 64] {
        let bb_data = u64::from(self);
        [
            bb_data & (1u64 << 63) != 0,
            bb_data & (1u64 << 62) != 0,
            bb_data & (1u64 << 61) != 0,
            bb_data & (1u64 << 60) != 0,
            bb_data & (1u64 << 59) != 0,
            bb_data & (1u64 << 58) != 0,
            bb_data & (1u64 << 57) != 0,
            bb_data & (1u64 << 56) != 0,
            bb_data & (1u64 << 55) != 0,
            bb_data & (1u64 << 54) != 0,
            bb_data & (1u64 << 53) != 0,
            bb_data & (1u64 << 52) != 0,
            bb_data & (1u64 << 51) != 0,
            bb_data & (1u64 << 50) != 0,
            bb_data & (1u64 << 49) != 0,
            bb_data & (1u64 << 48) != 0,
            bb_data & (1u64 << 47) != 0,
            bb_data & (1u64 << 46) != 0,
            bb_data & (1u64 << 45) != 0,
            bb_data & (1u64 << 44) != 0,
            bb_data & (1u64 << 43) != 0,
            bb_data & (1u64 << 42) != 0,
            bb_data & (1u64 << 41) != 0,
            bb_data & (1u64 << 40) != 0,
            bb_data & (1u64 << 39) != 0,
            bb_data & (1u64 << 38) != 0,
            bb_data & (1u64 << 37) != 0,
            bb_data & (1u64 << 36) != 0,
            bb_data & (1u64 << 35) != 0,
            bb_data & (1u64 << 34) != 0,
            bb_data & (1u64 << 33) != 0,
            bb_data & (1u64 << 32) != 0,
            bb_data & (1u64 << 31) != 0,
            bb_data & (1u64 << 30) != 0,
            bb_data & (1u64 << 29) != 0,
            bb_data & (1u64 << 28) != 0,
            bb_data & (1u64 << 27) != 0,
            bb_data & (1u64 << 26) != 0,
            bb_data & (1u64 << 25) != 0,
            bb_data & (1u64 << 24) != 0,
            bb_data & (1u64 << 23) != 0,
            bb_data & (1u64 << 22) != 0,
            bb_data & (1u64 << 21) != 0,
            bb_data & (1u64 << 20) != 0,
            bb_data & (1u64 << 19) != 0,
            bb_data & (1u64 << 18) != 0,
            bb_data & (1u64 << 17) != 0,
            bb_data & (1u64 << 16) != 0,
            bb_data & (1u64 << 15) != 0,
            bb_data & (1u64 << 14) != 0,
            bb_data & (1u64 << 13) != 0,
            bb_data & (1u64 << 12) != 0,
            bb_data & (1u64 << 11) != 0,
            bb_data & (1u64 << 10) != 0,
            bb_data & (1u64 << 9) != 0,
            bb_data & (1u64 << 8) != 0,
            bb_data & (1u64 << 7) != 0,
            bb_data & (1u64 << 6) != 0,
            bb_data & (1u64 << 5) != 0,
            bb_data & (1u64 << 4) != 0,
            bb_data & (1u64 << 3) != 0,
            bb_data & (1u64 << 2) != 0,
            bb_data & (1u64 << 1) != 0,
            bb_data & (1u64 << 0) != 0,
        ]
    }
}

/// Score a board as: # my pieces - # opponent pieces.
/// Faster than [`score_winner_gets_empties()`], but less common.
/// Undefined behavior if both players have a piece at the same location.
#[inline]
pub fn score_absolute_difference(active: Bitboard, opponent: Bitboard) -> i8 {
    (active.0.count_ones() as i8) - (opponent.0.count_ones() as i8)
}

/// Score a board as: # my spaces - # opponent spaces, where empty spaces are scored for the winner.
/// Undefined behavior if both players have a piece at the same location.
#[inline]
pub fn score_winner_gets_empties(active: Bitboard, opponent: Bitboard) -> i8 {
    let absolute_difference = score_absolute_difference(active, opponent);

    if absolute_difference.is_positive() {
        absolute_difference + ((active | opponent).0.count_zeros() as i8)
    } else if absolute_difference.is_negative() {
        absolute_difference - ((active | opponent).0.count_zeros() as i8)
    } else {
        0
    }
}

/// Compute a mask of the legal moves for the active player from
/// masks of the active player's pieces and the opponent's pieces.
/// Undefined behavior if an invalid Othello board is specified.
// Algorithm adapted from Sam Blazes' Coin, released under the Apache 2.0 license:
// https://github.com/Tenebryo/coin/blob/master/bitboard/src/find_moves_fast.rs
#[inline]
pub fn get_move_mask(active: Bitboard, opponent: Bitboard) -> Bitboard {
    // Shifts for each vectorized direction: E/W, N/S, NW/SE, NE/SW.
    // The first direction is handled by SHL, the second by SHR.
    const SHIFTS_1: u64x4 = u64x4::new(1, 8, 7, 9);
    const SHIFTS_2: u64x4 = u64x4::new(2, 16, 14, 18);
    const SHIFTS_4: u64x4 = u64x4::new(4, 32, 28, 36);

    // Mask to clip off the invalid wraparound pieces on the edge.
    const EDGE_MASK: u64 = 0x7E7E7E7E7E7E7E7Eu64;

    let opponent_edge_mask = EDGE_MASK & opponent.0;

    // Masks used for each shift direction. The same for SHL and SHR.
    let masks: u64x4 = u64x4::new(
        opponent_edge_mask,
        opponent.0,
        opponent_edge_mask,
        opponent_edge_mask,
    );

    // Pieces we flip while shifting along each direction.
    let mut flip_l = u64x4::splat(active.0);
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
    let empties = !(active | opponent).0;
    let captures_l = (flip_l & masks) << SHIFTS_1;
    let captures_r = (flip_r & masks) >> SHIFTS_1;

    Bitboard(empties & (captures_l | captures_r).or())
}

/// Compute an updated board after a given move is made, returning new bitboards
/// for the active player and the opponent. `move_mask` must be a one-hot bitboard
/// indicating the move location. Undefined behavior if an invalid Othello board
/// or `move_mask` is provided.
#[inline]
pub fn apply_move(
    active: Bitboard,
    opponent: Bitboard,
    move_mask: Bitboard,
) -> (Bitboard, Bitboard) {
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
        move_mask.0,
        move_mask.0,
        move_mask.0,
        move_mask.0,
        active.0,
        active.0,
        active.0,
        active.0,
    );
    let mut pro_left = u64x8::splat(opponent.0);

    // Generators and propagators for all RSH directions: W, S, SE, SW.
    // Two concatenated vectors: [self + opponent interactions, new + opponent interactions].
    let mut gen_right = u64x8::new(
        active.0,
        active.0,
        active.0,
        active.0,
        move_mask.0,
        move_mask.0,
        move_mask.0,
        move_mask.0,
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

    let new_active = Bitboard((active.0 ^ flip_mask) | move_mask.0);
    let new_opponent = Bitboard(opponent.0 ^ flip_mask);

    (new_active, new_opponent)
}

/// Iterator for the bits in a [`Bitboard`].
#[derive(Clone, Copy, Debug)]
pub struct Bits {
    remaining: usize,
    bitboard: Bitboard,
}

impl Iterator for Bits {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let bitmask = Bitboard::from(1u64 << (self.remaining - 1));
        let bit = !(self.bitboard & bitmask).is_empty();
        self.remaining -= 1;

        Some(bit)
    }
}

impl ExactSizeIterator for Bits {
    fn len(&self) -> usize {
        self.remaining
    }
}

/// Iterate over the bits in row-major order.
impl IntoIterator for Bitboard {
    type Item = bool;
    type IntoIter = Bits;

    fn into_iter(self) -> Self::IntoIter {
        Bits {
            remaining: NUM_SPACES,
            bitboard: self,
        }
    }
}
