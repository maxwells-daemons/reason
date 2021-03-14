//! Implements low-level bitboard operations. These are fast, but not ergonomic;
//! prefer to use abstractions under [`game`] if possible.
//!
//! Under the hood, all these operations work on u64 bitboards. By convention,
//! the MSB is the upper-left of the board, and uses row-major order.

use packed_simd::{u64x4, u64x8};

/// A wrapper type for bitboards, to ensure they aren't mixed with other numeric types.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Default)]
pub struct Bitboard(pub u64);

/// Compute a mask of the occupied locations on the board.
#[inline]
pub fn get_occupancy_mask(player_1: Bitboard, player_2: Bitboard) -> Bitboard {
    Bitboard(player_1.0 | player_2.0)
}

/// Compute a mask of empty squares on the board.
#[inline]
pub fn get_empty_mask(player_1: Bitboard, player_2: Bitboard) -> Bitboard {
    Bitboard(!(get_occupancy_mask(player_1, player_2).0))
}

/// Count the number of empty spaces on the board.
#[inline]
pub fn count_empties(player_1: Bitboard, player_2: Bitboard) -> i8 {
    get_occupancy_mask(player_1, player_2).0.count_zeros() as i8
}

/// Score a board as: # my pieces - # opponent pieces.
/// Faster than [`score_winner_gets_empties()`], but less common.
#[inline]
pub fn score_absolute_difference(active: Bitboard, opponent: Bitboard) -> i8 {
    (active.0.count_ones() as i8) - (opponent.0.count_ones() as i8)
}

/// Score a board as: # my spaces - # opponent spaces, where empty spaces are scored for the winner.
#[inline]
pub fn score_winner_gets_empties(active: Bitboard, opponent: Bitboard) -> i8 {
    let absolute_difference = score_absolute_difference(active, opponent);

    if absolute_difference.is_positive() {
        absolute_difference + count_empties(active, opponent)
    } else if absolute_difference.is_negative() {
        absolute_difference - count_empties(active, opponent)
    } else {
        0
    }
}

/// Compute a mask of the legal moves for the active player from
/// masks of the active player's stones and the opponent's stones.
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
    let empties = get_empty_mask(active, opponent).0;
    let captures_l = (flip_l & masks) << SHIFTS_1;
    let captures_r = (flip_r & masks) >> SHIFTS_1;

    Bitboard(empties & (captures_l | captures_r).or())
}

/// Compute an updated board after a given move is made, returning new bitboards
/// for the active player and the opponent. `move_mask` must be a one-hot bitboard
/// indicating the move location; otherwise, the behavior of this function is undefined.
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
