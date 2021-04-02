//! A C-compatible FFI for core bitboard functions.
use crate::bitboard::{self, Bitboard};
use crate::Location;

#[repr(C)]
pub struct ApplyMoveResult {
    pub new_active_mask: u64,
    pub new_opponent_mask: u64,
}

/// Given the bitboards of the active player and opponent, get the bitboard of legal moves.
#[no_mangle]
pub extern "C" fn ffi_get_move_mask(active_mask: u64, opponent_mask: u64) -> u64 {
    bitboard::get_move_mask(Bitboard::from(active_mask), Bitboard::from(opponent_mask)).into()
}

/// Given the bitboards of the active player and the opponent, apply a move at the given row and column.
#[no_mangle]
pub extern "C" fn ffi_apply_move(
    active_mask: u64,
    opponent_mask: u64,
    row: usize,
    col: usize,
) -> ApplyMoveResult {
    let move_loc = Location::from_coords(row, col);
    let (new_active_mask, new_opponent_mask) = bitboard::apply_move(
        Bitboard::from(active_mask),
        Bitboard::from(opponent_mask),
        move_loc.into(),
    );
    ApplyMoveResult {
        new_active_mask: new_active_mask.into(),
        new_opponent_mask: new_opponent_mask.into(),
    }
}
