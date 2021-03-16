use crate::bitboard;
use crate::game::Move;

#[repr(C)]
pub struct ApplyMoveResult {
    pub new_active_mask: u64,
    pub new_opponent_mask: u64,
}

#[no_mangle]
pub extern "C" fn ffi_get_move_mask(active_mask: u64, opponent_mask: u64) -> u64 {
    bitboard::get_move_mask(
        bitboard::Bitboard(active_mask),
        bitboard::Bitboard(opponent_mask),
    )
    .0
}

#[no_mangle]
pub extern "C" fn ffi_apply_move(
    active_mask: u64,
    opponent_mask: u64,
    row: u8,
    col: u8,
) -> ApplyMoveResult {
    let move_mask = Move::from_coords(row, col).unwrap().0;
    let (new_active_mask, new_opponent_mask) = bitboard::apply_move(
        bitboard::Bitboard(active_mask),
        bitboard::Bitboard(opponent_mask),
        move_mask,
    );
    ApplyMoveResult {
        new_active_mask: new_active_mask.0,
        new_opponent_mask: new_opponent_mask.0,
    }
}
