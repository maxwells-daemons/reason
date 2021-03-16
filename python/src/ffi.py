"""
A wrapper for moves provided by the `reason-othello` library through an FFI.
"""

import ctypes

import torch

LIB_PATH = "rust/target/release/libreason_othello.so"

_HIGH_BIT = 1 << 63
_BOARD_EDGE = 8


class ApplyMoveResult(ctypes.Structure):
    _fields_ = [
        ("new_active_mask", ctypes.c_uint64),
        ("new_opponent_mask", ctypes.c_uint64),
    ]


_LIB_HANDLE = ctypes.cdll.LoadLibrary(LIB_PATH)
_LIB_HANDLE.ffi_get_move_mask.restype = ctypes.c_uint64
_LIB_HANDLE.ffi_apply_move.restype = ApplyMoveResult


def _serialize_pieces(pieces: torch.Tensor) -> ctypes.c_uint64:
    bitboard = 0
    for piece in pieces.view(-1):
        bitboard <<= 1
        if piece:
            bitboard |= 1

    return ctypes.c_uint64(bitboard)


def _deserialize_pieces(bitboard: int) -> torch.Tensor:
    board = torch.zeros([_BOARD_EDGE, _BOARD_EDGE], dtype=bool)  # type: ignore

    for col in range(_BOARD_EDGE):
        for row in range(_BOARD_EDGE):
            board[col, row] = bool(bitboard & _HIGH_BIT)
            bitboard <<= 1

    return board


def _validate_board_shape(pieces: torch.Tensor) -> None:
    if pieces.shape != (2, _BOARD_EDGE, _BOARD_EDGE):
        raise ValueError("Invalid board shape.")


def get_move_mask(pieces: torch.Tensor) -> torch.Tensor:
    """
    Given the active player's pieces and the opponent's pieces,
    get a mask of the legal move locations.
    """
    _validate_board_shape(pieces)

    active_mask = _serialize_pieces(pieces[0])
    opponent_mask = _serialize_pieces(pieces[1])
    result_bitboard = _LIB_HANDLE.ffi_get_move_mask(active_mask, opponent_mask)
    return _deserialize_pieces(result_bitboard)


def apply_move(pieces: torch.Tensor, row: int, col: int) -> torch.Tensor:
    """
    Given the active player's pieces, the opponent's pieces, and a row and
    column index, compute the new piece masks.
    """
    _validate_board_shape(pieces)

    active_mask = _serialize_pieces(pieces[0])
    opponent_mask = _serialize_pieces(pieces[1])
    result: ApplyMoveResult = _LIB_HANDLE.ffi_apply_move(
        active_mask, opponent_mask, ctypes.c_uint8(col), ctypes.c_uint8(row)
    )

    active_pieces = _deserialize_pieces(result.new_active_mask)
    opponent_pieces = _deserialize_pieces(result.new_opponent_mask)
    return torch.stack([active_pieces, opponent_pieces])
