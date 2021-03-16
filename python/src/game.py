"""
Code for working with Othello game objects.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch

from src import ffi

BOARD_EDGE = 8
BOARD_SPACES = 64


class Player(Enum):
    BLACK = "Black"
    WHITE = "White"

    def other(self) -> "Player":
        return Player.BLACK if self == Player.WHITE else Player.WHITE


@dataclass
class GameState:
    """
    The complete state of an Othello game.
    """

    # All the pieces on the board, as the [active player, opponent].
    # Shape: [2, 8, 8]; bool.
    pieces: torch.Tensor
    active_player: Player
    just_passed: bool

    def player_view(self, player: Player) -> torch.Tensor:
        """Get the view of the board from the perspective of a particular player."""
        return (
            self.pieces if player == self.active_player else self.pieces[(1, 0), :, :]
        )

    def get_move_mask(self) -> torch.Tensor:
        return ffi.get_move_mask(self.pieces)

    def apply_pass(self) -> "GameState":
        return GameState(self.pieces[(1, 0), :, :], self.active_player.other(), True)

    def apply_move(self, row: int, col: int) -> "GameState":
        new_board = ffi.apply_move(self.pieces, row, col)
        return GameState(new_board[(1, 0), :, :], self.active_player.other(), False)

    def score_absolute_difference(self, player: Player) -> int:
        return _score_absolute_difference(self.player_view(player))

    def score_winner_gets_empties(self, player: Player) -> int:
        return _score_winner_gets_empties(self.player_view(player))

    def __str__(self) -> str:
        black_squares, white_squares = self.player_view(Player.BLACK)
        moves = self.get_move_mask()

        fmt = f"{self.active_player.value} to move"
        if self.just_passed:
            fmt += " (last move passed)"
        fmt += "\n  A B C D E F G H\n"

        for row in range(BOARD_EDGE):
            fmt += str(row + 1) + " "
            for col in range(BOARD_EDGE):
                if black_squares[row, col]:
                    fmt += "# "
                elif white_squares[row, col]:
                    fmt += "O "
                elif moves[row, col]:
                    fmt += "- "
                else:
                    fmt += ". "
            fmt += "\n"

        return fmt


def parse_move(move: str) -> Tuple[int, int]:
    col = "abcdefgh".index(move[0].lower())
    row = int(move[1]) - 1
    return (row, col)


def starting_state() -> GameState:
    pieces = torch.zeros((2, BOARD_EDGE, BOARD_EDGE), dtype=bool)  # type: ignore
    pieces[0, 3, 4] = 1
    pieces[0, 4, 3] = 1
    pieces[1, 3, 3] = 1
    pieces[1, 4, 4] = 1
    return GameState(pieces, Player.BLACK, False)


def _score_absolute_difference(pieces: torch.Tensor) -> int:
    return (pieces[0].sum() - pieces[1].sum()).item()  # type: ignore


def _score_winner_gets_empties(pieces: torch.Tensor) -> int:
    my_score = pieces[0].sum().item()
    opp_score = pieces[1].sum().item()
    empties = 64 - (my_score + opp_score)

    if my_score > opp_score:
        return my_score - opp_score + empties  # type: ignore
    elif my_score < opp_score:
        return my_score - opp_score - empties  # type: ignore
    return 0