"""
Defines the data format and loading utilities for imitation-learning data.
"""

from typing import NamedTuple

import torch
import torchvision


class Example(NamedTuple):
    """
    An example used for imitation training, or a batch of examples.
    When batched, all dimensions include a [batch] dimension as their first axis.

    Attributes
    ----------
    board
        The active player's view of the board. Shape: [2, 8, 8].
    score
        The game's final score for the active player. Shape: [].
        Canonically, the "absolute difference" score.
    move_mask
        The target move mask out of this state. Shape: [8, 8].
    """

    board: torch.Tensor
    score: torch.Tensor
    move_mask: torch.Tensor

    def clone(self) -> "Example":
        return Example(*map(torch.clone, self))


def augment_square_symmetries(example: Example) -> Example:
    """
    Randomly alter an example with the symmetries of the square: flips and rotations.
    """
    board, score, move_mask = example.clone()

    if torch.rand(1) < 0.5:
        torchvision.transforms.RandomHorizontalFlip()
        board = torchvision.transforms.functional.hflip(board)
        move_mask = torchvision.transforms.functional.hflip(move_mask)

    if torch.rand(1) < 0.5:
        board = torchvision.transforms.functional.vflip(board)
        move_mask = torchvision.transforms.functional.vflip(move_mask)

    rotations = torch.randint(size=(1,), low=0, high=4).item()
    board = torch.rot90(board, k=rotations, dims=[1, 2])  # type: ignore
    move_mask = torch.rot90(move_mask, k=rotations, dims=[0, 1])  # type: ignore

    return Example(board, score, move_mask)


def augment_swap_players(example: Example) -> Example:
    """
    Randomly alter an example by swapping the player roles.
    """
    board, score, move_mask = example.clone()

    if torch.rand(1) < 0.5:
        board = board[[1, 0], :, :]
        score *= -1

    return Example(board, score, move_mask)