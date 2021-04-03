"""
Defines the data format and associated utilities for data samples and batches.
"""

from typing import NamedTuple

import torch
import torchvision


class Example(NamedTuple):
    """
    An example used for agent training, or a batch of examples.
    When batched, all dimensions include a [batch] dimension as their first axis.

    Attributes
    ----------
    board
        The active player's view of the board. Shape: [2, 8, 8].
    score
        The game's final score for the active player. Shape: [].
        Canonically, the "absolute difference" score.
    policy_target
        The (possibly unnormalized) target distribution for policy learning.
        Shape: [8, 8].
    """

    board: torch.Tensor
    score: torch.Tensor
    policy_target: torch.Tensor

    def clone(self) -> "Example":
        return Example(*map(torch.clone, self))


def augment_square_symmetries(example: Example) -> Example:
    """
    Randomly alter a batch with the symmetries of the square: flips and rotations.
    """
    board, score, policy_target = example.clone()

    if torch.rand(1) < 0.5:
        board = torchvision.transforms.functional.hflip(board)
        policy_target = torchvision.transforms.functional.hflip(policy_target)

    if torch.rand(1) < 0.5:
        board = torchvision.transforms.functional.vflip(board)
        policy_target = torchvision.transforms.functional.vflip(policy_target)

    rotations = torch.randint(size=(1,), low=0, high=4).item()
    board = torch.rot90(board, k=rotations, dims=[2, 3])  # type: ignore
    policy_target = torch.rot90(policy_target, k=rotations, dims=[1, 2])  # type: ignore

    return Example(board, score, policy_target)