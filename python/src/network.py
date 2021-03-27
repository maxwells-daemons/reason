"""
Defines the neural network architecture.
"""

from typing import Tuple

import torch

from src import game


class InputBlock(torch.nn.Sequential):
    def __init__(self, n_channels: int):
        super(InputBlock, self).__init__(
            torch.nn.Conv2d(2, n_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(n_channels),
            torch.nn.ReLU(),
        )


class ResidualBlock(torch.nn.Module):
    def __init__(self, n_channels: int):
        super(ResidualBlock, self).__init__()
        self.inner_path = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(n_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(n_channels),
        )
        self.post_residual_activation = torch.nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pre_activation = inputs + self.inner_path(inputs)
        return self.post_residual_activation(pre_activation)


class PolicyHead(torch.nn.Sequential):
    def __init__(self, n_channels: int):
        super(PolicyHead, self).__init__(
            torch.nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(n_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_channels, 1, kernel_size=1),
        )


class ValueHead(torch.nn.Sequential):
    def __init__(self, n_channels: int):
        super(ValueHead, self).__init__(
            torch.nn.Conv2d(n_channels, 1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(game.BOARD_SPACES, game.BOARD_SPACES),
            torch.nn.ReLU(),
            torch.nn.Linear(game.BOARD_SPACES, 1),
            torch.nn.Tanh(),
        )


class AgentModel(torch.nn.Module):
    def __init__(self, n_channels: int, n_blocks: int):
        super(AgentModel, self).__init__()

        self.input_block = InputBlock(n_channels)
        trunk_blocks = [ResidualBlock(n_channels) for _ in range(n_blocks)]
        self.trunk = torch.nn.Sequential(*trunk_blocks)
        self.policy_head = PolicyHead(n_channels)
        self.value_head = ValueHead(n_channels)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        board_rep = self.trunk(self.input_block(inputs))
        policy = self.policy_head(board_rep).squeeze(dim=1)
        value = self.value_head(board_rep).squeeze(dim=1)
        return policy, value
