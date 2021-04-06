"""
Defines the neural network architecture.
"""

from typing import Tuple

import torch

from python import game, data


# Pattern from: https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
class ConvBNReLU(torch.nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = 1,
    ):
        super(ConvBNReLU, self).__init__(
            torch.nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(),
        )

    def fuse_inplace(self):
        torch.quantization.fuse_modules(self, ["0", "1", "2"], inplace=True)


class ResidualBlock(torch.nn.Module):
    def __init__(self, n_channels: int):
        super(ResidualBlock, self).__init__()
        self.inner_path = torch.nn.Sequential(
            ConvBNReLU(n_channels, n_channels),
            ConvBNReLU(n_channels, n_channels),
        )
        self.skip_connection = torch.nn.quantized.FloatFunctional()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.skip_connection.add(inputs, self.inner_path(inputs))

    def fuse_inplace(self):
        for layer in self.inner_path:
            layer.fuse_inplace()


class PolicyHead(torch.nn.Sequential):
    def __init__(self, n_channels: int):
        super(PolicyHead, self).__init__(
            ConvBNReLU(n_channels, n_channels),
            torch.nn.Conv2d(n_channels, 1, kernel_size=1),
        )

    def fuse_inplace(self):
        self[0].fuse_inplace()


class ValueHead(torch.nn.Sequential):
    def __init__(self, n_channels: int):
        super(ValueHead, self).__init__(
            ConvBNReLU(n_channels, 1, kernel_size=1, padding=0),
            torch.nn.Flatten(),
            torch.nn.Linear(game.BOARD_SPACES, game.BOARD_SPACES),
            torch.nn.ReLU(),
            torch.nn.Linear(game.BOARD_SPACES, 1),
            torch.nn.Tanh(),
        )

    def fuse_inplace(self):
        # ConvBNReLU
        self[0].fuse_inplace()

        # Linear -> ReLU
        torch.quantization.fuse_modules(self, ["2", "3"], inplace=True)


class AgentModel(torch.nn.Module):
    def __init__(self, n_channels: int, n_blocks: int):
        super(AgentModel, self).__init__()

        trunk_blocks = [
            ConvBNReLU(
                input_channels=data.example.N_BOARD_FEATURES, output_channels=n_channels
            )
        ]
        for _ in range(n_blocks):
            trunk_blocks.append(ResidualBlock(n_channels))  # type: ignore

        self.trunk = torch.nn.Sequential(*trunk_blocks)
        self.policy_head = PolicyHead(n_channels)
        self.value_head = ValueHead(n_channels)

        self.quantize = torch.quantization.QuantStub()
        self.dequantize = torch.quantization.DeQuantStub()

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        board_rep = self.trunk(self.quantize(inputs))
        policy = self.policy_head(board_rep).squeeze(dim=1)
        value = self.value_head(board_rep).squeeze(dim=1)
        return self.dequantize(policy), self.dequantize(value)

    def fuse_inplace(self):
        for block in self.trunk:
            block.fuse_inplace()

        self.policy_head.fuse_inplace()
        self.value_head.fuse_inplace()