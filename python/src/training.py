"""
Defines the training strategy.
"""

import torch
import pytorch_lightning as pl

from src.network import AgentModel
from src.data.logistello import LogistelloDataModule


class TrainingModule(pl.LightningModule):
    def __init__(
        self, learning_rate: float = 1e-4, n_channels: int = 128, n_blocks: int = 5
    ):
        super(TrainingModule, self).__init__()
        self.save_hyperparameters()

        self.model = AgentModel(n_channels, n_blocks)
        self.value_loss_fn = torch.nn.MSELoss()
        self.policy_loss_fn = torch.nn.CrossEntropyLoss()

        self.policy_acc_train = pl.metrics.Accuracy()
        self.policy_acc_val = pl.metrics.Accuracy()

    def forward(self, board):
        return self.model(board)

    def training_step(self, batch, _):
        board, move_index, outcome = batch
        policy, value = self(board)

        policy_flat = policy.view(policy.size(0), -1)
        policy_loss = self.policy_loss_fn(policy_flat, move_index)
        self.log("policy_loss/training", policy_loss)

        value_loss = self.value_loss_fn(value, outcome)
        self.log("value_loss/training", value_loss)

        loss = policy_loss + value_loss
        self.log("total_loss/training", loss)

        policy_dist = torch.softmax(policy_flat, dim=-1)
        self.policy_acc_train(policy_dist, move_index)
        self.log("policy_acc/training", self.policy_acc_train)

        self.log("value/mean", value.mean())
        self.log("value/max", value.max())
        self.log("value/min", value.min())

        return loss

    def validation_step(self, batch, _):
        board, move_index, outcome = batch
        policy, value = self(board)

        policy_flat = policy.view(policy.size(0), -1)
        policy_loss = self.policy_loss_fn(policy_flat, move_index)
        self.log("policy_loss/validation", policy_loss)

        value_loss = self.value_loss_fn(value, outcome)
        self.log("value_loss/validation", value_loss)

        loss = policy_loss + value_loss
        self.log("total_loss/validation", loss)

        policy_dist = torch.softmax(policy_flat, dim=-1)
        self.policy_acc_val(policy_dist, move_index)
        self.log("policy_acc/validation", self.policy_acc_train)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def train(
    learning_rate: float, n_channels: int = 128, n_blocks: int = 5, batch_size: int = 32
):
    trainer = pl.Trainer(gpus=-1, callbacks=[], gradient_clip_val=0.1)
    model = TrainingModule(learning_rate, n_channels, n_blocks)
    data_module = LogistelloDataModule(batch_size=batch_size)  # type: ignore
    trainer.fit(model, datamodule=data_module)  # type: ignore


# TODO: remove
if __name__ == "__main__":
    train(learning_rate=1e-3, n_channels=128, n_blocks=5, batch_size=256)