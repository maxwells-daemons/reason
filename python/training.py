"""
Defines the training strategy.
"""

import logging

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from python import ffi
from python.data import example, logistello, utils, wthor
from python.data.example import Example
from python.network import AgentModel

_logger = logging.getLogger(__name__)


class TrainingModule(pl.LightningModule):
    """
    A module with state and functionality needed for optimization during training.
    """

    def __init__(self, learning_rate: float, model: AgentModel):
        super(TrainingModule, self).__init__()
        _logger.debug("Building training module.")

        self.model = model
        self._learning_rate = learning_rate

    def forward(self, board):
        return self.model(board)

    def training_step(self, batch, _):
        board, target_score, target_move_probs = batch
        target_outcome = target_score.sign()

        policy_scores, value = self(board)

        # TODO: mask out invalid moves
        policy_loss = self._policy_loss(policy_scores, target_move_probs)
        self.log("loss/policy", policy_loss)

        # TODO: try WLD
        # TODO: try cross-entropy
        # TODO: try predicting score
        value_loss = torch.nn.functional.mse_loss(value, target_outcome)
        self.log("loss/value", value_loss)

        # TODO: loss weighting
        loss = policy_loss + value_loss
        self.log("loss/total", loss)

        return {
            "loss": loss,
            "policy_scores": policy_scores,
            "value": value,
            "batch": batch,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    def _policy_loss(self, policy_scores, target_probs):
        return self._soft_crossentropy(
            policy_scores.flatten(1), target_probs.flatten(1)
        )

    @staticmethod
    def _soft_crossentropy(predicted_scores, target_probs):
        """
        Cross-entropy loss capable of handling soft target probabilities.
        """
        predicted_logprobs = torch.nn.functional.log_softmax(predicted_scores, dim=1)
        return -(target_probs * predicted_logprobs).sum(1).mean(0)


class ImitationData(pl.LightningDataModule):
    """
    A DataModule which samples from Logistello and WTHOR data, applies data
    augmentation, and buffer-shuffles.
    """

    def __init__(
        self,
        batch_size: int,
        wthor_weight: float,
        augment_square_symmetries: bool,
        shuffle_buffer_size: int,
        data_workers: int,
        wthor_glob: str,
        logistello_path: str,
    ):
        super().__init__()

        self._batch_size = batch_size
        self._wthor_weight = wthor_weight
        self._augment_square_symmetries = augment_square_symmetries
        self._shuffle_buffer_size = shuffle_buffer_size
        self._data_workers = data_workers
        self._wthor_glob = wthor_glob
        self._logistello_path = logistello_path

    def setup(self, stage=None):
        _logger.debug("Building imitation training dataset.")
        wthor_data = wthor.WthorDataset(self._wthor_glob)
        logistello_data = logistello.LogistelloDataset(self._logistello_path)
        combined_data = utils.SamplingDataset(
            [(wthor_data, self._wthor_weight), (logistello_data, 1.0)]
        )
        self._dataset = torch.utils.data.BufferedShuffleDataset(
            combined_data, self._shuffle_buffer_size
        )

    def on_before_batch_transfer(self, batch: Example, _):
        if self._augment_square_symmetries:
            batch = example.augment_square_symmetries(batch)

        return batch

    def train_dataloader(self):
        if self._data_workers == 0:
            self.set_random_seed(0)
            return torch.utils.data.DataLoader(
                self._dataset, self._batch_size, pin_memory=True
            )

        return torch.utils.data.DataLoader(
            self._dataset,
            self._batch_size,
            num_workers=self._data_workers,
            worker_init_fn=self.set_random_seed,
            pin_memory=True,
        )

    @staticmethod
    def set_random_seed(worker_id: int) -> None:
        torch.random.manual_seed(1337)


class VisualizePredictions(pl.Callback):
    """
    Periodically visualize the distribution of model predictions on training data.
    """

    def __init__(self, batch_period: int):
        self._batch_period = batch_period

    def on_train_batch_end(
        self, trainer, module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx % self._batch_period:
            return

        outputs = outputs[0][0]["extra"]
        policy_scores = outputs["policy_scores"].detach().cpu()
        value = outputs["value"].detach().cpu()

        # Visualize model predictions on the first example in the batch
        board = outputs["batch"].board[0]
        board_img = torch.zeros([3, 8, 8], dtype=float, device="cpu")
        board_img[0] = board[0]  # Active player's stones are red
        board_img[2] = board[1]  # Opponent's stones are blue
        board_img[1] = ffi.get_move_mask(board.cpu())  # Legal moves are green

        policy_target = outputs["batch"].policy_target[0]
        policy_preds = (
            policy_scores[0].flatten(1).softmax(1).view(policy_target.size(0), -1)
        )

        module.logger.experiment.log(
            {
                "policy.distribution": wandb.Histogram(policy_scores),
                "value.distribution": wandb.Histogram(value),
                "trainer/global_step": trainer.global_step,
                "visualization/board": wandb.Image(board_img, caption="Board"),
                "visualization/policy_preds": wandb.Image(
                    policy_preds, caption="Policy probabilities"
                ),
                "visualization/policy_target": wandb.Image(
                    policy_target, caption="Target probabilities"
                ),
            }
        )


@hydra.main(config_path="config", config_name="training")
def train(config: DictConfig):
    # Avoid duplicate logs
    logging.getLogger("lightning").propagate = False

    _logger.info("Beginning a new training run.")
    _logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    _logger.debug("Setting up W&B integration.")
    logger = hydra.utils.instantiate(config.logger)
    logger.experiment.config.update(
        OmegaConf.to_container(config.log_hparams, resolve=True)
    )

    _logger.info("Initializing model.")
    model = hydra.utils.instantiate(config.model)
    training_module = hydra.utils.instantiate(config.training, model=model)

    _logger.info("Initializing trainer.")
    callbacks = [hydra.utils.instantiate(config.visualize_callback)]
    trainer = hydra.utils.instantiate(
        config.trainer, logger=logger, callbacks=callbacks
    )

    _logger.info("Initializing data.")
    imitation_data = hydra.utils.instantiate(config.imitation_data)

    _logger.info("Starting training.")
    trainer.fit(training_module, datamodule=imitation_data)
    _logger.info("Training complete.")


if __name__ == "__main__":
    train()
