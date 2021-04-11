"""
Defines the training strategy.
"""

import logging

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from python import game, utils
from python.data import example, logistello, wthor
from python.data.example import Example
from python.network import AgentModel

_logger = logging.getLogger(__name__)


RANDOM_SEED = 1337


class TrainingModule(pl.LightningModule):
    """
    A module with state and functionality needed for optimization during training.

    Parameters
    ----------
    learning_rate
        Learning rate to use.
    value_loss_weight
        Amount to scale value loss relative to policy loss.
    value_target
        What to train the value function to predict. One of:
         - 'piece_difference': the game's final piece difference, scaled to -1 to 1.
         - 'outcome': outcome of the game for the active player.
           1 for win, -1 for loss, 0 for draw.
    mask_invalid_moves
        Whether to mask out invalid moves before computing the policy loss.
    """

    def __init__(
        self,
        n_channels: int,
        n_blocks: int,
        learning_rate: float,
        value_loss_weight: float,
        value_target: str,
        mask_invalid_moves: bool,
    ):
        if value_target not in {"piece_difference", "outcome"}:
            raise ValueError("Unrecognized setting for `value_target`.")

        super(TrainingModule, self).__init__()
        self.save_hyperparameters()
        _logger.debug("Building training module.")

        self.model = AgentModel(n_channels=n_channels, n_blocks=n_blocks)
        self.value_accuracy = pl.metrics.classification.Accuracy()

    def forward(self, board):
        return self.model(board)

    def training_step(self, batch, _):
        board, target_score, target_move_probs, valid_move_mask = batch

        # Forward pass
        policy_scores, value = self(board)

        # Policy loss and accuracy
        policy_scores_flat = policy_scores.flatten(1)
        target_move_probs_flat = target_move_probs.flatten(1)
        policy_logprobs_flat = utils.masked_log_softmax(
            policy_scores_flat,
            valid_move_mask.flatten(1) if self.hparams.mask_invalid_moves else None,
        )

        policy_loss = utils.soft_crossentropy(
            policy_logprobs_flat, target_move_probs_flat
        )

        # For policy accuracy, assume the "policy target" is the greedy max
        policy_accuracy = (
            (
                policy_logprobs_flat.max(dim=1).indices
                == target_move_probs_flat.max(dim=1).indices
            )
            .float()
            .mean()
        )

        # TODO: try WLD classification
        # Value loss and accuracy
        if self.hparams.value_target == "piece_difference":
            value_target = target_score / game.BOARD_SPACES
        elif self.hparams.value_target == "outcome":
            value_target = torch.sign(target_score)
        else:
            raise AssertionError

        value_loss = torch.nn.functional.mse_loss(value, value_target)

        # "Value accuracy": treat V>0 as predicted win and V<0 as predicted loss;
        # treat draws as always classified correctly.
        draws = value_target == 0
        value_cls_preds = (value >= 0) | draws
        value_cls_targets = (value_target >= 0) | draws
        self.value_accuracy(value_cls_preds, value_cls_targets)

        loss = self._total_loss(policy_loss, value_loss)

        self.log_dict(
            {
                "loss/policy": policy_loss,
                "loss/value": value_loss,
                "loss/total": loss,
                "policy/accuracy_argmax": policy_accuracy,
                "value/accuracy": self.value_accuracy,
            }
        )

        return {
            "loss": loss,
            "policy_scores_flat": policy_scores_flat.float(),
            "value": value.float(),
            "batch": batch,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    # -- 300 cpu fwd passes on [4, 6, 8, 8], best of 5
    # Regular model: 1.21s
    # No-grad: 1.13s
    # Frozen:  0.71s
    # Frozen + no-grad + fuser2 + optimized execution ("kitchen sink"): 0.68s
    # Kitchen sink + dynamic quantization: 0.68s
    # Kitchen sink + manual fusing: 0.59s
    # Kitchen sink + manual fusing + static quantization: 0.32s
    def export_model(
        self, path: str, quantize: bool, n_calibration_batches: int = 64
    ) -> None:
        _logger.info("Starting model export.")

        _logger.debug("Setting up model.")
        training_mode = self.model.training
        model = self.model.eval().cpu()
        model.fuse_inplace()

        if quantize:
            _logger.debug("Running quantization and calibration.")
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(model, inplace=True)
            calibration_batches = iter(self.train_dataloader())
            for _ in range(n_calibration_batches):
                model(next(calibration_batches))
            torch.quantization.convert(model, inplace=True)

        _logger.debug("Tracing and optimizing.")
        # TODO: move board features into the scripted function as constants
        with torch.jit.optimized_execution(True), torch.jit.fuser("fuser2"):
            traced = torch.jit.trace(
                model,
                example_inputs=torch.zeros(
                    [1, example.N_BOARD_FEATURES, game.BOARD_EDGE, game.BOARD_EDGE],
                    device="cpu",
                ),
            )
            traced = torch.jit.freeze(traced)

        _logger.info("Exporting model script to:", path)
        traced.save(path)

        # Restore training mode
        self.model.train(training_mode)

    def _total_loss(self, policy_loss, value_loss):
        unnormalized = policy_loss + (self.hparams.value_loss_weight * value_loss)
        return unnormalized / (1 + self.hparams.value_loss_weight)


# TODO: move somewhere else
# TODO: support sampling ratios
class CompressedGameDataset(torch.utils.data.Dataset):
    def __init__(self, wthor_data: np.ndarray, logistello_data: np.ndarray):
        self._wthor_data = wthor_data
        self._logistello_data = logistello_data

    def __getitem__(self, index) -> Example:
        wthor_lines = self._wthor_data.shape[0]

        if index < wthor_lines:
            compressed = self._wthor_data[index]
        else:
            compressed = self._logistello_data[index - wthor_lines]

        return Example.decompress(compressed)

    def __len__(self):
        return self._wthor_data.shape[0] + self._logistello_data.shape[0]


class ImitationData(pl.LightningDataModule):
    """
    A DataModule which samples from Logistello and WTHOR data and applies data
    augmentation.
    """

    def __init__(
        self,
        wthor_data_path: str,
        logistello_data_path: str,
        val_frac: float,
        batch_size: int,
        augment_square_symmetries: bool,
        data_workers: int,
    ):
        super().__init__()
        self._wthor_data_path = wthor_data_path
        self._logistello_data_path = logistello_data_path
        self._val_frac = val_frac
        self._batch_size = batch_size
        self._augment_square_symmetries = augment_square_symmetries
        self._data_workers = data_workers

    def setup(self, stage=None):
        _logger.debug("Loading imitation training dataset.")

        wthor_data = np.load(self._wthor_data_path)
        logistello_data = np.load(self._logistello_data_path)
        data = CompressedGameDataset(wthor_data, logistello_data)

        val_lines = int(len(data) * self._val_frac)
        self._train_ds, self._val_ds = torch.utils.data.random_split(
            data,
            [len(data) - val_lines, val_lines],
            generator=torch.Generator().manual_seed(RANDOM_SEED),
        )

        self._board_features = example.get_board_features("cpu")

    def on_before_batch_transfer(self, batch: Example, _):
        # Augmentation, if applied
        if self._augment_square_symmetries:
            batch = example.augment_square_symmetries(batch)

        # Add board features, which aren't rotation/flip invariant
        batch_size = batch.board.size(0)
        board_features = self._board_features.unsqueeze(0).repeat([batch_size, 1, 1, 1])
        batch = batch._replace(board=torch.cat([batch.board, board_features], dim=1))

        return batch

    # TODO: val loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_ds,
            self._batch_size,
            num_workers=self._data_workers,
            pin_memory=True,
        )


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

        # Visualize model predictions on the first example in the batch
        board = outputs["batch"].board[0, :2]
        board_img = torch.zeros([3, 8, 8], dtype=float, device="cpu")
        board_img[0] = board[0]  # Active player's stones are red
        board_img[2] = board[1]  # Opponent's stones are blue

        # Show legal moves in green
        legal_moves = torch.clone(board_img)
        legal_moves[1] = outputs["batch"].move_mask[0].detach().cpu()

        # Show policy target in green
        policy_target = torch.clone(board_img)
        policy_target[1] = outputs["batch"].policy_target[0]

        # Show policy predictions in green
        policy_scores_flat = outputs["policy_scores_flat"].detach().cpu()
        if module.hparams.mask_invalid_moves:
            move_mask_flat = outputs["batch"].move_mask.flatten(1).detach().cpu()
            policy_scores_flat[~move_mask_flat] = float("-inf")
        policy_probs_flat = policy_scores_flat.softmax(1)

        policy_preds = torch.clone(board_img)
        policy_preds[1] = policy_probs_flat[0].reshape(
            [game.BOARD_EDGE, game.BOARD_EDGE]
        )

        module.logger.experiment.log(
            {
                "policy/distribution": wandb.Histogram(policy_probs_flat),
                "value/distribution": wandb.Histogram(outputs["value"].detach().cpu()),
                "trainer/global_step": trainer.global_step,
                "visualization/legal_moves": wandb.Image(
                    legal_moves, caption="Legal moves"
                ),
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
    training_module = hydra.utils.instantiate(config.model)

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
