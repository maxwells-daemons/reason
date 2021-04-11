"""
Defines the training strategy.
"""

import logging
from typing import List

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from python import game, utils
from python.data import example
from python.data.example import Example
from python.network import AgentModel

_logger = logging.getLogger(__name__)


# Hardcoded for experiment repeatability
RANDOM_SEED = 1337
VAL_FRAC = 0.1


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

    def forward(self, board):
        return self.model(board)

    def _shared_step(self, batch, mask_invalid_moves, log_suffix):
        """Do work common to `training_step` and `validation_step`."""

        # Forward pass
        board, target_score, target_move_probs, valid_move_mask = batch
        policy_scores, value = self(board)

        # Policy loss and accuracy
        policy_scores_flat = policy_scores.flatten(1)
        target_move_probs_flat = target_move_probs.flatten(1)
        policy_logprobs_flat = utils.masked_log_softmax(
            policy_scores_flat,
            valid_move_mask.flatten(1) if mask_invalid_moves else None,
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
        value_accuracy = (value_cls_preds == value_cls_targets).float().mean()

        # Total loss
        loss = self._total_loss(policy_loss, value_loss)

        # Log with a suffix
        self.log_dict(
            {
                f"loss/policy.{log_suffix}": policy_loss,
                f"loss/value.{log_suffix}": value_loss,
                f"loss/total.{log_suffix}": loss,
                f"policy/accuracy.{log_suffix}": policy_accuracy,
                f"value/accuracy.{log_suffix}": value_accuracy,
            }
        )

        return {
            "loss": loss,
            "policy_scores_flat": policy_scores_flat,
            "policy_loss": policy_loss,
            "policy_accuracy": policy_accuracy,
            "value": value,
            "value_loss": value_loss,
            "value_accuracy": value_accuracy,
        }

    def training_step(self, batch, _):
        shared_outs = self._shared_step(
            batch,
            mask_invalid_moves=self.hparams.mask_invalid_moves,
            log_suffix="training",
        )

        return {
            "loss": shared_outs["loss"],
            "policy_scores_flat": shared_outs["policy_scores_flat"].float(),
            "value": shared_outs["value"].float(),
            "batch": batch,
        }

    def validation_step(self, batch, _):
        self._shared_step(batch, mask_invalid_moves=True, log_suffix="validation")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

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
class CompressedGameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: np.ndarray,
        augment_square_symmetries: bool,
    ):
        self._data = data
        self._augment_square_symmetries = augment_square_symmetries
        self._board_features = example.get_board_features("cpu")

    def __getitem__(self, index) -> Example:
        ex = Example.decompress(self._data[index])

        if self._augment_square_symmetries:
            ex = example.augment_square_symmetries(ex)

        # Add board features, which aren't rotation/flip invariant
        ex = ex._replace(board=torch.cat([ex.board, self._board_features], dim=0))

        return ex

    def __len__(self):
        return self._data.shape[0]


# TODO: support sampling ratios
class ImitationData(pl.LightningDataModule):
    """
    A DataModule which samples from Logistello and WTHOR data and applies data
    augmentation.
    """

    def __init__(
        self,
        data_paths: List[str],
        batch_size: int,
        augment_square_symmetries: bool,
        data_workers: int,
    ):
        super().__init__()
        self._data_paths = data_paths
        self._batch_size = batch_size
        self._augment_square_symmetries = augment_square_symmetries
        self._data_workers = data_workers

    def setup(self, stage=None):
        _logger.debug("Loading imitation training dataset.")
        data_by_path = [np.load(path) for path in self._data_paths]
        all_data = np.concatenate(data_by_path, axis=0)

        # Take the first VAL_FRAC positions as validation data.
        # Don't split randomly, because outcomes within games are correlated.
        # NOTE: this means the order of `data_paths` matters.
        val_lines = int(all_data.shape[0] * VAL_FRAC)
        val_data = all_data[:val_lines, :]
        train_data = all_data[val_lines:, :]

        # Pre-shuffle data
        rng = np.random.default_rng(seed=RANDOM_SEED)
        rng.shuffle(train_data)
        rng.shuffle(val_data)

        # Build datasets
        self._train_ds = CompressedGameDataset(
            train_data, self._augment_square_symmetries
        )
        self._val_ds = CompressedGameDataset(val_data, False)  # Don't augment val data

        _logger.info(
            f"Loaded {train_data.shape[0]} training positions and {val_data.shape[0]} validation positions."
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_ds,
            self._batch_size,
            num_workers=self._data_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_ds,
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
