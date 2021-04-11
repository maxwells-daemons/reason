"""
Defines the training strategy.
"""

import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from python import game, utils
from python.data import example
from python.network import AgentModel

_logger = logging.getLogger(__name__)


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
