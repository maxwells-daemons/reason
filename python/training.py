"""
Defines the training strategy.
"""

import logging

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from python import ffi, game
from python.data import example, logistello, utils, wthor
from python.data.example import Example
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
    """

    def __init__(
        self,
        n_channels: int,
        n_blocks: int,
        learning_rate: float,
        value_loss_weight: float,
        value_target: str,
    ):
        if value_target not in {"piece_difference", "outcome"}:
            raise ValueError("Unrecognized setting for `value_target`.")

        super(TrainingModule, self).__init__()
        self.save_hyperparameters()
        _logger.debug("Building training module.")

        self.model = AgentModel(n_channels=n_channels, n_blocks=n_blocks)
        self.policy_accuracy = pl.metrics.classification.Accuracy()
        self.value_accuracy = pl.metrics.classification.Accuracy()

    def forward(self, board):
        return self.model(board)

    def training_step(self, batch, _):
        board, target_score, target_move_probs = batch

        # Forward pass
        policy_scores, value = self(board)

        # Policy loss and accuracy
        # TODO: try masking out invalid moves
        policy_scores_flat = policy_scores.flatten(1)
        target_move_probs_flat = target_move_probs.flatten(1)
        policy_loss = self._soft_crossentropy(
            policy_scores_flat, target_move_probs_flat
        )

        # For policy accuracy, assume the "policy target" is the greedy max
        policy_probs_flat = policy_scores_flat.softmax(dim=1)
        policy_argmax_indices = target_move_probs_flat.max(dim=1).indices
        self.policy_accuracy(policy_probs_flat, policy_argmax_indices)

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
                "policy/accuracy_argmax": self.policy_accuracy,
                "value/accuracy": self.value_accuracy,
            }
        )

        return {
            "loss": loss,
            "policy_scores": policy_scores,
            "value": value,
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
        board = outputs["batch"].board[0, :2]
        board_img = torch.zeros([3, 8, 8], dtype=float, device="cpu")
        board_img[0] = board[0]  # Active player's stones are red
        board_img[2] = board[1]  # Opponent's stones are blue

        # Show legal moves in green
        legal_moves = torch.clone(board_img)
        legal_moves[1] = ffi.get_move_mask(board.cpu().bool())

        # Show policy target in green
        policy_target = torch.clone(board_img)
        policy_target[1] = outputs["batch"].policy_target[0]

        # Show policy predictions in green
        policy_preds = torch.clone(board_img)
        policy_preds[1] = (
            policy_scores[0].flatten(1).softmax(1).view(policy_scores.size(1), -1)
        )

        module.logger.experiment.log(
            {
                "policy/distribution": wandb.Histogram(policy_scores),
                "value/distribution": wandb.Histogram(value),
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
