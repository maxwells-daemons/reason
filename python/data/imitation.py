"""
Defines how preprocessed imitation training data is loaded.
"""

import logging
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from python.data import example
from python.data.example import Example

# Hardcoded for experiment repeatability
RANDOM_SEED = 1337
VAL_FRAC = 0.1


_logger = logging.getLogger(__name__)


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
