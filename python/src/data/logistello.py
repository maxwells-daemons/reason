"""
Code for parsing loading Logistello self-play data.
"""

import itertools
import random
from typing import List, Optional, Tuple

import more_itertools
import pytorch_lightning as pl
import torch
import torchvision
from src import game
from src.game import Player

LOGISTELLO_PATH = "resources/logistello/logbook.gam"


def parse_game(
    line: str,
) -> Tuple[List[game.GameState], List[Optional[Tuple[int, int]]], int]:
    """
    Parse a Logistello game into:
        - A list of game states.
        - A move out of each position, as a row and column, or None if pass.
        - A final "absolute difference" score from Black's perspective.
    """
    state = game.starting_state()
    states = [state]
    moves: List[Optional[Tuple[int, int]]] = []

    board_str, score_str, _ = line.split()
    for [player_str, *move_str] in more_itertools.ichunked(board_str[:-1], 3):
        player = game.Player.BLACK if player_str == "+" else game.Player.WHITE

        move = game.parse_move(move_str)  # type: ignore

        if player != state.active_player:
            state = state.apply_pass()
            moves.append(None)
            states.append(state)

        moves.append(move)  # NOTE: associated to the _prior_ board
        state = state.apply_move(*move)
        states.append(state)

    moves.append(None)  # No move out of the last board
    return states, moves, int(score_str)


class LogistelloDataset(torch.utils.data.IterableDataset):
    """
    A Dataset of Logistello games.
    It's recommended to access this through the LogistelloDataModule.
    """

    def __init__(self, lines: List[str]):
        super(LogistelloDataset, self).__init__()
        self._lines = lines

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_lines = iter(self._lines)
        else:
            worker_lines = itertools.islice(
                self._lines, worker_info.id, None, worker_info.num_workers
            )

        return itertools.chain.from_iterable(map(self._game_examples, worker_lines))

    def _game_examples(self, game: str):
        states, moves, black_score = parse_game(game)
        states_and_moves = list(zip(states, moves))

        # Shuffle within a single game
        random.shuffle(states_and_moves)

        draw = black_score == 0
        black_wins = black_score > 0

        for (state, move) in states_and_moves:
            # Skip pass moves
            if move is None:
                continue

            board = state.board

            # NOTE: getting the move index by explicitly building the board is
            # convenient, but inefficient
            move_mask = torch.zeros([8, 8], dtype=bool)  # type: ignore
            row, col = move
            move_mask[row, col] = True

            if draw:
                outcome = torch.tensor(0.0)
            elif black_wins and (state.active_player == Player.BLACK):
                outcome = torch.tensor(1.0)
            elif (not black_wins) and (state.active_player == Player.WHITE):
                outcome = torch.tensor(1.0)
            else:
                outcome = torch.tensor(-1.0)

            # Augment the board and move mask together with random flips and rotations
            if torch.rand(1) < 0.5:
                board = torchvision.transforms.functional.hflip(board)
                move_mask = torchvision.transforms.functional.hflip(move_mask)

            if torch.rand(1) < 0.5:
                board = torchvision.transforms.functional.vflip(board)
                move_mask = torchvision.transforms.functional.vflip(move_mask)

            rotations = torch.randint(size=(1,), low=0, high=4).item()
            board = torch.rot90(board, k=rotations, dims=[1, 2])  # type: ignore
            move_mask = torch.rot90(move_mask, k=rotations, dims=[0, 1])  # type: ignore

            # TODO: re-enable
            # # Augment the board and outcome by flipping the active player
            # if torch.rand(1) < 0.5:
            #     board = board[[1, 0], :, :]
            #     outcome *= -1

            # Get the move index by argmax
            move_index = move_mask.reshape(-1).int().argmax()

            yield board.float(), move_index, outcome


class LogistelloDataModule(pl.LightningDataModule):
    """
    A DataModule handling splitting games into training and validation sets.
    """

    def __init__(
        self,
        path: str = LOGISTELLO_PATH,
        batch_size: int = 32,
        val_frac: float = 0.3,
        data_workers: int = 12,
    ):
        super(LogistelloDataModule, self).__init__()
        self.path = path
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.data_workers = data_workers

    def setup(self, stage: Optional[str] = None):
        with open(self.path, "r") as f:
            games = f.readlines()

        random.shuffle(games)
        n_val_games = int(self.val_frac * len(games))

        self._train_ds = LogistelloDataset(games[n_val_games:])
        self._val_ds = LogistelloDataset(games[:n_val_games])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            num_workers=self.data_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            num_workers=self.data_workers,
            pin_memory=True,
        )