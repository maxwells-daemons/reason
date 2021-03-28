"""
Code for parsing loading Logistello self-play data.
"""

import itertools
from typing import Iterator, List

import more_itertools
import torch
from src import game
from src.data.example import Example


def parse_game(line: str) -> Iterator[Example]:
    """
    Parse a Logistello game into a list of examples.
    """
    last_state = game.starting_state()
    states: List[game.GameState] = []
    move_masks: List[torch.Tensor] = []

    board_str, score_str, _ = line.split()
    for [player_str, *move_str] in more_itertools.ichunked(board_str[:-1], 3):
        player = game.Player.BLACK if player_str == "+" else game.Player.WHITE
        move = game.parse_move(move_str)  # type: ignore

        # Passing does not create an example
        if player != last_state.active_player:
            last_state = last_state.apply_pass()

        states.append(last_state)
        move_masks.append(game.make_move_mask(*move))
        last_state = last_state.apply_move(*move)

    # Internal consistency check
    assert last_state.score_absolute_difference(game.Player.BLACK) == int(score_str)

    # NOTE: no example for the last board
    for state, move_mask in zip(states, move_masks):
        yield Example(
            state.board.float(),
            torch.tensor(
                last_state.score_absolute_difference(state.active_player),
                dtype=torch.float,
            ),
            move_mask.float(),
        )


class LogistelloDataset(torch.utils.data.IterableDataset):
    """
    A Dataset of Logistello games.
    """

    def __init__(self, path: str):
        super(LogistelloDataset, self).__init__()

        with open(path, "r") as f:
            self._lines = f.readlines()

        self._n_examples = 0
        for line in self._lines:
            moves, _ = line.split(":")
            assert len(moves) % 3 == 0
            self._n_examples += len(moves) // 3

    def __len__(self):
        return self._n_examples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_lines = self._lines
        else:
            worker_lines = itertools.islice(
                self._lines, worker_info.id, None, worker_info.num_workers
            )

        yield from itertools.chain.from_iterable(map(parse_game, worker_lines))
