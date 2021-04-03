"""
Code for working with the WThor database, available at
http://www.ffothello.org/informatique/la-base-wthor/.

Adapted from my (Aidan Swope) code originally at
https://github.com/maxwells-daemons/seldon/blob/master/src/scripts/parse_wthor.py
released under the MIT license.
"""

import itertools
from glob import glob
from typing import Iterator, Tuple

import torch
from python import game
from python.data.example import Example

DB_HEADER_BYTES = 16  # Bytes in the header of a WTHOR database
GAME_BYTES = 68  # Bytes in a WTHOR game's data
GAME_HEADER_BYTES = 8  # Bytes in a WTHOR game's header


def parse_move(move_encoding: int) -> Tuple[int, int]:
    col = move_encoding % 10 - 1
    row = move_encoding // 10 - 1
    return row, col


def parse_game(game_bytes: bytes) -> Iterator[Example]:
    assert len(game_bytes) == GAME_BYTES
    move_bytes = game_bytes[GAME_HEADER_BYTES:]

    last_state = game.starting_state()
    states = []
    move_masks = []

    for byte in move_bytes:
        move = parse_move(byte)
        if move == (-1, -1):  # Game's over
            break

        if not last_state.get_move_mask().any():  # No moves: this player must pass
            last_state = last_state.apply_pass()

        states.append(last_state)
        move_masks.append(game.make_move_mask(*move))
        last_state = last_state.apply_move(*move)

    for state, move_mask in zip(states, move_masks):
        yield Example(
            state.board.float(),
            torch.tensor(
                last_state.score_absolute_difference(state.active_player),
                dtype=torch.float,
            ),
            move_mask.float(),
        )


class WthorDataset(torch.utils.data.IterableDataset):
    """
    A dataset of WTHOR games.
    """

    def __init__(self, path_glob):
        super(WthorDataset, self).__init__()

        self._paths = glob(path_glob)

        self._n_examples = 0
        for path in self._paths:
            with open(path, "rb") as f:
                db_bytes = f.read()

            data_bytes = db_bytes[DB_HEADER_BYTES:]
            for i in range(len(data_bytes) // GAME_BYTES):
                game_bytes = list(data_bytes[i * GAME_BYTES : (i + 1) * GAME_BYTES])

                self._n_examples += 61 - game_bytes.count(0)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_paths = self._paths
        else:
            worker_paths = itertools.islice(
                self._paths, worker_info.id, None, worker_info.num_workers
            )

        for path in worker_paths:
            yield from self._shard_examples(path)

    @staticmethod
    def _shard_examples(path: str) -> Iterator[Example]:
        with open(path, "rb") as f:
            db_bytes = f.read()

        data_bytes = db_bytes[DB_HEADER_BYTES:]
        for i in range(len(data_bytes) // GAME_BYTES):
            game_bytes = data_bytes[i * GAME_BYTES : (i + 1) * GAME_BYTES]
            yield from parse_game(game_bytes)