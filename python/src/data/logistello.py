"""
Code for parsing loading Logistello self-play data.
"""


from typing import List, Tuple

import more_itertools
from src import game

LOGISTELLO_PATH = "resources/logistello/logbook.gam"


def parse_game(line: str) -> Tuple[List[game.GameState], int]:
    """
    Parse a Logistello game into a list of game states and a final score.
    The score is an "absolute difference" score from Black's perspective.
    """
    state = game.starting_state()
    states = [state]

    board_str, score_str, _ = line.split()
    for [player_str, *move] in more_itertools.ichunked(board_str[:-1], 3):
        player = game.Player.BLACK if player_str == "+" else game.Player.WHITE
        row, col = game.parse_move(move)  # type: ignore

        if player != state.active_player:
            state = state.apply_pass()
            states.append(state)

        state = state.apply_move(row, col)
        states.append(state)

    return states, int(score_str)