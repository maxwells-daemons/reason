"""
Debugging script to play against the policy network.
"""

# TODO: move this somewhere else

import argparse
import functools


from python import training, network, game, data
import torch


def load_model(path):
    module = training.TrainingModule(
        learning_rate=0,
        value_loss_weight=0,
        value_target="outcome",
        model=network.AgentModel(n_channels=128, n_blocks=3),
    )
    ckpt = torch.load(path)
    module.load_state_dict(ckpt["state_dict"])
    return module.model.eval()


def featurize(game):
    return torch.cat([game.board, data.example.get_board_features("cpu")], dim=0)


def get_human_move(game_state):
    mv_mask = game_state.get_move_mask()

    while True:
        mv_str = input("Enter a move please: ").strip()

        try:
            row, col = game.parse_move(mv_str)
        except Exception as e:
            print("Internal error:\n", e)
            continue

        if not mv_mask[row, col].item():
            print("Invalid move")
            continue

        return row, col


def get_agent_move(game_state, model):
    with torch.no_grad():
        features = featurize(game_state)
        policy, value = model(features.unsqueeze(0))
        policy = policy[0]
        value = value[0]

    move_mask = game_state.get_move_mask()
    policy[~move_mask] = float("-inf")
    policy = policy.flatten().softmax(0).reshape(policy.shape)

    header = f"------- {game_state.active_player} -------"
    print(header)
    print((policy * 100).int().numpy())
    print("Value:", value.item())
    print("-" * len(header) + "\n")

    out = (policy == torch.max(policy)).nonzero(as_tuple=False)

    return tuple(out[0].numpy())


def play_game(black_move_fn, white_move_fn):
    g = game.starting_state()

    move_fns = {
        game.Player.BLACK: black_move_fn,
        game.Player.WHITE: white_move_fn,
    }

    while True:
        print("\n", g)
        move_mask = g.get_move_mask()

        if not move_mask.any().item():
            if g.just_passed:
                break

            g = g.apply_pass()
            continue

        mv = move_fns[g.active_player](g)
        print(f"{g.active_player} played: {game.format_move(*mv)}")
        g = g.apply_move(*mv)

    black_score = g.score_absolute_difference(game.Player.BLACK)
    if black_score == 0:
        print("Draw.")
    elif black_score > 0:
        print("Black wins.")
    else:
        print("White wins.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("black_ckpt", type=str)
    parser.add_argument("white_ckpt", type=str)
    args = parser.parse_args()

    if args.black_ckpt == "human":
        black_move_fn = get_human_move
    else:
        print("Loading black model...")
        black_model = load_model(args.black_ckpt)
        black_move_fn = functools.partial(get_agent_move, model=black_model)

    if args.white_ckpt == "human":
        white_move_fn = get_human_move
    else:
        print("Loading black model...")
        white_model = load_model(args.white_ckpt)
        white_move_fn = functools.partial(get_agent_move, model=white_model)

    print("Begin play!")
    play_game(black_move_fn, white_move_fn)


if __name__ == "__main__":
    main()