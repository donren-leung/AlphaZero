import argparse
import json
import logging
import pathlib
import sys

import numpy as np
import torch

from .games.TicTacToe import TicTacToeGame
from .games.ConnectFour import ConnectFourGame
from .games.GameBase import GameBase
from .MCTS.MCTS_multichild import MCTS_Factory
from .MCTS.MCTS_AlphaZero import MCTS_Factory as MCTS_Factory_AZ
from .models.model import ResNet


# Return action -1 to quit
def human_gamelogic(move_legalities, curr_player) -> int:
    while True:
        try:
            cmd = input(f"Human (player {curr_player}):").strip()
        except EOFError:
            return -1
        if cmd == "q" or cmd == "":
            return -1

        try:
            action = int(cmd)
        except ValueError:
            print(f"Your input: {cmd} (numbers only). Press q to quit.")
            continue

        if move_legalities[action] == 0:
            print("move not legal\n")
            continue

        return action

def main(args) -> None:
    game: GameBase
    match args.game:
        case "tictac":
            game = TicTacToeGame()
        case "c4":
            game = ConnectFourGame()
        case _:
            raise ValueError("Unknown game '{args.game}'")

    MCTS_factory: MCTS_Factory | MCTS_Factory_AZ
    model: ResNet | None = None
    if args.model is not None:
        if args.model_args is None:
            raise ValueError("If --model is specified, --model-args must also be specified.")
        model_args_path = args.model_args
        if not model_args_path.exists():
            raise FileNotFoundError(f"Model args file {model_args_path} does not exist.")
        with open(model_args_path, "r") as f:
            model_args = json.load(f)

        model = ResNet(**model_args)
        model.load_state_dict(torch.load(args.model))
        model.to("cuda")
        model.eval()

        MCTS_factory = MCTS_Factory_AZ(args.rollouts)
    else:
        MCTS_factory = MCTS_Factory(args.rollouts, args.multi_sims, args.processes)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
        MCTS_factory.set_debug_state(args.debug)

    if args.exploration is not None:
        MCTS_factory.set_exploration_param(args.exploration)

    # state = TicTacToeState(np.asarray([[1, 0, 0], [-1, -1, 0], [0, 0, 0]]))
    # game = TicTacToeGame(state)

    curr_player = 1
    if args.first:
        print("Human selected to go first.")
        computer_players = [-1]
    elif args.second:
        print("Computer selected to go first.")
        computer_players = [1]
    elif args.both_human:
        print("Human self-play selected.")
        computer_players = []
    else:
        print("Computer self-play selected.")
        computer_players = [-1, 1]

    while True:
        print(game, flush=True)
        curr_player = game.current_player
        move_legalities = game.get_legal_actions()
        print("legal moves", [i for i in range(game.action_size) if move_legalities[i]])

        if curr_player in computer_players:
            mcts_instance = MCTS_factory.make_instance(game=game, model=model)
            result = mcts_instance.search()

            action = result.best_action
            print(f"Computer (player {curr_player}): {action}.\nThe visits were:")
            print(result)
        else:
            action = human_gamelogic(move_legalities, curr_player)
            if action == -1:
                print("Quitting game...")
                break

        game.make_move(action)
        value, terminated = game.get_value_and_terminated(action)

        if terminated:
            print(game)
            if value == 1:
                print(f"Player {curr_player} won!")
            else:
                print("Draw")
            break

        # curr_player = game.get_opponent(curr_player)
        print()

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='TicTacToe/C4 with MCTS')

    parser.add_argument('game', choices=['tictac', 'c4'], type=str,
                        help='Play tictactoe or connect four')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--f', '--1', '--first', dest='first', action='store_true',
                       help='Player goes first')
    group.add_argument('--s', '--2', '--second', dest='second', action='store_true',
                       help='Player goes second')

    group.add_argument('--h', '--both-human', dest='both_human', action='store_true',
                       help='Both Players are human (self-play)')
    group.add_argument('--c', '--both-comp', dest='both_comp', action='store_true',
                       help='Both Players are PC (self-play)')


    parser.add_argument('-r', '--rollouts', dest='rollouts', type=int, default=1000,
                        help='Number of MCTS rollouts per move for the PC (default: %(default)s)')

    parser.add_argument('-e', '--exploration', dest='exploration', type=float, default=MCTS_Factory.DEFAULT_EXPLORATION_PARAM,
                        help='Exploration parameter for MCTS node selection')

    parser.add_argument('-m', '--multi', dest='multi_sims', type=int, default=1,
                        help='How many simulations to make in simulation phase (default: %(default)s)')

    parser.add_argument('-p', '--processes', dest='processes', type=int, default=1,
                        help='Number of CPUs to utilise (default: %(default)s)')

    parser.add_argument('-d', '--debug', dest='debug', choices=[1, 2], type=int,
                        help='Enable debug mode')

    parser.add_argument('-w', '--model-weights', dest='model', type=pathlib.Path, default=None,
                        help='Path to model file for MCTS evaluation (default: %(default)s)')
    parser.add_argument('--model-args', dest='model_args', type=pathlib.Path, default=None,
                        help='Path to model args file for MCTS evaluation (default: %(default)s)')

    # Parse the command-line arguments
    args = parser.parse_args()
    main(args)
