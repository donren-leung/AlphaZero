import argparse
import json
import logging
import pathlib
import sys

import torch
import numpy as np
from numpy.typing import NDArray

from .games.TicTacToe import TicTacToeGame
from .games.ConnectFour import ConnectFourGame
from .games.Chess import ChessGame
from .games.ChessUtils import decode_action, encode_move
from .games.GameBase import GameBase
# from .MCTS.MCTS_multichild import MCTS_Factory
from .MCTS.MCTS_AlphaZero import MCTS_Factory
from .models.model import ResNet
from .utils import parse_game_type, game_types

# Return action -1 to quit
def human_gamelogic(move_legalities: NDArray[np.bool_], game: GameBase) -> int:
    while True:
        try:
            cmd = input(f"Human (player {game.current_player}):").strip()
        except EOFError:
            return -1
        if cmd == "q" or cmd == "":
            return -1

        try:
            action = game.parse_move_str(cmd)
        except ValueError as e:
            print(f"Your input: {cmd}. Error: {e}. Press q to quit.")
            continue

        if move_legalities[action] == 0:
            print("move not legal\n")
            continue

        return action

def main(args) -> None:
    game: GameBase = args.game_type()
    # game = ChessGame()

    MCTS_factory: MCTS_Factory | None = None
    model: ResNet | None = None

    if not args.both_human:
        if args.model_args is None:
            # check for args.json in the same directory as the model file
            model_args_path = args.model.parent / "args.json"
            if not model_args_path.exists():
                raise ValueError("Cannot automatically find model args file. --model-args must also be specified.")
        else:
            model_args_path = args.model_args

        if not model_args_path.exists():
            raise FileNotFoundError(f"Model args file {model_args_path} does not exist.")
        with open(model_args_path, "r") as f:
            model_args = json.load(f)

        model = ResNet(**model_args)
        model.load_state_dict(torch.load(args.model))
        model.to("cuda")
        model.eval()

        MCTS_factory = MCTS_Factory(args.rollouts)

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
        legal_actions = game.get_legal_actions()
        legal_moves = game.get_legal_moves(legal_actions)
        print("legal moves", legal_moves)

        if curr_player in computer_players:
            assert MCTS_factory is not None
            mcts_instance = MCTS_factory.make_instance(game=game, model=model)
            result = mcts_instance.search(game.state)

            action = result.best_action
            best_move_str = result.best_move_str
            print(f"Computer (player {curr_player}): {best_move_str}.\nThe visits were:")
            print(result)
        else:
            action = human_gamelogic(legal_actions, game)
            if action == -1:
                print("Quitting game...")
                break

        game.make_move(action)
        value, terminated = game.get_value_and_terminated()

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
    parser = argparse.ArgumentParser(description='Two player games with AlphaZero using MCTS')

    parser.add_argument(
        "--game-type", "-game",
        type=parse_game_type,
        default=ChessGame,
        metavar=f"{','.join(game_types.keys())}",
        help="Type of game to play (default: chess)",
    )

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

    parser.add_argument('-w', '--model-weights', dest='model', type=pathlib.Path,
                        help='Path to model file for MCTS evaluation (default: %(default)s)')
    parser.add_argument('--model-args', dest='model_args', type=pathlib.Path, default=None,
                        help='Path to model args file for MCTS evaluation (default: %(default)s)')

    # Parse the command-line arguments
    args = parser.parse_args()

    if not args.model and not args.both_human:
        parser.error("If not both human players, --model-weights must be specified.")

    main(args)
