import numpy as np
from TicTacToe import TicTacToeGame
from MCTS import MCTS_Instance

import argparse
import logging
import sys

np.__version__

# action -1 to quit
def human_gamelogic(move_legalities, curr_player) -> int:
    while True:
        try:
            cmd = input(f"Player {curr_player}:").strip()
            action = int(cmd)
        except ValueError:
            if cmd == "q" or cmd == "":
                return -1
            print(f"Your input: {cmd} (numbers only). Press q to quit.")
            continue
        except EOFError:
            return -1

        if move_legalities[action] == 0:
            print("move not legal\n")
            continue
        return action

def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    position = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    game = TicTacToeGame()

    curr_player = 1
    if args.first:
        computer_player = -1
    else:
        computer_player = 1

    while True:
        print(game, flush=True)
        move_legalities = game.get_legal_actions()
        print("legal moves", [i for i in range(game.action_size) if move_legalities[i]])

        if curr_player == computer_player:
            mcts_instance = MCTS_Instance(args.rollouts, game.state, curr_player)
            result = mcts_instance.search()

            action = result.best_action
            print(f"Computer: {action}.\nThe visits were:")
            print(result)
        else:
            action = human_gamelogic(move_legalities, curr_player)
            if action == -1:
                print("Quitting game...")
                break

        game.make_move(action, curr_player)
        value, terminated = game.get_value_and_terminated(action)

        if terminated:
            print(game)
            if value == 1:
                print(f"Player {curr_player} won!")
            else:
                print("Draw")
            break
        
        curr_player = game.get_opponent(curr_player)
        print()

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='TicTacToe against MCTS')

    # Add a flag argument. The action "store_true" means it will be set to True if the flag is provided.
    parser.add_argument('-r', '--rollouts', dest='rollouts', type=int, required=True, help='Number of rollouts per move')
    parser.add_argument('--f', '--first', dest='first', action='store_true', help='Player goes first')
    parser.add_argument('--d', '--debug', dest='debug', action='store_true', help='Enable debug mode')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    main(args)
