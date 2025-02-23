import numpy as np
from TicTacToe import TicTacToeGame
from MCTS import MCTS_Instance

import argparse
import logging
import sys

np.__version__

# Return action -1 to quit
def human_gamelogic(move_legalities, curr_player) -> int:
    while True:
        try:
            cmd = input(f"Human (player {curr_player}):").strip()
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
        move_legalities = game.get_legal_actions()
        print("legal moves", [i for i in range(game.action_size) if move_legalities[i]])

        if curr_player in computer_players:
            mcts_instance = MCTS_Instance(args.rollouts, game.state, curr_player)
            result = mcts_instance.search()

            action = result.best_action
            print(f"Computer (player {curr_player}): {action}.\nThe visits were:")
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
                print(f"Human player {curr_player} won!")
            else:
                print("Draw")
            break
        
        curr_player = game.get_opponent(curr_player)
        print()

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='TicTacToe with MCTS')


    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--f', '--1', '--first', dest='first', action='store_true',
                       help='Player goes first')
    group.add_argument('--s', '--2', '--second', dest='second', action='store_true',
                       help='Player goes second')
    
    group.add_argument('--h', '--both-human', dest='both_human', action='store_true',
                       help='Both Players are human (self-play)')
    group.add_argument('--c', '--both-comp', dest='both_comp', action='store_true',
                       help='Both Players are PC (self-play)')

    
    parser.add_argument('-r', '--rollouts', dest='rollouts', type=int, default=4000,
                        help='Number of MCTS rollouts per move for the PC (default: %(default)s)')
    
    parser.add_argument('--d', '--debug', dest='debug', action='store_true', help='Enable debug mode')


    # Parse the command-line arguments
    args = parser.parse_args()    
    main(args)
