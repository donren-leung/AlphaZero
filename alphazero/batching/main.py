import logging
import sys

import argparse
from multiprocessing import Process
from typing import Type

from alphazero.games.GameBase import GameBase
from alphazero.games.TicTacToe import TicTacToeGame
from alphazero.games.ConnectFour import ConnectFourGame
from alphazero.MCTS_batch import MCTS_Factory

from .NodeBatch import NodeBatchRequest, NodeBatchResponse
from .GameWorker import GameWorker, mpQueueGen
from .CPUPool import CPUPool


GameType = type[TicTacToeGame] | type[ConnectFourGame]

def parse_game_type(value: str) -> GameType:
    game_types: dict[str, GameType] = {
        "TicTacToe": TicTacToeGame,
        "ConnectFour": ConnectFourGame,
    }

    try:
        return game_types[value]
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"invalid game type {value!r}; "
            f"choose from: {', '.join(game_types)}"
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batched parallel MCTS self-play."
    )

    parser.add_argument(
        "--rollouts",
        type=int,
        default=1000,
        help="Number of MCTS rollouts/visits per move (default: 1000)",
    )

    parser.add_argument(
        "--multi-sims",
        type=int,
        default=10,
        help=(
            "Number of random simulations performed for each position "
            "in an evaluation batch (default: 10)"
        ),
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=8,
        help="Number of node-evaluation worker processes (default: 8)",
    )

    parser.add_argument(
        "--game-workers",
        type=int,
        default=2,
        help="Number of GameWorker processes (default: 2)",
    )

    parser.add_argument(
        "--games-per-worker",
        type=int,
        default=8,
        help="Number of concurrent games/threads per GameWorker (default: 8)",
    )

    parser.add_argument(
        "--game-type",
        type=parse_game_type,
        default=ConnectFourGame,
        metavar="{TicTacToe,ConnectFour}",
        help="Type of game to play (default: ConnectFour)",
    )

    return parser.parse_args()

def main(game_type: Type[GameBase], ROLLOUTS: int, MULTI_SIMS: int, PROCESSES: int, TARGET_GAME_WORKERS: int, GAME_WORKER_GAMES: int) -> None:
    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    ## Init
    # CPU or (in the future) GPU
    pool_type = CPUPool

    MCTS_factory = MCTS_Factory(ROLLOUTS, MULTI_SIMS, PROCESSES)
    target_eval_workers = MCTS_factory.processes
    # eval_workers: list[CPUPool] = []

    game_worker_ps: list[Process] = []
    all_game_results: mpQueueGen[tuple[str, GameBase]] = mpQueueGen()

    # 1 queue for ALL game_workers --- sending to ---> ALL eval_workers
    request_queue: mpQueueGen[NodeBatchRequest] = mpQueueGen()
    # N queues for ALL eval_workers --- sending to ---> N * game_workers queues
    results_queues: list[mpQueueGen[NodeBatchResponse]] = [mpQueueGen()
                                                    for _ in range(TARGET_GAME_WORKERS)]

    for i in range(target_eval_workers):
        eval_worker = pool_type(request_queue, results_queues)
        p = Process(target=eval_worker.run, name=f"EvalWorker_{i}", daemon=True)
        p.start()

    for i, results_queue in enumerate(results_queues):
        game_worker = GameWorker(game_type,
                                 num_games=GAME_WORKER_GAMES, output_games=all_game_results,
                                 in_queue=results_queue,
                                 out_queue=request_queue,
                                 worker_id=i,
                                 MCTS_factory=MCTS_factory)
        p = Process(target=game_worker.run, name=f"GameWorker_{i}")
        game_worker_ps.append(p)

    for game_worker_p in game_worker_ps:
        game_worker_p.start()

    expecting_results = TARGET_GAME_WORKERS * GAME_WORKER_GAMES
    for _ in range(expecting_results):
        id_, game = all_game_results.get()

        print(f"{id_=}")
        print(game.action_history)
        print(game.state)

    for game_worker_p in game_worker_ps:
        game_worker_p.join()

if __name__ == "__main__":
    args = parse_args()
    main(args.game_type, args.rollouts, args.multi_sims, args.processes, args.game_workers, args.games_per_worker)
