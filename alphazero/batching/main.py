import logging
import sys

from math import nan
from multiprocessing import Process
from queue import Queue
from threading import Thread, local, Condition
from typing import Generic, TypeVar, Type

from batching.NodeBatch import NodeBatchRequest, NodeBatchResponse
from batching.GameWorker import GameWorker, mpQueueGen
from batching.CPUPool import CPUPool
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from games.GameBase import GameBase
from games.GameStateBase import GameStateBase
from games.TicTacToe import TicTacToeGame
from games.ConnectFour import ConnectFourGame
from MCTS_batch import MCTS_Factory, Node, simulate_

def main() -> None:
    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    game_type = ConnectFourGame
    MCTS_factory = MCTS_Factory(1000, 10, 15)

    target_eval_workers = MCTS_factory.processes
    # eval_workers: list[CPUPool] = []
    pool_type = CPUPool

    target_game_workers = 1
    game_worker_ps: list[Process] = []
    all_game_results: mpQueueGen[tuple[str, GameBase]] = mpQueueGen()

    # 1 * all game_workers -> all eval_worker
    request_queue: mpQueueGen[NodeBatchRequest] = mpQueueGen()
    # N * all eval_worker -> one game_workers
    results_queues: list[mpQueueGen[NodeBatchResponse]] = [mpQueueGen()
                                                    for _ in range(target_game_workers)]

    for i in range(target_eval_workers):
        eval_worker = pool_type(request_queue, results_queues)
        p = Process(target=eval_worker.run, name=f"EvalWorker_{i}", daemon=True)
        p.start()

    for i, results_queue in enumerate(results_queues):
        game_worker = GameWorker(game_type,
                                 num_games=15, output_games=all_game_results,
                                 in_queue=results_queue,
                                 out_queue=request_queue,
                                 worker_id=i,
                                 MCTS_factory=MCTS_factory)
        p = Process(target=game_worker.run, name=f"GameWorker_{i}")
        game_worker_ps.append(p)

    for game_worker_p in game_worker_ps:
        game_worker_p.start()

    # TODO: rework
    for game_worker_p in game_worker_ps:
        p.join()

    while not all_game_results.empty():
        id_, game = all_game_results.get()

        print(f"{id_=}")
        print(game.action_history)
        print(game.state)

if __name__ == "__main__":
    main()
