import logging
import sys
import queue
import time
from threading import Thread

import argparse
import multiprocessing as mp
from typing import Type

import torch

from alphazero.games.GameBase import GameBase
from alphazero.games.TicTacToe import TicTacToeGame
from alphazero.games.ConnectFour import ConnectFourGame
from alphazero.MCTS.MCTS_AlphaZero import MCTS_Factory
from alphazero.models.model import ResNet

from .batching.NodeBatch import AZ_NodeBatchRequest, AZ_NodeBatchResponse
from .batching.AZ_GameWorker import GameWorker, mpQueueGen
from .batching.Pool import CPU_RandomRollout_Worker, GPU_AZ_Worker, PoolFactory

GameType = type[TicTacToeGame] | type[ConnectFourGame]

def parse_game_type(value: str) -> GameType:
    game_types: dict[str, GameType] = {
        "ttt": TicTacToeGame,
        "c4": ConnectFourGame,
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
        "-r",
        type=int,
        default=1000,
        help="Number of MCTS rollouts/visits per move (default: 1000)",
    )

    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=8,
        help="Number of node-evaluation worker processes (default: 8)",
    )

    parser.add_argument(
        "--game-workers",
        "-gw",
        type=int,
        default=2,
        help="Number of GameWorker processes (default: 2)",
    )

    parser.add_argument(
        "--games-per-worker",
        "-gpw",
        type=int,
        default=8,
        help="Number of concurrent games/threads per GameWorker (default: 8)",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="GPU batch size for node evaluation (default: 8)",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for node evaluation (default: False)",
    )

    parser.add_argument(
        "--game-type",
        type=parse_game_type,
        default=ConnectFourGame,
        metavar="{TicTacToe,ConnectFour}",
        help="Type of game to play (default: ConnectFour)",
    )

    return parser.parse_args()

def main(game_type: Type[GameBase],
         ROLLOUTS: int,
         PROCESSES: int,
         TARGET_GAME_WORKERS: int,
         GAME_WORKER_GAMES: int,
         BATCH_SIZE: int,
         GPU: bool) -> None:
    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    ## Init
    # CPU or (in the future) GPU
    ctx = mp.get_context("spawn")
    if GPU and torch.cuda.is_available():
        print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'} for node evaluation.")
        model_args = {
            "game_type": game_type,
            "num_resBlocks": 3,
            "num_channels": 32,
            "device": "cpu",
            "batch_size": BATCH_SIZE
        }
        pool_factory = PoolFactory(GPU_AZ_Worker if torch.cuda.is_available() else CPU_RandomRollout_Worker, model_args=model_args)
    else:
        pool_factory = PoolFactory(CPU_RandomRollout_Worker)

    MCTS_factory = MCTS_Factory(ROLLOUTS)

    game_worker_ps: list[mp.context.SpawnProcess] = []
    all_game_results: mpQueueGen[tuple[str, GameBase]] = mpQueueGen(ctx)

    # 1 queue for ALL game_workers --- sending to ---> ALL eval_workers
    request_queue: mpQueueGen[list[AZ_NodeBatchRequest] | None] = mpQueueGen(ctx)
    # N queues for ALL eval_workers --- sending to ---> N * game_workers queues
    results_queues: list[mpQueueGen[AZ_NodeBatchResponse]] = [mpQueueGen(ctx)
                                                    for _ in range(TARGET_GAME_WORKERS)]
    metrics_queue: mpQueueGen[tuple[int, int]] = mpQueueGen(ctx)

    for i in range(PROCESSES):
        eval_worker = pool_factory.create_AZ_GPU_worker(request_queue, results_queues)
        p = ctx.Process(target=eval_worker.run, name=f"EvalWorker_{i}", daemon=True)
        p.start()

    for i, results_queue in enumerate(results_queues):
        game_worker = GameWorker(game_type,
                                 num_games=GAME_WORKER_GAMES,
                                 output_games=all_game_results,
                                 in_queue=results_queue,
                                 out_queue=request_queue,
                                 metrics_queue=metrics_queue,
                                 worker_id=i,
                                 MCTS_factory=MCTS_factory)
        p = ctx.Process(target=game_worker.run, name=f"GameWorker_{i}")
        game_worker_ps.append(p)

    for game_worker_p in game_worker_ps:
        game_worker_p.start()

    # Start the metrics daemon
    metrics_daemon_thread = Thread(target=metrics_daemon, args=(metrics_queue, ROLLOUTS), daemon=True)
    metrics_daemon_thread.start()

    expecting_results = TARGET_GAME_WORKERS * GAME_WORKER_GAMES
    for _ in range(expecting_results):
        id_, game = all_game_results.get()

        print(f"{id_=}")
        print(game.action_history)
        print(game.state)

    for game_worker_p in game_worker_ps:
        game_worker_p.join()

def metrics_daemon(
    metrics_queue: mpQueueGen[tuple[int, int]],
    ROLLOUTS: int,
    print_interval_s: float = 4.0,
) -> None:
    """
    Consume cumulative GameWorker metric snapshots and periodically print
    per-worker and total move throughput.

    Queue messages:
        (worker_id, total_moves)
    """
    latest_totals: dict[int, int] = {}
    previous_totals: dict[int, int] = {}

    interval_start = time.perf_counter()
    next_print = interval_start + print_interval_s

    while True:
        now = time.perf_counter()
        timeout = max(0.0, next_print - now)

        try:
            worker_id, total_moves = metrics_queue.get(timeout=timeout)
            latest_totals[worker_id] = total_moves
        except queue.Empty:
            pass

        now = time.perf_counter()
        if now < next_print:
            continue

        elapsed = now - interval_start

        worker_rates: dict[int, float] = {}
        total_moves_delta = 0

        for worker_id, total_moves in latest_totals.items():
            previous = previous_totals.get(worker_id)
            if previous is None:
                # First observation establishes the baseline.
                previous_totals[worker_id] = total_moves
                continue

            previous_total_moves = previous
            delta_moves = total_moves - previous_total_moves
            worker_rates[worker_id] = delta_moves / elapsed
            total_moves_delta += delta_moves

            previous_totals[worker_id] = total_moves

        if worker_rates:
            total_rate = total_moves_delta / elapsed

            workers_text = " ".join(
                f"W{worker_id}={rate:<2,.0f}"
                for worker_id, rate in sorted(worker_rates.items())
            )

            print(
                f"[Game throughput] "
                f"{elapsed:1.3f}s :: "
                f"{total_rate:3,.0f} moves/s :: "
                f"{total_rate * ROLLOUTS:5,.0f} nodes/s "
                f"| {workers_text}",
                flush=True,
            )

        interval_start = now
        next_print = now + print_interval_s

if __name__ == "__main__":
    args = parse_args()

    main(
        args.game_type,
        args.rollouts,
        args.processes,
        args.game_workers,
        args.games_per_worker,
        args.batch_size,
        args.gpu
    )
