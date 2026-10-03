import logging
import sys
import queue
import time
from threading import Thread

import argparse
import multiprocessing as mp
from multiprocessing.process import BaseProcess
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
        default=400,
        help="Number of MCTS nodes per move (default: 400)",
    )

    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=4,
        help="Number of node-evaluation worker processes (default: 4)",
    )

    parser.add_argument(
        "--game-workers",
        "-gw",
        type=int,
        default=8,
        help="Number of GameWorker processes (default: 8)",
    )

    parser.add_argument(
        "--games-per-worker",
        "-gpw",
        type=int,
        default=64,
        help="Number of concurrent games/threads per GameWorker (default: 64)",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="GPU batch size for node evaluation (default: 64)",
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
            "num_resBlocks": 12,
            "num_channels": 64,
            "device": "cpu",
            "batch_size": BATCH_SIZE
        }
        pool_factory = PoolFactory(GPU_AZ_Worker if torch.cuda.is_available() else CPU_RandomRollout_Worker, model_args=model_args)
    else:
        pool_factory = PoolFactory(CPU_RandomRollout_Worker)

    MCTS_factory = MCTS_Factory(ROLLOUTS)

    game_worker_ps: list[BaseProcess] = []
    all_game_results: mpQueueGen[tuple[str, GameBase]] = mpQueueGen(ctx)

    # 1 queue for ALL game_workers --- sending to ---> ALL eval_workers
    request_queue: mpQueueGen[list[AZ_NodeBatchRequest] | None] = mpQueueGen(ctx)
    # N queues for ALL eval_workers --- sending to ---> N * game_workers queues
    results_queues: list[mpQueueGen[AZ_NodeBatchResponse]] = [mpQueueGen(ctx)
                                                    for _ in range(TARGET_GAME_WORKERS)]

    # Cumulative moves
    worker_metrics_queue: mpQueueGen[tuple[int, int, int, int, int, int]] = mpQueueGen(ctx)
    # Batch total, batches
    eval_metrics_queue: mpQueueGen[tuple[int, int]] = mpQueueGen(ctx)

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
                                 metrics_queue=worker_metrics_queue,
                                 worker_id=i,
                                 MCTS_factory=MCTS_factory)
        p = ctx.Process(target=game_worker.run, name=f"GameWorker_{i}")
        game_worker_ps.append(p)

    for game_worker_p in game_worker_ps:
        game_worker_p.start()

    # Start the metrics daemon
    metrics_daemon_thread = Thread(target=metrics_daemon, args=(worker_metrics_queue,), daemon=True)
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
    metrics_queue: mpQueueGen[tuple[int, int, int, int, int, int]],
    print_interval_s: float = 2.5,
) -> None:
    """
    Consume cumulative GameWorker metric snapshots and periodically print
    per-worker and total move throughput.

    Queue messages:
        (worker_id, total_moves, sent_batches, sent_requests, received_batches, received_results)
    """
    latest_totals: dict[int, tuple[int, int, int, int, int]] = {}
    previous_totals: dict[int, tuple[int, int, int, int, int]] = {}

    interval_start = time.perf_counter()
    next_print = interval_start + print_interval_s

    while True:
        now = time.perf_counter()
        timeout = max(0.0, next_print - now)

        try:
            worker_id, total_moves, sent_batches, sent_requests, received_batches, received_results = metrics_queue.get(timeout=timeout)
            latest_totals[worker_id] = (total_moves, sent_batches, sent_requests, received_batches, received_results)
        except queue.Empty:
            pass

        now = time.perf_counter()
        if now < next_print:
            continue

        elapsed = now - interval_start

        worker_sent_eff: dict[int, float] = {}
        worker_rec_node_rates: dict[int, float] = {}
        worker_rec_eff: dict[int, float] = {}

        total_moves_delta = 0
        total_sent_batches = 0
        total_sent_requests = 0
        total_received_batches = 0
        total_received_results = 0

        for worker_id, (total_moves, sent_batches, sent_requests, received_batches, received_results) in latest_totals.items():
            previous = previous_totals.get(worker_id)
            if previous is None:
                # First observation establishes the baseline.
                previous_totals[worker_id] = (total_moves, sent_batches, sent_requests, received_batches, received_results)
                continue

            deltas = tuple(total - prev for total, prev in zip((total_moves, sent_batches, sent_requests, received_batches, received_results), previous))
            worker_sent_eff[worker_id] = deltas[2] / (deltas[1]) if deltas[1] > 0 else 0.0
            worker_rec_node_rates[worker_id] = deltas[4] / elapsed
            worker_rec_eff[worker_id] = deltas[4] / (deltas[3]) if deltas[3] > 0 else 0.0

            total_moves_delta += deltas[0]
            total_sent_batches += deltas[1]
            total_sent_requests += deltas[2]
            total_received_batches += deltas[3]
            total_received_results += deltas[4]

            previous_totals[worker_id] = (total_moves, sent_batches, sent_requests, received_batches, received_results)

        if worker_rec_node_rates:
            total_move_rate = total_moves_delta / elapsed
            total_sent_batch_rate = total_sent_batches / elapsed
            total_sent_node_rate = total_sent_requests / elapsed
            total_received_node_rate = total_received_results / elapsed
            total_received_batch_rate = total_received_batches / elapsed
            # total_sent_eff = total_sent_requests / elapsed

            node_rates = list(worker_rec_node_rates.values())
            sent_effs = list(worker_sent_eff.values())
            rec_effs = list(worker_rec_eff.values())

            min_node_rate = min(node_rates)
            mean_node_rate = sum(node_rates) / len(node_rates)
            max_node_rate = max(node_rates)

            min_sent_efficiency = min(sent_effs)
            mean_sent_efficiency = sum(sent_effs) / len(sent_effs)
            max_sent_efficiency = max(sent_effs)

            min_rec_efficiency = min(rec_effs)
            mean_rec_efficiency = sum(rec_effs) / len(rec_effs)
            max_rec_efficiency = max(rec_effs)

            def human_readable(num: float) -> str:
                for threshold, suffix in [(1_000_000, "M"), (1_000, "K"), (1, "")]:
                    if num >= threshold:
                        scaled = num / threshold
                        # Format to 3 significant figures and strip trailing zeros/dot
                        formatted = f"{scaled:.3f}"[:4].rstrip(".")
                        return f"{formatted}{suffix}"
                return f"{num:.3g}"

            print(
                f"[Game throughput] "
                f"{elapsed:1.3f}s :: "
                f"{total_move_rate:3.0f} moves/s :: "
                f"SENT: "
                f"{human_readable(total_sent_node_rate)} nodes/s "
                f"{human_readable(total_sent_batch_rate)} batches/s "
                f"/batch {min_sent_efficiency:3.0f}-{mean_sent_efficiency:3.0f}-{max_sent_efficiency:3.0f} "
                f"| REC: "
                f"{human_readable(total_received_node_rate)} nodes/s "
                f"{human_readable(total_received_batch_rate)} batches/s "
                f"nodes/worker={min_node_rate:4.0f}-{mean_node_rate:4.0f}-{max_node_rate:4.0f} "
                f"/batch {min_rec_efficiency:3.0f}-{mean_rec_efficiency:3.0f}-{max_rec_efficiency:3.0f} ",

                # f"| {workers_text}",
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
