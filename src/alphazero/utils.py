import argparse
from collections import Counter
from pathlib import Path
import queue
import threading
import time

import numpy as np

from alphazero.games.GameBase import GameBase
from alphazero.games.TicTacToe import TicTacToeGame
from alphazero.games.ConnectFour import ConnectFourGame
from alphazero.games.Chess import ChessGame

from .batching.Queues import mpQueueGen

GameType = type[GameBase]
game_types: dict[str, GameType] = {
    "ttt": TicTacToeGame,
    "c4": ConnectFourGame,
    "chess": ChessGame,
}

def parse_game_type(value: str) -> GameType:
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
        default=1,
        help="Number of node-evaluation worker processes (default: 1)",
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
        "-game",
        type=parse_game_type,
        default=ChessGame,
        metavar=f"{','.join(game_types.keys())}",
        help="Type of game to play (default: chess)",
    )

    parser.add_argument(
        "--iter",
        "-i",
        type=int,
        default=5,
        help="Number of self-play iterations to run",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=4,
        help="Number of epochs to train the model per iteration",
    )

    parser.add_argument(
        "--blocks",
        "-bl",
        type=int,
        default=9,
        help="Number of residual blocks in the model (default: 9)",
    )

    parser.add_argument(
        "--channels",
        "-ch",
        type=int,
        default=256,
        help="Number of channels in the model (default: 256)",
    )

    parser.add_argument(
        "--head-channels",
        "-hch",
        type=int,
        default=64,
        help="Number of channels in the model heads (default: 64)",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a saved model to resume training from (default: None)",
    )

    return parser.parse_args()

class ThreadSafeCounter:
    def __init__(self, initial_value=0):
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self, amount=1):
        with self._lock:
            self._value += amount
            return self._value

    @property
    def value(self):
        with self._lock:
            return self._value

SLEEP_S = 0.1
def metrics_daemon(
    stop_event: threading.Event,
    metrics_queues: list[mpQueueGen[tuple[int, int, int, int, int, int]]],
    eval_metrics_queue: mpQueueGen[tuple[int, list[tuple], dict]],
    counter: ThreadSafeCounter,
    expected_results: int,
    print_interval_s: float = 2.5,
) -> None:
    """
    Consume cumulative GameWorker metric snapshots and periodically print
    per-worker and total move throughput.

    Queue messages:
        (worker_id, total_moves, sent_batches, sent_requests, received_batches, received_results)
    Evaluation Queue messages:
        (nodes, timings, batch_sizes)
    """
    latest_totals: dict[int, tuple[int, int, int, int, int]] = {}
    previous_totals: dict[int, tuple[int, int, int, int, int]] = {}

    eval_records: list[tuple[int, list[tuple], dict]] = []

    interval_start = time.perf_counter()
    next_print = interval_start + print_interval_s

    while not stop_event.is_set():
        now = time.perf_counter()
        # timeout = max(0.0, next_print - now)

        # Drain available evaluation metrics without blocking.
        while True:
            try:
                record = eval_metrics_queue.get(block=False)
                eval_records.append(record)
            except queue.Empty:
                break

        for metrics_queue in metrics_queues:
            for _ in range(10):  # Drain up to 10 messages per queue to avoid blocking
                try:
                    worker_id, total_moves, sent_batches, sent_requests, received_batches, received_results = metrics_queue.get(block=False)
                    latest_totals[worker_id] = (total_moves, sent_batches, sent_requests, received_batches, received_results)
                except queue.Empty:
                    break

        now = time.perf_counter()
        if now < next_print:
            stop_event.wait(min(SLEEP_S, next_print - now))
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

        # Evaluation metrics
        total_eval_nodes = 0
        all_eval_timings: list[tuple] = []
        batch_sizes_counter: Counter[int] = Counter()

        for record in eval_records:
            eval_nodes, eval_timings, eval_batch_sizes = record
            total_eval_nodes += eval_nodes
            all_eval_timings.extend(eval_timings)
            batch_sizes_counter.update(eval_batch_sizes)

        eval_node_rate = total_eval_nodes / elapsed if elapsed > 0 else 0.0
        total_eval_batches = sum(batch_sizes_counter.values())
        eval_batch_rate = total_eval_batches / elapsed if elapsed > 0 else 0.0
        unique_batch_sizes = sorted(batch_sizes_counter.items())

        # Timing statistics: mean and p95 per stage.
        if all_eval_timings:
            timings = np.asarray(all_eval_timings, dtype=np.float64)
            mean_timings = np.mean(timings, axis=0)*1000
            p95_timings = np.percentile(timings, 95, axis=0)*1000
        else:
            mean_timings = np.zeros(3)
            p95_timings = np.zeros(3)

        eval_records.clear()

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
                f"[Throughput] "
                f"{elapsed:1.1f}s :: "
                f"{counter.value:>5}/{expected_results} games ::"
                f"{total_move_rate:4.0f} moves/s :: "
                f"SENT: "
                f"{human_readable(total_sent_node_rate)} nps "
                f"{human_readable(total_sent_batch_rate)} bps "
                f"/batch {min_sent_efficiency:3.0f}-{mean_sent_efficiency:3.0f}-{max_sent_efficiency:3.0f} "
                f"| REC: "
                f"{human_readable(total_received_node_rate)} nps "
                f"{human_readable(total_received_batch_rate)} bps "
                f"nodes/worker={min_node_rate:4.0f}-{mean_node_rate:4.0f}-{max_node_rate:4.0f} "
                f"/batch {min_rec_efficiency:3.0f}-{mean_rec_efficiency:3.0f}-{max_rec_efficiency:3.0f} ",

                # f"| {workers_text}",
                flush=True,
            )

            print(
                f"[Evaluation] "
                f"{elapsed:1.1f}s :: "
                f"{human_readable(eval_node_rate)} nps "
                f"{human_readable(eval_batch_rate)} bps "
                f":: batches: {unique_batch_sizes}"
                f":: receive {mean_timings[0]:.1f}/{p95_timings[0]:.1f}ms "
                f"compute {mean_timings[1]:.1f}/{p95_timings[1]:.1f}ms "
                f"submit {mean_timings[2]:.1f}/{p95_timings[2]:.1f}ms",
                flush=True,
            )

        interval_start = now
        next_print = now + print_interval_s

def metrics_daemon_CPU(
    stop_event: threading.Event,
    metrics_queue: mpQueueGen[tuple[int, int, int, int, int, int]],
    counter: ThreadSafeCounter,
    expected_results: int,
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

    while not stop_event.is_set():
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
                f"[Throughput] "
                f"{elapsed:1.1f}s :: "
                f"{counter.value:>5}/{expected_results} games ::"
                f"{total_move_rate:4.0f} moves/s :: "
                f"SENT: "
                f"{human_readable(total_sent_node_rate)} nps "
                f"{human_readable(total_sent_batch_rate)} bps "
                f"/batch {min_sent_efficiency:3.0f}-{mean_sent_efficiency:3.0f}-{max_sent_efficiency:3.0f} "
                f"| REC: "
                f"{human_readable(total_received_node_rate)} nps "
                f"{human_readable(total_received_batch_rate)} bps "
                f"nodes/worker={min_node_rate:4.0f}-{mean_node_rate:4.0f}-{max_node_rate:4.0f} "
                f"/batch {min_rec_efficiency:3.0f}-{mean_rec_efficiency:3.0f}-{max_rec_efficiency:3.0f} ",

                # f"| {workers_text}",
                flush=True,
            )

        interval_start = now
        next_print = now + print_interval_s
