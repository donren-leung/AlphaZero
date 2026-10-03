import argparse
import queue
import threading
import time

from alphazero.games.TicTacToe import TicTacToeGame
from alphazero.games.ConnectFour import ConnectFourGame

from .batching.AZ_GameWorker import GameWorker, mpQueueGen

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

def metrics_daemon(
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

