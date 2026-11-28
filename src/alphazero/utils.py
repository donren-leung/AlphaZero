import argparse
from collections import Counter
import json
from pathlib import Path
import queue
import re
import threading
import time

import numpy as np

from alphazero.games.GameBase import GameBase
from alphazero.games.TicTacToe import TicTacToeGame
from alphazero.games.ConnectFour import ConnectFourGame
from alphazero.games.Chess import ChessGame

from .batching.Metrics import WorkerMetrics, WorkerTotals
from .batching.Queues import mpQueueGen
from .models.model import (
    MODEL_ARCHITECTURES, DEFAULT_ARCHITECTURE,
    DEFAULT_BLOCKS, DEFAULT_CHANNELS, DEFAULT_HEAD_CHANNELS
)

DEFAULT_ROLLOUTS = 400
DEFAULT_PROCESSES = 1
DEFAULT_GAME_WORKERS = 8
DEFAULT_GAMES_PER_WORKER = 256
DEFAULT_BATCH_SIZE = 256
DEFAULT_INFERENCE_MAX_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GPU = True
DEFAULT_ITERATIONS = 6
DEFAULT_EPOCHS = 2
DEFAULT_SEED = 42

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

def training_model_args(args: argparse.Namespace) -> dict:
    game_properties = {
        "game_type_str": args.game_type.__name__.removesuffix("Game").lower(),
        "game_input_channels": args.game_type.input_channels,
        "game_spatial_size": args.game_type.row_count * args.game_type.col_count,
        "game_action_size": args.game_type.action_size,
    }
    saved = {}
    if args.resume is not None:
        metadata_path = args.resume.expanduser().resolve().with_name("args.json")
        if metadata_path.is_file():
            with metadata_path.open() as f:
                saved = json.load(f)

            # Backwards compatibility with pre-factory args.json files
            # Implied ResNet architecture if field is not present.
            saved.setdefault("architecture", DEFAULT_ARCHITECTURE)
            # Similarly for game_type
            prefix = str(metadata_path.parent.name.partition("_")[0])
            saved.setdefault("game_type_str", prefix)

            # Resuming from a checkpoint, ensure past model input/output shapes are compatible.
            for key, value in game_properties.items():
                if key in saved and saved[key] != value:
                    raise ValueError(f"Checkpoint {key}={saved[key]} conflicts with the selected game ({value})")

    model_args = dict(game_properties)
    for key, field, default in (
        ("architecture", args.architecture, DEFAULT_ARCHITECTURE),
        ("num_resBlocks", args.blocks, DEFAULT_BLOCKS),
        ("num_channels", args.channels, DEFAULT_CHANNELS),
        ("head_hidden_size", args.head_channels, DEFAULT_HEAD_CHANNELS),
    ):
        if key in saved:
            if field is not None and saved[key] != field:
                raise ValueError(f"Requested {key}={field} conflicts with checkpoint {key}={saved[key]}")
            model_args[key] = saved[key]
        else:
            model_args[key] = field if field is not None else default

    return {
        **model_args,
        "batch_size": args.batch_size,
        "inference_max_batch_size": args.inference_max_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "random_seed": args.seed
    }

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        description="Run batched parallel MCTS self-play."
    )

    parser.add_argument(
        "game_type",
        type=parse_game_type,
        metavar=f"{','.join(game_types.keys())}",
        help=f"Type of game to play ({','.join(game_types.keys())})",
    )

    parser.add_argument(
        "--rollouts",
        "-r",
        type=int,
        help="Number of MCTS nodes per move (default: 400)",
    )

    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        help="Number of node-evaluation worker processes (default: 1)",
    )

    parser.add_argument(
        "--game-workers",
        "-gw",
        type=int,
        help="Number of GameWorker processes (default: 8)",
    )

    parser.add_argument(
        "--games-per-worker",
        "-gpw",
        type=int,
        help="Number of concurrent games/threads per GameWorker (default: 64)",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        help="GPU batch size for supervised training (default: 256)",
    )

    parser.add_argument(
        "--inference-max-batch-size",
        "-ib",
        type=int,
        help="GPU batch size for node evaluation (default: 256)",
    )

    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        help="Learning rate for the optimizer (default: 0.001)",
    )

    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        help="Weight decay for the optimizer (default: 1e-4)",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for node evaluation (default: False)",
    )

    parser.add_argument(
        "--iter",
        "-i",
        type=int,
        help="Number of self-play iterations to run",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        help="Number of epochs to train the model per iteration",
    )

    parser.add_argument(
        "--architecture",
        "-a",
        type=str,
        choices=MODEL_ARCHITECTURES.keys(),
        help="Model architecture (default: resnet for new runs, saved architecture on resume)",
    )

    parser.add_argument(
        "--blocks",
        "-bl",
        type=int,
        help=f"Number of residual blocks in the model (default: {DEFAULT_BLOCKS})",
    )

    parser.add_argument(
        "--channels",
        "-ch",
        type=int,
        help=f"Number of channels in the model (default: {DEFAULT_CHANNELS})",
    )

    parser.add_argument(
        "--head-channels",
        "-hch",
        type=int,
        help=f"Number of channels in the model heads (default: {DEFAULT_HEAD_CHANNELS})",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to a saved model to resume training from (default: None)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for worker processes (default: 42)",
    )

    args = parser.parse_args(argv)
    # With only the game, --resume, and -i, continue in the checkpoint folder.
    # -i counts additional iterations and can differ from the original run.
    # Supplying any other option selects a separate output directory.
    resume_in_place = set(vars(args)) == {"game_type", "resume", "iter"}
    defaults = dict(rollouts=DEFAULT_ROLLOUTS, processes=DEFAULT_PROCESSES,
                    game_workers=DEFAULT_GAME_WORKERS, games_per_worker=DEFAULT_GAMES_PER_WORKER,
                    batch_size=DEFAULT_BATCH_SIZE, inference_max_batch_size=DEFAULT_INFERENCE_MAX_BATCH_SIZE,
                    learning_rate=DEFAULT_LEARNING_RATE, weight_decay=DEFAULT_WEIGHT_DECAY,
                    gpu=DEFAULT_GPU, iter=DEFAULT_ITERATIONS, epochs=DEFAULT_EPOCHS,
                    # None means user omitted the setting.
                    # training_model_args() can then use the checkpoint’s saved setting or,
                    # for a fresh run, the corresponding DEFAULT_*
                    architecture=None, blocks=None,
                    channels=None, head_channels=None,
                    resume=None, seed=DEFAULT_SEED)
    if resume_in_place:
        directory = args.resume.expanduser().resolve().parent
        # Older checkpoints encode the search and training settings in the run name.
        match = re.match(r"^[^_]+_r(\d+)_p(\d+)_gw(\d+)_gpw(\d+)_b(\d+)_e(\d+)(?:_|$)", directory.name)
        if match:
            keys = ("rollouts", "processes", "game_workers", "games_per_worker", "batch_size", "epochs")
            defaults.update(zip(keys, map(int, match.groups()), strict=True))
        metadata = directory / "args.json"
        if metadata.is_file():
            saved = json.loads(metadata.read_text())
            for key in ("inference_max_batch_size", "learning_rate", "weight_decay"):
                if key in saved:
                    defaults[key] = saved[key]
            defaults["seed"] = saved.get("random_seed", defaults["seed"])
    return argparse.Namespace(**(defaults | vars(args)), resume_in_place=resume_in_place)

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

def _sum_worker_totals(latest: dict[int, WorkerTotals]) -> WorkerTotals:
    return WorkerTotals(*(sum(column) for column in zip(*latest.values())))

def _reuse_report(totals: WorkerTotals, *, final: bool = False) -> str:
    """
    Report on the reuse of MCTS search results across all workers.
    Reused statistics are kept cumulatively across time.
    """
    skipped_percent = f"{totals.fast_skipped_rounds / totals.fast_budget_rounds:.0%}" if totals.fast_budget_rounds else "N/A"
    terminal_percent = f"{totals.fast_terminal_rounds / totals.fast_budget_rounds:.0%}" if totals.fast_budget_rounds else "N/A"
    fast = (
        f"{totals.fast_skipped_rounds} skipped + {totals.fast_terminal_rounds} terminal/{totals.fast_budget_rounds} total "
        f"({skipped_percent} + {terminal_percent}, {totals.fast_moves} moves)"
    )

    full_all = totals.full_cached_evals + totals.full_terminal_evals + totals.full_requested_evals
    cached_percent = f"{totals.full_cached_evals / full_all:.0%}" if full_all else "N/A"
    terminal_percent = f"{totals.full_terminal_evals / full_all:.0%}" if full_all else "N/A"
    full = (
        f"{totals.full_cached_evals} cached + {totals.full_terminal_evals} terminal/{full_all} total "
        f"({cached_percent} + {terminal_percent}, {totals.full_moves} moves)"
    )

    label = "[Search reuse total]" if final else "[Search reuse]"
    return f"{label} fast: {fast} | full: {full}"

SLEEP_S = 0.1
def metrics_daemon(
    stop_event: threading.Event,
    metrics_queues: list[mpQueueGen[WorkerMetrics]],
    eval_metrics_queue: mpQueueGen[tuple[int, list[tuple], dict]] | None,
    counter: ThreadSafeCounter,
    expected_results: int,
    print_interval_s: float = 2.5,
) -> None:
    """
     Consume cumulative GameWorker and EvalWorker metric snapshots and periodically print
    per-worker and total move throughput.
          
    Queue messages:
        WorkerMetrics object containing cumulative metrics for a single worker.
    Evaluation Queue messages:
        (nodes, timings, batch_sizes)
    """
    latest_totals: dict[int, WorkerTotals] = {}
    previous_totals: dict[int, WorkerTotals] = {}
    # Sum across all workers
    previous_sum = WorkerTotals()

    eval_records: list[tuple[int, list[tuple], dict]] = []

    interval_start = time.perf_counter()
    next_print = interval_start + print_interval_s

    while True:
        # The parent sets stop_event after joining producers, so the last drain
        # includes the final snapshot from every worker.
        finished = stop_event.is_set()
        while eval_metrics_queue is not None:
            try:
                eval_records.append(eval_metrics_queue.get(block=False))
            except queue.Empty:
                break

        for metrics_queue in metrics_queues:
            while True:
                try:
                    snapshot = metrics_queue.get(block=False)
                except queue.Empty:
                    break
                latest_totals[snapshot.worker_id] = snapshot.totals

        if finished:
            print(_reuse_report(_sum_worker_totals(latest_totals), final=True), flush=True)
            return

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

        for worker_id, totals in latest_totals.items():
            previous = previous_totals.get(worker_id)
            previous_totals[worker_id] = totals
            if previous is None:
                continue

            delta = totals.since(previous)
            worker_sent_eff[worker_id] = delta.sent_requests / delta.sent_batches if delta.sent_batches else 0.0
            worker_rec_node_rates[worker_id] = delta.received_results / elapsed
            worker_rec_eff[worker_id] = delta.received_results / delta.received_batches if delta.received_batches else 0.0
            total_moves_delta += delta.moves
            total_sent_batches += delta.sent_batches
            total_sent_requests += delta.sent_requests
            total_received_batches += delta.received_batches
            total_received_results += delta.received_results

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
            mean_timings = np.zeros(5)
            p95_timings = np.zeros(5)

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

            if eval_metrics_queue is not None:
                print(
                    f"[Evaluation] "
                    f"{elapsed:1.1f}s :: "
                    f"{human_readable(eval_node_rate)} nps "
                    f"{human_readable(eval_batch_rate)} bps "
                    f":: batches: {unique_batch_sizes}"
                    f":: rec1 {mean_timings[0]:.1f}/{p95_timings[0]:.1f}ms "
                    f"rec2 {mean_timings[1]:.1f}/{p95_timings[1]:.1f}ms "
                    f"prp {mean_timings[2]:.1f}/{p95_timings[2]:.1f}ms "
                    f"compute {mean_timings[3]:.1f}/{p95_timings[3]:.1f}ms "
                    f"submit {mean_timings[4]:.1f}/{p95_timings[4]:.1f}ms",
                    flush=True,
                )

        if latest_totals:
            current_sum = _sum_worker_totals(latest_totals)
            print(_reuse_report(current_sum.since(previous_sum)), flush=True)
            previous_sum = current_sum

        interval_start = now
        next_print = now + print_interval_s

# def metrics_daemon_CPU(
#     stop_event: threading.Event,
#     metrics_queue: mpQueueGen[WorkerMetrics],
#     counter: ThreadSafeCounter,
#     expected_results: int,
#     print_interval_s: float = 2.5,
# ) -> None:
#     metrics_daemon(stop_event, [metrics_queue], None, counter, expected_results, print_interval_s)
