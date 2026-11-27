r"""Benchmark inference on real chess positions, without starting game workers.

Run in the project's Python environment, from any directory::

    python scripts/isolated_gpu.py --checkpoint /path/to/model_011.pt \
        --selfplay-nps 10000 --output-json working_files/inference_benchmark.json

For a quick CUDA smoke test, add:
    --games 8 --batch-sizes 4 --warmup 2 --iterations 3 --repeats 1

The self-play rate is an external reference, not a measurement by this script.
Run without other GPU workloads for meaningful ceiling measurements.
"""
# mypy: ignore-errors

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bulletchess
import chess
import chess.pgn
import numpy as np
import torch

from alphazero.batching.NodeBatch import AZ_NodeRequest, AZ_NodeBatchRequest
from alphazero.batching.Pool import GPU_AZ_Worker
from alphazero.games.ChessUtils import board_ply
from alphazero.games.Chess import ChessState
from alphazero.models.model import create_model


class DirectEvaluator(GPU_AZ_Worker):
    """Use the production compute method with an already-loaded model, no queues."""

    def __init__(self, model: torch.nn.Module, *, compile_model: bool=False, compile_mode: str="reduce-overhead"):
        self.model = model
        self.compiled_model = torch.compile(model, mode=compile_mode) if compile_model else model
        self.device = torch.device("cuda")
        self.compiled_compute = (
            torch.compile(
                self.compute_tensors,
                mode=compile_mode,
                fullgraph=True,
            )
            if compile_model
            else self.compute_tensors
        )

def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pgn", type=Path, default=ROOT / "working_files/lichess_db_standard_rated_2015-01.pgn")
    parser.add_argument("--games", type=positive_int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-args", type=Path, help="Defaults to args.json beside the checkpoint")
    parser.add_argument("--batch-sizes", nargs="+", type=positive_int, default=[64, 128, 256])
    parser.add_argument("--dtypes", nargs="+", choices=["fp32", "fp16", "bf16"], default=["fp32", "fp16", "bf16"])
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--iterations", type=positive_int, default=200)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--selfplay-nps", type=float, help="Externally measured reference; may have different batch/precision")
    parser.add_argument("--output-json", type=Path)

    parser.add_argument(
        "--compile",
        action="store_true",
        dest="compile_model",
        help="Benchmark torch.compile instead of eager execution",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="Compilation mode used with --compile",
    )

    args = parser.parse_args(argv)
    if args.games > 65536:
        parser.error("--games must be <= 65536 (request IDs are uint16)")
    if args.selfplay_nps is not None and (not math.isfinite(args.selfplay_nps) or args.selfplay_nps <= 0):
        parser.error("--selfplay-nps must be finite and positive")
    args.batch_sizes = list(dict.fromkeys(args.batch_sizes))
    args.dtypes = list(dict.fromkeys(args.dtypes))
    return args


def sample_board(game: chess.pgn.Game, rng: random.Random) -> ChessState | None:
    """A random permutation makes the first eligible ply uniform among eligible plies.

    Usually only one candidate needs checking, avoiding expensive draw-claim
    checks on every ply. Replaying from the root preserves the full history.
    """
    moves = list(game.mainline_moves())
    candidates = list(range(1, len(moves) + 1))
    rng.shuffle(candidates)
    for ply in candidates:
        board = bulletchess.Board.from_fen(game.board().fen())
        for move in moves[:ply]:
            board.apply(bulletchess.Move.from_uci(move.uci()))
        state = ChessState(board)
        if not state.get_value_and_terminated()[1]:
            return state
    return None


def load_requests(path: Path, count: int, seed: int) -> tuple[list[AZ_NodeRequest], dict]:
    rng = random.Random(seed)
    requests = []
    stats = {"games_read": 0, "malformed": 0, "nonstandard": 0, "no_eligible_position": 0}
    samples = []
    with path.open(encoding="utf-8-sig") as handle:
        while len(requests) < count:
            game = chess.pgn.read_game(handle)
            if game is None:
                raise ValueError(f"PGN exhausted: found {len(requests)} usable games, need {count}; {stats}")
            stats["games_read"] += 1
            if game.errors:
                stats["malformed"] += 1
                continue
            root = game.board()
            if type(root) is not chess.Board or root.chess960:
                stats["nonstandard"] += 1
                continue
            if not root.is_valid():
                stats["malformed"] += 1
                continue
            state = sample_board(game, rng)
            if state is None:
                stats["no_eligible_position"] += 1
                continue
            board = state.board
            player = 1 if board.turn == bulletchess.WHITE else -1
            request_id = len(requests)
            requests.append(AZ_NodeRequest(request_id, player, state.get_legal_actions(), state.encode_for_model(player)))
            samples.append({"request_id": request_id, "game_number": stats["games_read"],
                            "ply": board_ply(board), "history_plies": len(board.history)})
            if len(requests) % 100 == 0:
                print(f"Prepared {len(requests)}/{count} positions", flush=True)
    rng.shuffle(requests)
    stats.update({"seed": seed, "unique_positions": count, "samples": samples})
    return requests, stats


def prepare_batches(requests: list[AZ_NodeRequest], sizes: list[int]) -> tuple[dict, int]:
    # Use one common padded sequence for every size, including custom CLI sizes.
    multiple = math.lcm(*sizes)
    total = math.ceil(len(requests) / multiple) * multiple
    padded = [requests[i % len(requests)] for i in range(total)]
    return {size: [AZ_NodeBatchRequest.from_NodeRequests(0, padded[i:i + size])
                   for i in range(0, total, size)] for size in sizes}, total


def read_model_args(checkpoint: Path, config_path: Path | None) -> tuple[dict, Path]:
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    path = config_path or checkpoint.parent / "args.json"
    if not path.is_file():
        raise ValueError(f"Model configuration does not exist: {path}; use --model-args")
    config = json.loads(path.read_text())
    keys = ("game_input_channels", "game_spatial_size", "game_action_size",
            "num_resBlocks", "num_channels", "head_hidden_size")
    if not isinstance(config, dict) or any(key not in config for key in keys):
        raise ValueError(f"Model configuration must contain: {', '.join(keys)}")
    if any(type(config[key]) is not int or config[key] <= 0 for key in keys):
        raise ValueError("Model dimensions and residual block count must be positive integers")
    if tuple(config[key] for key in keys[:3]) != (119, 64, 4672):
        raise ValueError("Checkpoint architecture must use chess inputs (119, 8, 8) and 4672 actions")
    return {key: config[key] for key in (*keys, "architecture") if key in config}, path


def precision_context(dtype: str):
    return (nullcontext() if dtype == "fp32" else
            torch.autocast("cuda", dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]))


def validate_policy(batch: AZ_NodeBatchRequest, response) -> None:
    valid = np.arange(batch.padded_legal_actions.shape[1])[None, :] < batch.lengths[:, None]
    np.testing.assert_array_equal(response.thread_idxs, batch.thread_idxs)
    np.testing.assert_array_equal(response.lengths, batch.lengths)
    np.testing.assert_array_equal(response.legal_actions, batch.padded_legal_actions[valid])
    if response.values.shape != (len(batch.thread_idxs),) or not np.isfinite(response.values).all():
        raise ValueError("Invalid evaluator values")
    if not np.isfinite(response.probs).all() or np.any(response.probs < 0) or np.any(response.probs > 1):
        raise ValueError("Invalid evaluator probabilities")
    if response.probs.shape != response.legal_actions.shape:
        raise ValueError("Evaluator probability/action lengths differ")
    offsets = np.concatenate(([0], np.cumsum(batch.lengths, dtype=np.int64)))
    for start, end in zip(offsets[:-1], offsets[1:]):
        np.testing.assert_allclose(response.probs[start:end].sum(), 1.0, atol=1e-5, rtol=1e-5)


def benchmark(evaluator: DirectEvaluator, batches: list[AZ_NodeBatchRequest], mode: str,
              dtype: str, args: argparse.Namespace) -> dict:
    size = len(batches[0].thread_idxs)
    # Transfers/allocation happen before the GPU-only timing region.
    inputs = ([torch.as_tensor(batch.states, device="cuda", dtype=torch.float32) for batch in batches]
              if mode == "forward" else batches)
    operation = evaluator.compiled_model if mode == "forward" else evaluator.gpu_compute
    timings = []
    with torch.inference_mode(), precision_context(dtype):
        # Validate every prepared batch once, outside warm-up and measurement.
        for batch, item in zip(batches, inputs):
            result = operation(item)
            if mode == "forward":
                logits, values = result
                if logits.shape != (size, 4672) or values.shape != (size, 1):
                    raise ValueError("Unexpected model output shapes")
                if not torch.isfinite(logits).all().item() or not torch.isfinite(values).all().item():
                    raise ValueError(f"Non-finite model outputs in {dtype}")
            else:
                validate_policy(batch, result)
            del result
        # No correctness-check tensors should remain live while benchmarking.
        if mode == "forward":
            del logits, values
        for repeat in range(args.repeats):
            for i in range(args.warmup):
                operation(inputs[i % len(inputs)])
            start_event = torch.cuda.Event(enable_timing=True) if mode == "forward" else None
            end_event = torch.cuda.Event(enable_timing=True) if mode == "forward" else None
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            if start_event is not None:
                start_event.record()
            for i in range(args.iterations):
                operation(inputs[i % len(inputs)])
            if end_event is not None:
                end_event.record()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            timings.append({"repeat": repeat + 1, "wall_seconds": elapsed,
                            "wall_ms_per_batch": elapsed * 1000 / args.iterations,
                            "nodes_per_second": size * args.iterations / elapsed,
                            "cuda_ms_per_batch": (start_event.elapsed_time(end_event) / args.iterations
                                                  if start_event is not None else None)})
    rates = [row["nodes_per_second"] for row in timings]
    mean_rate = statistics.mean(rates)
    total_seconds = sum(row["wall_seconds"] for row in timings)
    rate = size * args.iterations * args.repeats / total_seconds
    return {"mode": mode, "dtype": dtype, "batch_size": size, "status": "ok",
            "wall_ms_per_batch": total_seconds * 1000 / (args.iterations * args.repeats),
            "cuda_ms_per_batch": (statistics.mean(row["cuda_ms_per_batch"] for row in timings)
                                  if mode == "forward" else None),
            "nodes_per_second": rate, "nps_min": min(rates), "nps_max": max(rates),
            "nps_cv_percent": 100 * statistics.pstdev(rates) / mean_rate,
            "selfplay_ratio": rate / args.selfplay_nps if args.selfplay_nps is not None else None,
            "repetitions": timings}


def print_result(row: dict) -> None:
    prefix = f"{row['mode']:<12} {row['dtype']:<6} {row['batch_size']:>6}"
    if row["status"] != "ok":
        print(f"{prefix}  {row['status']}: {row['reason']}", flush=True)
        return
    cuda = f"{row['cuda_ms_per_batch']:.3f}" if row["cuda_ms_per_batch"] is not None else "—"
    ratio = f"{row['selfplay_ratio']:.2f}x" if row["selfplay_ratio"] is not None else "—"
    print(f"{prefix} {row['wall_ms_per_batch']:>10.3f} {cuda:>10} {row['nodes_per_second']:>12,.0f} "
          f"{row['nps_cv_percent']:>7.2f}% {row['nps_min']:>10,.0f}–{row['nps_max']:<10,.0f} {ratio:>9}", flush=True)


def run(args: argparse.Namespace) -> dict:
    config, config_path = read_model_args(args.checkpoint, args.model_args)
    if not args.pgn.is_file():
        raise ValueError(f"PGN does not exist: {args.pgn}")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Run this script in a CUDA-enabled project environment.")
    # Explicit IEEE FP32 baseline, without changing settings in production code.
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    model = create_model(**config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().to("cuda")
    evaluator = DirectEvaluator(model, compile_model=args.compile_model, compile_mode=args.compile_mode)
    requests, sampling = load_requests(args.pgn, args.games, args.seed)
    batches, total = prepare_batches(requests, args.batch_sizes)
    sampling.update({"batched_positions": total, "repeated_positions": total - len(requests),
                     "request_order": [int(request.thread_idx) for request in requests]})
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    metadata = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "gpu": properties.name,
                "gpu_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
                "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "checkpoint": str(args.checkpoint.resolve()), "model_args_path": str(config_path.resolve()),
                "architecture": config}
    print(f"GPU: {properties.name}; PyTorch {torch.__version__}; CUDA {torch.version.cuda}")
    execution = f"compiled ({args.compile_mode})" if args.compile_model else "eager"
    print(f"FP32: IEEE (TF32 off); cuDNN benchmark=False; {execution}, contiguous inputs"
    )
    print(f"Samples: {len(requests)} unique + {total - len(requests)} repeated = {total} per batch-size dataset")
    print(f"Skipped games: malformed={sampling['malformed']}, nonstandard={sampling['nonstandard']}, "
          f"no eligible position={sampling['no_eligible_position']}")
    print(f"Per configuration: {args.warmup} warm-up + {args.iterations} timed calls, repeated {args.repeats} times")
    if args.selfplay_nps is not None:
        print(f"External self-play baseline: {args.selfplay_nps:,.0f} nodes/s (not measured here; batch/precision may differ)")
    print(f"{'Mode':<12} {'Dtype':<6} {'Batch':>6} {'Wall ms/B':>10} {'CUDA ms/B':>10} "
          f"{'Nodes/s':>12} {'NPS CV':>8} {'Repeat NPS min–max':>21} {'/baseline':>9}")
    results = []
    for dtype in args.dtypes:
        for size in args.batch_sizes:
            for mode in ("forward", "gpu_compute"):
                row = {"mode": mode, "dtype": dtype, "batch_size": size}
                if dtype == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
                    row.update(status="unsupported", reason="GPU does not support native BF16")
                else:
                    try:
                        row = benchmark(evaluator, batches[size], mode, dtype, args)
                    except torch.cuda.OutOfMemoryError:
                        row.update(status="out_of_memory", reason="Insufficient CUDA memory; no precision substitution")
                    except (ValueError, AssertionError) as exc:
                        row.update(status="validation_failed", reason=str(exc))
                # Run after the exception handler so failed frames no longer hold tensors.
                gc.collect()
                torch.cuda.empty_cache()
                results.append(row)
                print_result(row)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = {"metadata": metadata, "settings": settings, "sampling": sampling, "results": results,
              "baseline_note": "User-supplied self-play rate, not measured here; batch/precision may differ."}
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Saved {args.output_json}")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 1
    return int(any(row["status"] == "validation_failed" for row in report["results"])
               or not any(row["status"] == "ok" for row in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
