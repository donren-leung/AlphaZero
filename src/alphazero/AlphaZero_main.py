import argparse
import multiprocessing as mp
import json
import re
from multiprocessing.process import BaseProcess
from pathlib import Path
import random
from threading import Thread, Event
from typing import Any, TypeVar

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange

from .batching.NodeBatch import AZ_NodeBatchRequest, AZ_NodeBatchResponse, SparseLegalPolicy
from .batching.AZ_GameWorker import GameWorker
from .batching.Pool import GPU_AZ_Worker, PoolFactory
from .batching.Queues import mpQueueGen
from .games.GameBase import GameBase
from .games.GameStateBase import GameStateBase, CompactRepr
from .MCTS.MCTS_AlphaZero import MCTS_Factory
from .models.model import DEFAULT_ARCHITECTURE, create_model
from .utils import parse_args, training_model_args, metrics_daemon, ThreadSafeCounter

GameStateT = TypeVar("GameStateT", bound=GameStateBase)

def self_play(
         game_type: type[GameBase[GameStateT]],
         ROLLOUTS: int,
         PROCESSES: int,
         TARGET_GAME_WORKERS: int,
         GAME_WORKER_GAMES: int,
         model_args: dict,
         RANDOM_SEED: int
    ) -> list[tuple[CompactRepr, SparseLegalPolicy, int]]:
    ctx = mp.get_context("spawn")
    assert torch.cuda.is_available(), "GPU is not available."
    print(f"Using GPU for node evaluation.")
    pool_factory = PoolFactory(GPU_AZ_Worker, model_args=model_args)

    MCTS_factory: MCTS_Factory[GameStateT, Any] = MCTS_Factory(ROLLOUTS)

    game_worker_ps: list[BaseProcess] = []
    eval_worker_ps: list[BaseProcess] = []
    all_game_results: mpQueueGen[tuple[str, GameBase]] = mpQueueGen(ctx)

    # N queues for each game_worker --- sending to ---> ALL eval_workers
    request_queues: list[mpQueueGen[AZ_NodeBatchRequest | None]] = [
        mpQueueGen(ctx) for _ in range(TARGET_GAME_WORKERS)
    ]

    # N queues for ALL eval_workers --- sending to ---> N * game_workers queues
    results_queues: list[mpQueueGen[AZ_NodeBatchResponse]] = [
        mpQueueGen(ctx) for _ in range(TARGET_GAME_WORKERS)
    ]

    # Cumulative moves
    worker_metrics_queues: list[mpQueueGen[tuple[int, int, int, int, int, int]]] = []
    # Batch total, batches
    eval_metrics_queue: mpQueueGen[tuple[int, list[tuple], dict]] = mpQueueGen(ctx)

    for i in range(PROCESSES):
        eval_worker = pool_factory.create_AZ_GPU_worker(request_queues, results_queues, eval_metrics_queue)
        p = ctx.Process(target=eval_worker.run, name=f"EvalWorker_{i}", daemon=True)
        p.start()
        eval_worker_ps.append(p)

    for i, results_queue in enumerate(results_queues):
        worker_metrics_queue: mpQueueGen[tuple[int, int, int, int, int, int]] = mpQueueGen(ctx)
        worker_metrics_queues.append(worker_metrics_queue)
        game_worker = GameWorker(game_type,
                                 num_games=GAME_WORKER_GAMES,
                                 output_games=all_game_results,
                                 in_queue=results_queue,
                                 out_queue=request_queues[i],
                                 metrics_queue=worker_metrics_queue,
                                 worker_id=i,
                                 MCTS_factory=MCTS_factory,
                                 random_seed=RANDOM_SEED)
        p = ctx.Process(target=game_worker.run, name=f"GameWorker_{i}")
        game_worker_ps.append(p)

    for game_worker_p in game_worker_ps:
        game_worker_p.start()

    # Start the metrics daemon
    metrics_stop_event = Event()
    counter = ThreadSafeCounter()
    expecting_results = TARGET_GAME_WORKERS * GAME_WORKER_GAMES
    metrics_daemon_thread = Thread(target=metrics_daemon, args=(metrics_stop_event, worker_metrics_queues, eval_metrics_queue, counter, expecting_results), daemon=True)
    metrics_daemon_thread.start()

    memory: list[tuple[CompactRepr, SparseLegalPolicy, int]] = []
    for _ in range(expecting_results):
        _, game = all_game_results.get()
        counter.increment()
        memory.extend(zip(game.history['states'], game.history['MCTS_results'], game.history['outcome'], strict=True))

    for i, game_worker_p in enumerate(game_worker_ps):
        game_worker_p.join()
        request_queues[i].put(None)

    for eval_worker_p in eval_worker_ps:
        eval_worker_p.join()

    metrics_stop_event.set()
    metrics_daemon_thread.join()
    return memory

class AlphaZero:
    def __init__(self,
                 args: argparse.Namespace,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 model_args: dict
):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.model_args = model_args

        game_name = args.game_type.__name__.removesuffix("Game").lower()
        settings_name = (
            f"{game_name}"
            f"_r{args.rollouts}"
            f"_p{args.processes}"
            f"_gw{args.game_workers}"
            f"_gpw{args.games_per_worker}"
            f"_b{args.batch_size}"
            f"_e{args.epochs}"
        )
        architecture = model_args.get("architecture", DEFAULT_ARCHITECTURE)
        if architecture != DEFAULT_ARCHITECTURE:
            settings_name += f"_{architecture}"
        if args.resume is not None:
            settings_name += f"_resume_{args.resume.parent.name}_{args.resume.stem}"

        project_root = Path(__file__).resolve().parent.parent.parent
        self.artifact_dir = project_root / "artifacts" / settings_name
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.start_iteration = 0
        if args.resume is not None:
            model_path = args.resume.expanduser().resolve()

            if not model_path.is_file():
                raise FileNotFoundError(model_path)

            # Derive optimizer checkpoint from model checkpoint.
            if not re.fullmatch(r"model_\d+\.pt", model_path.name):
                raise ValueError(
                    "Resume checkpoint must be named model_NNN.pt"
                )

            optimizer_path = model_path.with_name(
                model_path.name.replace("model_", "optimizer_", 1)
            )

            if not optimizer_path.is_file():
                raise FileNotFoundError(optimizer_path)

            # Restoring weights verifies parameter/buffer shape compatibility.
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )

            # Restore optimizer state.
            # Optimizer is still associated with the model's CPU parameters.
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location="cpu", weights_only=True)
            )

            # Continue from the saved iteration number.
            checkpoint_iteration = int(model_path.stem.removeprefix("model_"))
            self.start_iteration = checkpoint_iteration + 1

            print(f"Resumed from {model_path}")

        args_payload = {k: v for k, v in model_args.items() if k not in ("batch_size", "state_dict")}
        args_payload["architecture"] = architecture
        args_payload["resume_from"] = (str(args.resume.expanduser().resolve()) if args.resume is not None else None)

        args_path = self.artifact_dir / "args.json"
        with open(args_path, "w") as f:
            json.dump(args_payload, f, indent=4)

    def move_optimizer_state(self, device: torch.device | str) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    def self_play(self,
                  GAME_TYPE: type[GameBase[GameStateBase]],
                  ROLLOUTS: int,
                  PROCESSES: int,
                  TARGET_GAME_WORKERS: int,
                  GAME_WORKER_GAMES: int,
                  RANDOM_SEED: int) -> list[tuple[CompactRepr, SparseLegalPolicy, int]]:
        worker_args = {**self.model_args, "state_dict": self.model.state_dict()}
        return self_play(GAME_TYPE, ROLLOUTS, PROCESSES, TARGET_GAME_WORKERS, GAME_WORKER_GAMES, worker_args, RANDOM_SEED)

    def train(self, GAME_TYPE: type[GameBase[GameStateBase]], memory: list[tuple[CompactRepr, SparseLegalPolicy, int]]):
        random.shuffle(memory)
        for batchIdx in trange(0, len(memory), self.args.batch_size, mininterval=2.5, desc="Training"):
            sample = memory[batchIdx:batchIdx + self.args.batch_size]

            states = [compact_repr.unpack() for compact_repr, _, _ in sample]
            policy_targets = [policy.unpack(GAME_TYPE.action_size) for _, policy, _ in sample]
            value_targets = [value for _, _, value in sample]

            state_tensor = torch.from_numpy(np.stack(states)).to("cuda").to(dtype=torch.float32)
            policy_targets_tensor = torch.from_numpy(np.stack(policy_targets)).to("cuda").to(dtype=torch.float32)
            value_targets_tensor = torch.tensor(value_targets).reshape(-1, 1).to("cuda").to(dtype=torch.float32)

            self.optimizer.zero_grad(set_to_none=True)

            out_policy, out_value = self.model(state_tensor)
            policy_loss = F.cross_entropy(out_policy, policy_targets_tensor)
            value_loss = F.mse_loss(out_value, value_targets_tensor)
            loss = policy_loss + value_loss

            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.start_iteration, self.start_iteration + self.args.iter):
            print_cuda_memory("Before cleanup")
            self.optimizer.zero_grad(set_to_none=True)
            self.model.to("cpu")
            self.move_optimizer_state("cpu")

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            print_cuda_memory("After cleanup")

            memory = self.self_play(
                self.args.game_type,
                self.args.rollouts,
                self.args.processes,
                self.args.game_workers,
                self.args.games_per_worker,
                self.args.seed
            )

            self.model.to("cuda")
            self.move_optimizer_state("cuda")
            self.model.train()
            for epoch in trange(self.args.epochs):
                self.train(self.args.game_type,memory)

            torch.save(
                self.model.state_dict(),
                self.artifact_dir / f"model_{iteration:03d}.pt",
            )

            torch.save(
                self.optimizer.state_dict(),
                self.artifact_dir / f"optimizer_{iteration:03d}.pt",
            )

            del memory

def print_cuda_memory(label: str):
    if not torch.cuda.is_initialized():
        print(f"{label}: CUDA not initialized")
        return

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2

    print(f"{label}: allocated={allocated:.1f} MiB, reserved={reserved:.1f} MiB")

def main():
    args = parse_args()
    model_args = training_model_args(args)
    model = create_model(**model_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_args["learning_rate"], weight_decay=model_args["weight_decay"])

    az = AlphaZero(args, model, optimizer, model_args)
    az.learn()
