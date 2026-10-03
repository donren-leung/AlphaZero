import argparse
import multiprocessing as mp
from multiprocessing.process import BaseProcess
from pathlib import Path
import random
from threading import Thread, Event

import torch
import torch.nn.functional as F
import numpy as np
from numpy.typing import NDArray
from tqdm import trange

from alphazero.games.GameBase import GameBase
from alphazero.MCTS.MCTS_AlphaZero import MCTS_Factory

from .batching.NodeBatch import AZ_NodeBatchRequest, AZ_NodeBatchResponse
from .batching.AZ_GameWorker import GameWorker, mpQueueGen
from .batching.Pool import GPU_AZ_Worker, PoolFactory
from .games.GameStateBase import GameStateBase
from .models.model import ResNet
from .utils import parse_args, metrics_daemon, ThreadSafeCounter

def self_play(
         ROLLOUTS: int,
         PROCESSES: int,
         TARGET_GAME_WORKERS: int,
         GAME_WORKER_GAMES: int,
         model_args: dict,
    ) -> list[tuple[GameStateBase, NDArray[np.float32], int]]:
    ctx = mp.get_context("spawn")
    assert torch.cuda.is_available(), "GPU is not available."
    print(f"Using GPU for node evaluation.")
    pool_factory = PoolFactory(GPU_AZ_Worker, model_args=model_args)

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
        game_worker = GameWorker(model_args["game_type"],
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
    metrics_stop_event = Event()
    counter = ThreadSafeCounter()
    expecting_results = TARGET_GAME_WORKERS * GAME_WORKER_GAMES
    metrics_daemon_thread = Thread(target=metrics_daemon, args=(metrics_stop_event, worker_metrics_queue, counter, expecting_results), daemon=True)
    metrics_daemon_thread.start()

    memory: list[tuple[GameStateBase, NDArray[np.float32], int]] = []
    for _ in range(expecting_results):
        _, game = all_game_results.get()
        counter.increment()
        memory.extend(zip(game.state_history, game.action_prob_history, game.outcome, strict=True))

    for game_worker_p in game_worker_ps:
        game_worker_p.join()

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

        self.model_args["state_dict"] = self.model.state_dict()

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

        project_root = Path(__file__).resolve().parent.parent
        self.artifact_dir = project_root / "artifacts" / settings_name
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def self_play(self, ROLLOUTS: int, PROCESSES: int, TARGET_GAME_WORKERS: int, GAME_WORKER_GAMES: int) -> list[tuple[GameStateBase, NDArray[np.float32], int]]:
        self.model_args["state_dict"] = self.model.state_dict()
        return self_play(ROLLOUTS, PROCESSES, TARGET_GAME_WORKERS, GAME_WORKER_GAMES, self.model_args)

    def train(self, memory: list[tuple[GameStateBase, NDArray[np.float32], int]]):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args.batch_size):
            sample = memory[batchIdx:batchIdx + self.args.batch_size]

            states = [state for state, _, _ in sample]
            policy_targets = [policy for _, policy, _ in sample]
            value_targets = [value for _, _, value in sample]

            state_tensor = torch.stack([state.to_tensor() for state in states]).to(dtype=torch.float32)

            policy_targets_tensor = torch.from_numpy(np.stack(policy_targets)).to(dtype=torch.float32)

            value_targets_tensor = torch.tensor(value_targets, dtype=torch.float32).reshape(-1, 1).to(dtype=torch.float32)

            out_policy, out_value = self.model(state_tensor)

            policy_loss = F.cross_entropy(out_policy, policy_targets_tensor)
            value_loss = F.mse_loss(out_value, value_targets_tensor)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args.iter):
            memory = self_play(
                self.args.rollouts,
                self.args.processes,
                self.args.game_workers,
                self.args.games_per_worker,
                self.model_args
            )

            self.model.train()
            for epoch in trange(self.args.epochs):
                self.train(memory)

                torch.save(
                    self.model.state_dict(),
                    self.artifact_dir / f"model_{iteration:03d}.pt",
                )

                torch.save(
                    self.optimizer.state_dict(),
                    self.artifact_dir / f"optimizer_{iteration:03d}.pt",
                )

if __name__ == "__main__":
    args = parse_args()

    model_args = {
        "game_type": args.game_type,
        "num_resBlocks": 3,
        "num_channels": 32,
        "head_hidden_size": 16,
        "device": "cpu",
        "batch_size": args.batch_size,
        "state_dict": None,
    }

    model = ResNet(**model_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    az = AlphaZero(args, model, optimizer, model_args)
    az.learn()
