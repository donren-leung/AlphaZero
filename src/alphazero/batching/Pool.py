from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from queue import Queue
import time
from threading import Thread
from typing import Any, Callable, Counter, Self, cast

import torch
import numpy as np

from alphazero.models.model import create_model

from .NodeBatch import AZ_NodeBatchRequest, AZ_NodeBatchResponse, BatchSparseLegalPolicy
from .Queues import mpQueueGen

@dataclass(frozen=True)
class Route:
    worker_id: int
    start: int
    stop: int

@dataclass(frozen=True)
class CombinedBatch:
    states: np.ndarray
    thread_idxs: np.ndarray
    lengths: np.ndarray
    padded_legal_actions: np.ndarray
    routes: tuple[Route, ...]

    @classmethod
    def combine(cls, batches: list[AZ_NodeBatchRequest]) -> Self:
        if not batches:
            raise ValueError("Cannot combine an empty batch list")

        real_bs = sum(len(b.thread_idxs) for b in batches)
        real_width = max(b.padded_legal_actions.shape[1] for b in batches)

        batch_size = max(4, 1 << (real_bs - 1).bit_length())
        width = 1 << (max(64, real_width) - 1).bit_length()

        # [B', C, H, W]
        states = np.zeros((batch_size, *batches[0].states.shape[1:]), dtype=np.uint8)
        # [B', K']
        actions = np.zeros((batch_size, width), dtype=np.uint16)
        # [B']
        lengths = np.zeros(batch_size, dtype=np.uint16)

        routes: list[Route] = []
        offset = 0

        for batch in batches:
            batch_count, local_width = batch.padded_legal_actions.shape
            stop = offset + batch_count

            states[offset:stop] = batch.states
            actions[offset:stop, :local_width] = batch.padded_legal_actions
            lengths[offset:stop] = batch.lengths

            routes.append(Route(batch.worker_id, offset, stop))
            offset = stop

        return cls(
            states=states,
            thread_idxs=np.concatenate([b.thread_idxs for b in batches]),
            lengths=lengths,
            padded_legal_actions=actions,
            routes=tuple(routes),
        )

    @classmethod
    def deliver(cls, result: BatchSparseLegalPolicy, routes: tuple[Route, ...], outboxes: list[mpQueueGen[AZ_NodeBatchResponse]]) -> None:
        offsets = np.empty(len(result.lengths) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(result.lengths, dtype=np.int64, out=offsets[1:])

        for route in routes:
            rows = slice(route.start, route.stop)
            actions = slice(offsets[route.start], offsets[route.stop])

            response = BatchSparseLegalPolicy(
                thread_idxs=result.thread_idxs[rows],
                lengths=result.lengths[rows],
                legal_actions=result.legal_actions[actions],
                probs=result.probs[actions],
                values=result.values[rows],
            )
            outboxes[route.worker_id].put(
                AZ_NodeBatchResponse(route.worker_id, response)
            )

class GPU_AZ_Worker(object):
    METRICS_INTERVAL_S = 0.5
    MAX_PREFETCH = 2
    MAX_WAIT_S = 0.02

    def __init__(self,
                 inboxes: list[mpQueueGen[AZ_NodeBatchRequest | None]],
                 outboxes: list[mpQueueGen[AZ_NodeBatchResponse]],
                 metrics_queue: mpQueueGen[tuple[int, list[tuple], dict]],
                 model_args: dict):
        self.inboxes = inboxes
        self.outboxes = outboxes
        self.metrics_queue = metrics_queue

        self.batch_size = model_args["inference_max_batch_size"]
        self.device = torch.device("cuda")
        self.model = create_model(**model_args)

    def run(self) -> None:
        self.model.to(self.device)
        self.model.eval()
        # self.compiled_model: Callable[..., Any] = torch.compile(self.model, mode="reduce-overhead")
        self.compiled_compute = torch.compile(self.compute_tensors, mode="reduce-overhead", fullgraph=True)

        torch.set_num_threads(2)
        print("Evaluator CPU intra-op threads:", torch.get_num_threads(), flush=True)

        # Each row: (receive seconds, compute seconds, submit seconds)
        timings = []
        batch_sizes: Counter[tuple[int, int]] = Counter()
        nodes = 0

        self.ready_batches: Queue[
            tuple[CombinedBatch, tuple[float, ...]] | Exception | None
        ] = Queue(maxsize=self.MAX_PREFETCH)
        def collect_requests():
            try:
                self.request_batch()
            except Exception as error:
                self.ready_batches.put(error)
        collector_thread = Thread(target=collect_requests, daemon=True)
        collector_thread.start()

        window_start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            while True:
                item = self.ready_batches.get()
                if item is None:
                    collector_thread.join()
                    break

                if isinstance(item, Exception):
                    collector_thread.join()
                    raise RuntimeError("Request batching failed") from item

                combined_batch, (t0, t1, t2) = item

                t3 = time.perf_counter()
                response = self.gpu_compute(combined_batch)
                t4 = time.perf_counter()

                CombinedBatch.deliver(response, combined_batch.routes, self.outboxes)
                t5 = time.perf_counter()

                timings.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4))
                size = tuple(combined_batch.padded_legal_actions.shape)[:2]
                batch_sizes[size] += 1
                nodes += len(combined_batch.thread_idxs)

                elapsed = t4 - window_start
                if elapsed >= self.METRICS_INTERVAL_S:
                    # Send metrics
                    self.metrics_queue.put((
                        nodes,
                        timings,
                        dict(batch_sizes),
                    ))

                    timings = []
                    batch_sizes.clear()
                    nodes = 0
                    window_start = time.perf_counter()

        # Flush remaining metrics.
        if nodes:
            self.metrics_queue.put((
                nodes,
                timings,
                dict(batch_sizes),
            ))

    # NOTE: Can sometimes exceed batch size.
    def request_batch(self) -> None:
        def fish(wait_time: float | None) -> AZ_NodeBatchRequest | None:
            # Wait for any producer.
            ready = wait(reader_to_worker, wait_time)
            if not ready:
                return None

            reader = cast(Connection, ready[0])
            worker_id = reader_to_worker.pop(reader)

            batch = self.inboxes[worker_id].get()

            if batch is None:
                # Producer terminated normally.
                return None

            # Reinsert at end for fairer scheduling.
            reader_to_worker[reader] = worker_id
            assert batch.worker_id == worker_id
            return batch

        reader_to_worker = {
            inbox._reader(): worker_id
            for worker_id, inbox in enumerate(self.inboxes)
        }

        while reader_to_worker:
            t0 = time.perf_counter()
            batch = fish(None)
            t1 = time.perf_counter()

            if batch is None:
                continue

            batches = [batch]
            batch_size = len(batches[0].thread_idxs)
            deadline = time.perf_counter() + self.MAX_WAIT_S

            while reader_to_worker and batch_size < self.batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break

                new_batch = fish(remaining)
                if new_batch is None:
                    continue

                batches.append(new_batch)
                batch_size += len(new_batch.thread_idxs)

            t2 = time.perf_counter()
            combined_batch = CombinedBatch.combine(batches)
            self.ready_batches.put((combined_batch, (t0, t1, t2)))

        self.ready_batches.put(None)

    def gpu_compute(self, batch: CombinedBatch | AZ_NodeBatchRequest) -> BatchSparseLegalPolicy:
        # Transfer compact inputs to GPU.
        # (B, C, H, W)
        x = torch.as_tensor(batch.states, device=self.device).to(torch.float32)
        # (B, K)
        action_idx = torch.as_tensor(batch.padded_legal_actions, device=self.device).to(torch.int64)
        # (B,)
        lengths_gpu = torch.as_tensor(batch.lengths, device=self.device).to(torch.int32)

        probs, values = self.compiled_compute(x, action_idx, lengths_gpu)

        # Transfer back to CPU
        probs_cpu = probs.cpu().numpy()
        values_cpu = values.cpu().numpy()

        # Remove padding
        real_size = len(batch.thread_idxs)
        # (R,)
        real_lengths = batch.lengths[:real_size]
        # (R, K)
        real_actions = batch.padded_legal_actions[:real_size]
        # (R, K) boolean mask
        valid_cpu = np.arange(real_actions.shape[1])[None, :] < real_lengths[:, None]

        return BatchSparseLegalPolicy(
            thread_idxs=batch.thread_idxs,
            lengths=real_lengths,
            legal_actions=real_actions[valid_cpu],
            probs=probs_cpu[:real_size][valid_cpu],
            values=values_cpu[:real_size],
        )

    def compute_tensors(self, x, action_idx, lengths_gpu):
        # Evaluate.
        # policy is (B, A) where A is e.g. 64*73=4672 and values is (B,)
        policy_logits, values = self.model(x)

        # assert torch.isfinite(policy_logits).all()
        # assert torch.isfinite(values).all()

        K = action_idx.shape[1]
        # Gather only legal-action logits via broadcasting -> (B, K)
        valid = (
            # (,K)
            torch.arange(K, device=self.device)[None, :]
            # (B,)
            < lengths_gpu[:, None]
        )

        # Gather dim=1: along action dimension, for each batch row -> (B, K)
        masked_logits = policy_logits.gather(1, action_idx).float().masked_fill(~valid, -torch.inf)

        # Avoid NaNs for positions with zero legal moves.
        safe_logits = torch.where(
            lengths_gpu[:, None] > 0,
            masked_logits,
            torch.zeros_like(masked_logits),
        )

        probs = torch.softmax(safe_logits, dim=1)
        # probs = probs.masked_fill(~valid, 0.0) Not necessary if CPU filters out padding.
        return probs, values.float().reshape(-1)

class PoolFactory(object):
    def __init__(self,
                 pool_type: type[GPU_AZ_Worker],
                 model_args: dict | None = None):

        self.pool_type = pool_type
        self.model_args = model_args

        if self.pool_type == GPU_AZ_Worker and self.model_args is None:
            raise ValueError("GPUPool requires a model args dictionary to be provided.")

    def create_AZ_GPU_worker(self,
                            inboxes: list[mpQueueGen[AZ_NodeBatchRequest | None]],
                            outboxes: list[mpQueueGen[AZ_NodeBatchResponse]],
                            metrics_queue: mpQueueGen[tuple[int, list[tuple], dict]]) -> GPU_AZ_Worker:
        assert self.pool_type in [GPU_AZ_Worker]
        assert self.model_args is not None
        return GPU_AZ_Worker(inboxes, outboxes, metrics_queue, self.model_args,)
