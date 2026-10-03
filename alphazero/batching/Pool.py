from collections import defaultdict
from multiprocessing.context import BaseContext
from typing import Generic, TypeVar
import queue
from queue import Queue
import time
from threading import Thread

import torch

from alphazero.MCTS.MCTS_batch import simulate_
from alphazero.models.model import ResNet

# from .GameWorker import mpQueueGen
from .NodeBatch import NodeBatchRequest, NodeBatchResponse,  SimulationReturnType
from .NodeBatch import AZ_NodeBatchRequest, AZ_NodeBatchResponse, AZ_SimulationReturnType

T = TypeVar("T")

class mpQueueGen(Generic[T]):
    """
    Generic wrapper around the multiprocessing Queue.
    """
    def __init__(self, ctx: BaseContext, *args, **kwargs):
        self._queue = ctx.Queue(*args, **kwargs)
        # Lock for consumers to ensure only one is waiting on the queue at a time
        self._consumer_lock = ctx.Lock()

    def put(self, obj: T, *args, **kwargs) -> None:
        try:
            self._queue.put(obj, *args, **kwargs)
        except Exception as e:
            raise e

    def get(self, *args, **kwargs) -> T:
        try:
            res = self._queue.get(*args, **kwargs)
        except Exception as e:
            raise e
        return res

    def consumer_lock(self):
        return self._consumer_lock

    def empty(self):
        return self._queue.empty()

class CPU_RandomRollout_Worker(object):
    def __init__(self,
                 inbox: mpQueueGen[NodeBatchRequest],
                 outboxes: list[mpQueueGen[NodeBatchResponse]]):
        self.inbox = inbox
        self.outboxes = outboxes

    def run(self) -> None:
        while True:
            request = self.inbox.get()
            worker_id, thread_id = request.worker_id, request.thread_id
            results = [simulate_(state,
                                 request.curr_player,
                                 action,
                                 target_sims=request.target_sims)
                       for action, state
                       in request.states_and_actions]

            response = NodeBatchResponse(worker_id, thread_id, results)
            self.outboxes[worker_id].put(response)

class GPU_AZ_Worker(object):
    # 5 ms after first request
    MAX_WAIT_S = 0.005
    MAX_PREFETCH = 3
    RESPONSE_COOL_DOWN_S = 0.005
    METRICS_INTERVAL_S = 0.5

    def __init__(self,
                 inbox: mpQueueGen[list[AZ_NodeBatchRequest] | None],
                 outboxes: list[mpQueueGen[AZ_NodeBatchResponse]],
                 model_args: dict):
        self.inbox = inbox
        self.outboxes = outboxes

        self.batch_size = model_args["batch_size"]
        model_args = {k: v for k, v in model_args.items() if k != "batch_size"}
        self.model = ResNet(**model_args)

    def run(self) -> None:
        device = torch.device("cuda")
        self.model = self.model.to(device)
        self.model.device = device
        self.model.eval()

        self.ready_batches: Queue[list[AZ_NodeBatchRequest]] = Queue(maxsize=self.MAX_PREFETCH)
        Thread(target=self.request_batch_d, args=[], daemon=True).start()

        self.ready_responses: Queue[tuple[int, int,  AZ_SimulationReturnType]] = Queue()
        Thread(target=self.response_d, args=[], daemon=True).start()

        while True:
            ready_batch = self.ready_batches.get()
            if ready_batch is None:
                break
            self.gpu_compute(ready_batch)

    def gpu_compute(self, batch: list[AZ_NodeBatchRequest]) -> None:
        # Batch up the game state and action pairs into a tensor
        x = torch.stack([request.state.to_tensor() for request in batch]).to(self.model.device)

        # Call GPU
        with torch.inference_mode():
            policy_batch, value_batch = self.model(x)
            policy_batch = torch.softmax(policy_batch, dim=1)

        policy_batch = policy_batch.cpu()
        value_batch = value_batch.cpu()

        for idx, request in enumerate(batch):
            policy = policy_batch[idx].numpy()
            value = value_batch[idx].item()
            result = (policy, value)
            self.ready_responses.put((request.worker_id, request.thread_id, result))

    def request_batch_d(self) -> None:
        pending: list[AZ_NodeBatchRequest] = []

        while True:
            # Start the next batch with any overflow from the previous one.
            requests = pending[:self.batch_size]
            pending = pending[self.batch_size:]

            if len(requests) == self.batch_size:
                # If we already have a full batch, don't block for more.
                self.ready_batches.put(requests)
                continue

            with self.inbox.consumer_lock():
                # Block indefinitely for the FIRST request.
                first = self.inbox.get()

                if first is None:
                    break
                requests.extend(first)
                deadline = time.monotonic() + self.MAX_WAIT_S

                # Once at least one request exists, don't block indefinitely.
                while len(requests) < self.batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break

                    try:
                        request = self.inbox.get(timeout=remaining)
                    except queue.Empty:
                        break

                    if request is None:
                        break

                    space = self.batch_size - len(requests)
                    requests.extend(request[:space])
                    pending.extend(request[space:])

            self.ready_batches.put(requests)

    def response_d(self) -> None:
        # Per-worker list of pending responses containing tuples of (thread_id, result)
        pending: defaultdict[int, list[tuple[int, AZ_SimulationReturnType]]] = defaultdict(list)

        while True:
            # Block until at least one result exists.
            worker_id, thread_id, result = self.ready_responses.get()
            pending[worker_id].append((thread_id, result))

            # Then drain everything currently available.
            while True:
                try:
                    worker_id, thread_id, result = self.ready_responses.get(block=False)
                except queue.Empty:
                    break

                pending[worker_id].append((thread_id, result))

            # Process the pending responses
            # Try every worker rather than getting stuck on one.
            for worker_id in list(pending):
                thread_result = pending[worker_id]
                if len(thread_result) == 0:
                    del pending[worker_id]
                    continue

                batch_response = AZ_NodeBatchResponse(worker_id, thread_result)
                try:
                    self.outboxes[worker_id].put(batch_response, block=False)
                except queue.Full:
                    # If the queue is full, we will try again next time.
                    continue

                del pending[worker_id]
            time.sleep(self.RESPONSE_COOL_DOWN_S)

class PoolFactory(object):
    def __init__(self, pool_type: type[CPU_RandomRollout_Worker] | type[GPU_AZ_Worker], model_args: dict | None = None):
        self.pool_type = pool_type
        self.model_args = model_args

        if self.pool_type == GPU_AZ_Worker and self.model_args is None:
            raise ValueError("GPUPool requires a model args dictionary to be provided.")

    def create_CPU_rollout_worker(self,
                                inbox: mpQueueGen[NodeBatchRequest],
                                outboxes: list[mpQueueGen[NodeBatchResponse]]) -> CPU_RandomRollout_Worker:
        assert self.pool_type in [CPU_RandomRollout_Worker]
        return CPU_RandomRollout_Worker(inbox, outboxes)

    def create_AZ_GPU_worker(self,
                            inbox: mpQueueGen[list[AZ_NodeBatchRequest] | None],
                            outboxes: list[mpQueueGen[AZ_NodeBatchResponse]]) -> GPU_AZ_Worker:
        assert self.pool_type in [GPU_AZ_Worker]
        assert self.model_args is not None
        return GPU_AZ_Worker(inbox, outboxes, self.model_args)
