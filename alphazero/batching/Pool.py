from multiprocessing.context import BaseContext
from typing import Generic, TypeVar

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

    def put(self, item: T) -> None:
        self._queue.put(item)

    def get(self) -> T:
        return self._queue.get()

    def empty(self):
        return self._queue.empty()

class CPU_RandomRollout_Pool(object):
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

class GPU_AZ_Pool(object):
    def __init__(self,
                 inbox: mpQueueGen[AZ_NodeBatchRequest],
                 outboxes: list[mpQueueGen[AZ_NodeBatchResponse]],
                 model_args: dict):
        self.inbox = inbox
        self.outboxes = outboxes
        self.model = ResNet(**model_args)

    def run(self) -> None:
        device = torch.device("cuda")
        self.model = self.model.to(device)
        self.model.device = device
        self.model.eval()

        while True:
            # Grab queue lock first (only 1 consumer should be actively waiting on the queue at a time)
            requests: list[AZ_NodeBatchRequest] = []
            states = 0
            while states < 1:  # Wait for at least 1 request to come in
                request = self.inbox.get()
                requests.append(request)
                states += 1

            # Wait for B requests to come and retrieve
            # Release queue lock

            responses = self.gpu_compute(requests)  # For now, just process the first request in the batch
            for response in responses:
                self.outboxes[response.worker_id].put(response)

    def gpu_compute(self, requests: list[AZ_NodeBatchRequest]) -> list[AZ_NodeBatchResponse]:
        # Batch up the game state and action pairs into a tensor
        x = torch.stack([request.state.to_tensor() for request in requests]).to(self.model.device)
        print(f"GPU_Pool.gpu_compute: x.shape={x.shape}, device={x.device}")
        B = x.size(0)

        # Call GPU
        with torch.inference_mode():
            policy_batch, value_batch = self.model(x)

        policy_batch = policy_batch.cpu()
        value_batch = value_batch.cpu()

        # Unpack
        results: list[AZ_NodeBatchResponse] = []
        for idx, request in enumerate(requests):
            policy = policy_batch[idx].numpy()
            value = value_batch[idx].item()
            result = (policy, value)
            results.append(AZ_NodeBatchResponse(request.worker_id, request.thread_id, result))

        return results

class PoolFactory(object):
    def __init__(self, pool_type: type[CPU_RandomRollout_Pool] | type[GPU_AZ_Pool], model_args: dict | None = None):
        self.pool_type = pool_type
        self.model_args = model_args

        if self.pool_type == GPU_AZ_Pool and self.model_args is None:
            raise ValueError("GPUPool requires a model args dictionary to be provided.")

    def create_pool(self,
                    inbox: mpQueueGen[NodeBatchRequest],
                    outboxes: list[mpQueueGen[NodeBatchResponse]]) -> CPU_RandomRollout_Pool:
        assert self.pool_type in [CPU_RandomRollout_Pool]
        return CPU_RandomRollout_Pool(inbox, outboxes)

    def create_AZpool(self,
                      inbox: mpQueueGen[AZ_NodeBatchRequest],
                      outboxes: list[mpQueueGen[AZ_NodeBatchResponse]]) -> GPU_AZ_Pool:
        assert self.pool_type in [AZ_NodeBatchRequest]
        assert self.model_args is not None
        return GPU_AZ_Pool(inbox, outboxes, self.model_args)
