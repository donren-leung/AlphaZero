import torch

from alphazero.MCTS_batch import simulate_
from alphazero.models.model import ResNet

from .GameWorker import mpQueueGen
from .NodeBatch import NodeBatchRequest, NodeBatchResponse, SimulationReturnType

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

class GPU_Pool(object):
    def __init__(self,
                 inbox: mpQueueGen[NodeBatchRequest],
                 outboxes: list[mpQueueGen[NodeBatchResponse]],
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
            requests: list[NodeBatchRequest] = []
            states = 0
            while states < 1:  # Wait for at least 1 request to come in
                request = self.inbox.get()
                requests.append(request)
                states += len(request.states_and_actions)

            # Wait for B requests to come and retrieve
            # Release queue lock

            for request in requests:
                results = self.gpu_compute(request)  # For now, just process the first request in the batch
                response = NodeBatchResponse(request.worker_id, request.thread_id, results)
                self.outboxes[request.worker_id].put(response)

    def gpu_compute(self, request: NodeBatchRequest) -> list[SimulationReturnType]:
        # Batch up the game state and action pairs into a tensor
        x = torch.stack([state.to_tensor() for _, state in request.states_and_actions]).to(self.model.device)
        print(f"GPU_Pool.gpu_compute: x.shape={x.shape}, device={x.device}")
        B = x.size(0)

        # Call GPU
        with torch.inference_mode():
            policy_batch, value_batch = self.model(x)

        policy_batch = policy_batch.cpu()
        value_batch = value_batch.cpu()

        # Unpack
        unpacked = []
        for i in range(B):
            policy = policy_batch[i].numpy()
            value = value_batch[i].item()
            action, state = request.states_and_actions[i]
            _, terminal = state.get_value_and_terminated(action)
            unpacked.append((policy, value, terminal))

        return unpacked

class PoolFactory(object):
    def __init__(self, pool_type: type[CPU_RandomRollout_Pool] | type[GPU_Pool], model_args: dict | None = None):
        self.pool_type = pool_type
        self.model_args = model_args

        if self.pool_type == GPU_Pool and self.model_args is None:
            raise ValueError("GPUPool requires a model args dictionary to be provided.")

    def create_pool(self,
                    inbox: mpQueueGen[NodeBatchRequest],
                    outboxes: list[mpQueueGen[NodeBatchResponse]]) -> CPU_RandomRollout_Pool | GPU_Pool:
        if self.pool_type == GPU_Pool:
            assert self.model_args is not None
            return GPU_Pool(inbox, outboxes, self.model_args)
        else:
            return CPU_RandomRollout_Pool(inbox, outboxes)
