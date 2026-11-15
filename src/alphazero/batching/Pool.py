import torch
import numpy as np

from alphazero.models.model import ResNet

from .NodeBatch import AZ_NodeBatchRequest, AZ_NodeBatchResponse, BatchSparseLegalPolicy
from .Queues import mpQueueGen

class GPU_AZ_Worker(object):
    # 5 ms after first request
    MAX_PREFETCH = 2
    METRICS_INTERVAL_S = 0.5

    def __init__(self,
                 inbox: mpQueueGen[AZ_NodeBatchRequest | None],
                 outboxes: list[mpQueueGen[AZ_NodeBatchResponse]],
                 model_args: dict):
        self.inbox = inbox
        self.outboxes = outboxes

        self.batch_size = model_args["batch_size"]
        model_args = {k: v for k, v in model_args.items() if k != "batch_size"}
        self.device = torch.device("cuda")
        self.model = ResNet(**model_args)

    def run(self) -> None:
        self.model.to(self.device)
        self.model.eval()

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            while True:
                batch = self.inbox.get()
                if batch is None:
                    break
                response = self.gpu_compute(batch)
                self.outboxes[batch.worker_id].put(AZ_NodeBatchResponse(batch.worker_id, response))

    def gpu_compute(self, batch: AZ_NodeBatchRequest) -> BatchSparseLegalPolicy:
        K = batch.padded_legal_actions.shape[1]

        # Transfer compact inputs to GPU.
        # (B, C, H, W)
        x = torch.as_tensor(batch.states, device=self.device, dtype=torch.float32)
        # (B, K)
        action_idx = torch.as_tensor(batch.padded_legal_actions, device=self.device, dtype=torch.int64)
        # (B,)
        lengths_gpu = torch.as_tensor(batch.lengths, device=self.device, dtype=torch.int32)

        # Evaluate.
        # policy is (B, A) where A is e.g. 64*73=4672 and values is (B,)
        policy_logits, values = self.model(x)

        # assert torch.isfinite(policy_logits).all()
        # assert torch.isfinite(values).all()

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

        # Transfer fixed-shape outputs to CPU.
        # This synchronizes so that the arrays are ready.
        probs_cpu = probs.cpu().numpy()
        values_cpu = values.float().reshape(-1).cpu().numpy()

        # Remove padding on CPU.
        valid_cpu = np.arange(K)[None, :] < batch.lengths[:, None]

        return BatchSparseLegalPolicy(
            thread_idxs=batch.thread_idxs,
            lengths=batch.lengths,
            legal_actions=batch.padded_legal_actions[valid_cpu],
            probs=probs_cpu[valid_cpu],
            values=values_cpu,
        )

class PoolFactory(object):
    def __init__(self,
                 pool_type: type[GPU_AZ_Worker],
                 model_args: dict | None = None):

        self.pool_type = pool_type
        self.model_args = model_args

        if self.pool_type == GPU_AZ_Worker and self.model_args is None:
            raise ValueError("GPUPool requires a model args dictionary to be provided.")

    def create_AZ_GPU_worker(self,
                            inbox: mpQueueGen[AZ_NodeBatchRequest | None],
                            outboxes: list[mpQueueGen[AZ_NodeBatchResponse]]) -> GPU_AZ_Worker:
        assert self.pool_type in [GPU_AZ_Worker]
        assert self.model_args is not None
        return GPU_AZ_Worker(inbox, outboxes, self.model_args)
