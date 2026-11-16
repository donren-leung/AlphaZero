from __future__ import annotations

from dataclasses import dataclass
from typing import Self, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
if TYPE_CHECKING:
    import torch

@dataclass(frozen=True, slots=True)
class BatchSparseLegalPolicy:
    """
    
    """
    thread_idxs: NDArray[np.uint16]
    lengths: NDArray[np.uint16]

    legal_actions: NDArray[np.uint16]
    probs: NDArray[np.float32]
    values: NDArray[np.float32]

    @classmethod
    def from_dense(cls,
                   legal_masks: torch.Tensor,
                   logits: torch.Tensor,
                   values: torch.Tensor,
                   thread_idxs: NDArray[np.uint16]) -> Self:
        """
        Convert dense batched policies into packed sparse policies.

        Args:
            legal_masks: (B, A) boolean legal-action mask.
            logits:      (B, A) dense policy.
            values:      (B, 1) value estimates.
            thread_idxs: (B,) thread identifiers.

        Returns:
            Packed representation where:
                worker_id:     (B,)
                lengths:       (B,)
                legal_actions: (N,)
                probs:         (N,)

            N = total number of legal actions across the batch.
        """
        import torch
        assert legal_masks.shape == logits.shape
        assert legal_masks.ndim == 2
        assert legal_masks.dtype == torch.bool
        assert values.shape == (legal_masks.shape[0], 1)
        assert thread_idxs.shape == (legal_masks.shape[0],)

        with torch.inference_mode():
            masked_logits = logits.float().masked_fill(~legal_masks, -torch.inf)

            # Handle terminal positions with zero legal actions.
            has_legal = legal_masks.any(dim=1, keepdim=True)
            masked_logits = torch.where(
                has_legal,
                masked_logits,
                torch.zeros_like(masked_logits),
            )

            probs = torch.softmax(masked_logits, dim=1)
            probs = probs.masked_fill(~legal_masks, 0.0)

            # Number of legal actions in each position.
            lengths = legal_masks.sum(dim=1)

            # Extract legal action indices and probabilities.
            # Drop the batch axis. We extract the indices of legal actions.
            legal_actions = torch.nonzero(legal_masks)[:, 1]
            sparse_probs = probs[legal_masks]
            assert torch.all((sparse_probs >= 0.0) & (sparse_probs <= 1.0)), "Sparse policy probabilities must be in range [0, 1]"

            # Transfer only sparse data to CPU.
            lengths_np = lengths.cpu().numpy().astype(np.uint16)
            actions_np = legal_actions.cpu().numpy().astype(np.uint16)
            probs_np = sparse_probs.cpu().numpy().astype(np.float32)

        return cls(
            thread_idxs=thread_idxs,
            lengths=lengths_np,
            legal_actions=actions_np,
            probs=probs_np,
            values=values.cpu().numpy().astype(np.float32),
        )

    def unpack(self) -> list[tuple[int, SparseLegalPolicy]]:
        """
        Unpack the batched sparse policy into a list of individual sparse policies.
        """
        offset = 0
        ret: list[tuple[int, SparseLegalPolicy]] = []
        for worker_idx, length, value in zip(self.thread_idxs, self.lengths, self.values, strict=True):
            end = offset + int(length)
            ret.append((
                int(worker_idx),
                SparseLegalPolicy(
                    legal_actions=self.legal_actions[offset:end], 
                    probs=self.probs[offset:end],
                    value=float(value)
                )
            ))

            offset = end

        return ret

@dataclass(slots=True)
class SparseLegalPolicy:
    legal_actions: NDArray[np.uint16]
    probs: NDArray[np.float32]
    value: float

    def unpack(self, action_size: int) -> NDArray[np.float32]:
        policy = np.zeros(action_size, dtype=np.float32)
        policy[self.legal_actions] = self.probs
        return policy

    @classmethod
    def value_only(cls, value: float) -> Self:
        return cls(
            legal_actions=np.empty(0, dtype=np.uint16),
            probs=np.empty(0, dtype=np.float32),
            value=value
        )

@dataclass(slots=True, frozen=True)
class AZ_NodeBatchRequest:
    worker_id: int

    # With batch axis at dim 0
    thread_idxs: NDArray[np.uint16]
    curr_players: NDArray[np.int8]
    # legal_masks: NDArray[np.bool_]

    # (B, K) packed legal actions, where K is the max legal actions.
    # Padded with zeros. Lengths tracks the number of legal actions.
    padded_legal_actions: NDArray[np.uint16]
    lengths: NDArray[np.uint16]
    
    states: NDArray[np.uint8]

    @classmethod
    def from_NodeRequests(cls, worker_id: int, requests: list[AZ_NodeRequest]) -> Self:
        thread_idxs = np.array([r.thread_idx for r in requests], dtype=np.uint16)
        curr_players = np.array([r.curr_player for r in requests], dtype=np.int8)
        legal_masks = np.stack([r.legal_mask for r in requests], axis=0, dtype=np.bool_)
        states = np.stack([r.state for r in requests], axis=0, dtype=np.uint8)

        B, A = legal_masks.shape
        assert B == len(thread_idxs) > 0
        assert A <= np.iinfo(np.uint16).max

        # 1. Find legal actions on CPU
        lengths = np.count_nonzero(legal_masks, axis=1).astype(np.uint16)
        # A flat array of all legal actions across the batch.
        rows, cols = np.nonzero(legal_masks)
        K = max(1, int(lengths.max()))

        # Pack actions into a padded (B, K) matrix.
        padded_legal_actions = np.zeros((B, K), dtype=np.uint16)

        # Offsets will tell us where in the flat array is the start of each batch's legal actions.
        offsets = np.empty(B, dtype=np.intp)
        offsets[0] = 0
        np.cumsum(lengths[:-1], dtype=np.intp, out=offsets[1:])

        # Offsets[rows] gets the offset for the row, for each legal action.
        # Subtracting that from arange means: this is the nth legal action for this row. 
        positions = (
            np.arange(cols.size, dtype=np.intp)
            - offsets[rows]
        )
        padded_legal_actions[rows, positions] = cols

        return cls(
            worker_id=worker_id,
            thread_idxs=thread_idxs,
            curr_players=curr_players,
            padded_legal_actions=padded_legal_actions,
            lengths=lengths,
            states=states
        )

@dataclass(slots=True, frozen=True)
class AZ_NodeRequest():
    thread_idx: int
    curr_player: int
    legal_mask: NDArray[np.bool_]
    state: NDArray[np.uint8]

@dataclass(slots=True, frozen=True)
class AZ_NodeBatchResponse:
    worker_id:  int
    batch_sparse_legal_policy: BatchSparseLegalPolicy

    def unpack(self) -> list[tuple[int, SparseLegalPolicy]]:
        return self.batch_sparse_legal_policy.unpack()

    def __len__(self) -> int:
        return len(self.batch_sparse_legal_policy.thread_idxs)

def mask_and_norm(legal: NDArray[np.bool_], policy: NDArray[np.float32]) -> NDArray[np.float32]:
    assert legal.shape == policy.shape

    masked_policy = np.where(legal, policy, 0.0)
    sums = masked_policy.sum(axis=-1, keepdims=True)
    legal_counts = legal.sum(axis=-1, keepdims=True)

    fallback = np.divide(
        legal.astype(np.float32),
        legal_counts,
        out=np.zeros_like(masked_policy),
        where=legal_counts != 0,
    )

    return np.divide(
        masked_policy,
        sums,
        out=fallback,
        where=sums > 0,
    )
