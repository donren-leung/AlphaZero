from typing import NamedTuple


class WorkerTotals(NamedTuple):
    moves: int = 0
    sent_batches: int = 0
    sent_requests: int = 0
    received_batches: int = 0
    received_results: int = 0

    fast_moves: int = 0
    fast_skipped_rounds: int = 0
    fast_terminal_rounds: int = 0
    fast_budget_rounds: int = 0

    full_moves: int = 0
    full_cached_evals: int = 0
    full_terminal_evals: int = 0
    full_requested_evals: int = 0

    def since(self, previous: "WorkerTotals") -> "WorkerTotals":
        return WorkerTotals(*(new - old for new, old in zip(self, previous, strict=True)))

class WorkerMetrics(NamedTuple):
    worker_id: int
    totals: WorkerTotals
