from queue import Queue
from threading import Event
from unittest.mock import Mock

import pytest

from alphazero import utils
from alphazero.batching.Metrics import WorkerMetrics, WorkerTotals


def test_interval_reports_weight_workers_and_count_each_snapshot_once(monkeypatch, capsys):
    small = WorkerMetrics(0, WorkerTotals(fast_moves=1, fast_skipped_rounds=2, fast_budget_rounds=2,
                                          full_moves=1, full_cached_evals=1, full_terminal_evals=1, full_requested_evals=1))
    large = WorkerMetrics(1, WorkerTotals(fast_moves=3, fast_skipped_rounds=1, fast_budget_rounds=8,
                                          full_moves=4, full_cached_evals=2, full_terminal_evals=3, full_requested_evals=6))
    queues = [Queue(), Queue()]
    queues[0].put(small)
    queues[1].put(large)
    clock = Mock()
    clock.perf_counter.side_effect = [0, 1, 2]
    monkeypatch.setattr(utils, "time", clock)
    stop = Mock()

    def idle_interval():
        queues[0].put(small)
        return False

    def final_interval():
        queues[1].put(WorkerMetrics(1, large.totals._replace(full_moves=5, full_cached_evals=4, full_terminal_evals=4)))
        return True

    phases = iter([lambda: False, idle_interval, final_interval])
    stop.is_set.side_effect = lambda: next(phases)()
    utils.metrics_daemon(stop, queues, Queue(), utils.ThreadSafeCounter(), 2, print_interval_s=1)

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[Search reuse")]
    assert lines == [
        "[Search reuse] fast: 3 skipped/10 total(30%, 4 moves) | full: 3 cached + 4 terminal/14 total(21% + 29%, 5 moves)",
        "[Search reuse] fast: 0 skipped/0 total(N/A, 0 moves) | full: 0 cached + 0 terminal/0 total(N/A + N/A, 0 moves)",
        "[Search reuse total] fast: 3 skipped/10 total(30%, 4 moves) | full: 5 cached + 5 terminal/17 total(29% + 29%, 6 moves)",
    ]


@pytest.mark.parametrize("cpu_only", [True, False])
def test_short_run_drains_worker_queue_before_final_report(cpu_only, capsys):
    stop = Event()
    stop.set()
    snapshots = Queue()
    for moves in range(1, 13):
        snapshots.put(WorkerMetrics(0, WorkerTotals(full_moves=moves, full_terminal_evals=moves,
                                                     full_requested_evals=2 * moves)))
    utils.metrics_daemon(stop, [snapshots], None if cpu_only else Queue(), utils.ThreadSafeCounter(), 1)
    assert capsys.readouterr().out.strip() == (
        "[Search reuse total] fast: 0 skipped/0 total(N/A, 0 moves) | "
        "full: 0 cached + 12 terminal/36 total(0% + 33%, 12 moves)"
    )
