from threading import Lock
from unittest.mock import Mock

import numpy as np
import pytest

from alphazero.batching.AZ_GameWorker import GameWorker
from alphazero.batching.NodeBatch import AZ_NodeRequest, SparseLegalPolicy
from alphazero.games.TicTacToe import TicTacToeGame
from alphazero.MCTS.MCTS_AlphaZero import MCTS_Factory

"""Test subtree reuse across consecutive self-play moves.

A full search starts with fresh visit counts and uses the retained child
as a cache of network evaluations. With a deterministic evaluator and the
same random seed, the cached and cold searches produce identical root
statistics and recorded policy and value. The cached search issues fewer
evaluator requests in this position.

A fast search uses the retained child as its root and counts inherited
visits toward its rollout budget, adds no Dirichlet noise to the new root
priors, and it records no training example.
"""

def evaluate(request: AZ_NodeRequest) -> SparseLegalPolicy:
    actions = np.flatnonzero(request.legal_mask).astype(np.uint16)
    weights = actions.astype(np.float32) + 1
    value = float(request.state[2].sum()) / 9
    return SparseLegalPolicy(actions, weights / weights.sum(), value)


@pytest.fixture
def worker_evaluator(monkeypatch):
    worker = GameWorker(
        TicTacToeGame, num_games=1, worker_id=0, random_seed=0, iteration=0,
        MCTS_factory=MCTS_Factory(64), in_queue=Mock(), out_queue=Mock(),
        output_games=Mock(), metrics_queue=Mock(),
    )
    evaluator = Mock(side_effect=evaluate)
    monkeypatch.setattr(worker, "_evaluate_leaf", evaluator)
    worker.metrics_lock = Lock()
    return worker, evaluator


@pytest.fixture
def retained_position(worker_evaluator):
    worker, evaluator = worker_evaluator
    game = TicTacToeGame()
    actions, policy, search = worker.search_move(0, game, None, np.random.default_rng(1), fast=False)
    action = int(actions[policy.argmax()])
    child = search.get_child_by_action(action)
    search.dispose(kept_child=child)
    game.make_move(action)
    assert worker.full_moves == 1
    assert worker.full_cached_evals == 0
    assert worker.full_requested_evals == evaluator.call_count
    evaluator.reset_mock()
    return worker, evaluator, game, child


def test_cached_full_search_matches_cold_search(retained_position):
    worker, evaluator, game, retained = retained_position
    raw_priors = retained.floats[0].copy()
    seed = 37

    _, _, cold = worker.search_move(0, game, None, np.random.default_rng(seed), fast=False)
    cold_requests = evaluator.call_count
    requests_before_cached_search = worker.full_requested_evals
    terminals_before_cached_search = worker.full_terminal_evals
    cold_target = game.history["MCTS_results"][-1]
    evaluator.reset_mock()

    _, _, cached = worker.search_move(0, game, retained, np.random.default_rng(seed), fast=False)
    # Save requests for both the promoted root and at least one descendant.
    assert 0 < evaluator.call_count <= cold_requests - 2
    assert worker.full_cached_evals == cold_requests - evaluator.call_count
    assert worker.full_requested_evals - requests_before_cached_search == evaluator.call_count
    terminal_rounds = worker.full_terminal_evals - terminals_before_cached_search
    assert worker.full_cached_evals + terminal_rounds + evaluator.call_count == worker.MCTS_factory.rollouts + 1
    assert worker.full_moves == 3
    assert cached.root is not retained
    assert cached.root.ints[1].sum() == worker.MCTS_factory.rollouts
    np.testing.assert_array_equal(cached.root.ints, cold.root.ints)
    np.testing.assert_array_equal(cached.root.floats, cold.root.floats)
    assert cached.root.value_sum == cold.root.value_sum

    assert not np.array_equal(cached.root.floats[0], raw_priors)
    np.testing.assert_array_equal(retained.floats[0], raw_priors)
    assert len(game.history["MCTS_results"]) == 3
    cached_target = game.history["MCTS_results"][-1]
    np.testing.assert_array_equal(cached_target.legal_actions, cold_target.legal_actions)
    np.testing.assert_array_equal(cached_target.probs, cold_target.probs)
    assert cached_target.value == cold_target.value


@pytest.mark.parametrize("fast_factor", [1 / 64, 1.0])
def test_fast_search_reuses_visits_and_skips_training_example(retained_position, fast_factor):
    worker, _, game, retained = retained_position
    worker.MCTS_factory.fast_rollouts_factor = fast_factor
    full_counts = (worker.full_moves, worker.full_cached_evals, worker.full_terminal_evals, worker.full_requested_evals)
    old_visits = retained.visits
    raw_priors = retained.floats[0].copy()
    _, _, search = worker.search_move(0, game, retained, np.random.default_rng(2), fast=True)

    assert search.root is retained
    assert search.root.visits == max(old_visits, search.num_rollouts(fast=True) + 1)
    new_rounds = search.root.visits - old_visits
    assert (new_rounds == 0) == (fast_factor == 1 / 64)
    assert worker.fast_budget_rounds == search.num_rollouts(fast=True) + 1
    assert worker.fast_skipped_rounds == worker.fast_budget_rounds - new_rounds
    assert worker.fast_moves == 1
    assert (worker.full_moves, worker.full_cached_evals, worker.full_terminal_evals, worker.full_requested_evals) == full_counts
    np.testing.assert_array_equal(search.root.floats[0], raw_priors)
    assert len(game.history["MCTS_results"]) == 1


def test_terminal_backups_do_not_count_as_model_evaluations(worker_evaluator):
    worker, evaluator = worker_evaluator
    game = TicTacToeGame()
    for action in [0, 1, 2, 4, 3, 5, 7, 6]:
        game.make_move(action)
    _, _, search = worker.search_move(0, game, None, np.random.default_rng(1), fast=False)
    assert search.root.visits == worker.MCTS_factory.rollouts + 1
    assert worker.full_requested_evals == evaluator.call_count == 1
    assert worker.full_cached_evals == 0
    assert worker.full_terminal_evals == worker.MCTS_factory.rollouts


def test_worker_publishes_final_metrics_after_games_finish(worker_evaluator, monkeypatch):
    worker, _ = worker_evaluator
    monkeypatch.setattr(worker, "inbound_mailman_d", lambda: None)
    monkeypatch.setattr(worker, "outbound_mailman_d", lambda: None)
    worker.metrics_interval_s = 3600
    worker.num_games = 2
    worker.run()
    snapshot, = worker.metrics_queue.put.call_args.args
    totals = snapshot.totals
    assert worker.output_games.put.call_count == 2
    full_moves = sum(len(call.args[0][1]) for call in worker.output_games.put.call_args_list)
    assert totals.full_moves == full_moves
    full_rounds = totals.full_cached_evals + totals.full_terminal_evals + totals.full_requested_evals
    assert full_rounds == full_moves * (worker.MCTS_factory.rollouts + 1)
    assert totals.fast_moves + full_moves == totals.moves == worker.total_moves


def test_chess_search_matches_b1fd8df(worker_evaluator):
    from alphazero.games.Chess import ChessGame

    worker, _ = worker_evaluator
    worker.MCTS_factory.rollouts = 32
    game = ChessGame()
    actions, policy, search = worker.search_move(0, game, None, np.random.default_rng(37), fast=False)
    # Generated on b1fd8df with the same evaluator, player, seed and rollout budget.
    expected_visits = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 20, 1, 1]
    assert search.root.ints[1].tolist() == expected_visits
    assert search.root.value_sum == pytest.approx(1.1111111111111107)
    assert actions[policy.argmax()] == 3585
    assert search.game_state.board.history == []
    np.testing.assert_array_equal(search.game_state.encode_for_model(), game.encode_for_model())


def _finish_chess_game(output):
    from alphazero.games.Chess import ChessGame

    game = ChessGame()
    for san in ("f3", "e5", "g4"):
        game.make_move(game.parse_move_str(san))
    mate = game.parse_move_str("Qh4#")
    factory = MCTS_Factory(4)
    factory.dirichlet_epsilon = 0
    worker = GameWorker(ChessGame, num_games=1, worker_id=0, random_seed=1, iteration=0, MCTS_factory=factory,
                        in_queue=Mock(), out_queue=Mock(), output_games=output, metrics_queue=Mock())
    worker.metrics_lock = Lock()

    def force_mate(request):
        actions = np.flatnonzero(request.legal_mask).astype(np.uint16)
        return SparseLegalPolicy(actions, (actions == mate).astype(np.float32), 0.0)

    worker._evaluate_leaf = force_mate
    worker.run_game(0, game)


def test_completed_chess_examples_cross_spawn_queue():
    import multiprocessing
    from alphazero.batching.Queues import mpQueueGen
    from alphazero.games.Chess import ChessGame

    ctx = multiprocessing.get_context("spawn")
    queue = mpQueueGen(ctx)
    child = ctx.Process(target=_finish_chess_game, args=(queue,))
    child.start()
    try:
        game_id, examples = queue.get(timeout=45)
    finally:
        child.join(timeout=10)
        if child.is_alive():
            child.terminate()
            child.join()
    assert child.exitcode == 0
    assert game_id == "0_0"
    (compact, policy, outcome), = examples
    game = ChessGame()
    for san in ("f3", "e5", "g4"):
        game.make_move(game.parse_move_str(san))
    np.testing.assert_array_equal(compact.unpack(), game.encode_for_model())
    assert policy.unpack(game.action_size).argmax() == game.parse_move_str("Qh4#")
    assert outcome == 1
