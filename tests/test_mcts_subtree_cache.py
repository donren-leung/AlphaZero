from unittest.mock import Mock

import numpy as np
import pytest

from alphazero.batching.AZ_GameWorker import GameWorker
from alphazero.batching.NodeBatch import AZ_NodeRequest, SparseLegalPolicy
from alphazero.games.TicTacToe import TicTacToeGame
from alphazero.MCTS.MCTS_AlphaZero import MCTS_Factory


def evaluate(request: AZ_NodeRequest) -> SparseLegalPolicy:
    actions = np.flatnonzero(request.legal_mask).astype(np.uint16)
    weights = actions.astype(np.float32) + 1
    value = float(request.state[2].sum()) / 9
    return SparseLegalPolicy(actions, weights / weights.sum(), value)


@pytest.fixture
def retained_position(monkeypatch):
    worker = GameWorker(
        TicTacToeGame, num_games=1, worker_id=0, random_seed=0,
        MCTS_factory=MCTS_Factory(64), in_queue=Mock(), out_queue=Mock(),
        output_games=Mock(), metrics_queue=Mock(),
    )
    evaluator = Mock(side_effect=evaluate)
    monkeypatch.setattr(worker, "_evaluate_leaf", evaluator)
    game = TicTacToeGame()
    actions, policy, search = worker.search_move(0, game, None, np.random.default_rng(1), fast=False)
    action = int(actions[policy.argmax()])
    child = search.get_child_by_action(action)
    search.dispose(kept_child=child)
    game.make_move(action)
    evaluator.reset_mock()
    return worker, evaluator, game, child


def test_cached_full_search_matches_cold_search(retained_position):
    worker, evaluator, game, retained = retained_position
    raw_priors = retained.floats[0].copy()
    seed = 37

    _, _, cold = worker.search_move(0, game, None, np.random.default_rng(seed), fast=False)
    cold_requests = evaluator.call_count
    cold_target = game.history["MCTS_results"][-1]
    evaluator.reset_mock()

    _, _, cached = worker.search_move(0, game, retained, np.random.default_rng(seed), fast=False)
    # Save requests for both the promoted root and at least one descendant.
    assert 0 < evaluator.call_count <= cold_requests - 2
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


def test_fast_search_reuses_visits_and_skips_training_example(retained_position):
    worker, _, game, retained = retained_position
    old_visits = retained.visits
    raw_priors = retained.floats[0].copy()
    _, _, search = worker.search_move(0, game, retained, np.random.default_rng(2), fast=True)

    assert search.root is retained
    assert search.root.visits == max(old_visits, search.num_rollouts(fast=True) + 1)
    np.testing.assert_array_equal(search.root.floats[0], raw_priors)
    assert len(game.history["MCTS_results"]) == 1
