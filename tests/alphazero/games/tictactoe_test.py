
"""Run with: pytest -q tests/alphazero/games/tictactoe_test.py"""

import copy

import numpy as np
import pytest

from alphazero.games.TicTacToe import TicTacToeState, TicTacToeGame


def make_state(board: list[list[int]]) -> TicTacToeState:
    return TicTacToeState(np.array(board, dtype=np.int8))


def test_initial_state():
    state = TicTacToeGame.get_initial_state()

    assert isinstance(state, TicTacToeState)
    assert state.board.shape == (3, 3)
    assert state.board.dtype == np.int8
    assert np.all(state.board == 0)
    assert state.get_value_and_terminated() == (0, False)


def test_legal_actions():
    state = make_state([
        [ 1,  0, -1],
        [ 0,  1,  0],
        [-1,  0,  0],
    ])

    expected = np.array([
        False, True,  False,
        True,  False, True,
        False, True,  True,
    ])

    legal = state.get_legal_actions()

    assert legal.shape == (9,)
    assert legal.dtype == np.bool_
    np.testing.assert_array_equal(legal, expected)


@pytest.mark.parametrize("action", range(9))
@pytest.mark.parametrize("player", [1, -1])
def test_next_state(action: int, player: int):
    state = TicTacToeGame.get_initial_state()
    next_state = state.get_next_state(action, player)

    expected = np.zeros((3, 3), dtype=np.int8)
    expected.flat[action] = player

    np.testing.assert_array_equal(next_state.board, expected)
    assert not next_state.get_legal_actions()[action]

    # Default copy=True must not modify the original.
    assert next_state is not state
    assert np.all(state.board == 0)


def test_next_state_inplace():
    state = TicTacToeGame.get_initial_state()

    result = state.get_next_state(4, 1, copy=False)

    assert result is state
    assert state.board[1, 1] == 1
    assert state.get_legal_actions().sum() == 8


def test_invalid_action():
    state = TicTacToeGame.get_initial_state()
    state.get_next_state(4, 1, copy=False)

    with pytest.raises(AssertionError):
        state.get_next_state(4, -1)


def test_copy():
    state = make_state([
        [ 1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  0],
    ])

    copied = copy.copy(state)

    assert copied is not state
    assert copied.board is not state.board
    np.testing.assert_array_equal(copied.board, state.board)

    copied.get_next_state(1, 1, copy=False)
    assert state.board[0, 1] == 0


def test_full_game_win_player1():
    state = TicTacToeGame.get_initial_state()

    # X wins across the top row.
    actions = [0, 3, 1, 4, 2]

    for i, action in enumerate(actions):
        player = 1 if i % 2 == 0 else -1
        state = state.get_next_state(action, player)

        expected = (1, True) if i == len(actions) - 1 else (0, False)
        assert state.get_value_and_terminated() == expected


def test_full_game_win_player2():
    state = TicTacToeGame.get_initial_state()

    # O wins across the top row.
    actions = [6, 0, 3, 1, 4, 2]

    for i, action in enumerate(actions):
        player = 1 if i % 2 == 0 else -1
        state = state.get_next_state(action, player)

        expected = (1, True) if i == len(actions) - 1 else (0, False)
        assert state.get_value_and_terminated() == expected


def test_full_game_draw():
    state = TicTacToeGame.get_initial_state()

    actions = [0, 1, 2, 4, 3, 5, 7, 6, 8]

    for i, action in enumerate(actions):
        player = 1 if i % 2 == 0 else -1
        state = state.get_next_state(action, player)

        expected = (0, True) if i == len(actions) - 1 else (0, False)
        assert state.get_value_and_terminated() == expected

    assert not state.get_legal_actions().any()


@pytest.mark.parametrize("player", [1, -1])
def test_model_encoding(player: int):
    state = make_state([
        [ 1,  0, -1],
        [ 0, -1,  1],
        [ 0,  0,  0],
    ])

    original = state.board.copy()
    encoded = state.encode_for_model(player)
    neutral = original * player

    assert encoded.shape == (3, 3, 3)
    assert encoded.dtype == np.uint8

    np.testing.assert_array_equal(encoded[0], neutral == -1)
    np.testing.assert_array_equal(encoded[1], neutral == 0)
    np.testing.assert_array_equal(encoded[2], neutral == 1)

    # Exactly one plane should be active at every square.
    np.testing.assert_array_equal(
        encoded.sum(axis=0),
        np.ones((3, 3)),
    )

    # Encoding must not modify the original state.
    np.testing.assert_array_equal(state.board, original)


@pytest.mark.parametrize("player", [1, -1])
def test_compact_repr_roundtrip(player: int):
    state = make_state([
        [ 1,  0, -1],
        [ 0,  1,  0],
        [-1,  0,  0],
    ])

    compact = state.to_compact_repr(player)
    restored = compact.unpack()
    expected = state.encode_for_model(player)

    np.testing.assert_array_equal(restored, expected)
    assert restored.dtype == np.uint8
    assert restored.shape == (3, 3, 3)


def test_game_delegation():
    game = TicTacToeGame(TicTacToeGame.get_initial_state())

    assert game.get_legal_actions().sum() == 9
    assert game.get_value_and_terminated() == (0, False)

    game._make_move(0, 1)

    assert game.state.board[0, 0] == 1
    assert game.get_legal_actions().sum() == 8
    assert game.get_value_and_terminated() == (0, False)
