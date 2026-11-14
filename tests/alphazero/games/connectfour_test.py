
"""Run with: pytest -q tests/alphazero/games/connect_four_test.py"""

import copy
import random

import numpy as np
import pytest

from alphazero.games.ConnectFour import ConnectFourState, ConnectFourGame


ROWS = 6
COLS = 7


def make_state(board) -> ConnectFourState:
    return ConnectFourState(np.array(board, dtype=np.int8, copy=True))


def reference_result(board: np.ndarray) -> tuple[int, bool]:
    """Independent, comprehensive terminal-state evaluator."""
    for row in range(ROWS):
        for col in range(COLS):
            player = board[row, col]
            if player == 0:
                continue

            # Horizontal, vertical, both diagonals.
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if all(
                    0 <= row + i * dr < ROWS
                    and 0 <= col + i * dc < COLS
                    and board[row + i * dr, col + i * dc] == player
                    for i in range(4)
                ):
                    return 1, True

    if not np.any(board == 0):
        return 0, True

    return 0, False


def test_initial_state():
    state = ConnectFourGame.get_initial_state()

    assert isinstance(state, ConnectFourState)
    assert state.board.shape == (ROWS, COLS)
    assert state.board.dtype == np.int8
    assert np.all(state.board == 0)
    assert state.get_value_and_terminated() == (0, False)

    legal = state.get_legal_actions()
    assert legal.shape == (COLS,)
    assert legal.dtype == np.bool_
    assert legal.all()


@pytest.mark.parametrize("action", range(COLS))
@pytest.mark.parametrize("player", [1, -1])
def test_next_state(action: int, player: int):
    state = ConnectFourGame.get_initial_state()
    next_state = state.get_next_state(action, player)

    expected = np.zeros((ROWS, COLS), dtype=np.int8)
    expected[-1, action] = player

    np.testing.assert_array_equal(next_state.board, expected)

    # The column is still legal after one piece.
    assert next_state.get_legal_actions()[action]

    # copy=True must leave the original unchanged.
    assert next_state is not state
    assert np.all(state.board == 0)


def test_next_state_inplace():
    state = ConnectFourGame.get_initial_state()

    result = state.get_next_state(3, 1, copy=False)

    assert result is state
    assert state.board[-1, 3] == 1

    result = state.get_next_state(3, -1, copy=False)

    assert result is state
    assert state.board[-2, 3] == -1
    assert state.board[-1, 3] == 1


def test_gravity_and_full_column():
    state = ConnectFourGame.get_initial_state()

    for depth in range(ROWS):
        player = 1 if depth % 2 == 0 else -1
        state = state.get_next_state(2, player)

        expected = np.zeros(ROWS, dtype=np.int8)
        expected[ROWS - depth - 1:] = [
            1 if i % 2 == 0 else -1
            for i in range(depth, -1, -1)
        ]

        np.testing.assert_array_equal(state.board[:, 2], expected)

    expected_legal = np.ones(COLS, dtype=np.bool_)
    expected_legal[2] = False

    np.testing.assert_array_equal(
        state.get_legal_actions(), expected_legal
    )

    with pytest.raises((AssertionError, ValueError)):
        state.get_next_state(2, 1)


def test_copy():
    state = ConnectFourGame.get_initial_state()
    state = state.get_next_state(3, 1)

    copied = copy.copy(state)

    assert copied is not state
    assert copied.board is not state.board
    np.testing.assert_array_equal(copied.board, state.board)

    copied.get_next_state(4, -1, copy=False)

    assert copied.board[-1, 4] == -1
    assert state.board[-1, 4] == 0


# These are legal, alternating-player games.
# Every sequence ends with the specified player winning.
WIN_SEQUENCES = [
    ("horizontal", 1, [0, 6, 1, 6, 2, 5, 3]),
    ("horizontal", -1, [6, 0, 6, 1, 5, 2, 5, 3]),
    ("vertical", 1, [0, 6, 0, 6, 0, 5, 0]),
    ("vertical", -1, [0, 6, 0, 6, 1, 6, 1, 6]),
    (
        "diagonal_down_right", 1,
        [0, 0, 1, 1, 3, 0, 0, 2, 2, 4, 1],
    ),
    (
        "diagonal_down_right", -1,
        [2, 4, 2, 3, 2, 2, 3, 5, 1, 3, 2, 4],
    ),
    (
        "diagonal_down_left", 1,
        [2, 2, 2, 3, 0, 1, 3, 6, 1, 3, 3],
    ),
    (
        "diagonal_down_left", -1,
        [2, 0, 2, 3, 1, 6, 3, 2, 3, 3, 6, 1],
    ),
]


@pytest.mark.parametrize(
    ("direction", "winner", "actions"),
    WIN_SEQUENCES,
    ids=[f"{direction}_{winner}" for direction, winner, _ in WIN_SEQUENCES],
)
def test_all_winning_directions(direction, winner, actions):
    state = ConnectFourGame.get_initial_state()

    for i, action in enumerate(actions):
        player = 1 if i % 2 == 0 else -1
        state = state.get_next_state(action, player)

        expected = (
            (1, True) if i == len(actions) - 1
            else (0, False)
        )

        assert state.get_value_and_terminated() == expected, (
            f"{direction}, move {i}, action {action}"
        )

    assert player == winner


@pytest.mark.parametrize("player", [1, -1])
@pytest.mark.parametrize(
    "cells",
    [
        [(5, 0), (5, 1), (5, 2)],  # Three horizontally
        [(3, 0), (4, 0), (5, 0)],  # Three vertically
        [(5, 0), (4, 1), (3, 2)],  # Three diagonally
        [(5, 0), (5, 1), (5, 3), (5, 4)],  # Gap
    ],
)
def test_no_false_wins(player, cells):
    board = np.zeros((ROWS, COLS), dtype=np.int8)

    for row, col in cells:
        board[row, col] = player

    state = ConnectFourState(board)
    assert state.get_value_and_terminated() == (0, False)


# Full board, equal piece counts, no four-in-a-row.
DRAW_BOARD = np.array([
    [ 1,  1, -1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1, -1,  1],
    [ 1,  1, -1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1, -1,  1],
    [ 1,  1, -1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1, -1,  1],
], dtype=np.int8)


def test_draw_position():
    state = make_state(DRAW_BOARD)

    assert state.get_value_and_terminated() == (0, True)
    assert not state.get_legal_actions().any()

def test_full_game_draw():
    """A legal 42-move draw with alternating players."""
    actions = [
        2, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3,
        6,
        4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6,
    ]

    assert len(actions) == ROWS * COLS

    state = ConnectFourGame.get_initial_state()

    for i, action in enumerate(actions):
        player = 1 if i % 2 == 0 else -1

        assert state.get_legal_actions()[action]
        state = state.get_next_state(action, player)

        expected = (
            (0, True) if i == len(actions) - 1
            else (0, False)
        )
        assert state.get_value_and_terminated() == expected

    np.testing.assert_array_equal(state.board, DRAW_BOARD)
    assert not state.get_legal_actions().any()


@pytest.mark.parametrize("player", [1, -1])
def test_model_encoding(player: int):
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[-1, 0] = 1
    board[-1, 1] = -1
    board[-2, 0] = 1

    state = ConnectFourState(board.copy())
    encoded = state.encode_for_model(player)
    neutral = board * player

    assert encoded.shape == (3, ROWS, COLS)
    assert encoded.dtype == np.uint8

    np.testing.assert_array_equal(encoded[0], neutral == -1)
    np.testing.assert_array_equal(encoded[1], neutral == 0)
    np.testing.assert_array_equal(encoded[2], neutral == 1)

    # Exactly one plane must be active at each square.
    np.testing.assert_array_equal(
        encoded.sum(axis=0),
        np.ones((ROWS, COLS), dtype=np.uint8),
    )

    np.testing.assert_array_equal(state.board, board)


@pytest.mark.parametrize("player", [1, -1])
def test_compact_repr_roundtrip(player: int):
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    board[-1, 0] = 1
    board[-1, 1] = -1
    board[-2, 0] = 1

    state = ConnectFourState(board)

    compact = state.to_compact_repr(player)
    restored = compact.unpack()

    np.testing.assert_array_equal(
        restored,
        state.encode_for_model(player),
    )

    assert restored.dtype == np.uint8
    assert restored.shape == (3, ROWS, COLS)


def test_game_delegation():
    game = ConnectFourGame(ConnectFourGame.get_initial_state())

    assert game.get_legal_actions().sum() == COLS
    assert game.get_value_and_terminated() == (0, False)

    game._make_move(3, 1)

    assert game.state.board[-1, 3] == 1
    assert game.get_legal_actions().sum() == COLS
    assert game.get_value_and_terminated() == (0, False)


def test_random_games_against_reference():
    """Compare against independent full-board evaluation."""
    rng = random.Random(42)

    for _ in range(128):
        state = ConnectFourGame.get_initial_state()

        for turn in range(ROWS * COLS):
            assert state.get_value_and_terminated() == (
                reference_result(state.board)
            )

            legal = state.get_legal_actions()
            assert legal.dtype == np.bool_
            assert legal.shape == (COLS,)

            available = np.flatnonzero(legal).tolist()
            assert available

            action = rng.choice(available)
            player = 1 if turn % 2 == 0 else -1

            before = state.board.copy()

            # Independent gravity calculation.
            empty_rows = np.flatnonzero(before[:, action] == 0)
            assert len(empty_rows) > 0
            landing_row = int(empty_rows[-1])

            state = state.get_next_state(action, player)

            expected_board = before.copy()
            expected_board[landing_row, action] = player

            np.testing.assert_array_equal(
                state.board, expected_board
            )

            assert state.get_value_and_terminated() == (
                reference_result(state.board)
            )

            value, terminated = state.get_value_and_terminated()
            if terminated:
                break
