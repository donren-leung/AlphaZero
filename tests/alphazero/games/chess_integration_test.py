"""
Tests for the ChessState/ChessGame wrappers.
"""

import chess
import numpy as np
import pytest

from alphazero.games.Chess import ChessGame, ChessState
from alphazero.games.ChessUtils import CHESS_ACTION_SIZE, encode_move

def _player(board: chess.Board) -> int:
    return 1 if board.turn == chess.WHITE else -1

@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        chess.STARTING_FEN.replace(" w ", " b "),
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    ],
)
def test_legal_mask_matches_codec_exactly(fen: str) -> None:
    board = chess.Board(fen)
    state = ChessState(board)
    mask = state.get_legal_actions()

    expected_actions = {
        encode_move(board, move, _player(board)) for move in board.legal_moves
    }
    assert mask.shape == (CHESS_ACTION_SIZE,)
    assert mask.dtype == np.dtype(np.bool_)
    assert {int(i) for i in np.flatnonzero(mask)} == expected_actions
    assert np.count_nonzero(mask) == board.legal_moves.count()

def test_next_state_copy_does_not_mutate_original_board() -> None:
    state = ChessState(chess.Board())
    original_fen = state.state.fen()
    original_board = state.state
    action = encode_move(original_board, chess.Move.from_uci("e2e4"), 1)

    next_state = state.get_next_state(action, player=1, copy=True)

    assert state.state is original_board
    assert state.state.fen() == original_fen
    assert next_state is not state
    assert next_state.state is not original_board
    assert next_state.state.peek().uci() == "e2e4"
    assert next_state.state.turn == chess.BLACK

def test_next_state_without_copy_mutates_original_board() -> None:
    state = ChessState(chess.Board())
    original_board = state.state
    action = encode_move(original_board, chess.Move.from_uci("e2e4"), 1)

    next_state = state.get_next_state(action, player=1, copy=False)

    assert next_state.state is original_board
    assert original_board.peek().uci() == "e2e4"
    assert original_board.turn == chess.BLACK

def test_next_state_rejects_wrong_player_and_illegal_action() -> None:
    state = ChessState(chess.Board())
    initial_fen = state.state.fen()
    legal_action = encode_move(state.state, chess.Move.from_uci("e2e4"), 1)

    with pytest.raises(AssertionError):
        state.get_next_state(legal_action, player=-1)
    with pytest.raises(ValueError):
        state.get_next_state(0, player=1)
    assert state.state.fen() == initial_fen

@pytest.mark.parametrize(
    ("fen", "move_uci", "expected"),
    [
        pytest.param(
            chess.STARTING_FEN,
            "e2e4",
            (0, False),
            id="opening-e4-not-terminal",
        ),
        pytest.param(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "h5f7",
            (1, True),
            id="scholars-mate-qxf7-checkmate",
        ),
        pytest.param(
            "7k/5K2/8/6Q1/8/8/8/8 w - - 0 1",
            "g5g6",
            (0, True),
            id="qg6-stalemate",
        ),
        pytest.param(
            "4k3/8/8/8/8/8/4K3/4q3 w - - 0 1",
            "e2e1",
            (0, True),
            id="kxe1-insufficient-material",
        ),
        pytest.param(
            "r4b1r/ppp3pp/8/4p3/2Pq4/3P4/PP2QPPP/2k1K2R w K - 0 18",
            "e1g1",
            (1, True),
            id="morphys-castling-checkmate",
        ),
    ],
)
def test_state_value_and_termination_after_move(
    fen: str, move_uci: str, expected: tuple[int, bool]
) -> None:
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)

    assert not board.is_game_over(claim_draw=True)
    assert move in board.legal_moves

    player = _player(board)
    action = encode_move(board, move, player)
    next_state = ChessState(board).get_next_state(action, player)

    assert next_state.state.peek() == move
    assert next_state.get_value_and_terminated(action) == expected

def test_threefold_repetition_is_claimable_draw() -> None:
    board = chess.Board()
    for _ in range(2):
        for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
            board.push_san(san)

    assert board.can_claim_threefold_repetition()
    assert ChessState(board).get_value_and_terminated(action=0) == (0, True)

def test_fifty_move_rule_is_claimable_draw() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 100 1")
    assert board.can_claim_fifty_moves()
    assert ChessState(board).get_value_and_terminated(action=0) == (0, True)

def test_game_move_updates_state_and_keeps_starting_board_unchanged() -> None:
    start = chess.Board()
    game = ChessGame(ChessState(start))
    move = chess.Move.from_uci("e2e4")
    action = encode_move(start, move, 1)
    initial_fen = start.fen()

    assert game.get_legal_actions()[action]
    game._make_move(action, 1)

    assert start.fen() == initial_fen
    assert game.state.state.peek() == move
    assert game.state.state.turn == chess.BLACK
    assert game.get_legal_actions().shape == (CHESS_ACTION_SIZE,)
